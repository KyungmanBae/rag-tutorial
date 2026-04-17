"""
===================================================
RAG 실습 Step2: 외부 검색 + LLM 생성
- 검색기 1: DuckDuckGo (실시간 웹 검색)
- 검색기 2: Wikipedia   (백과사전 문서 검색)
- 각 검색기 Top-5 결과를 UI에 표시
- 원하는 결과를 체크박스로 선택 → LLM에 컨텍스트로 전달 → 답변 생성

[UI 구성]
  ┌─ 상단: 모델 로딩 영역 ──────────────────────┐
  │  로컬 모델 드롭다운 / 양자화 / 로딩 버튼      │
  └────────────────────────────────────────────┘
  ┌─ 하단: 외부 검색 + 생성 영역 ───────────────┐
  │  질문 입력 → 두 검색기 동시 실행             │
  │  DuckDuckGo Top5 | Wikipedia Top5 표시      │
  │  체크박스로 원하는 결과 선택                  │
  │  → 선택된 컨텍스트 + 질문을 LLM에 전달       │
  │  → 생성 결과 출력                            │
  └────────────────────────────────────────────┘

[필수 설치]
pip install gradio transformers huggingface_hub
pip install bitsandbytes accelerate
pip install ddgs
pip install wikipedia-api
pip install torch --index-url https://download.pytorch.org/whl/cu118
===================================================
"""

import os
import time
import threading
import gradio as gr
import torch

# ── Gradio 버전 호환성 유틸 (./colab/gradio_compat.py) ──
# Colab 환경 감지(share=True 자동) + Gradio 버전별 theme 위치 자동 처리
from colab.gradio_compat import make_blocks, safe_launch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)

# 외부 검색 라이브러리
from ddgs import DDGS
import wikipediaapi

# ──────────────────────────────────────────────
# 전역 상태
# ──────────────────────────────────────────────
LOCAL_MODEL_ROOT = "./hf_model"  # 로컬 모델 저장 루트 폴더

# 로딩된 모델 정보: {"model_id", "pipe", "status"} 또는 None
current_model = {"info": None}

# Wikipedia API 클라이언트 (한국어 우선, 영어 fallback)
wiki_ko = wikipediaapi.Wikipedia(language="ko", user_agent="rag-practice/1.0")
wiki_en = wikipediaapi.Wikipedia(language="en", user_agent="rag-practice/1.0")

# 마지막 검색 결과를 저장 (체크박스 선택 시 본문 재사용)
# {"ddg": [...], "wiki": [...]}  각 항목: {"title", "source", "body"}
last_search_results = {"ddg": [], "wiki": []}


# ──────────────────────────────────────────────
# [유틸] GPU / CPU 정보
# ──────────────────────────────────────────────
def get_device_info():
    if torch.cuda.is_available():
        name  = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        free  = (torch.cuda.get_device_properties(0).total_memory
                 - torch.cuda.memory_allocated(0)) / (1024**3)
        return f"GPU: {name} | 전체: {total:.1f}GB | 여유: {free:.1f}GB"
    return "GPU 없음 — CPU 모드"


# ──────────────────────────────────────────────
# [유틸] 로컬 모델 폴더 스캔
#   config.json이 있는 폴더를 유효한 모델로 판단
# ──────────────────────────────────────────────
def scan_local_models(root: str = LOCAL_MODEL_ROOT) -> list:
    if not os.path.isdir(root):
        return []
    found = []
    for entry in sorted(os.scandir(root), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        if os.path.isfile(os.path.join(entry.path, "config.json")):
            found.append(entry.path)
        else:
            # 조직명/모델명 2단계 구조 대응
            for sub in sorted(os.scandir(entry.path), key=lambda e: e.name):
                if sub.is_dir() and os.path.isfile(os.path.join(sub.path, "config.json")):
                    found.append(sub.path)
    return found


# ──────────────────────────────────────────────
# 1. 모델 로딩
# ──────────────────────────────────────────────
def load_model(model_path, quant_mode, trust_remote):
    """
    model_path  : 로컬 폴더 경로 또는 HF Hub 모델 ID
    quant_mode  : '4bit' | '8bit' | 'fp16' | 'fp32'
    trust_remote: trust_remote_code 여부
    """
    if not model_path or not model_path.strip():
        return "⚠️ 모델 경로를 선택하거나 입력하세요."

    # 기존 모델 언로드
    if current_model["info"]:
        old = current_model["info"]
        try:
            del old["pipe"].model
            del old["pipe"]
        except Exception:
            pass
        current_model["info"] = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    src = model_path.strip()
    try:
        # ── 양자화 설정 ──────────────────────────
        quant_cfg = None
        dtype = torch.float32
        if quant_mode == "4bit":
            # NF4 양자화: RTX 2080 VRAM 절약에 효과적
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif quant_mode == "8bit":
            quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
        elif quant_mode == "fp16":
            dtype = torch.float16

        # ── 토크나이저 ───────────────────────────
        tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=trust_remote)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ── 모델 ─────────────────────────────────
        model_kwargs = {"trust_remote_code": trust_remote, "device_map": "auto"}
        if quant_cfg:
            model_kwargs["quantization_config"] = quant_cfg
        else:
            model_kwargs["torch_dtype"] = dtype

        model = AutoModelForCausalLM.from_pretrained(src, **model_kwargs)
        model.eval()

        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

        model_name = os.path.basename(src) if os.path.isdir(src) else src
        current_model["info"] = {"model_id": model_name, "pipe": pipe, "status": quant_mode}

        return f"✅ 로딩 완료: {model_name} ({quant_mode})\n{get_device_info()}"

    except Exception as e:
        return f"❌ 로딩 오류:\n{e}"


def unload_model():
    """모델 언로드 및 GPU 메모리 해제"""
    if not current_model["info"]:
        return "⚠️ 로딩된 모델이 없습니다."
    name = current_model["info"]["model_id"]
    try:
        del current_model["info"]["pipe"].model
        del current_model["info"]["pipe"]
    except Exception:
        pass
    current_model["info"] = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return f"✅ 언로드 완료: {name}"


def refresh_model_list():
    """로컬 모델 폴더 재스캔"""
    paths = scan_local_models()
    msg = f"✅ {len(paths)}개 모델 발견" if paths else f"⚠️ {LOCAL_MODEL_ROOT} 에 모델 없음"
    return gr.Dropdown(choices=paths, value=None), msg


# ──────────────────────────────────────────────
# 2. DuckDuckGo 검색
# ──────────────────────────────────────────────
def search_duckduckgo(query: str, top_k: int = 5) -> list:
    """
    DuckDuckGo로 웹을 검색합니다.
    반환: [{"title": str, "source": str(URL), "body": str}, ...]
    """
    results = []
    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=top_k))
        for r in raw:
            results.append({
                "title":  r.get("title", "(제목 없음)"),
                "source": r.get("href",  ""),
                "body":   r.get("body",  ""),
            })
    except Exception as e:
        results.append({"title": f"DuckDuckGo 오류: {e}", "source": "", "body": ""})
    return results


# ──────────────────────────────────────────────
# 3. Wikipedia 검색
# ──────────────────────────────────────────────
def search_wikipedia(query: str, top_k: int = 5) -> list:
    """
    Wikipedia에서 쿼리와 관련된 페이지를 검색합니다.
    한국어 Wikipedia를 우선 시도하고, 결과가 없으면 영어로 fallback합니다.
    반환: [{"title": str, "source": str(URL), "body": str(요약문)}, ...]
    """
    results = []
    try:
        # Wikipedia 검색: search API로 관련 페이지 제목 목록 획득
        # wikipediaapi는 직접 검색 API가 없으므로 requests로 검색 후 페이지 로딩
        import requests

        # MediaWiki API를 이용한 제목 검색 (무료, 인증 불필요)
        def mw_search(lang: str, q: str, limit: int) -> list:
            url = f"https://{lang}.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "list":   "search",
                "srsearch": q,
                "srlimit": limit,
                "format": "json",
            }
            # User-Agent 필수: 없으면 Wikipedia가 빈 응답 또는 차단 반환
            headers = {"User-Agent": "rag-practice/1.0 (https://github.com/example)"}
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            # 응답이 비어있거나 JSON이 아닌 경우 방어
            if resp.status_code != 200 or not resp.text.strip():
                return []
            try:
                data = resp.json()
            except Exception:
                return []
            return [item["title"] for item in data.get("query", {}).get("search", [])]

        # 한국어 검색 시도
        titles = mw_search("ko", query, top_k)
        wiki_client = wiki_ko
        lang = "ko"

        # 한국어 결과 부족하면 영어 보충
        if len(titles) < top_k:
            en_titles = mw_search("en", query, top_k - len(titles))
            titles += en_titles
            # 영어 결과는 영어 클라이언트로 처리 (아래에서 구분)

        for title in titles[:top_k]:
            # 한국어 페이지 시도
            page = wiki_ko.page(title)
            if not page.exists():
                # 영어 페이지 시도
                page = wiki_en.page(title)
            if page.exists():
                # 요약문 최대 500자 사용 (너무 길면 LLM 컨텍스트 초과)
                summary = page.summary[:500] if len(page.summary) > 500 else page.summary
                results.append({
                    "title":  page.title,
                    "source": page.fullurl,
                    "body":   summary,
                })
            else:
                results.append({
                    "title":  title,
                    "source": "",
                    "body":   "(페이지를 찾을 수 없습니다.)",
                })
            time.sleep(0.1)  # Wikipedia API 요청 간격 준수

    except Exception as e:
        results.append({"title": f"Wikipedia 오류: {e}", "source": "", "body": ""})
    return results


# ──────────────────────────────────────────────
# 4. 검색 실행 (DuckDuckGo + Wikipedia 동시)
# ──────────────────────────────────────────────
def run_search(query: str, top_k: int = 5):
    """
    질문으로 두 검색기를 동시에 실행하고
    각각 Top-K 결과를 체크박스 목록으로 반환합니다.
    검색 후 모든 결과를 기본 선택 상태로 설정합니다.

    Returns:
        ddg_checks   : DuckDuckGo 체크박스 (전체 선택)
        wiki_checks  : Wikipedia  체크박스 (전체 선택)
        status_msg   : 검색 상태 메시지
    """
    if not query.strip():
        return gr.CheckboxGroup(choices=[]), gr.CheckboxGroup(choices=[]), "⚠️ 질문을 입력하세요."

    top_k = int(top_k)
    ddg_results  = []
    wiki_results = []

    def fetch_ddg():
        ddg_results.extend(search_duckduckgo(query, top_k=top_k))

    def fetch_wiki():
        wiki_results.extend(search_wikipedia(query, top_k=top_k))

    # 두 검색기를 스레드로 동시 실행
    t1 = threading.Thread(target=fetch_ddg)
    t2 = threading.Thread(target=fetch_wiki)
    t1.start(); t2.start()
    t1.join();  t2.join()

    # 전역 저장 (생성 단계에서 본문 재사용)
    last_search_results["ddg"]  = ddg_results
    last_search_results["wiki"] = wiki_results

    # 체크박스 레이블 생성: "[PREFIX] 번호. 제목\n   출처" 형태
    def make_labels(results, prefix):
        labels = []
        for i, r in enumerate(results):
            source_short = r["source"][:60] + "..." if len(r["source"]) > 60 else r["source"]
            labels.append(f"[{prefix}] {i+1}. {r['title']}\n   {source_short}")
        return labels

    ddg_labels  = make_labels(ddg_results,  "DDG")
    wiki_labels = make_labels(wiki_results, "WIKI")

    status = (f"✅ 검색 완료 — "
              f"DuckDuckGo: {len(ddg_results)}건 / Wikipedia: {len(wiki_results)}건")

    # value=labels 로 설정 → 검색 후 전체 선택 상태
    return (
        gr.CheckboxGroup(choices=ddg_labels,  value=ddg_labels,  label="DuckDuckGo 결과 (선택)"),
        gr.CheckboxGroup(choices=wiki_labels, value=wiki_labels, label="Wikipedia 결과 (선택)"),
        status,
    )


# ──────────────────────────────────────────────
# 5. 선택된 검색 결과 미리보기
# ──────────────────────────────────────────────
def preview_selected(ddg_selected: list, wiki_selected: list) -> str:
    """
    체크박스에서 선택된 항목의 본문을 미리보기로 반환합니다.
    LLM에 전달될 컨텍스트를 사용자가 확인할 수 있습니다.
    """
    lines = []

    # DuckDuckGo 선택 항목 본문 추출
    for label in ddg_selected:
        # 레이블에서 번호 파싱: "[DDG] 1. 제목..." → 인덱스 0
        try:
            idx = int(label.split("]")[1].strip().split(".")[0]) - 1
            r = last_search_results["ddg"][idx]
            lines.append(f"[DuckDuckGo] {r['title']}\n{r['body']}")
        except Exception:
            lines.append(label)

    # Wikipedia 선택 항목 본문 추출
    for label in wiki_selected:
        try:
            idx = int(label.split("]")[1].strip().split(".")[0]) - 1
            r = last_search_results["wiki"][idx]
            lines.append(f"[Wikipedia] {r['title']}\n{r['body']}")
        except Exception:
            lines.append(label)

    if not lines:
        return "선택된 검색 결과가 없습니다. 위에서 항목을 선택하세요."
    return "\n\n---\n\n".join(lines)


# ──────────────────────────────────────────────
# 6. RAG 생성 (선택된 컨텍스트 + 질문 → LLM)
# ──────────────────────────────────────────────
def generate_with_context(
    query: str,
    ddg_selected: list,
    wiki_selected: list,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
):
    """
    선택된 검색 결과를 컨텍스트로 조합하여 LLM에 전달하고 답변을 생성합니다.

    RAG 프롬프트 구조:
      [참고 정보]
      컨텍스트1 ...
      컨텍스트2 ...
      [질문]
      사용자 질문
      [답변]
    """
    if not current_model["info"]:
        return "⚠️ 모델이 로딩되지 않았습니다. 상단의 모델 로딩 영역에서 먼저 모델을 로딩하세요."
    if not query.strip():
        return "⚠️ 질문을 입력하세요."

    # 선택된 컨텍스트 본문 수집
    context_parts = []

    for label in ddg_selected:
        try:
            idx = int(label.split("]")[1].strip().split(".")[0]) - 1
            r = last_search_results["ddg"][idx]
            context_parts.append(f"[출처: {r['title']}]\n{r['body']}")
        except Exception:
            pass

    for label in wiki_selected:
        try:
            idx = int(label.split("]")[1].strip().split(".")[0]) - 1
            r = last_search_results["wiki"][idx]
            context_parts.append(f"[출처: {r['title']}]\n{r['body']}")
        except Exception:
            pass

    # ── RAG 프롬프트 구성 ─────────────────────
    if context_parts:
        context_text = "\n\n".join(context_parts)
        prompt = (
            "다음 참고 정보를 바탕으로 질문에 답하세요.\n\n"
            f"[참고 정보]\n{context_text}\n\n"
            f"[질문]\n{query}\n\n"
            "[답변]\n"
        )
    else:
        # 검색 결과 미선택 시 순수 LLM 생성 (RAG 없음)
        prompt = f"[질문]\n{query}\n\n[답변]\n"

    # ── LLM 추론 ─────────────────────────────
    try:
        pipe = current_model["info"]["pipe"]
        out = pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            do_sample=do_sample,
            pad_token_id=pipe.tokenizer.eos_token_id,
            return_full_text=False,  # 프롬프트 제외하고 생성 부분만 반환
        )
        answer = out[0]["generated_text"].strip()

        # 사용된 컨텍스트 수 정보 첨부
        ctx_count = len(context_parts)
        suffix = f"\n\n─────\n📚 사용된 컨텍스트: {ctx_count}건" if ctx_count else "\n\n─────\n⚠️ 컨텍스트 없이 생성됨 (순수 LLM)"
        return answer + suffix

    except Exception as e:
        return f"❌ 생성 오류: {e}"


# ──────────────────────────────────────────────
# 7. Gradio UI
# ──────────────────────────────────────────────
def build_ui():
    # make_blocks(): Gradio 버전에 따라 theme 위치를 자동 조정
    with make_blocks(title="외부 검색 RAG 실습", theme=gr.themes.Soft()) as demo:

        gr.Markdown("# 🔍 외부 검색 RAG 실습 — Step 2\nDuckDuckGo + Wikipedia 검색 결과를 선택하여 LLM 답변 생성")

        # ════════════════════════════════════════
        # 상단: 모델 로딩 영역
        # ════════════════════════════════════════
        with gr.Group():
            gr.Markdown("## 📦 모델 로딩")

            # 디바이스 정보
            gr.Textbox(
                label="디바이스 정보",
                value=get_device_info,
                interactive=False,
                every=30,
            )

            with gr.Row():
                # 로컬 모델 드롭다운
                model_dd = gr.Dropdown(
                    label=f"📂 로컬 모델 선택 ({LOCAL_MODEL_ROOT})",
                    choices=scan_local_models(),
                    interactive=True,
                    allow_custom_value=True,   # HF Hub ID 직접 입력도 허용
                    scale=5,
                    info="드롭다운 선택 또는 HF Hub 모델 ID 직접 입력",
                )
                refresh_btn = gr.Button("🔄 새로고침", scale=1, min_width=90)
                scan_status = gr.Textbox(label="스캔 결과", interactive=False, scale=2)

            with gr.Row():
                quant_mode = gr.Radio(
                    choices=["4bit", "8bit", "fp16", "fp32"],
                    value="4bit",
                    label="양자화 모드",
                    info="RTX 2080 → 4bit 권장",
                    scale=3,
                )
                trust_remote = gr.Checkbox(
                    label="trust_remote_code",
                    value=False,
                    info="EXAONE 등 커스텀 코드 모델",
                    scale=1,
                )

            with gr.Row():
                load_btn   = gr.Button("🚀 모델 로딩", variant="primary", scale=2)
                unload_btn = gr.Button("🗑️ 언로드",   variant="stop",    scale=1)

            load_status = gr.Textbox(label="로딩 상태", lines=2, interactive=False)

            # 이벤트 연결
            refresh_btn.click(fn=refresh_model_list, outputs=[model_dd, scan_status])
            load_btn.click(
                fn=load_model,
                inputs=[model_dd, quant_mode, trust_remote],
                outputs=[load_status],
            )
            unload_btn.click(fn=unload_model, outputs=[load_status])

        gr.Markdown("---")

        # ════════════════════════════════════════
        # 하단: 외부 검색 + RAG 생성 영역
        # ════════════════════════════════════════
        gr.Markdown("## 🌐 외부 검색 + RAG 생성")

        # ── 질문 입력 & 검색 실행 ────────────────
        with gr.Row():
            query_box = gr.Textbox(
                label="질문 입력  ※ 줄바꿈: Shift+Enter",
                placeholder="예: 인공지능의 역사는?",
                lines=3,
                max_lines=10,
                submit_btn=False,
                scale=5,
            )
            with gr.Column(scale=1, min_width=120):
                top_k_slider = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1,
                    label="검색 결과 수 (Top-K)",
                    info="각 검색기당 가져올 결과 수",
                )
                search_run_btn = gr.Button(
                    "🔍 검색 실행",
                    variant="primary",
                    min_width=100,
                )

        search_status = gr.Textbox(label="검색 상태", interactive=False, lines=1)

        # ── 검색 결과: DuckDuckGo | Wikipedia 나란히 ─
        gr.Markdown("### 검색 결과 — 컨텍스트로 사용할 항목을 선택하세요")
        with gr.Row():
            ddg_checks = gr.CheckboxGroup(
                label="🦆 DuckDuckGo Top-5",
                choices=[],
                interactive=True,
                scale=1,
            )
            wiki_checks = gr.CheckboxGroup(
                label="📖 Wikipedia Top-5",
                choices=[],
                interactive=True,
                scale=1,
            )

        # ── 선택된 컨텍스트 미리보기 ─────────────
        with gr.Accordion("📄 선택된 컨텍스트 미리보기 (LLM에 전달될 내용)", open=False):
            preview_btn = gr.Button("미리보기 갱신", size="sm")
            context_preview = gr.Textbox(
                label="컨텍스트 내용",
                lines=8,
                interactive=False,
            )

        gr.Markdown("### 생성 설정")

        # ── 생성 파라미터 ─────────────────────────
        with gr.Row():
            max_new_tokens = gr.Slider(
                32, 512, value=256, step=32,
                label="max_new_tokens",
                info="생성할 최대 토큰 수",
            )
            temperature = gr.Slider(
                0.1, 2.0, value=0.7, step=0.1,
                label="temperature",
            )
            do_sample = gr.Checkbox(label="do_sample", value=True)

        generate_btn = gr.Button("✨ RAG 생성", variant="primary", size="lg")

        # ── 생성 결과 출력 ────────────────────────
        output_box = gr.Textbox(
            label="📝 생성 결과",
            lines=12,
            interactive=False,
        )

        # ────────────────────────────────────────
        # 이벤트 연결
        # ────────────────────────────────────────

        # 검색 실행 → 두 체크박스 그룹 업데이트
        search_run_btn.click(
            fn=run_search,
            inputs=[query_box, top_k_slider],
            outputs=[ddg_checks, wiki_checks, search_status],
        )

        # 미리보기 갱신
        preview_btn.click(
            fn=preview_selected,
            inputs=[ddg_checks, wiki_checks],
            outputs=[context_preview],
        )

        # 생성 실행
        generate_btn.click(
            fn=generate_with_context,
            inputs=[query_box, ddg_checks, wiki_checks,
                    max_new_tokens, temperature, do_sample],
            outputs=[output_box],
        )

    return demo


# ──────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(LOCAL_MODEL_ROOT, exist_ok=True)
    demo = build_ui()
    # safe_launch(): Gradio 버전 및 Colab 환경을 자동 감지하여 안전하게 실행
    safe_launch(
        demo,
        theme=gr.themes.Soft(),
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
