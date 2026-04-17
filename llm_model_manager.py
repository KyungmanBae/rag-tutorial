"""
===================================================
RAG 실습 Step1: LLM 기반 생성
- 허깅페이스 모델 검색 / 다운로드 / 로딩
- 최대 3개 모델 동시 추론
- RTX 2080 (VRAM ~8~11GB) 환경 대응: 4bit/8bit 양자화 지원
- Gradio 6.0 호환 버전

[UI 변경 사항]
- 검색 결과 선택 시 로컬 저장 경로 자동 완성 (./hf_model/<모델명>)
- 모델 로딩 슬롯: 로컬 폴더(./hf_model) 모델 목록 드롭다운 제공
===================================================
[필수 설치]
pip install gradio transformers huggingface_hub
pip install bitsandbytes accelerate
pip install torch --index-url https://download.pytorch.org/whl/cu118
"""

import os
import threading
import gradio as gr
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
from huggingface_hub import HfApi, snapshot_download

# ──────────────────────────────────────────────
# 전역 상태: 로딩된 모델을 슬롯(0~2) 별로 관리
# ──────────────────────────────────────────────
MAX_SLOTS = 3                      # 동시에 올릴 수 있는 최대 모델 수
LOCAL_MODEL_ROOT = "./hf_model"    # 로컬 모델 저장 기본 루트 폴더

# 각 슬롯: {"model_id": str, "pipe": pipeline 객체, "status": str}
loaded_models = {i: None for i in range(MAX_SLOTS)}

# 허깅페이스 API 클라이언트 (모델 검색용)
hf_api = HfApi()


# ──────────────────────────────────────────────
# [유틸] GPU / CPU 정보 문자열 반환
# ──────────────────────────────────────────────
def get_device_info():
    if torch.cuda.is_available():
        name  = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        free  = (torch.cuda.get_device_properties(0).total_memory
                 - torch.cuda.memory_allocated(0)) / (1024**3)
        return f"GPU: {name} | 전체: {total:.1f}GB | 여유: {free:.1f}GB"
    return "GPU 없음 - CPU 모드로 동작합니다"


# ──────────────────────────────────────────────
# [유틸] 현재 로딩된 모델 슬롯 요약 문자열
# ──────────────────────────────────────────────
def get_loaded_summary():
    lines = []
    for i in range(MAX_SLOTS):
        info = loaded_models[i]
        if info:
            lines.append(f"슬롯 {i+1}: [로딩됨] {info['model_id']} ({info['status']})")
        else:
            lines.append(f"슬롯 {i+1}: [비어있음]")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# [유틸] 로컬 모델 폴더 목록 스캔
#   LOCAL_MODEL_ROOT 아래 하위 디렉토리 중
#   config.json 이 있는 폴더만 유효한 모델로 판단
# ──────────────────────────────────────────────
def scan_local_models(root: str = LOCAL_MODEL_ROOT) -> list[str]:
    """
    root 폴더 아래에서 HuggingFace 모델 폴더를 재귀적으로 탐색합니다.
    config.json 파일이 존재하는 폴더를 유효한 모델로 간주합니다.

    Returns:
        유효한 모델 폴더 절대경로 리스트
    """
    if not os.path.isdir(root):
        return []

    found = []
    for entry in sorted(os.scandir(root), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        # config.json 존재 여부로 모델 폴더 판단
        if os.path.isfile(os.path.join(entry.path, "config.json")):
            found.append(entry.path)
        else:
            # 한 단계 더 탐색 (조직명/모델명 구조 대응: ./hf_model/meta-llama/Llama-3-8B)
            for sub in sorted(os.scandir(entry.path), key=lambda e: e.name):
                if sub.is_dir() and os.path.isfile(os.path.join(sub.path, "config.json")):
                    found.append(sub.path)
    return found


# ──────────────────────────────────────────────
# [유틸] 모델 ID → 로컬 저장 경로 자동 생성
#   "mistralai/Mistral-7B-Instruct-v0.1"
#   → "./hf_model/mistralai--Mistral-7B-Instruct-v0.1"
#   슬래시(/)는 '--' 로 치환하여 단일 폴더명으로 사용
# ──────────────────────────────────────────────
def model_id_to_local_path(model_id: str) -> str:
    """모델 ID를 로컬 폴더 경로로 변환합니다."""
    if not model_id or not model_id.strip():
        return ""
    safe_name = model_id.strip().replace("/", "--")
    return os.path.join(LOCAL_MODEL_ROOT, safe_name)


# ──────────────────────────────────────────────
# 1. 허깅페이스 모델 검색
# ──────────────────────────────────────────────
def search_models(query, max_results=20):
    """
    HF Hub에서 모델을 검색합니다.

    [수정 이유]
    huggingface_hub >= 0.24 이후 list_models()에서
      - task=      파라미터가 삭제됨 → 제거
      - direction= 파라미터가 deprecated → 제거
    대신 search= 키워드로 모델명을 필터링한 뒤,
    결과의 pipeline_tag 속성으로 text-generation 모델만 클라이언트 측에서 필터링합니다.

    Returns:
        (Dropdown 업데이트, 상태 문자열)
    """
    if not query.strip():
        return gr.Dropdown(choices=[], value=None), "검색어를 입력하세요."
    try:
        raw = list(hf_api.list_models(
            search=query,
            sort="downloads",
            limit=max_results * 3,  # 필터 후 max_results 확보를 위해 넉넉히
        ))

        # pipeline_tag == "text-generation" 인 모델만 선택
        models = [
            m for m in raw
            if getattr(m, "pipeline_tag", None) == "text-generation"
        ][:max_results]

        if not models:
            models = raw[:max_results]  # 필터 결과 없으면 전체 반환

        # id 또는 modelId 속성 (버전별 차이 대응)
        choices = [getattr(m, "id", getattr(m, "modelId", str(m))) for m in models]
        return gr.Dropdown(choices=choices, value=None), f"✅ {len(choices)}개 모델 검색됨"

    except Exception as e:
        return gr.Dropdown(choices=[], value=None), f"❌ 검색 오류: {e}"


# ──────────────────────────────────────────────
# [콜백] 드롭다운 선택 → 모델 ID 박스 + 로컬 경로 박스 동시 업데이트
# ──────────────────────────────────────────────
def on_model_selected(model_id: str):
    """
    검색 결과 드롭다운에서 모델을 선택하면
      1. 선택된 모델 ID 텍스트박스에 값 반영
      2. 로컬 저장 경로를 ./hf_model/<모델명> 형식으로 자동 생성
    """
    local_path = model_id_to_local_path(model_id)
    return model_id, local_path   # (selected_model_box, local_save_dir)


# ──────────────────────────────────────────────
# 2. 모델 다운로드
# ──────────────────────────────────────────────
def download_model(model_id, save_type, local_dir):
    """
    model_id  : 허깅페이스 모델 경로 (예: 'mistralai/Mistral-7B-Instruct-v0.1')
    save_type : 'cache' → HF 기본 캐시 / 'local' → local_dir 폴더에 저장
    local_dir : save_type='local' 일 때 저장 폴더 경로
    """
    if not model_id or not model_id.strip():
        return "⚠️ 다운로드할 모델을 선택하거나 입력하세요."
    try:
        kwargs = {"repo_id": model_id.strip(), "repo_type": "model"}
        if save_type == "local":
            if not local_dir.strip():
                return "⚠️ 저장 폴더 경로를 입력하세요."
            kwargs["local_dir"] = local_dir.strip()
            # 저장 폴더가 없으면 미리 생성
            os.makedirs(local_dir.strip(), exist_ok=True)
        # snapshot_download: 파일만 내려받고 로딩은 별도 → 오류 원인 분리에 유리
        path = snapshot_download(**kwargs)
        return f"✅ 다운로드 완료!\n저장 경로: {path}"
    except Exception as e:
        return f"❌ 다운로드 오류: {e}"


# ──────────────────────────────────────────────
# [콜백] 로컬 모델 폴더 목록 새로고침
# ──────────────────────────────────────────────
def refresh_local_models():
    """
    ./hf_model 폴더를 스캔하여 드롭다운 목록을 반환합니다.
    모델 로딩 슬롯의 '로컬 모델 새로고침' 버튼에 연결됩니다.
    """
    paths = scan_local_models(LOCAL_MODEL_ROOT)
    if not paths:
        return gr.Dropdown(choices=[], value=None), f"⚠️ {LOCAL_MODEL_ROOT} 에 모델이 없습니다."
    return gr.Dropdown(choices=paths, value=None), f"✅ {len(paths)}개 로컬 모델 발견"


# ──────────────────────────────────────────────
# 3. 모델 로딩 (슬롯에 적재)
# ──────────────────────────────────────────────
def load_model_to_slot(model_id, slot_idx, quant_mode, local_path, trust_remote):
    """
    model_id    : HF 모델 ID (local_path 비어있을 때 HF Hub에서 로딩)
    slot_idx    : 0~2 (UI에서 1~3으로 표시)
    quant_mode  : '4bit' | '8bit' | 'fp16' | 'fp32'
    local_path  : 로컬 폴더 경로 (입력 시 로컬 우선, 비우면 HF Hub 사용)
    trust_remote: trust_remote_code 옵션 (EXAONE 등 커스텀 코드 모델)
    """
    src = local_path.strip() if local_path.strip() else model_id.strip()
    if not src:
        return "⚠️ 모델 ID 또는 로컬 경로를 입력하세요."

    if loaded_models[slot_idx]:
        unload_model(slot_idx)

    try:
        # ── 양자화 설정 ──────────────────────────
        quant_cfg = None
        dtype = torch.float32
        if quant_mode == "4bit":
            # NF4 양자화: RTX 2080 VRAM 절약에 효과적
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,  # 이중 양자화로 추가 절약
                bnb_4bit_quant_type="nf4",
            )
        elif quant_mode == "8bit":
            quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
        elif quant_mode == "fp16":
            dtype = torch.float16

        # ── 토크나이저 로딩 ──────────────────────
        tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=trust_remote)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # GPT 계열 대비

        # ── 모델 로딩 ────────────────────────────
        model_kwargs = {"trust_remote_code": trust_remote, "device_map": "auto"}
        if quant_cfg:
            model_kwargs["quantization_config"] = quant_cfg
        else:
            model_kwargs["torch_dtype"] = dtype

        model = AutoModelForCausalLM.from_pretrained(src, **model_kwargs)
        model.eval()

        # ── pipeline 생성 ────────────────────────
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

        loaded_models[slot_idx] = {
            "model_id": os.path.basename(src) if os.path.isdir(src) else src,
            "pipe": pipe,
            "status": quant_mode,
        }
        return f"✅ 슬롯 {slot_idx+1} 로딩 완료: {src} ({quant_mode})\n\n{get_loaded_summary()}"

    except Exception as e:
        return f"❌ 슬롯 {slot_idx+1} 로딩 오류:\n{e}"


# ──────────────────────────────────────────────
# 4. 모델 언로드 (슬롯 비우기 + GPU 메모리 해제)
# ──────────────────────────────────────────────
def unload_model(slot_idx):
    info = loaded_models[slot_idx]
    if not info:
        return f"슬롯 {slot_idx+1}은 이미 비어있습니다."
    name = info["model_id"]
    del info["pipe"].model
    del info["pipe"]
    loaded_models[slot_idx] = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return f"✅ 슬롯 {slot_idx+1} ({name}) 언로드 완료\n\n{get_loaded_summary()}"


# ──────────────────────────────────────────────
# 5. 단일 모델 추론 (내부 헬퍼)
# ──────────────────────────────────────────────
def _infer_one(slot_idx, prompt, max_new_tokens, temperature, do_sample):
    info = loaded_models[slot_idx]
    if not info:
        return f"[슬롯 {slot_idx+1}] 모델 없음"
    try:
        pipe = info["pipe"]
        out = pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            do_sample=do_sample,
            pad_token_id=pipe.tokenizer.eos_token_id,
            return_full_text=False,  # 입력 프롬프트를 결과에서 제거
        )
        return out[0]["generated_text"]
    except Exception as e:
        return f"[슬롯 {slot_idx+1}] ❌ 추론 오류: {e}"


# ──────────────────────────────────────────────
# 6. 멀티 모델 동시 추론 (스레드 병렬 실행)
# ──────────────────────────────────────────────
def run_all_models(prompt, max_new_tokens, temperature, do_sample):
    """
    로딩된 슬롯만 Thread로 동시 실행하여 결과를 비교합니다.
    비어있는 슬롯은 실행을 건너뛰고 안내 메시지를 반환합니다.
    """
    if not prompt.strip():
        return "⚠️ 질문을 입력하세요.", "", ""

    # 로딩된 슬롯이 하나도 없으면 즉시 반환
    active = [i for i in range(MAX_SLOTS) if loaded_models[i] is not None]
    if not active:
        return "⚠️ 로딩된 모델이 없습니다. 먼저 모델을 로딩하세요.", "", ""

    results = [""] * MAX_SLOTS
    threads = []

    def worker(idx):
        if loaded_models[idx] is None:
            # 비어있는 슬롯은 건너뜀
            results[idx] = f"[슬롯 {idx+1}] 모델 없음 — 로딩 관리 탭에서 모델을 로딩하세요."
            return
        results[idx] = _infer_one(idx, prompt, max_new_tokens, temperature, do_sample)

    # 로딩된 슬롯만 스레드 생성 (비어있는 슬롯은 메시지만 채움)
    for i in range(MAX_SLOTS):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    return results[0], results[1], results[2]


# ──────────────────────────────────────────────
# 7. Gradio UI 구성
# ──────────────────────────────────────────────
def build_ui():
    with gr.Blocks(title="LLM 모델 매니저 (RAG 실습)") as demo:

        gr.Markdown("# LLM 모델 매니저 — RAG 실습 Step 1\n허깅페이스 모델 검색 → 다운로드 → 슬롯 로딩 → 동시 추론")

        gr.Textbox(
            label="디바이스 정보",
            value=get_device_info,
            interactive=False,
            every=30,
        )

        # ═══════════════════════════════════════
        # 탭 1: 모델 로딩 관리
        # ═══════════════════════════════════════
        with gr.Tab("모델 로딩 관리"):

            gr.Textbox(
                label="슬롯 현황",
                value=get_loaded_summary,
                interactive=False,
                every=5,
                lines=3,
            )

            for slot_i in range(MAX_SLOTS):
                with gr.Accordion(f"슬롯 {slot_i+1}", open=(slot_i == 0)):

                    # ── 로컬 모델 선택 드롭다운 ──────────────────
                    with gr.Row():
                        local_model_dd = gr.Dropdown(
                            label=f"📂 로컬 모델 선택 ({LOCAL_MODEL_ROOT})",
                            choices=scan_local_models(),  # 시작 시 초기 스캔
                            interactive=True,
                            allow_custom_value=False,
                            scale=5,
                            info="아래 모델 ID / 로컬 경로에 자동 반영됩니다.",
                        )
                        refresh_btn = gr.Button(
                            "🔄 새로고침",
                            scale=1,
                            min_width=90,
                        )
                        refresh_status = gr.Textbox(
                            label="스캔 결과",
                            interactive=False,
                            scale=2,
                        )

                    # ── 모델 ID / 로컬 경로 입력 ─────────────────
                    with gr.Row():
                        s_model_id = gr.Textbox(
                            label="모델 ID (HF Hub — 로컬 경로 비어있을 때 사용)",
                            placeholder="예: mistralai/Mistral-7B-Instruct-v0.1",
                            scale=3,
                        )
                        s_local_path = gr.Textbox(
                            label="로컬 폴더 경로 (우선 사용 — 비우면 HF Hub 로딩)",
                            placeholder="예: ./hf_model/mistralai--Mistral-7B-Instruct-v0.1",
                            scale=3,
                        )

                    with gr.Row():
                        s_quant = gr.Radio(
                            choices=["4bit", "8bit", "fp16", "fp32"],
                            value="4bit",
                            label="양자화 모드",
                            info="RTX 2080 (8~11GB VRAM) → 4bit 권장",
                        )
                        s_trust = gr.Checkbox(
                            label="trust_remote_code",
                            value=False,
                            info="EXAONE 등 커스텀 코드 모델에 필요",
                        )

                    with gr.Row():
                        load_btn   = gr.Button(f"슬롯 {slot_i+1} 로딩", variant="primary")
                        unload_btn = gr.Button(f"슬롯 {slot_i+1} 언로드", variant="stop")

                    s_out = gr.Textbox(label="로딩 결과", lines=4, interactive=False)

                    # 로컬 드롭다운 선택 → 로컬 경로 박스에 자동 반영
                    local_model_dd.change(
                        fn=lambda p: p if p else "",
                        inputs=[local_model_dd],
                        outputs=[s_local_path],
                    )

                    # 새로고침 버튼 → 드롭다운 목록 갱신
                    refresh_btn.click(
                        fn=refresh_local_models,
                        outputs=[local_model_dd, refresh_status],
                    )

                    # 클로저로 slot_i 값 캡처
                    def make_load_fn(idx):
                        def fn(mid, quant, lpath, trust):
                            return load_model_to_slot(mid, idx, quant, lpath, trust)
                        return fn

                    def make_unload_fn(idx):
                        def fn():
                            return unload_model(idx)
                        return fn

                    load_btn.click(
                        fn=make_load_fn(slot_i),
                        inputs=[s_model_id, s_quant, s_local_path, s_trust],
                        outputs=[s_out],
                    )
                    unload_btn.click(
                        fn=make_unload_fn(slot_i),
                        outputs=[s_out],
                    )

        # ═══════════════════════════════════════
        # 탭 2: 모델 검색 & 다운로드
        # ═══════════════════════════════════════
        with gr.Tab("모델 검색 & 다운로드"):

            # ── 검색 한 줄: 검색어 | 버튼 | 상태 | 드롭다운 ──
            with gr.Row(equal_height=True):
                search_query = gr.Textbox(
                    label="검색어",
                    placeholder="llama, gemma, qwen ...",
                    scale=2,
                    min_width=120,
                )
                search_btn = gr.Button("🔍 검색", variant="primary", scale=1, min_width=80)
                search_status = gr.Textbox(
                    label="검색 상태",
                    interactive=False,
                    scale=2,
                    min_width=150,
                )
                search_results = gr.Dropdown(
                    label="검색 결과 — 선택하세요",
                    choices=[],
                    interactive=True,
                    allow_custom_value=True,
                    scale=5,
                )

            gr.Markdown("---")
            gr.Markdown("### 다운로드 설정")

            # 선택된 모델 ID (드롭다운 선택 시 자동 입력, 직접 수정 가능)
            selected_model_box = gr.Textbox(
                label="다운로드할 모델 ID (드롭다운 선택 시 자동 입력 / 직접 수정 가능)",
                placeholder="위 드롭다운에서 모델을 선택하면 자동으로 채워집니다.",
                interactive=True,
            )

            with gr.Row():
                save_type = gr.Radio(
                    choices=["cache", "local"],
                    value="local",          # 기본값을 local로 설정 (경로 자동완성 활용)
                    label="저장 방식",
                    info="cache: HF 기본 캐시 | local: 아래 지정 폴더에 저장",
                    scale=1,
                )
                local_save_dir = gr.Textbox(
                    label=f"로컬 저장 경로 (기본: {LOCAL_MODEL_ROOT}/<모델명>)",
                    placeholder="모델 선택 시 자동으로 채워집니다.",
                    interactive=True,   # 자동완성 후 수동 수정도 가능
                    scale=3,
                )

            download_btn = gr.Button("⬇️ 다운로드 시작", variant="primary", size="lg")
            download_status = gr.Textbox(label="다운로드 상태", lines=3, interactive=False)

            # 검색 버튼 → 드롭다운 갱신
            search_btn.click(
                fn=search_models,
                inputs=[search_query],
                outputs=[search_results, search_status],
            )
            # 드롭다운 선택 → 모델 ID 박스 + 로컬 저장 경로 동시 자동 완성
            search_results.change(
                fn=on_model_selected,
                inputs=[search_results],
                outputs=[selected_model_box, local_save_dir],
            )
            # 다운로드 버튼
            download_btn.click(
                fn=download_model,
                inputs=[selected_model_box, save_type, local_save_dir],
                outputs=[download_status],
            )

        # ═══════════════════════════════════════
        # 탭 3: 동시 추론
        # ═══════════════════════════════════════
        with gr.Tab("동시 추론"):

            gr.Markdown("로딩된 모든 슬롯 모델에 동일한 질문을 전송하여 결과를 비교합니다.")

            with gr.Row():
                max_new_tokens = gr.Slider(32, 512, value=200, step=32,
                    label="max_new_tokens", info="생성할 최대 토큰 수")
                temperature    = gr.Slider(0.1, 2.0, value=0.7, step=0.1,
                    label="temperature", info="높을수록 창의적, 낮을수록 일관된 답변")
                do_sample      = gr.Checkbox(label="do_sample", value=True,
                    info="False → greedy decoding")

            prompt_in = gr.Textbox(
                label="질문 (Prompt)  ※ 줄바꿈: Shift+Enter",
                placeholder="예: 인공지능이란 무엇인가요?",
                lines=6,
                max_lines=20,       # 내용이 늘어나면 자동으로 박스 확장
                submit_btn=False,   # Enter 키를 전송이 아닌 줄바꿈으로 처리
            )
            run_btn = gr.Button("전체 슬롯 동시 실행", variant="primary", size="lg")

            with gr.Row():
                out0 = gr.Textbox(label="슬롯 1 결과", lines=10)
                out1 = gr.Textbox(label="슬롯 2 결과", lines=10)
                out2 = gr.Textbox(label="슬롯 3 결과", lines=10)

            run_btn.click(
                fn=run_all_models,
                inputs=[prompt_in, max_new_tokens, temperature, do_sample],
                outputs=[out0, out1, out2],
            )

        # ═══════════════════════════════════════
        # 탭 4: 개별 슬롯 추론
        # ═══════════════════════════════════════
        with gr.Tab("개별 슬롯 추론"):

            gr.Markdown("특정 슬롯의 모델만 개별로 테스트합니다.")

            single_slot = gr.Radio(
                choices=[f"슬롯 {i+1}" for i in range(MAX_SLOTS)],
                value="슬롯 1", label="슬롯 선택")
            single_prompt = gr.Textbox(
                label="질문  ※ 줄바꿈: Shift+Enter",
                placeholder="테스트할 질문을 입력하세요.",
                lines=6,
                max_lines=20,
                submit_btn=False,   # Enter 키를 전송이 아닌 줄바꿈으로 처리
            )

            with gr.Row():
                s_max_tokens = gr.Slider(32, 512, value=200, step=32, label="max_new_tokens")
                s_temp       = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="temperature")
                s_sample     = gr.Checkbox(label="do_sample", value=True)

            single_run = gr.Button("실행", variant="primary")
            single_out = gr.Textbox(label="결과", lines=10)

            def run_single(slot_str, prompt, max_tok, temp, sample):
                idx = int(slot_str.replace("슬롯 ", "")) - 1
                return _infer_one(idx, prompt, max_tok, temp, sample)

            single_run.click(
                fn=run_single,
                inputs=[single_slot, single_prompt, s_max_tokens, s_temp, s_sample],
                outputs=[single_out],
            )

    return demo


# ──────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # 로컬 모델 루트 폴더가 없으면 미리 생성
    os.makedirs(LOCAL_MODEL_ROOT, exist_ok=True)

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )
