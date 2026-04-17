"""
===================================================
RAG 실습 Step3: 내부 검색 + LLM 생성
- 벡터 검색 : FAISS + sentence-transformers 임베딩
- 키워드 검색: BM25 (원문 토큰 vs 형태소 분석 토큰 비교)
- 데이터     : Wikipedia 한국어 문서 자동 수집 (5개 주제)
- UI 구성    :
    탭1. 데이터 수집 & 청킹
    탭2. 색인 구축 (FAISS / BM25)
    탭3. 검색 비교 (FAISS | BM25 나란히)
    탭4. RAG 생성  (검색 결과 선택 → LLM)

[필수 설치]
pip install gradio transformers huggingface_hub
pip install bitsandbytes accelerate
pip install faiss-cpu sentence-transformers
pip install rank-bm25
pip install konlpy
pip install wikipedia-api
pip install torch --index-url https://download.pytorch.org/whl/cu118
===================================================
"""

import os
import re
import json
import pickle
import threading
import numpy as np
import gradio as gr

# ── Gradio 버전 호환성 유틸 (./colab/gradio_compat.py) ──
# Colab 환경 감지(share=True 자동) + Gradio 버전별 theme 위치 자동 처리
from colab.gradio_compat import make_blocks, safe_launch
import torch
import requests

# ── 벡터 검색 ──────────────────────────────────
import faiss
from sentence_transformers import SentenceTransformer

# ── BM25 키워드 검색 ───────────────────────────
from rank_bm25 import BM25Okapi

# ── 형태소 분석 (한국어 BM25용) ───────────────
try:
    from konlpy.tag import Okt
    okt = Okt()
    KONLPY_AVAILABLE = True
except Exception:
    okt = None
    KONLPY_AVAILABLE = False

# ── Wikipedia 수집 ─────────────────────────────
import wikipediaapi

# ── LLM ───────────────────────────────────────
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, pipeline,
)

# ──────────────────────────────────────────────
# 전역 상태
# ──────────────────────────────────────────────
LOCAL_MODEL_ROOT = "./hf_model"
INDEX_SAVE_DIR   = "./rag_index"

wiki_ko = wikipediaapi.Wikipedia(language="ko", user_agent="internal-rag/1.0")
wiki_en = wikipediaapi.Wikipedia(language="en", user_agent="internal-rag/1.0")

chunks_store: list = []   # [{"text", "source", "url", "chunk_id"}, ...]

faiss_index    = None
faiss_model    = None
faiss_model_id = ""

bm25_raw   = None
bm25_morph = None

current_model = {"info": None}

_last_faiss:      list = []
_last_bm25_raw:   list = []   # BM25 원문 토큰 검색 결과
_last_bm25_morph: list = []   # BM25 형태소 분석 검색 결과

DEFAULT_TOPICS = ["인공지능", "기계학습", "자연어처리", "딥러닝", "대한민국"]

EMBED_MODELS = [
    "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    "jhgan/ko-sroberta-multitask",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
]


# ══════════════════════════════════════════════
# [유틸]
# ══════════════════════════════════════════════

def get_device_info() -> str:
    if torch.cuda.is_available():
        name  = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        free  = (torch.cuda.get_device_properties(0).total_memory
                 - torch.cuda.memory_allocated(0)) / (1024**3)
        return f"GPU: {name} | 전체: {total:.1f}GB | 여유: {free:.1f}GB"
    return "GPU 없음 — CPU 모드"


def scan_local_models(root: str = LOCAL_MODEL_ROOT) -> list:
    """로컬 모델 폴더 스캔 (config.json 존재 여부 기준)"""
    if not os.path.isdir(root):
        return []
    found = []
    for e in sorted(os.scandir(root), key=lambda x: x.name):
        if not e.is_dir():
            continue
        if os.path.isfile(os.path.join(e.path, "config.json")):
            found.append(e.path)
        else:
            for sub in sorted(os.scandir(e.path), key=lambda x: x.name):
                if sub.is_dir() and os.path.isfile(os.path.join(sub.path, "config.json")):
                    found.append(sub.path)
    return found


def clean_text(text: str) -> str:
    """Wikipedia 본문 정제: 과도한 빈 줄·연속 공백 제거"""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ══════════════════════════════════════════════
# 1. Wikipedia 데이터 수집
# ══════════════════════════════════════════════

def fetch_wikipedia_pages(topics: list) -> list:
    """주제 목록으로 Wikipedia 페이지 수집 (한국어 우선, 없으면 영어)"""
    docs = []
    for topic in topics:
        topic = topic.strip()
        if not topic:
            continue
        page = wiki_ko.page(topic)
        if not page.exists():
            page = wiki_en.page(topic)
        if page.exists():
            docs.append({
                "title": page.title,
                "text":  clean_text(page.text),
                "url":   page.fullurl,
            })
    return docs


def collect_data(topics_str: str):
    topics = [t.strip() for t in topics_str.split(",") if t.strip()]
    if not topics:
        return "⚠️ 주제를 입력하세요.", "", "문서 0개"
    try:
        docs = fetch_wikipedia_pages(topics)
        if not docs:
            return "❌ 수집된 문서가 없습니다.", "", "문서 0개"
        os.makedirs(INDEX_SAVE_DIR, exist_ok=True)
        with open(os.path.join(INDEX_SAVE_DIR, "raw_docs.json"), "w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)
        preview = "\n\n---\n\n".join(
            f"[{d['title']}]\n{d['text'][:200]}..." for d in docs
        )
        msg = f"✅ {len(docs)}개 문서 수집 완료 ({', '.join(d['title'] for d in docs)})"
        return msg, preview, f"문서 {len(docs)}개"
    except Exception as e:
        return f"❌ 수집 오류: {e}", "", "문서 0개"


# ══════════════════════════════════════════════
# 2. 청킹
# ══════════════════════════════════════════════

def chunk_by_chars(text: str, size: int, overlap: int) -> list:
    """문자 수 기준 청킹. overlap만큼 앞 청크와 겹쳐서 문맥 연속성 유지."""
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def chunk_by_sentences(text: str, n_sent: int, overlap: int) -> list:
    """문장 수 기준 청킹. 한국어 문장 구분: .!? 뒤 공백/줄바꿈"""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    step   = max(1, n_sent - overlap)
    chunks = []
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i: i + n_sent])
        if chunk:
            chunks.append(chunk)
    return chunks


def build_chunks(chunk_method: str, chunk_size: int, overlap: int, min_len: int):
    global chunks_store
    raw_path = os.path.join(INDEX_SAVE_DIR, "raw_docs.json")
    if not os.path.exists(raw_path):
        return "⚠️ 먼저 데이터를 수집하세요.", "", ""

    with open(raw_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    chunks_store = []
    for doc in docs:
        if chunk_method == "문자 수":
            raw_chunks = chunk_by_chars(doc["text"], int(chunk_size), int(overlap))
        else:
            raw_chunks = chunk_by_sentences(doc["text"], int(chunk_size), int(overlap))
        for c in raw_chunks:
            c = c.strip()
            if len(c) >= int(min_len):
                chunks_store.append({
                    "text":     c,
                    "source":   doc["title"],
                    "url":      doc.get("url", ""),
                    "chunk_id": len(chunks_store),
                })

    with open(os.path.join(INDEX_SAVE_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks_store, f, ensure_ascii=False, indent=2)

    lengths = [len(c["text"]) for c in chunks_store]
    avg_len = int(np.mean(lengths)) if lengths else 0
    stats   = (f"총 {len(chunks_store)}개 청크 | 평균 {avg_len}자 | "
               f"최소 {min(lengths) if lengths else 0}자 | "
               f"최대 {max(lengths) if lengths else 0}자")

    preview = "\n\n---\n\n".join(
        f"[청크 {c['chunk_id']} | 출처: {c['source']}]\n{c['text'][:200]}..."
        for c in chunks_store[:5]
    )
    return f"✅ 청킹 완료: {stats}", preview, stats


# ══════════════════════════════════════════════
# 3-A. FAISS 벡터 색인
# ══════════════════════════════════════════════

def build_faiss_index(embed_model_name: str, progress=gr.Progress()):
    """청크 임베딩 후 FAISS IndexFlatIP(코사인 유사도) 구축"""
    global faiss_index, faiss_model, faiss_model_id
    if not chunks_store:
        return "⚠️ 먼저 청킹을 완료하세요."
    try:
        progress(0, desc="임베딩 모델 로딩 중...")
        if faiss_model is None or faiss_model_id != embed_model_name:
            faiss_model    = SentenceTransformer(embed_model_name)
            faiss_model_id = embed_model_name

        texts = [c["text"] for c in chunks_store]
        progress(0.2, desc=f"임베딩 생성 중... ({len(texts)}개 청크)")
        embeddings = faiss_model.encode(
            texts, batch_size=32, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        progress(0.8, desc="FAISS 인덱스 구축 중...")
        dim   = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)   # 정규화된 벡터의 내적 = 코사인 유사도
        index.add(embeddings.astype(np.float32))
        faiss_index = index

        os.makedirs(INDEX_SAVE_DIR, exist_ok=True)
        faiss.write_index(index, os.path.join(INDEX_SAVE_DIR, "faiss.index"))
        progress(1.0, desc="완료")
        return (f"✅ FAISS 색인 완료\n모델: {embed_model_name}\n"
                f"벡터 차원: {dim} | 청크 수: {index.ntotal}")
    except Exception as e:
        return f"❌ FAISS 색인 오류: {e}"


# ══════════════════════════════════════════════
# 3-B. BM25 토큰화 함수
# ══════════════════════════════════════════════

def tokenize_raw(text: str) -> list:
    """공백 기준 단순 분리 — 형태소 분석 없음"""
    return text.split()


def tokenize_morph(text: str) -> list:
    """
    Okt 형태소 분석기로 명사+동사+형용사만 추출.
    조사·어미를 제거하여 BM25 매칭 정확도를 높이는 핵심 단계.
    konlpy 미설치 시 tokenize_raw로 fallback.
    """
    if not KONLPY_AVAILABLE or okt is None:
        return tokenize_raw(text)
    tokens = [
        word for word, pos in okt.pos(text, norm=True, stem=True)
        if pos.startswith(("N", "V", "Adjective"))
    ]
    return tokens if tokens else tokenize_raw(text)


# ══════════════════════════════════════════════
# 3-C. BM25 저장 (pickle + JSON 동시)
# ══════════════════════════════════════════════

def save_bm25(bm25: BM25Okapi, corpus: list, name: str):
    """
    BM25 객체를 pickle과 JSON 두 형식으로 저장합니다.

    pickle : 로딩용 — 통계 포함 객체 전체 저장, 재계산 없이 즉시 검색 가능
    JSON   : 검토·디버깅용 — idf/tf/토큰 리스트를 사람이 읽을 수 있는 형태로 저장
    """
    os.makedirs(INDEX_SAVE_DIR, exist_ok=True)
    base = os.path.join(INDEX_SAVE_DIR, name)

    # ── pickle 저장 ───────────────────────────
    with open(f"{base}.pkl", "wb") as f:
        pickle.dump(bm25, f)

    # ── JSON 저장 ─────────────────────────────
    try:
        # numpy float → python float 변환 (json 직렬화를 위해 필수)
        idf_dict = {k: round(float(v), 6) for k, v in bm25.idf.items()}

        # doc_freqs: rank_bm25 버전에 따라 구조가 다를 수 있어 방어 처리
        tf_list = []
        if hasattr(bm25, "doc_freqs") and bm25.doc_freqs:
            for doc in bm25.doc_freqs:
                if isinstance(doc, dict):
                    tf_list.append({tok: round(float(freq), 4)
                                    for tok, freq in doc.items()})
                else:
                    tf_list.append({})
        else:
            tf_list = [{} for _ in corpus]

        data = {
            "stats": {
                "corpus_size": int(bm25.corpus_size),
                "avgdl":       round(float(bm25.avgdl), 4),
            },
            "idf":         idf_dict,                              # 토큰별 IDF 값
            "tf":          tf_list,                               # 청크별 토큰 빈도
            "corpus":      corpus,                                # 청크별 토큰 리스트
            "chunks_text": [c["text"] for c in chunks_store],    # 원본 청크 텍스트
        }
        with open(f"{base}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[INFO] JSON 저장 완료: {base}.json")

    except Exception as e:
        # JSON 저장 실패해도 pickle은 이미 완료됐으므로 경고만 출력
        print(f"[WARNING] JSON 저장 실패 ({name}): {e}")


# ══════════════════════════════════════════════
# 3-D. BM25 색인 구축
# ══════════════════════════════════════════════

def build_bm25_index(use_raw: bool, use_morph: bool):
    """
    선택된 방식으로 BM25 색인을 구축합니다.
    use_raw  : 원문 공백 토큰 BM25
    use_morph: 형태소 분석 BM25 (konlpy 필요)
    각 색인은 pickle + JSON으로 동시 저장됩니다.
    """
    global bm25_raw, bm25_morph

    if not chunks_store:
        return "⚠️ 먼저 청킹을 완료하세요."
    if not use_raw and not use_morph:
        return "⚠️ 원문 토큰 또는 형태소 분석 중 하나 이상을 선택하세요."
    if use_morph and not KONLPY_AVAILABLE:
        return "⚠️ konlpy가 설치되지 않았습니다.\npip install konlpy\n형태소 없이 진행하려면 '원문 토큰'만 선택하세요."

    try:
        texts = [c["text"] for c in chunks_store]
        lines = [f"청크 수: {len(texts)}개\n"]

        # ── 원문 토큰 BM25 ────────────────────────
        if use_raw:
            raw_corpus = [tokenize_raw(t) for t in texts]
            bm25_raw   = BM25Okapi(raw_corpus)
            save_bm25(bm25_raw, raw_corpus, "bm25_raw")   # pickle + JSON 동시 저장
            sample = " ".join(tokenize_raw(texts[0])[:10])
            lines.append(f"✅ 원문 토큰 BM25 완료\n   저장: bm25_raw.pkl / bm25_raw.json\n   토큰 샘플: {sample}...")
        else:
            bm25_raw = None
            lines.append("⬜ 원문 토큰 BM25 — 미구축")

        # ── 형태소 분석 BM25 ──────────────────────
        if use_morph:
            morph_corpus = [tokenize_morph(t) for t in texts]
            bm25_morph   = BM25Okapi(morph_corpus)
            save_bm25(bm25_morph, morph_corpus, "bm25_morph")  # pickle + JSON 동시 저장
            sample = " ".join(tokenize_morph(texts[0])[:10])
            lines.append(f"✅ 형태소 BM25 완료\n   저장: bm25_morph.pkl / bm25_morph.json\n   토큰 샘플: {sample}...")
        else:
            bm25_morph = None
            lines.append("⬜ 형태소 분석 BM25 — 미구축")

        return "\n".join(lines)
    except Exception as e:
        return f"❌ BM25 색인 오류: {e}"


# ══════════════════════════════════════════════
# 4. 검색
# ══════════════════════════════════════════════

def search_faiss(query: str, top_k: int = 5) -> list:
    """FAISS 코사인 유사도 검색"""
    if faiss_index is None or faiss_model is None:
        return [{"text": "FAISS 색인이 없습니다.", "source": "", "score": 0.0, "chunk_id": -1}]
    q_vec = faiss_model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    scores, indices = faiss_index.search(q_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunks_store):
            continue
        c = chunks_store[idx]
        results.append({"text": c["text"], "source": c["source"],
                         "chunk_id": c["chunk_id"], "score": float(score)})
    return results


def search_bm25(query: str, top_k: int = 5, use_morph: bool = True) -> list:
    """
    BM25 키워드 검색.
    색인과 동일한 토큰화 방식으로 쿼리를 처리해야 올바른 점수가 나옵니다.
    """
    bm25 = bm25_morph if use_morph else bm25_raw
    if bm25 is None:
        mode = "형태소" if use_morph else "원문 토큰"
        return [{"text": f"BM25({mode}) 색인이 없습니다. 색인 구축 탭에서 먼저 구축하세요.",
                 "source": "", "score": 0.0, "chunk_id": -1}]

    q_tokens    = tokenize_morph(query) if use_morph else tokenize_raw(query)
    scores      = bm25.get_scores(q_tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        c = chunks_store[idx]
        results.append({"text": c["text"], "source": c["source"],
                         "chunk_id": c["chunk_id"], "score": float(scores[idx])})
    return results


def run_search_compare(query: str, top_k: int):
    """
    FAISS / BM25 원문 / BM25 형태소를 동시에 실행하고
    각각의 결과를 체크박스 형태로 반환합니다.
    색인이 없는 방식은 안내 메시지를 표시합니다.
    """
    global _last_faiss, _last_bm25_raw, _last_bm25_morph

    if not query.strip():
        empty = gr.CheckboxGroup(choices=[])
        return empty, empty, empty, "", "⚠️ 질문을 입력하세요."

    top_k = int(top_k)
    faiss_res, raw_res, morph_res = [], [], []

    # 세 검색기를 스레드로 동시 실행
    def run_f():     faiss_res.extend(search_faiss(query, top_k))
    def run_raw():   raw_res.extend(search_bm25(query, top_k, use_morph=False))
    def run_morph(): morph_res.extend(search_bm25(query, top_k, use_morph=True))

    threads = [
        threading.Thread(target=run_f),
        threading.Thread(target=run_raw),
        threading.Thread(target=run_morph),
    ]
    for t in threads: t.start()
    for t in threads: t.join()

    def fmt_label(r, idx):
        """체크박스 레이블: 번호. [출처] score / 본문 앞 80자"""
        preview = r["text"][:80].replace("\n", " ")
        return f"{idx+1}. [{r['source']}] score={r['score']:.3f}\n   {preview}..."

    faiss_labels = [fmt_label(r, i) for i, r in enumerate(faiss_res)]
    raw_labels   = [fmt_label(r, i) for i, r in enumerate(raw_res)]
    morph_labels = [fmt_label(r, i) for i, r in enumerate(morph_res)]

    # 쿼리 토큰 분석 (교육용: 원문 vs 형태소 토큰 직접 비교)
    q_raw   = " | ".join(tokenize_raw(query)[:15])
    q_morph = (" | ".join(tokenize_morph(query)[:15])
               if KONLPY_AVAILABLE else "konlpy 미설치")
    token_info = (
        f"[쿼리 토큰 분석]\n"
        f"  원문 토큰   ({len(tokenize_raw(query))}개): {q_raw}\n"
        f"  형태소 토큰 ({len(tokenize_morph(query)) if KONLPY_AVAILABLE else '?'}개): {q_morph}"
    )

    status = (
        f"✅ 검색 완료 — "
        f"FAISS: {len(faiss_res)}건 | "
        f"BM25 원문: {len(raw_res)}건 | "
        f"BM25 형태소: {len(morph_res)}건"
    )

    # 전역 저장 (생성 탭에서 재사용)
    _last_faiss      = faiss_res
    _last_bm25_raw   = raw_res
    _last_bm25_morph = morph_res

    return (
        gr.CheckboxGroup(choices=faiss_labels,  value=faiss_labels),   # 1. FAISS
        gr.CheckboxGroup(choices=raw_labels,    value=raw_labels),     # 2. BM25 원문
        gr.CheckboxGroup(choices=morph_labels,  value=morph_labels),   # 3. BM25 형태소
        token_info,   # 4. 쿼리 토큰 분석
        status,       # 5. 검색 상태
    )


# ══════════════════════════════════════════════
# 5. LLM 로딩 / 언로드
# ══════════════════════════════════════════════

def load_model(model_path: str, quant_mode: str, trust_remote: bool) -> str:
    if not model_path or not model_path.strip():
        return "⚠️ 모델 경로를 선택하거나 입력하세요."
    if current_model["info"]:
        try:
            del current_model["info"]["pipe"].model
            del current_model["info"]["pipe"]
        except Exception:
            pass
        current_model["info"] = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    src = model_path.strip()
    try:
        quant_cfg, dtype = None, torch.float32
        if quant_mode == "4bit":
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            )
        elif quant_mode == "8bit":
            quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
        elif quant_mode == "fp16":
            dtype = torch.float16

        tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=trust_remote)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        mk = {"trust_remote_code": trust_remote, "device_map": "auto"}
        if quant_cfg:
            mk["quantization_config"] = quant_cfg
        else:
            mk["torch_dtype"] = dtype

        model = AutoModelForCausalLM.from_pretrained(src, **mk)
        model.eval()
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        name = os.path.basename(src) if os.path.isdir(src) else src
        current_model["info"] = {"model_id": name, "pipe": pipe, "status": quant_mode}
        return f"✅ 로딩 완료: {name} ({quant_mode})\n{get_device_info()}"
    except Exception as e:
        return f"❌ 로딩 오류:\n{e}"


def unload_model() -> str:
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
    paths = scan_local_models()
    msg = f"✅ {len(paths)}개 모델" if paths else "⚠️ 모델 없음"
    return gr.Dropdown(choices=paths, value=None), msg


# ══════════════════════════════════════════════
# 6. RAG 생성
# ══════════════════════════════════════════════

def generate_answer(query: str,
                    faiss_selected: list,
                    bm25_raw_selected: list,
                    bm25_morph_selected: list,
                    max_new_tokens: int, temperature: float,
                    do_sample: bool) -> str:
    """
    선택된 FAISS / BM25 원문 / BM25 형태소 결과를
    컨텍스트로 조합하여 LLM 답변을 생성합니다.
    세 검색기 결과를 함께 선택하면 상호 보완적인 컨텍스트를 구성할 수 있습니다.
    중복 청크는 자동으로 제거합니다.
    """
    if not current_model["info"]:
        return "⚠️ 먼저 LLM을 로딩하세요."
    if not query.strip():
        return "⚠️ 질문을 입력하세요."

    context_parts = []

    def extract(selected, source_list):
        """선택된 레이블에서 청크 본문을 추출하여 context_parts에 추가"""
        for label in selected:
            try:
                idx = int(label.split(".")[0].strip()) - 1
                if 0 <= idx < len(source_list):
                    r    = source_list[idx]
                    text = f"[출처: {r['source']}]\n{r['text']}"
                    if text not in context_parts:   # 중복 제거
                        context_parts.append(text)
            except Exception:
                pass

    extract(faiss_selected,      _last_faiss)
    extract(bm25_raw_selected,   _last_bm25_raw)
    extract(bm25_morph_selected, _last_bm25_morph)

    if context_parts:
        ctx    = "\n\n".join(context_parts)
        prompt = (f"다음 참고 정보를 바탕으로 질문에 답하세요.\n\n"
                  f"[참고 정보]\n{ctx}\n\n[질문]\n{query}\n\n[답변]\n")
    else:
        prompt = f"[질문]\n{query}\n\n[답변]\n"

    try:
        pipe   = current_model["info"]["pipe"]
        out    = pipe(prompt, max_new_tokens=max_new_tokens,
                      temperature=temperature if do_sample else 1.0,
                      do_sample=do_sample,
                      pad_token_id=pipe.tokenizer.eos_token_id,
                      return_full_text=False)
        answer = out[0]["generated_text"].strip()
        suffix = (f"\n\n─────\n📚 컨텍스트: {len(context_parts)}건 사용"
                  if context_parts else "\n\n─────\n⚠️ 컨텍스트 없이 생성 (순수 LLM)")
        return answer + suffix
    except Exception as e:
        return f"❌ 생성 오류: {e}"


# ──────────────────────────────────────────────
# 시작 시 기존 색인 파일 자동 로딩
# ──────────────────────────────────────────────
def load_existing_indexes():
    """
    프로그램 재시작 시 ./rag_index 폴더에 저장된 색인 파일을 자동으로 로딩합니다.
    - chunks.json   : 청크 목록 (FAISS/BM25 검색 결과 본문 복원에 필요)
    - faiss.index   : FAISS 벡터 인덱스
    - bm25_raw.pkl  : BM25 원문 토큰 인덱스
    - bm25_morph.pkl: BM25 형태소 분석 인덱스

    각 파일이 없으면 건너뛰고 로딩된 항목만 보고합니다.
    """
    global chunks_store, faiss_index, faiss_model, faiss_model_id
    global bm25_raw, bm25_morph

    loaded = []
    skipped = []

    # ── 1. 청크 목록 ─────────────────────────
    chunk_path = os.path.join(INDEX_SAVE_DIR, "chunks.json")
    if os.path.exists(chunk_path):
        with open(chunk_path, "r", encoding="utf-8") as f:
            chunks_store = json.load(f)
        loaded.append(f"청크 {len(chunks_store)}개")
    else:
        skipped.append("chunks.json")

    # ── 2. FAISS 인덱스 ───────────────────────
    faiss_path = os.path.join(INDEX_SAVE_DIR, "faiss.index")
    if os.path.exists(faiss_path):
        try:
            faiss_index = faiss.read_index(faiss_path)
            # 임베딩 모델은 검색 시 필요하므로 기본 모델로 미리 로딩
            faiss_model    = SentenceTransformer(EMBED_MODELS[0])
            faiss_model_id = EMBED_MODELS[0]
            loaded.append(f"FAISS ({faiss_index.ntotal}개 벡터)")
        except Exception as e:
            skipped.append(f"faiss.index ({e})")
    else:
        skipped.append("faiss.index")

    # ── 3. BM25 원문 토큰 인덱스 ─────────────
    raw_path = os.path.join(INDEX_SAVE_DIR, "bm25_raw.pkl")
    if os.path.exists(raw_path):
        try:
            with open(raw_path, "rb") as f:
                bm25_raw = pickle.load(f)
            loaded.append("BM25 원문")
        except Exception as e:
            skipped.append(f"bm25_raw.pkl ({e})")
    else:
        skipped.append("bm25_raw.pkl")

    # ── 4. BM25 형태소 분석 인덱스 ───────────
    morph_path = os.path.join(INDEX_SAVE_DIR, "bm25_morph.pkl")
    if os.path.exists(morph_path):
        try:
            with open(morph_path, "rb") as f:
                bm25_morph = pickle.load(f)
            loaded.append("BM25 형태소")
        except Exception as e:
            skipped.append(f"bm25_morph.pkl ({e})")
    else:
        skipped.append("bm25_morph.pkl")

    # 결과 출력
    if loaded:
        print(f"[INFO] 기존 색인 자동 로딩 완료: {', '.join(loaded)}")
    if skipped:
        print(f"[INFO] 없거나 로딩 실패 (건너뜀): {', '.join(skipped)}")


def get_index_status() -> str:
    """
    현재 메모리에 로딩된 색인 상태를 문자열로 반환합니다.
    Gradio Textbox의 value=get_index_status, every=5 옵션으로
    5초마다 자동 갱신하는 데 사용됩니다.
    """
    lines = []

    # ── 청크 상태 ─────────────────────────────
    if chunks_store:
        lines.append(f"✅ 청크: {len(chunks_store)}개 로딩됨")
    else:
        lines.append("❌ 청크: 없음 (데이터 수집 & 청킹 탭에서 먼저 실행)")

    # ── FAISS 상태 ────────────────────────────
    if faiss_index is not None:
        lines.append(f"✅ FAISS: {faiss_index.ntotal}개 벡터 | 모델: {faiss_model_id or '?'}")
    else:
        lines.append("❌ FAISS: 없음 (색인 구축 탭에서 먼저 실행)")

    # ── BM25 원문 상태 ────────────────────────
    if bm25_raw is not None:
        lines.append(f"✅ BM25 원문: 로딩됨 (corpus_size={bm25_raw.corpus_size})")
    else:
        lines.append("❌ BM25 원문: 없음")

    # ── BM25 형태소 상태 ──────────────────────
    if bm25_morph is not None:
        lines.append(f"✅ BM25 형태소: 로딩됨 (corpus_size={bm25_morph.corpus_size})")
    else:
        lines.append("❌ BM25 형태소: 없음")

    return "\n".join(lines)


def build_ui():
    # make_blocks(): Gradio 버전에 따라 theme 위치를 자동 조정
    with make_blocks(title="내부 검색 RAG 실습", theme=gr.themes.Soft()) as demo:

        gr.Markdown("# 📚 내부 검색 RAG 실습 — Step 3\n"
                    "FAISS(벡터 검색) vs BM25(키워드 검색) 비교 | 한국어 형태소 분석 효과 확인")
        gr.Textbox(label="디바이스 정보", value=get_device_info,
                   interactive=False, every=30)

        # ── 탭 1: 데이터 수집 & 청킹 ─────────────
        with gr.Tab("📥 데이터 수집 & 청킹"):
            gr.Markdown("### 1단계: Wikipedia 문서 수집")
            gr.Markdown("쉼표로 구분된 주제 입력 → Wikipedia 자동 수집 (한국어 우선, 없으면 영어)")

            with gr.Row():
                topics_box  = gr.Textbox(label="수집 주제 (쉼표 구분)",
                                         value=", ".join(DEFAULT_TOPICS), scale=4)
                collect_btn = gr.Button("📥 수집 시작", variant="primary", scale=1)
            with gr.Row():
                collect_status = gr.Textbox(label="수집 상태", interactive=False, scale=2)
                doc_count_lbl  = gr.Textbox(label="수집 결과", interactive=False, scale=1)
            collect_preview = gr.Textbox(label="수집 문서 미리보기", lines=8, interactive=False)

            gr.Markdown("---\n### 2단계: 청킹 설정")
            gr.Markdown("- **문자 수**: 정확한 크기 제어  |  **문장 수**: 의미 단위 유지\n"
                        "- **Overlap**: 앞 청크와 겹치는 양 → 경계 문맥 손실 방지")
            with gr.Row():
                chunk_method  = gr.Radio(["문자 수", "문장 수"], value="문자 수",
                                         label="청킹 방식", scale=1)
                chunk_size    = gr.Slider(100, 1000, value=300, step=50,
                                          label="청크 크기", scale=2)
                chunk_overlap = gr.Slider(0, 200, value=50, step=10,
                                          label="Overlap", scale=2)
                chunk_min_len = gr.Slider(10, 200, value=50, step=10,
                                          label="최소 청크 길이(자)", scale=1)
            chunk_btn     = gr.Button("✂️ 청킹 실행", variant="primary")
            chunk_status  = gr.Textbox(label="청킹 상태 및 통계", interactive=False)
            chunk_preview = gr.Textbox(label="청크 미리보기 (처음 5개)",
                                       lines=10, interactive=False)

            collect_btn.click(fn=collect_data, inputs=[topics_box],
                              outputs=[collect_status, collect_preview, doc_count_lbl])
            chunk_btn.click(fn=build_chunks,
                            inputs=[chunk_method, chunk_size, chunk_overlap, chunk_min_len],
                            outputs=[chunk_status, chunk_preview, gr.Textbox(visible=False)])

        # ── 탭 2: 색인 구축 ───────────────────────
        with gr.Tab("🗂️ 색인 구축"):

            gr.Markdown("### FAISS 벡터 색인")
            gr.Markdown("sentence-transformers 임베딩 → FAISS IndexFlatIP(코사인 유사도)")
            with gr.Row():
                embed_model_dd = gr.Dropdown(choices=EMBED_MODELS, value=EMBED_MODELS[0],
                                             label="임베딩 모델", allow_custom_value=True, scale=4)
                faiss_btn      = gr.Button("⚡ FAISS 색인 구축", variant="primary", scale=1)
            faiss_status = gr.Textbox(label="FAISS 색인 상태", lines=3, interactive=False)

            gr.Markdown("---\n### BM25 키워드 색인")
            gr.Markdown("선택한 방식으로 색인을 구축합니다. 각 색인은 **pickle + JSON** 으로 자동 저장됩니다.\n\n"
                        "| 방식 | 토큰화 | 특징 |\n|---|---|---|\n"
                        "| 원문 토큰 | 공백 분리 | 빠름, 조사·어미 포함 |\n"
                        "| 형태소 분석 | Okt 명사+동사+형용사 | 정확도 향상, 한국어 특화 |")
            with gr.Row():
                bm25_use_raw   = gr.Checkbox(label="원문 토큰 BM25", value=True,
                                             info="konlpy 불필요")
                bm25_use_morph = gr.Checkbox(label="형태소 분석 BM25",
                                             value=KONLPY_AVAILABLE,
                                             info="konlpy 필요 / 시간 더 소요")
                konlpy_status  = gr.Textbox(
                    label="konlpy 상태",
                    value="✅ 사용 가능" if KONLPY_AVAILABLE else "❌ 미설치 (pip install konlpy)",
                    interactive=False)
            bm25_btn    = gr.Button("⚡ BM25 색인 구축", variant="primary")
            bm25_status = gr.Textbox(label="BM25 색인 상태", lines=6, interactive=False)

            faiss_btn.click(fn=build_faiss_index, inputs=[embed_model_dd],
                            outputs=[faiss_status])
            bm25_btn.click(fn=build_bm25_index, inputs=[bm25_use_raw, bm25_use_morph],
                           outputs=[bm25_status])

        # ── 탭 3: 검색 비교 ───────────────────────
        with gr.Tab("🔍 검색 비교 (FAISS vs BM25)"):
            gr.Markdown(
                "동일한 질문으로 **FAISS(벡터)**, **BM25 원문**, **BM25 형태소** 결과를 한눈에 비교합니다.\n"
                "색인이 구축된 방식만 결과가 표시됩니다."
            )

            # 색인 현황 (재시작 후 바로 확인 가능)
            with gr.Row():
                index_status_box = gr.Textbox(
                    label="📂 현재 로딩된 색인 상태",
                    value=get_index_status,   # 5초마다 자동 갱신
                    interactive=False,
                    every=5,
                    lines=4,
                    scale=3,
                )
                reload_index_btn = gr.Button(
                    "🔄 색인 다시 로딩",
                    scale=1,
                    min_width=120,
                )

            with gr.Row():
                search_query = gr.Textbox(
                    label="검색 질문  ※ Shift+Enter = 줄바꿈",
                    placeholder="예: 딥러닝과 기계학습의 차이는?",
                    lines=2, max_lines=6, submit_btn=False, scale=4,
                )
                with gr.Column(scale=1, min_width=120):
                    top_k_sl   = gr.Slider(1, 10, value=5, step=1, label="Top-K")
                    search_btn = gr.Button("🔍 검색", variant="primary")

            search_status_box = gr.Textbox(label="검색 상태", interactive=False)
            token_info_box    = gr.Textbox(
                label="🔤 쿼리 토큰 분석 (원문 vs 형태소)",
                lines=3, interactive=False,
            )

            gr.Markdown("### 검색 결과 3종 비교 — 원하는 항목을 선택하여 RAG 생성에 사용하세요")
            # 3개 체크박스 나란히 배치
            with gr.Row():
                faiss_checks = gr.CheckboxGroup(
                    label="🧠 FAISS (벡터 의미 검색)",
                    choices=[], interactive=True, scale=1,
                )
                bm25_raw_checks = gr.CheckboxGroup(
                    label="🔤 BM25 원문 토큰",
                    choices=[], interactive=True, scale=1,
                )
                bm25_morph_checks = gr.CheckboxGroup(
                    label="🔬 BM25 형태소 분석",
                    choices=[], interactive=True, scale=1,
                )

            search_btn.click(
                fn=run_search_compare,
                inputs=[search_query, top_k_sl],
                outputs=[
                    faiss_checks,       # 반환 1: FAISS 체크박스
                    bm25_raw_checks,    # 반환 2: BM25 원문 체크박스
                    bm25_morph_checks,  # 반환 3: BM25 형태소 체크박스
                    token_info_box,     # 반환 4: 쿼리 토큰 분석
                    search_status_box,  # 반환 5: 검색 상태
                ],
            )
            # 색인 다시 로딩 버튼 → load_existing_indexes 재실행 후 상태 갱신
            def reload_and_status():
                load_existing_indexes()
                return get_index_status()
            reload_index_btn.click(
                fn=reload_and_status,
                outputs=[index_status_box],
            )

        # ── 탭 4: RAG 생성 ────────────────────────
        with gr.Tab("✨ RAG 생성"):
            gr.Markdown("검색 결과에서 선택한 컨텍스트를 LLM에 전달하여 답변을 생성합니다.")

            with gr.Accordion("📦 LLM 로딩", open=True):
                with gr.Row():
                    model_dd    = gr.Dropdown(label=f"로컬 모델 ({LOCAL_MODEL_ROOT})",
                                              choices=scan_local_models(),
                                              interactive=True, allow_custom_value=True, scale=4)
                    refresh_btn = gr.Button("🔄", scale=1, min_width=50)
                    scan_lbl    = gr.Textbox(label="스캔", interactive=False, scale=1)
                with gr.Row():
                    quant_radio = gr.Radio(["4bit","8bit","fp16","fp32"], value="4bit",
                                           label="양자화", scale=3)
                    trust_chk   = gr.Checkbox(label="trust_remote_code", scale=1)
                    load_btn    = gr.Button("🚀 로딩", variant="primary", scale=1)
                    unload_btn  = gr.Button("🗑️ 언로드", variant="stop", scale=1)
                load_status = gr.Textbox(label="로딩 상태", lines=2, interactive=False)

                refresh_btn.click(fn=refresh_model_list, outputs=[model_dd, scan_lbl])
                load_btn.click(fn=load_model, inputs=[model_dd, quant_radio, trust_chk],
                               outputs=[load_status])
                unload_btn.click(fn=unload_model, outputs=[load_status])

            gr.Markdown("---")
            gr.Markdown("### 검색 탭 결과 가져오기 → 원하는 항목 선택 → RAG 생성")

            sync_btn = gr.Button("🔄 검색 탭 결과 가져오기", size="sm")

            # 3개 검색기 결과 나란히 (검색 탭과 동일한 레이아웃)
            with gr.Row():
                gen_faiss_checks = gr.CheckboxGroup(
                    label="🧠 FAISS 선택 결과", choices=[], interactive=True, scale=1)
                gen_bm25_raw_checks = gr.CheckboxGroup(
                    label="🔤 BM25 원문 선택 결과", choices=[], interactive=True, scale=1)
                gen_bm25_morph_checks = gr.CheckboxGroup(
                    label="🔬 BM25 형태소 선택 결과", choices=[], interactive=True, scale=1)

            gen_query = gr.Textbox(
                label="질문  ※ Shift+Enter = 줄바꿈",
                placeholder="검색 탭의 질문을 입력하세요.",
                lines=3, max_lines=8, submit_btn=False,
            )
            with gr.Row():
                max_tok    = gr.Slider(32, 512, value=256, step=32, label="max_new_tokens")
                temp_sl    = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="temperature")
                sample_chk = gr.Checkbox(label="do_sample", value=True)
            gen_btn = gr.Button("✨ RAG 생성", variant="primary", size="lg")
            gen_out = gr.Textbox(label="📝 생성 결과", lines=12, interactive=False)

            # 검색 탭 3개 체크박스 → 생성 탭으로 동기화
            sync_btn.click(
                fn=lambda f, r, m: (
                    gr.CheckboxGroup(choices=list(f), value=list(f)),
                    gr.CheckboxGroup(choices=list(r), value=list(r)),
                    gr.CheckboxGroup(choices=list(m), value=list(m)),
                ),
                inputs=[faiss_checks, bm25_raw_checks, bm25_morph_checks],
                outputs=[gen_faiss_checks, gen_bm25_raw_checks, gen_bm25_morph_checks],
            )
            gen_btn.click(
                fn=generate_answer,
                inputs=[gen_query,
                        gen_faiss_checks, gen_bm25_raw_checks, gen_bm25_morph_checks,
                        max_tok, temp_sl, sample_chk],
                outputs=[gen_out],
            )
    return demo


# ──────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(LOCAL_MODEL_ROOT, exist_ok=True)
    os.makedirs(INDEX_SAVE_DIR,   exist_ok=True)

    # 재시작 시 기존 색인 파일 자동 로딩 (청크 + FAISS + BM25 원문/형태소)
    load_existing_indexes()

    demo = build_ui()
    # safe_launch(): Colab이면 share=True 자동 설정 + inbrowser/server_name 제거
    #               로컬이면 아래 kwargs 그대로 사용
    safe_launch(
        demo,
        theme=gr.themes.Soft(),
        server_name="0.0.0.0",
        server_port=7862,
        share=False,       # Colab 환경에서는 safe_launch가 자동으로 True로 전환
        inbrowser=True,
    )
