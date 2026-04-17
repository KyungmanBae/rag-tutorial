"""
===========================================================
KorFactScore 실습 — 실제 저장소 연동 버전
===========================================================
KorFactScore GitHub 저장소의 실제 코드와 데이터를 활용합니다.

[사전 준비]
  # 1. 저장소 클론
  git clone https://github.com/ETRI-XAINLP/KorFactScore.git

  # 2. 패키지 설치
  pip install gradio transformers bitsandbytes accelerate
  pip install rank-bm25 sqlite-utils

  # 3. 위키백과 DB 다운로드 (KorFactScore README 참고)
  #    kowiki-20240301.db → KorFactScore/downloaded_files/ 에 위치

[실행]
  python korfactscore_lab.py --korfs_path ./KorFactScore

[데이터 흐름]
  ① k_labeled/*.jsonl   — atomic facts 포함된 GPT-4 생성 약력 (라벨링 완료)
     → LLM 없이도 실행 가능 (atomic facts 재사용)
  ② k_unlabeled/*.jsonl — atomic facts 미포함 GPT-4 생성 약력
     → LLM 로딩 시 atomic facts 분해 가능
  ③ k_truth_annotations — 인간 정답 라벨
     → 시스템 판단과 인간 정답 비교 가능

[탭 구성]
  탭1. 데이터 탐색     — 저장소 데이터 확인 및 샘플 조회
  탭2. 사실 검증       — 개별 문장 또는 데이터셋 배치 실행
  탭3. 성능 평가       — 시스템 판단 vs 인간 정답 비교
===========================================================
"""

import os
import sys
import re
import json
import sqlite3
import argparse
import gradio as gr
import torch
from pathlib import Path
from collections import defaultdict
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, pipeline,
)

# ──────────────────────────────────────────────
# 저장소 경로 설정
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--korfs_path", type=str,
    default="./KorFactScore",
    help="KorFactScore 저장소 경로",
)
args, _ = parser.parse_known_args()

KORFS_PATH = args.korfs_path

# KorFactScore 저장소 모듈 경로 추가
if os.path.isdir(KORFS_PATH):
    sys.path.insert(0, KORFS_PATH)

# 데이터 경로
DATA_LABELED   = os.path.join(KORFS_PATH, "data", "k_labeled")
DATA_UNLABELED = os.path.join(KORFS_PATH, "data", "k_unlabeled")
DATA_TRUTH     = os.path.join(KORFS_PATH, "data", "k_truth_annotations")
DB_PATH        = os.path.join(KORFS_PATH, "downloaded_files", "kowiki-20240301.db")
SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"

# ──────────────────────────────────────────────
# 전역 상태
# ──────────────────────────────────────────────
LOCAL_MODEL_ROOT = "./hf_model"
current_model    = {"pipe": None, "name": None}
bm25_cache: dict = {}   # {topic: (BM25Okapi, passages)}
_db_schema_cache: dict = {}


# ══════════════════════════════════════════════
# 유틸
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
                if sub.is_dir() and os.path.isfile(
                        os.path.join(sub.path, "config.json")):
                    found.append(sub.path)
    return found


def _normalize_title(title: str) -> str:
    title = (title or "").strip()
    title = title.replace("_", " ")
    title = re.sub(r"\s+", " ", title)
    return title


def _detect_db_schema(db_path: str) -> dict:
    if db_path in _db_schema_cache:
        return _db_schema_cache[db_path]

    schema = {
        "table": None,
        "title_col": "title",
        "text_col": "text",
        "splitter": None,
        "error": None,
    }

    if not os.path.exists(db_path):
        _db_schema_cache[db_path] = schema
        return schema

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cur.fetchall()}

        if "documents" in tables:
            schema.update({
                "table": "documents",
                "title_col": "title",
                "text_col": "text",
                "splitter": SPECIAL_SEPARATOR,
            })
        elif "passages" in tables:
            schema.update({
                "table": "passages",
                "title_col": "title",
                "text_col": "text",
                "splitter": None,
            })
        else:
            schema["error"] = f"지원되지 않는 테이블: {sorted(tables)}"

        conn.close()
    except Exception as e:
        schema["error"] = str(e)

    _db_schema_cache[db_path] = schema
    return schema


def check_repo_status() -> str:
    """KorFactScore 저장소 설치 상태를 확인합니다."""
    lines = ["=== KorFactScore 저장소 상태 ===\n"]

    # 저장소 존재 여부
    repo_ok = os.path.isdir(KORFS_PATH)
    lines.append(f"{'✅' if repo_ok else '❌'} 저장소: {KORFS_PATH}")

    # 데이터 파일
    for path, label in [
        (DATA_LABELED,   "라벨링 데이터 (k_labeled)"),
        (DATA_UNLABELED, "미라벨 데이터 (k_unlabeled)"),
        (DATA_TRUTH,     "정답 데이터 (k_truth_annotations)"),
    ]:
        exists = os.path.isdir(path)
        if exists:
            files = list(Path(path).glob("*.jsonl"))
            lines.append(f"  ✅ {label}: {len(files)}개 파일")
        else:
            lines.append(f"  ❌ {label}: 없음")

    # DB 파일 및 스키마
    db_ok = os.path.exists(DB_PATH)
    db_size = f"{os.path.getsize(DB_PATH)/1024**3:.1f}GB" if db_ok else ""
    lines.append(f"{'✅' if db_ok else '❌'} 위키백과 DB: {DB_PATH} {db_size}")
    if db_ok:
        schema = _detect_db_schema(DB_PATH)
        if schema["table"]:
            lines.append(f"  ✅ DB 스키마: {schema['table']}({schema['title_col']}, {schema['text_col']})")
        else:
            lines.append(f"  ⚠️ DB 스키마 확인 실패: {schema['error'] or '알 수 없음'}")

    if not repo_ok:
        lines.append(
            "\n[설치 방법]\n"
            "git clone https://github.com/ETRI-XAINLP/KorFactScore.git\n"
            "python korfactscore_lab.py --korfs_path ./KorFactScore"
        )
    if not db_ok:
        lines.append(
            "\n[DB 다운로드]\n"
            "KorFactScore README의 downloaded_files 안내를 따르세요."
        )

    return "\n".join(lines)


# ══════════════════════════════════════════════
# 데이터 로딩
# ══════════════════════════════════════════════

def load_jsonl(path: str) -> list:
    """JSONL 파일을 로딩합니다."""
    if not os.path.exists(path):
        return []
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def scan_data_files() -> dict:
    """
    저장소의 데이터 파일 목록을 반환합니다.

    Returns:
      {
        "labeled":   [경로, ...],   # atomic facts 포함
        "unlabeled": [경로, ...],   # atomic facts 미포함
        "truth":     [경로, ...],   # 인간 정답 라벨
      }
    """
    result = {"labeled": [], "unlabeled": [], "truth": []}
    for key, path in [
        ("labeled",   DATA_LABELED),
        ("unlabeled", DATA_UNLABELED),
        ("truth",     DATA_TRUTH),
    ]:
        if os.path.isdir(path):
            result[key] = sorted(str(p) for p in Path(path).glob("*.jsonl"))
    return result


def get_file_summary(filepath: str) -> str:
    """데이터 파일의 요약 정보를 반환합니다."""
    data = load_jsonl(filepath)
    if not data:
        return "파일이 비어있거나 로딩 실패"

    has_atomic = "annotations" in data[0]
    has_output = "output" in data[0]
    topics     = [d.get("topic", d.get("title", "?")) for d in data]

    lines = [
        f"총 {len(data)}개 문서",
        f"Atomic facts 포함: {'✅' if has_atomic else '❌'}",
        f"생성 텍스트 포함: {'✅' if has_output else '❌'}",
        f"주제 샘플: {', '.join(topics[:5])}{'...' if len(topics) > 5 else ''}",
    ]

    if has_atomic:
        n_facts = sum(
            len([a for sent in d.get("annotations", [])
                 for a in sent.get("model-atomic-facts", [])])
            for d in data
        )
        lines.append(f"전체 atomic facts: {n_facts}개")

    return "\n".join(lines)


def get_sample_record(filepath: str, idx: int) -> tuple:
    """
    데이터 파일에서 idx번째 레코드를 반환합니다.

    Returns:
      (topic, output_text, atomic_facts, display_text)
    """
    data = load_jsonl(filepath)
    if not data or idx >= len(data):
        return "", "", [], "범위를 벗어난 인덱스"

    d = data[idx]
    topic  = d.get("topic", d.get("title", ""))
    output = d.get("output", "")

    # atomic facts 추출 (라벨링된 파일인 경우)
    atomic_facts = []
    if "annotations" in d:
        for sent in d["annotations"]:
            for atom in sent.get("model-atomic-facts", []):
                atomic_facts.append(atom.get("text", ""))

    display = (
        f"주제: {topic}\n\n"
        f"생성 텍스트:\n{output}\n\n"
    )
    if atomic_facts:
        display += f"Atomic Facts ({len(atomic_facts)}개):\n"
        display += "\n".join(f"  {i+1}. {f}"
                             for i, f in enumerate(atomic_facts))

    return topic, output, atomic_facts, display


# ══════════════════════════════════════════════
# 모델 로딩
# ══════════════════════════════════════════════

def load_model(model_path: str, quant_mode: str,
               progress=gr.Progress()) -> str:
    """로컬 LLM을 로딩합니다."""
    global current_model

    if not model_path or not os.path.exists(model_path.strip()):
        return f"❌ 경로 없음: {model_path}"
    try:
        progress(0.1, desc="토크나이저 로딩 중...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path.strip(), trust_remote_code=True)

        progress(0.3, desc=f"모델 로딩 ({quant_mode})...")
        if quant_mode == "4bit":
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path.strip(), quantization_config=bnb_cfg,
                device_map="auto", trust_remote_code=True)
        elif quant_mode == "8bit":
            model = AutoModelForCausalLM.from_pretrained(
                model_path.strip(), load_in_8bit=True,
                device_map="auto", trust_remote_code=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path.strip(), torch_dtype=torch.float16,
                device_map="auto", trust_remote_code=True)

        progress(0.8, desc="파이프라인 구성 중...")
        current_model["pipe"] = pipeline(
            "text-generation", model=model, tokenizer=tokenizer,
            max_new_tokens=256, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_full_text=False,
        )
        current_model["name"] = os.path.basename(model_path.strip().rstrip("/\\"))

        progress(1.0)
        return (f"✅ 모델 로딩 완료\n"
                f"  {current_model['name']} ({quant_mode})\n"
                f"  {get_device_info()}")
    except Exception as e:
        return f"❌ 오류: {e}"


# ══════════════════════════════════════════════
# 검색 (BM25) — KorFactScore retrieval.py 방식
# ══════════════════════════════════════════════

def load_passages(topic: str) -> list:
    """
    KorFactScore DB에서 topic 단락을 로딩합니다.

    지원 스키마
    - documents(title, text): 원본 retrieval.py가 기대하는 스키마
    - passages(title, text): 실습용/배포본에서 흔한 스키마
    """
    if not os.path.exists(DB_PATH):
        return []

    schema = _detect_db_schema(DB_PATH)
    if not schema["table"]:
        print(f"[DB 오류] 스키마 확인 실패: {schema['error']}")
        return []

    normalized_topic = _normalize_title(topic)

    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()

        table = schema["table"]
        title_col = schema["title_col"]
        text_col = schema["text_col"]

        queries = [
            (
                f"SELECT {title_col}, {text_col} FROM {table} WHERE {title_col} = ?",
                (topic,),
            ),
            (
                f"SELECT {title_col}, {text_col} FROM {table} WHERE {title_col} = ?",
                (normalized_topic,),
            ),
            (
                f"SELECT {title_col}, {text_col} FROM {table} WHERE REPLACE({title_col}, '_', ' ') = ?",
                (normalized_topic,),
            ),
            (
                f"SELECT {title_col}, {text_col} FROM {table} WHERE {title_col} LIKE ? OR REPLACE({title_col}, '_', ' ') LIKE ? LIMIT 50",
                (f"%{topic}%", f"%{normalized_topic}%"),
            ),
        ]

        rows = []
        for query, params in queries:
            cur.execute(query, params)
            rows = cur.fetchall()
            if rows:
                break

        conn.close()

        if not rows:
            return []

        passages = []
        if schema["table"] == "documents":
            for title, text in rows:
                if not text:
                    continue
                for para in text.split(schema["splitter"] or SPECIAL_SEPARATOR):
                    para = para.strip()
                    if para:
                        passages.append({"title": _normalize_title(title), "text": para})
        else:
            for title, text in rows:
                if text and text.strip():
                    passages.append({"title": _normalize_title(title), "text": text.strip()})

        deduped = []
        seen = set()
        for p in passages:
            key = (p["title"], p["text"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(p)
        return deduped
    except Exception as e:
        print(f"[DB 오류] {e}")
        return []


def bm25_retrieve(query: str, topic: str, top_k: int = 5) -> list:
    """
    KorFactScore의 BM25 검색 방식을 그대로 재현합니다.

    KorFactScore retrieval.py 핵심:
      bm25 = BM25Okapi([psg["text"].split() for psg in passages])
      scores = bm25.get_scores(query.split())

    topic 페이지의 단락들만을 대상으로 검색합니다.
    (전체 위키가 아닌 해당 인물 페이지 내 검색)
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return []

    # 캐시 확인
    cache_topic = _normalize_title(topic)
    if cache_topic not in bm25_cache:
        passages = load_passages(topic)
        if not passages:
            return []
        # KorFactScore와 동일: <s>, </s> 태그 제거 후 공백 토큰화
        tokenized = [
            p["text"].replace("<s>", "").replace("</s>", "").split()
            for p in passages
        ]
        tokenized = [tokens if tokens else ["_"] for tokens in tokenized]
        bm25_cache[cache_topic] = (BM25Okapi(tokenized), passages)

    bm25, passages = bm25_cache[cache_topic]

    query_clean = query.replace("<s>", "").replace("</s>", "")
    scores  = bm25.get_scores(query_clean.split())
    ranked  = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    return [
        {"title": passages[i]["title"],
         "text":  passages[i]["text"],
         "score": float(scores[i])}
        for i in ranked[:top_k]
        if scores[i] > 0
    ]


# ══════════════════════════════════════════════
# Atomic Facts 분해
# ══════════════════════════════════════════════

def decompose_to_atomic_facts(text: str) -> list:
    """
    LLM으로 atomic facts를 분해합니다.

    KorFactScore의 atomic_facts_generator.py 프롬프트 방식 참고:
      "Please breakdown the following sentence into independent facts"

    LLM 없으면 문장 단위 분리 fallback.
    """
    if current_model["pipe"] is None:
        # 문장 단위 fallback
        sents = [s.strip() for s in re.split(r'[.。]', text) if s.strip()]
        return [s + "." if not s.endswith((".", "?", "!")) else s
                for s in sents]

    # KorFactScore atomic_facts_generator 스타일 프롬프트
    prompt = (
        "다음 문장을 독립적인 단일 사실(atomic facts)로 분해하세요.\n"
        "각 사실은 하나의 정보만 포함해야 합니다.\n"
        "JSON 배열로만 출력하세요.\n\n"
        f"문장: {text}\n"
        "단일 사실 목록:"
    )
    try:
        out = current_model["pipe"](prompt, max_new_tokens=256,
                                    do_sample=False, return_full_text=False)
        raw = out[0]["generated_text"].strip()
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            facts = json.loads(match.group())
            if isinstance(facts, list):
                return [f.strip() for f in facts if f.strip()]
        # 줄 단위 파싱 fallback
        return [l.strip().lstrip("-•· 0123456789.").strip()
                for l in raw.split("\n")
                if l.strip() and len(l.strip()) > 5]
    except Exception:
        sents = [s.strip() for s in re.split(r'[.。]', text) if s.strip()]
        return [s + "." if not s.endswith(".") else s for s in sents]


# ══════════════════════════════════════════════
# 사실 판단
# ══════════════════════════════════════════════

def verify_single_fact(atomic_fact: str, topic: str,
                        passages: list) -> dict:
    """
    KorFactScore의 판단 프롬프트 방식을 재현합니다.

    원본 FActScore/KorFactScore 프롬프트:
      "Answer the question about {topic} based on the given context.
       Title: {title}
       Text: {text}
       Input: {atomic_fact} True or False?
       Output:"

    Returns:
      {"label": "SUPPORTED"|"NOT SUPPORTED"|"UNKNOWN",
       "evidence": str, "score": float, "reasoning": str}
    """
    if not passages:
        return {"label": "UNKNOWN", "evidence": "",
                "score": 0.0, "reasoning": "검색 결과 없음"}

    best = passages[0]

    if current_model["pipe"] is None:
        # 데모 모드: 단어 겹침 기반
        words = set(w for w in atomic_fact.split() if len(w) > 1)
        overlap = sum(1 for w in words if w in best["text"])
        label = "SUPPORTED" if overlap >= 2 else "NOT SUPPORTED"
        return {"label": label, "evidence": best["text"],
                "score": best["score"],
                "reasoning": f"[데모] 겹침 단어 {overlap}개"}

    # KorFactScore / FActScore 원본 프롬프트 형식
    context = ""
    for p in passages[:3]:
        title = p["title"]
        text  = p["text"].replace("<s>", "").replace("</s>", "")
        context += f"Title: {title}\nText: {text}\n\n"

    definition = f"{topic}에 대한 다음 문서를 참고하여 질문에 답하세요.\n\n"
    prompt = (
        f"{definition}"
        f"{context}"
        f"Input: {atomic_fact} 참입니까, 거짓입니까?\n"
        f"Output:"
    )

    try:
        out = current_model["pipe"](prompt, max_new_tokens=50,
                                    do_sample=False, return_full_text=False)
        raw = out[0]["generated_text"].strip().lower()

        # "참" > "거짓" 위치 비교 — FActScore 원본 방식
        has_true  = "참" in raw or "true" in raw or "지지" in raw or "맞" in raw
        has_false = "거짓" in raw or "false" in raw or "틀" in raw or "아니" in raw

        if has_true and has_false:
            # 둘 다 있으면 먼저 나온 쪽
            t_pos = min(
                raw.index("참")  if "참"  in raw else 9999,
                raw.index("true") if "true" in raw else 9999,
            )
            f_pos = min(
                raw.index("거짓") if "거짓" in raw else 9999,
                raw.index("false") if "false" in raw else 9999,
            )
            label = "SUPPORTED" if t_pos < f_pos else "NOT SUPPORTED"
        elif has_true:
            label = "SUPPORTED"
        elif has_false:
            label = "NOT SUPPORTED"
        else:
            label = "UNKNOWN"

        return {"label": label, "evidence": best["text"],
                "score": best["score"], "reasoning": raw[:100]}
    except Exception as e:
        return {"label": "UNKNOWN", "evidence": best["text"],
                "score": best["score"], "reasoning": str(e)}


# ══════════════════════════════════════════════
# 핵심 실행 함수
# ══════════════════════════════════════════════

def run_on_record(topic: str, output: str,
                  atomic_facts_input: str,
                  top_k: int,
                  progress=gr.Progress()) -> tuple:
    """
    단일 레코드에 대해 KorFactScore 파이프라인을 실행합니다.

    atomic_facts_input이 있으면 분해 단계를 건너뜁니다.
    (k_labeled 데이터의 --use_atomic_facts 옵션과 동일)

    Returns: (결과 텍스트, 검색 상세, FactScore 수치)
    """
    bm25_cache.clear()

    if not topic.strip() or not output.strip():
        return "❌ 주제와 텍스트를 입력하세요.", "", "—"

    # ── Step 1: Atomic Facts ──────────────────
    progress(0.05, desc="Step 1: Atomic Facts 준비 중...")

    if atomic_facts_input.strip():
        # 이미 분해된 facts 사용 (k_labeled 데이터)
        atomic_facts = [
            f.strip() for f in atomic_facts_input.strip().split("\n")
            if f.strip()
        ]
        af_source = "데이터셋 제공 (LLM 분해 불필요)"
    else:
        # LLM으로 분해
        atomic_facts = decompose_to_atomic_facts(output)
        af_source = (f"LLM 분해: {current_model['name']}"
                     if current_model["pipe"] else "규칙 기반 분리 (LLM 미로딩)")

    # ── Step 2~3: 검색 + 판단 ─────────────────
    results = []
    retrieval_lines = []

    for i, fact in enumerate(atomic_facts):
        progress(0.1 + 0.8 * i / max(len(atomic_facts), 1),
                 desc=f"Step 2~3: [{i+1}/{len(atomic_facts)}] {fact[:30]}...")

        passages = bm25_retrieve(fact, topic.strip(), top_k=int(top_k))
        verdict  = verify_single_fact(fact, topic.strip(), passages)
        verdict["fact"] = fact
        verdict["passages"] = passages
        results.append(verdict)

        # 검색 상세 기록
        retrieval_lines.append(f"[{i+1}] {fact}")
        if passages:
            for j, p in enumerate(passages, 1):
                retrieval_lines.append(
                    f"  #{j} score={p['score']:.3f} | {p['text'][:80]}...")
        else:
            if not os.path.exists(DB_PATH):
                retrieval_lines.append("  ⚠️ DB 없음 — kowiki-20240301.db를 다운로드하세요")
            else:
                schema = _detect_db_schema(DB_PATH)
                if schema["table"]:
                    retrieval_lines.append(
                        f"  검색 결과 없음 (DB 스키마: {schema['table']}, topic 매칭 실패 가능)"
                    )
                else:
                    retrieval_lines.append(
                        f"  ⚠️ DB 스키마 오류: {schema['error'] or '알 수 없음'}"
                    )
        retrieval_lines.append("")

    # ── Step 4: FactScore ─────────────────────
    progress(0.95, desc="Step 4: FactScore 계산 중...")

    n_total     = len(results)
    n_supported = sum(1 for r in results if r["label"] == "SUPPORTED")
    n_not       = sum(1 for r in results if r["label"] == "NOT SUPPORTED")
    n_unknown   = sum(1 for r in results if r["label"] == "UNKNOWN")
    factscore   = n_supported / n_total if n_total > 0 else 0.0

    # ── 결과 포매팅 ───────────────────────────
    sep = "─" * 55
    lines = [
        "📊 KorFactScore 결과",
        f"  주제      : {topic}",
        f"  AF 출처   : {af_source}",
        f"  검색기    : BM25 Top-{top_k} (KorFactScore 방식)",
        f"  판단 모델 : {current_model['name'] or '데모 모드'}",
        sep,
        f"  전체 사실 : {n_total}개",
        f"  ✅ SUPPORTED    : {n_supported}개",
        f"  ❌ NOT SUPPORTED: {n_not}개",
        f"  ❓ UNKNOWN      : {n_unknown}개",
        sep,
        f"  🎯 FactScore    : {factscore:.1%}",
        sep, "",
        "=== 사실별 판단 ===",
    ]

    for i, r in enumerate(results, 1):
        icon = {"SUPPORTED": "✅", "NOT SUPPORTED": "❌",
                "UNKNOWN": "❓"}.get(r["label"], "❓")
        lines += [
            f"\n[{i}] {icon} {r['label']}",
            f"  사실  : {r['fact']}",
            f"  근거  : {r['evidence'][:100]}..." if r['evidence'] else "  근거  : 없음",
            f"  이유  : {r['reasoning'][:80]}" if r['reasoning'] else "",
        ]

    progress(1.0)
    return "\n".join(lines), "\n".join(retrieval_lines), f"{factscore:.1%}"


def run_batch_evaluation(filepath: str, n_samples: int,
                          top_k: int,
                          progress=gr.Progress()) -> str:
    """
    데이터 파일 전체(또는 일부)에 대해 배치 평가를 실행합니다.

    k_labeled 파일의 경우 atomic facts를 재사용합니다.
    KorFactScore README의 배치 실행 방식과 동일합니다.
    """
    data = load_jsonl(filepath)
    if not data:
        return "❌ 파일을 로딩할 수 없습니다."

    n = min(int(n_samples), len(data)) if n_samples > 0 else len(data)
    data = data[:n]

    total_facts     = 0
    total_supported = 0
    per_doc_scores  = []
    lines           = [f"배치 평가 — {os.path.basename(filepath)} ({n}개 문서)\n"]

    for i, d in enumerate(data):
        progress(i / n, desc=f"[{i+1}/{n}] {d.get('topic', '?')} 평가 중...")

        topic  = d.get("topic", d.get("title", ""))
        bm25_cache.clear()

        # atomic facts 추출
        if "annotations" in d:
            # k_labeled: 기존 atomic facts 재사용
            facts = [a.get("text", "")
                     for sent in d["annotations"]
                     for a in sent.get("model-atomic-facts", [])
                     if a.get("text")]
        else:
            # k_unlabeled: LLM 또는 규칙 기반 분해
            facts = decompose_to_atomic_facts(d.get("output", ""))

        if not facts:
            continue

        # 각 사실 검증
        n_sup = 0
        for fact in facts:
            passages = bm25_retrieve(fact, topic, top_k=int(top_k))
            verdict  = verify_single_fact(fact, topic, passages)
            if verdict["label"] == "SUPPORTED":
                n_sup += 1

        score = n_sup / len(facts) if facts else 0.0
        per_doc_scores.append(score)
        total_facts     += len(facts)
        total_supported += n_sup

        lines.append(f"[{i+1}] {topic}: {score:.1%} "
                     f"({n_sup}/{len(facts)} SUPPORTED)")

    # 전체 요약
    overall = total_supported / total_facts if total_facts > 0 else 0.0
    avg     = sum(per_doc_scores) / len(per_doc_scores) if per_doc_scores else 0.0
    lines += [
        "",
        "─" * 50,
        f"전체 FactScore (macro avg): {avg:.1%}",
        f"전체 FactScore (micro):     {overall:.1%}",
        f"총 사실 수: {total_facts}개 | SUPPORTED: {total_supported}개",
    ]

    return "\n".join(lines)


def compare_with_truth(system_filepath: str,
                        truth_filepath: str) -> str:
    """
    시스템 판단 결과와 인간 정답을 비교합니다.

    KorFactScore의 evaluate_system_vs_human_judgments.py 방식 참고:
      정답(ground truth)과 시스템 결과 간의 일치도를 계산합니다.
    """
    system_data = load_jsonl(system_filepath)
    truth_data  = load_jsonl(truth_filepath)

    if not system_data or not truth_data:
        return "❌ 파일을 로딩할 수 없습니다."

    # topic 기준으로 매칭
    truth_by_topic = {
        d.get("topic", d.get("title", "")): d
        for d in truth_data
    }

    agree = 0
    total = 0
    lines = ["=== 시스템 판단 vs 인간 정답 비교 ===\n"]

    for sd in system_data:
        topic = sd.get("topic", sd.get("title", ""))
        td    = truth_by_topic.get(topic)
        if not td:
            continue

        # 시스템 사실 판단 결과 추출
        sys_decisions = sd.get("decisions", [])
        # 인간 정답 라벨 추출
        truth_labels  = [
            a.get("human-judgment", "")
            for sent in td.get("annotations", [])
            for a in sent.get("model-atomic-facts", [])
        ]

        for sys_d, truth_l in zip(sys_decisions, truth_labels):
            total += 1
            sys_label   = "S" if sys_d else "NS"
            truth_label = truth_l
            if (sys_label == "S") == (truth_label == "S"):
                agree += 1

        lines.append(f"  {topic}: 매칭된 사실 {min(len(sys_decisions), len(truth_labels))}개")

    accuracy = agree / total if total > 0 else 0.0
    lines += [
        "",
        "─" * 50,
        f"전체 일치율: {accuracy:.1%} ({agree}/{total})",
    ]

    return "\n".join(lines)


# ══════════════════════════════════════════════
# Gradio UI
# ══════════════════════════════════════════════

def build_ui():

    files = scan_data_files()
    all_data_files = files["labeled"] + files["unlabeled"]

    with gr.Blocks(title="KorFactScore 실습", theme=gr.themes.Soft()) as demo:

        gr.Markdown(
            "# 🔍 KorFactScore 실습 (저장소 연동 버전)\n"
            "ETRI KorFactScore 저장소의 **실제 코드와 데이터**를 활용합니다.\n\n"
            "> **RAG 활용 사례**: Atomic Fact 검색(BM25 위키백과) → LLM 사실 판단"
        )

        # 저장소 상태
        repo_status = gr.Textbox(
            label="저장소 상태",
            value=check_repo_status(),
            lines=8, interactive=False,
        )
        refresh_btn = gr.Button("🔄 상태 새로고침", size="sm")
        refresh_btn.click(fn=check_repo_status, outputs=[repo_status])

        gr.Markdown("---")

        # ── 모델 로딩 (공통) ───────────────────
        with gr.Accordion("⚙️ 모델 로딩 (선택사항)", open=False):
            gr.Markdown(
                "로딩하면 Atomic Facts 분해와 사실 판단에 LLM을 사용합니다.  \n"
                "미로딩 시: k_labeled 데이터는 **기존 atomic facts 재사용**, "
                "판단은 규칙 기반으로 동작합니다."
            )
            with gr.Row():
                model_dd   = gr.Dropdown(
                    choices=scan_local_models(), value=None,
                    allow_custom_value=True, label="모델 경로", scale=4)
                model_scan = gr.Button("🔄", scale=1, min_width=40)
            with gr.Row():
                quant_r    = gr.Radio(["4bit","8bit","fp16"], value="4bit",
                                      label="양자화", scale=2)
                model_btn  = gr.Button("📥 로딩", variant="primary", scale=1)
            model_out = gr.Textbox(
                value=f"미로딩\n{get_device_info()}",
                lines=2, interactive=False)
            model_scan.click(fn=lambda: gr.Dropdown(choices=scan_local_models()),
                             outputs=[model_dd])
            model_btn.click(fn=load_model, inputs=[model_dd, quant_r],
                            outputs=[model_out])

        gr.Markdown("---")

        with gr.Tabs():

            # ════════════════════════════════
            # 탭1. 데이터 탐색
            # ════════════════════════════════
            with gr.Tab("📂 탭1. 데이터 탐색"):
                gr.Markdown(
                    "KorFactScore 저장소의 실제 데이터를 확인합니다.\n\n"
                    "- **k_labeled**: atomic facts 포함 (GPT-4 생성 + 인간 검수)\n"
                    "- **k_unlabeled**: atomic facts 미포함 (원본 생성 텍스트만)\n"
                    "- **k_truth_annotations**: 인간이 직접 라벨링한 정답"
                )
                with gr.Row():
                    file_dd = gr.Dropdown(
                        label="데이터 파일 선택",
                        choices=all_data_files,
                        value=all_data_files[0] if all_data_files else None,
                        scale=4,
                    )
                    file_scan_btn = gr.Button("🔄 스캔", scale=1)

                file_info = gr.Textbox(label="파일 요약", lines=4,
                                       interactive=False)

                with gr.Row():
                    idx_sl     = gr.Slider(0, 100, value=0, step=1,
                                           label="레코드 인덱스", scale=3)
                    load_rec_btn = gr.Button("📄 레코드 보기", scale=1)

                rec_display = gr.Textbox(label="레코드 내용", lines=15,
                                         interactive=False)

                # 탭2로 보내는 버튼
                send_btn = gr.Button("→ 이 레코드로 사실 검증하기", variant="primary")

                # 숨겨진 상태값 (탭2 전달용)
                state_topic = gr.State("")
                state_output = gr.State("")
                state_facts = gr.State([])

                def on_file_select(path):
                    if not path:
                        return "파일을 선택하세요."
                    return get_file_summary(path)

                def on_load_record(path, idx):
                    return get_sample_record(path, int(idx))

                file_scan_btn.click(
                    fn=lambda: gr.Dropdown(choices=scan_data_files()["labeled"]
                                           + scan_data_files()["unlabeled"]),
                    outputs=[file_dd])
                file_dd.change(fn=on_file_select, inputs=[file_dd],
                               outputs=[file_info])
                load_rec_btn.click(
                    fn=on_load_record,
                    inputs=[file_dd, idx_sl],
                    outputs=[state_topic, state_output, state_facts, rec_display],
                )

                # 파일 선택 시 자동 요약
                if all_data_files:
                    file_info.value = get_file_summary(all_data_files[0])

            # ════════════════════════════════
            # 탭2. 사실 검증 (단일)
            # ════════════════════════════════
            with gr.Tab("🔍 탭2. 사실 검증"):
                gr.Markdown(
                    "### 단일 레코드 검증\n"
                    "**탭1에서 레코드 불러오기** 또는 직접 입력할 수 있습니다.\n\n"
                    "k_labeled 데이터의 atomic facts를 붙여넣으면 "
                    "**LLM 없이도 BM25 검색 + 규칙 판단**으로 실행됩니다."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        topic_box = gr.Textbox(
                            label="주제 (위키백과 표제어)",
                            placeholder="예: 세종대왕",
                        )
                        output_box = gr.Textbox(
                            label="생성 텍스트",
                            lines=5, placeholder="LLM이 생성한 텍스트",
                        )
                        facts_box = gr.Textbox(
                            label="Atomic Facts (한 줄에 하나, 비우면 자동 분해)",
                            lines=6,
                            placeholder="k_labeled 데이터의 atomic facts를 붙여넣거나\n비워두면 LLM이 자동 분해합니다.",
                        )
                        with gr.Row():
                            topk_sl = gr.Slider(1, 10, value=5, step=1,
                                                label="BM25 Top-K")
                        run_single_btn = gr.Button("🚀 사실 검증 실행",
                                                   variant="primary", size="lg")

                    with gr.Column(scale=1):
                        score_box = gr.Textbox(label="🎯 FactScore",
                                               value="—", lines=1,
                                               interactive=False)
                        result_box = gr.Textbox(label="판단 결과",
                                                lines=20, interactive=False)
                        retrieval_box = gr.Textbox(label="BM25 검색 상세",
                                                   lines=8, interactive=False)

                # 탭1에서 데이터 받기
                def fill_from_tab1(topic, output, facts):
                    facts_str = "\n".join(facts) if facts else ""
                    return topic, output, facts_str

                send_btn.click(
                    fn=fill_from_tab1,
                    inputs=[state_topic, state_output, state_facts],
                    outputs=[topic_box, output_box, facts_box],
                )

                run_single_btn.click(
                    fn=run_on_record,
                    inputs=[topic_box, output_box, facts_box, topk_sl],
                    outputs=[result_box, retrieval_box, score_box],
                )

            # ════════════════════════════════
            # 탭3. 배치 평가
            # ════════════════════════════════
            with gr.Tab("📊 탭3. 배치 평가"):
                gr.Markdown(
                    "### 데이터 파일 전체 평가\n"
                    "KorFactScore 저장소의 실제 데이터로 배치 평가를 실행합니다.\n\n"
                    "k_labeled 파일 사용 시 atomic facts를 재사용하므로 빠릅니다."
                )

                with gr.Row():
                    batch_file_dd = gr.Dropdown(
                        label="평가할 데이터 파일",
                        choices=all_data_files,
                        value=all_data_files[0] if all_data_files else None,
                        scale=4,
                    )
                    batch_scan_btn = gr.Button("🔄", scale=1, min_width=40)

                with gr.Row():
                    n_samples_sl = gr.Slider(
                        0, 64, value=5, step=1,
                        label="평가 문서 수 (0=전체)",
                        info="0이면 파일 전체. 빠른 테스트용으로 5~10 권장",
                    )
                    batch_topk = gr.Slider(1, 10, value=5, step=1,
                                           label="BM25 Top-K")

                batch_btn = gr.Button("📊 배치 평가 실행", variant="primary",
                                      size="lg")
                batch_out = gr.Textbox(label="배치 평가 결과", lines=20,
                                       interactive=False)

                batch_scan_btn.click(
                    fn=lambda: gr.Dropdown(
                        choices=scan_data_files()["labeled"]
                               + scan_data_files()["unlabeled"]),
                    outputs=[batch_file_dd])
                batch_btn.click(
                    fn=run_batch_evaluation,
                    inputs=[batch_file_dd, n_samples_sl, batch_topk],
                    outputs=[batch_out],
                )

                # ── 인간 정답 비교 ──────────────
                gr.Markdown("---")
                gr.Markdown(
                    "### 시스템 판단 vs 인간 정답 비교\n"
                    "KorFactScore의 `evaluate_system_vs_human_judgments` 방식으로\n"
                    "시스템 결과와 인간 정답의 일치도를 계산합니다."
                )
                with gr.Row():
                    sys_file_dd = gr.Dropdown(
                        label="시스템 결과 파일 (decisions 포함)",
                        choices=files["labeled"],
                        value=files["labeled"][0] if files["labeled"] else None,
                        scale=3,
                    )
                    truth_file_dd = gr.Dropdown(
                        label="인간 정답 파일",
                        choices=files["truth"],
                        value=files["truth"][0] if files["truth"] else None,
                        scale=3,
                    )

                compare_btn = gr.Button("🔍 비교 실행")
                compare_out = gr.Textbox(label="비교 결과", lines=10,
                                         interactive=False)
                compare_btn.click(
                    fn=compare_with_truth,
                    inputs=[sys_file_dd, truth_file_dd],
                    outputs=[compare_out],
                )

    return demo


# ──────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(LOCAL_MODEL_ROOT, exist_ok=True)
    print(check_repo_status())

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7867,
        inbrowser=True,
    )
