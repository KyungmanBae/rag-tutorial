"""
===========================================================
KorQuAD v1.0 기반 RAG 실습 — Gradio UI 통합
===========================================================
[탭 구성]
  Step 1. 데이터 준비
          - KorQuAD train/dev JSON → 단락 corpus 색인
          - (query, positive, negative×N) 트리플렛 생성

  Step 2. 모델 로딩 & Fine-tuning
          - SentenceTransformer 베이스 모델 로딩 + FAISS 구축
          - TripletLoss / MultipleNegativesRankingLoss 선택 학습

  Step 3. 검색 결과 비교
          - 동일 질문으로 베이스/파인튜닝 모델 결과 나란히 비교

  Step 4. 정량 평가 (MRR@K / Recall@K)
          - dev 전체 질문으로 베이스/파인튜닝 모델 성능 수치 비교

[설치]
  pip install gradio sentence-transformers faiss-cpu torch numpy
===========================================================
"""

import os
import json
import random
import time
import numpy as np
import gradio as gr
import faiss
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# ──────────────────────────────────────────────
# 경로 / 기본값 설정
# ──────────────────────────────────────────────
OUTPUT_DIR   = "./data/korquad_output"  # 생성 파일 저장 위치
INDEX_DIR    = "./data/korquad_output/index"  # FAISS 인덱스 저장 전용 폴더
DEFAULT_MODEL = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"  # 기본 임베딩 모델

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INDEX_DIR,  exist_ok=True)

# ──────────────────────────────────────────────
# 전역 상태
# ──────────────────────────────────────────────
corpus:         list  = []    # 전체 단락 목록 [{para_id, title, context, split}]
triplets:       list  = []    # 학습 트리플렛 [{qa_id, query, pos_para_id, ...}]
eval_questions: list  = []    # 평가 질문 [{qa_id, question, pos_para_id, title}]
base_model:   SentenceTransformer = None
tuned_model:  SentenceTransformer = None
base_index:   faiss.Index = None
tuned_index:  faiss.Index = None
id_map:       list  = []    # FAISS 행 번호 → para_id

# 직접 입력 모델 캐시: {절대경로: (SentenceTransformer, faiss.Index)}
# 같은 경로를 다시 평가할 때 모델 로딩 + 임베딩 재생성을 건너뜁니다.
# corpus가 바뀌면 캐시도 무효화됩니다 (build_corpus 호출 시 자동 초기화).
_eval_model_cache: dict = {}


# ══════════════════════════════════════════════
# 공통 유틸
# ══════════════════════════════════════════════

def get_status() -> str:
    """현재 전역 상태를 한눈에 보여줍니다."""
    lines = [
        f"{'✅' if corpus        else '❌'} Corpus   : {len(corpus)}개 단락",
        f"{'✅' if triplets      else '❌'} 트리플렛 : {len(triplets)}개",
        f"{'✅' if eval_questions else '❌'} 평가 질문: {len(eval_questions)}개",
        f"{'✅' if base_model    else '❌'} 베이스 모델: {'로딩됨' if base_model  else '미로딩'}",
        f"{'✅' if tuned_model   else '❌'} 파인튜닝 모델: {'완료' if tuned_model else '미학습'}",
    ]
    return "\n".join(lines)


def encode_and_index(model: SentenceTransformer, corpus_data: list,
                     batch_size: int = 128):
    """
    corpus_data 전체를 임베딩하고 FAISS IndexFlatIP(코사인)를 구축합니다.
    L2 정규화 후 내적 = 코사인 유사도

    Returns:
        index  : faiss.Index
        imap   : List[str]  — FAISS 행 번호 → para_id 매핑
    """
    texts = [c["context"] for c in corpus_data]
    imap  = [c["para_id"] for c in corpus_data]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, imap


def save_index(index: faiss.Index, imap: list, name: str):
    """
    FAISS 인덱스와 para_id 매핑을 INDEX_DIR에 저장합니다.

    저장 파일:
      {INDEX_DIR}/{name}.faiss      — FAISS 인덱스 바이너리
      {INDEX_DIR}/{name}_imap.json  — 행 번호 → para_id 매핑

    name 예시: "base", "tuned_model_TL_ep3_bs16_n12000"
    """
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss_path = os.path.join(INDEX_DIR, f"{name}.faiss")
    imap_path  = os.path.join(INDEX_DIR, f"{name}_imap.json")
    faiss.write_index(index, faiss_path)
    with open(imap_path, "w", encoding="utf-8") as f:
        json.dump(imap, f, ensure_ascii=False)
    print(f"[INDEX] 저장 완료: {faiss_path}  ({index.ntotal}벡터)")


def load_index(name: str):
    """
    INDEX_DIR에서 FAISS 인덱스와 imap을 복원합니다.
    파일이 없거나 불일치하면 (None, None)을 반환합니다.

    name 예시: "base", "tuned_model_TL_ep3_bs16_n12000"
    """
    faiss_path = os.path.join(INDEX_DIR, f"{name}.faiss")
    imap_path  = os.path.join(INDEX_DIR, f"{name}_imap.json")
    if not os.path.exists(faiss_path) or not os.path.exists(imap_path):
        return None, None
    index = faiss.read_index(faiss_path)
    with open(imap_path, "r", encoding="utf-8") as f:
        imap = json.load(f)
    print(f"[INDEX] 로딩 완료: {faiss_path}  ({index.ntotal}벡터)")
    return index, imap



def search(query: str, model: SentenceTransformer,
           index: faiss.Index, imap: list, top_k: int) -> list:
    """
    단일 쿼리를 임베딩하고 FAISS로 top_k 결과를 반환합니다.
    imap: encode_and_index()가 반환한 행 번호 → para_id 매핑 리스트
    반환: [(para_id, score), ...]
    """
    q_vec = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    scores, indices = index.search(q_vec, top_k)
    return [(imap[idx], float(scores[0][r]))
            for r, idx in enumerate(indices[0]) if idx >= 0]


# ══════════════════════════════════════════════
# Step 1. 데이터 준비
# ══════════════════════════════════════════════

def build_corpus(train_path: str, dev_path: str,
                 progress=gr.Progress()) -> str:
    """
    train + dev JSON의 모든 단락과 질문을 하나로 합친 뒤,
    단락(context) 단위로 9:1 무작위 분할하여 corpus와 평가 질문을 생성합니다.

    변경 이유:
      원본 KorQuAD는 train/dev 단락이 완전히 분리되어 있습니다.
      이 경우 검색기 학습(트리플렛)에 사용한 단락과 평가에 사용하는 단락이
      항상 달라지므로, 파인튜닝 효과를 공정하게 측정하기 어렵습니다.
      train+dev를 합산한 뒤 9:1로 직접 분할하면 학습 단락과 평가 단락이
      같은 분포를 가지며, 색인(corpus)에는 모든 단락이 포함됩니다.

    para_id 규칙: "all-{global_idx:05d}"
      셔플 후 전체 일련번호를 사용합니다.

    생성 파일:
      corpus.json               — 전체 단락 (train90 + test10), 색인 대상
      eval_questions_train.json — 90% 단락에 속한 질문 (검색기 학습·성능 확인용)
      eval_questions_dev.json   — 10% 단락에 속한 질문 (홀드아웃 평가용)

    ※ corpus 재생성 후 메모리에 로딩된 모델(base/tuned)이 있으면
      FAISS 인덱스를 자동으로 재구축합니다.
      재구축하지 않으면 이전 para_id(train-XXXX, dev-XXXX)로 만든 인덱스가
      남아있어 평가 시 MRR/Recall이 0.0 이 됩니다.
    """
    global corpus, base_index, tuned_index, id_map

    if not train_path or not os.path.exists(train_path):
        return "❌ train 파일 경로를 확인하세요."
    if not dev_path or not os.path.exists(dev_path):
        return "❌ dev 파일 경로를 확인하세요."

    progress(0.0, desc="파일 읽는 중...")

    # ── 1단계: train + dev 전체 단락·질문 수집 ────────────────────────
    all_paras = []   # [{"para_id_tmp", "title", "context", "qas": [...]}]

    for src_split, fpath in [("train", train_path), ("dev", dev_path)]:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for title_idx, article in enumerate(data["data"]):
            title = article["title"]
            for para_idx, para in enumerate(article["paragraphs"]):
                all_paras.append({
                    "para_id_tmp": f"{src_split}-{title_idx:04d}-{para_idx:03d}",
                    "title":       title,
                    "context":     para["context"],
                    "qas":         [{"qa_id": qa["id"],
                                     "question": qa["question"]}
                                    for qa in para["qas"]],
                })

    progress(0.3, desc="9:1 분할 중...")

    # ── 2단계: 단락 단위 셔플 후 9:1 분할 ────────────────────────────
    random.seed(42)
    random.shuffle(all_paras)

    n_total = len(all_paras)
    n_train = int(n_total * 0.9)   # 앞 90% → 검색기 학습용

    corpus = []
    eval_train_qs = []   # 90% 단락의 질문 (학습 도메인 평가)
    eval_dev_qs   = []   # 10% 단락의 질문 (홀드아웃 평가)

    for global_idx, para in enumerate(all_paras):
        # 셔플 후 전체 일련번호 기반 para_id — 이후 모든 파일이 이 ID를 참조
        para_id = f"all-{global_idx:05d}"
        split   = "train" if global_idx < n_train else "test"

        corpus.append({
            "para_id": para_id,
            "title":   para["title"],
            "context": para["context"],
            "split":   split,   # "train" (90%) | "test" (10%)
        })

        # 해당 단락의 질문을 split에 맞게 분류
        for qa in para["qas"]:
            record = {
                "qa_id":       qa["qa_id"],
                "question":    qa["question"],
                "pos_para_id": para_id,
                "title":       para["title"],
            }
            if split == "train":
                eval_train_qs.append(record)
            else:
                eval_dev_qs.append(record)

    # corpus가 새로 만들어지면 외부 모델 캐시 무효화
    _eval_model_cache.clear()

    progress(0.6, desc="파일 저장 중...")

    # ── 3단계: 파일 저장 ──────────────────────────────────────────────
    corpus_path     = os.path.join(OUTPUT_DIR, "corpus.json")
    eval_train_path = os.path.join(OUTPUT_DIR, "eval_questions_train.json")
    eval_dev_path   = os.path.join(OUTPUT_DIR, "eval_questions_dev.json")

    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    with open(eval_train_path, "w", encoding="utf-8") as f:
        json.dump(eval_train_qs, f, ensure_ascii=False, indent=2)
    with open(eval_dev_path, "w", encoding="utf-8") as f:
        json.dump(eval_dev_qs, f, ensure_ascii=False, indent=2)

    # ── 4단계: 메모리에 로딩된 모델 FAISS 인덱스 자동 재구축 ─────────
    # corpus para_id가 바뀌었으므로 이전 인덱스는 반드시 재구축해야 합니다.
    # 재구축하지 않으면 imap에 이전 para_id(train-XXXX, dev-XXXX)가 남아
    # 평가 시 정답 para_id(all-XXXXX)와 불일치해 MRR/Recall이 0.0 이 됩니다.
    rebuild_msgs = []
    if base_model is not None:
        progress(0.75, desc="베이스 모델 FAISS 인덱스 재구축 중...")
        base_index, id_map = encode_and_index(base_model, corpus)
        save_index(base_index, id_map, "base")
        rebuild_msgs.append(f"  ✅ 베이스 인덱스 재구축·저장: {INDEX_DIR}/base.faiss")
    if tuned_model is not None:
        progress(0.90, desc="파인튜닝 모델 FAISS 인덱스 재구축 중...")
        tuned_index, tuned_imap = encode_and_index(tuned_model, corpus)
        save_index(tuned_index, tuned_imap, "tuned_model")
        rebuild_msgs.append(f"  ✅ 파인튜닝 인덱스 재구축·저장: {INDEX_DIR}/tuned_model.faiss")
    if not rebuild_msgs:
        rebuild_msgs.append("  ℹ️ 로딩된 모델 없음 — Step 2 버튼으로 모델 로딩 + 색인 생성을 진행하세요.")

    progress(1.0, desc="완료")

    n_tr = sum(1 for c in corpus if c["split"] == "train")
    n_te = sum(1 for c in corpus if c["split"] == "test")
    rebuild_str = "\n".join(rebuild_msgs)
    return (f"✅ Corpus + 평가 질문 생성 완료 (9:1 분할)\n"
            f"  전체 단락: {n_total}개 → train {n_tr}개 (90%) / test {n_te}개 (10%)\n"
            f"  corpus 저장: {corpus_path}\n"
            f"  eval (train 90%): {len(eval_train_qs)}개 → {eval_train_path}\n"
            f"  eval (test 10%) : {len(eval_dev_qs)}개 → {eval_dev_path}\n"
            f"\n[FAISS 인덱스 재구축]\n{rebuild_str}\n"
            f"  ※ 검색기 학습에는 train split 단락을, 최종 평가엔 test split을 사용하세요.")


def build_triplets(train_path: str, n_neg: int,
                   progress=gr.Progress()) -> str:
    """
    corpus의 train split 단락으로 (query, positive, negative×N) 트리플렛을 생성합니다.

    Negative 샘플링 전략 (2단계):
      1단계 — BM25 Hard Negative (가능한 경우):
        질문을 BM25로 검색해서 상위 K개 중 정답 단락을 제외한 나머지를 사용합니다.
        모델 입장에서 "헷갈리는" 단락이므로 gradient 신호가 강합니다.

      2단계 — Random Negative (Hard Negative가 부족할 때 보충):
        BM25 상위 결과만으로 n_neg를 못 채울 경우 무작위로 보충합니다.

    BM25가 없으면 전체 Random으로 fallback합니다.
    """
    global triplets

    if not corpus:
        return "❌ 먼저 Corpus를 생성하세요."

    eval_train_path = os.path.join(OUTPUT_DIR, "eval_questions_train.json")
    if not os.path.exists(eval_train_path):
        return "❌ eval_questions_train.json이 없습니다. 먼저 Corpus를 생성하세요."

    random.seed(42)

    corpus_by_id  = {c["para_id"]: c for c in corpus}
    all_para_ids  = [c["para_id"] for c in corpus]
    train_corpus  = [c for c in corpus if c["split"] == "train"]

    # title별 단락 ID 집합 (Negative 제외 — 같은 title 내 단락은 너무 유사)
    title_to_para_ids: dict = {}
    for c in corpus:
        title_to_para_ids.setdefault(c["title"], set()).add(c["para_id"])

    with open(eval_train_path, "r", encoding="utf-8") as f:
        train_questions = json.load(f)
    qs_by_para: dict = {}
    for q in train_questions:
        qs_by_para.setdefault(q["pos_para_id"], []).append(q)

    # ── BM25 인덱스 구축 (Hard Negative용) ───────────────────────────
    # 질문 키워드로 유사 단락을 찾아 hard negative로 사용합니다.
    # 완전 무작위 negative보다 훨씬 어렵고 학습 신호가 강합니다.
    progress(0.0, desc="BM25 인덱스 구축 중 (Hard Negative용)...")
    try:
        from rank_bm25 import BM25Okapi
        # 공백 기준 토큰화 (형태소 분석 없이도 충분한 수준)
        bm25_corpus_texts = [c["context"].split() for c in corpus]
        bm25_corpus_ids   = [c["para_id"]         for c in corpus]
        bm25 = BM25Okapi(bm25_corpus_texts)
        use_bm25 = True
    except ImportError:
        bm25 = None
        use_bm25 = False

    # BM25로 Hard Negative를 뽑을 때 상위 몇 개를 후보로 볼지
    # n_neg의 10배 풀에서 정답·동일title을 제외하고 n_neg개 선택
    BM25_POOL = max(n_neg * 10, 30)

    triplets = []
    total = len(train_corpus)
    hard_neg_count  = 0   # BM25 hard negative를 쓴 트리플렛 수
    random_neg_count = 0  # random으로 보충한 트리플렛 수

    for i, para in enumerate(train_corpus):
        if i % 500 == 0:
            progress(i / total, desc=f"트리플렛 생성 중... ({i}/{total})")

        para_id = para["para_id"]
        title   = para["title"]
        pos_ctx = para["context"]

        questions = qs_by_para.get(para_id, [])
        if not questions:
            continue

        same_title_ids = title_to_para_ids.get(title, set())

        for q in questions:
            query_text = q["question"]
            sampled_negs = []

            # ── 1단계: BM25 Hard Negative ─────────────────────
            if use_bm25:
                bm25_scores = bm25.get_scores(query_text.split())
                # 점수 높은 순으로 정렬, 정답·동일title 제외
                ranked = sorted(
                    range(len(bm25_corpus_ids)),
                    key=lambda x: bm25_scores[x],
                    reverse=True
                )
                for idx in ranked:
                    pid = bm25_corpus_ids[idx]
                    if pid == para_id:          # 정답 단락 제외
                        continue
                    if pid in same_title_ids:   # 같은 title 제외
                        continue
                    sampled_negs.append(pid)
                    if len(sampled_negs) >= n_neg:
                        break
                hard_neg_count += len(sampled_negs)

            # ── 2단계: 부족하면 Random으로 보충 ──────────────
            if len(sampled_negs) < n_neg:
                random_pool = [
                    pid for pid in all_para_ids
                    if pid not in same_title_ids
                    and pid != para_id
                    and pid not in sampled_negs
                ]
                need = n_neg - len(sampled_negs)
                sampled_negs += random.sample(random_pool, min(need, len(random_pool)))
                random_neg_count += (len(sampled_negs) - (len(sampled_negs) - need))

            triplets.append({
                "qa_id":        q["qa_id"],
                "query":        query_text,
                "pos_para_id":  para_id,
                "pos_context":  pos_ctx,
                "neg_para_ids": sampled_negs,
                "neg_contexts": [corpus_by_id[pid]["context"]
                                 for pid in sampled_negs],
            })

    out_path = os.path.join(OUTPUT_DIR, f"triplets_neg{n_neg}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(triplets, f, ensure_ascii=False, indent=2)

    progress(1.0, desc="완료")
    neg_strategy = (
        f"BM25 Hard Negative + Random 보충"
        if use_bm25 else "Random Negative (rank_bm25 미설치)"
    )
    return (
        f"✅ 트리플렛 생성 완료\n"
        f"  Negative 전략: {neg_strategy}\n"
        f"  대상 단락: train split {len(train_corpus)}개 / 전체 {len(corpus)}개\n"
        f"  생성된 트리플렛: {len(triplets)}개 | neg×{n_neg}개\n"
        f"  저장: {out_path}\n"
        f"  ※ rank_bm25 설치: pip install rank-bm25"
    )


def load_corpus_file(path: str) -> str:
    """이미 생성된 corpus.json을 불러옵니다."""
    global corpus
    if not path or not os.path.exists(path):
        return "❌ 파일 경로를 확인하세요."
    with open(path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    return f"✅ Corpus 로딩 완료: {len(corpus)}개 단락"


def load_triplets_file(path: str) -> str:
    """이미 생성된 triplets JSON을 불러옵니다."""
    global triplets
    if not path or not os.path.exists(path):
        return "❌ 파일 경로를 확인하세요."
    with open(path, "r", encoding="utf-8") as f:
        triplets = json.load(f)
    return f"✅ 트리플렛 로딩 완료: {len(triplets)}개"


def preview_triplets() -> str:
    """트리플렛 상위 3개를 미리 보여줍니다."""
    if not triplets:
        return "트리플렛이 없습니다."
    lines = []
    for t in triplets[:3]:
        lines.append(
            f"[질문] {t['query']}\n"
            f"  ✅ Positive ({t['pos_para_id']}): {t['pos_context'][:80]}...\n"
            f"  ❌ Negative[0] ({t['neg_para_ids'][0]}): {t['neg_contexts'][0][:80]}..."
        )
    return "\n\n".join(lines)


# ══════════════════════════════════════════════
# Step 2. 모델 로딩 & Fine-tuning
# ══════════════════════════════════════════════

def load_base_model(model_name: str, progress=gr.Progress()) -> str:
    """
    베이스 SentenceTransformer를 로딩하고 corpus 전체로 FAISS 인덱스를 구축합니다.
    베이스 모델은 ./korquad_output/base_model/에 저장하여 파인튜닝 후 비교에 사용합니다.
    """
    global base_model, base_index, id_map

    if not corpus:
        return "❌ 먼저 Corpus를 생성/로딩하세요."
    if not model_name.strip():
        return "❌ 모델명을 입력하세요."

    try:
        progress(0.1, desc="모델 로딩 중...")
        base_model = SentenceTransformer(model_name.strip())

        progress(0.4, desc="corpus 임베딩 생성 중 + FAISS 구축...")
        base_index, id_map = encode_and_index(base_model, corpus)

        # 인덱스를 INDEX_DIR/base.faiss 로 저장
        progress(0.85, desc="FAISS 인덱스 파일 저장 중...")
        save_index(base_index, id_map, "base")

        # 비교용으로 모델 원본 저장
        save_path = os.path.join(OUTPUT_DIR, "base_model")
        base_model.save(save_path)

        progress(1.0, desc="완료")
        dim = base_index.d
        return (f"✅ 베이스 모델 로딩 + 색인 완료\n"
                f"  모델: {model_name}\n"
                f"  임베딩 차원: {dim} | 단락: {base_index.ntotal}개\n"
                f"  모델 저장: {save_path}\n"
                f"  인덱스 저장: {INDEX_DIR}/base.faiss")
    except Exception as e:
        return f"❌ 오류: {e}"


def run_finetuning(loss_type: str, epochs: int, batch_size: int,
                   warmup_ratio: float, max_samples: int,
                   progress=gr.Progress()) -> str:
    """
    트리플렛 데이터로 Bi-Encoder를 파인튜닝합니다.

    loss_type:
      TripletLoss
        - (query, pos, neg) 명시적 트리플렛
        - distance_metric = COSINE 으로 명시 (기본값 EUCLIDEAN은 IndexFlatIP 검색과 불일치)
        - margin=0.5 사용 (코사인 거리 범위 0~2에 맞춘 값)

      MultipleNegativesRankingLoss (권장)
        - (query, pos) 쌍만 필요, 배치 내 다른 positive를 자동으로 negative로 사용
        - In-batch negative가 랜덤 negative보다 훨씬 어렵고 효과적
        - 배치 크기가 클수록 성능 향상 (최소 16 이상 권장)

    [주의] sentence-transformers 3.x 이상에서는 fit()이 deprecated됩니다.
           경고가 나와도 학습은 정상 동작합니다.
    """
    global tuned_model, tuned_index

    if not triplets:
        return "❌ 먼저 트리플렛을 생성/로딩하세요."
    if base_model is None:
        return "❌ 먼저 베이스 모델을 로딩하세요."

    import sentence_transformers as st_module
    st_version = st_module.__version__

    try:
        progress(0.05, desc="학습 모델 초기화 중 (베이스 모델 복사)...")
        # 베이스 모델 저장본을 복사해 원본 유지
        tuned_model = SentenceTransformer(os.path.join(OUTPUT_DIR, "base_model"))

        # 샘플 수 제한
        data = triplets
        if max_samples and max_samples < len(data):
            data = random.sample(data, max_samples)

        # ── InputExample 구성 ─────────────────────────────────────────
        if loss_type == "TripletLoss":
            # negative가 여러 개인 경우 각각 별도 트리플렛으로 확장
            examples = []
            for d in data:
                for neg_ctx in d["neg_contexts"]:
                    examples.append(
                        InputExample(texts=[d["query"], d["pos_context"], neg_ctx])
                    )
            # ★ distance_metric을 COSINE으로 명시
            # 기본값(EUCLIDEAN)은 IndexFlatIP(코사인 유사도) 검색과 목적함수가 불일치해
            # 학습해도 검색 성능이 오르지 않는 원인이 됩니다.
            loss_fn = losses.TripletLoss(
                model=tuned_model,
                distance_metric=losses.TripletDistanceMetric.COSINE,
                triplet_margin=0.5,   # 코사인 거리 범위(0~2) 기준 적정 margin
            )
        else:  # MultipleNegativesRankingLoss (권장)
            # (query, positive) 쌍만 사용
            # 배치 내 다른 샘플의 positive가 자동으로 hard negative가 됩니다.
            # 랜덤 negative보다 훨씬 어렵고 gradient 신호가 강합니다.
            examples = [
                InputExample(texts=[d["query"], d["pos_context"]])
                for d in data
            ]
            loss_fn = losses.MultipleNegativesRankingLoss(model=tuned_model)

        # ── Sanity check: 트리플렛 샘플 출력 ────────────────────────
        # 학습 전에 데이터가 올바르게 구성되었는지 확인합니다.
        sample = examples[0] if examples else None
        sanity = ""
        if sample:
            if loss_type == "TripletLoss":
                sanity = (
                    f"\n[Sanity] 첫 번째 트리플렛:\n"
                    f"  Query  : {sample.texts[0][:60]}\n"
                    f"  Pos    : {sample.texts[1][:60]}\n"
                    f"  Neg    : {sample.texts[2][:60]}"
                )
            else:
                sanity = (
                    f"\n[Sanity] 첫 번째 (query, pos) 쌍:\n"
                    f"  Query  : {sample.texts[0][:60]}\n"
                    f"  Pos    : {sample.texts[1][:60]}"
                )

        loader = DataLoader(examples, batch_size=int(batch_size), shuffle=True)
        warmup_steps = int(len(loader) * int(epochs) * float(warmup_ratio))

        # 저장 폴더명에 주요 파라미터 포함
        loss_abbr = "TL" if loss_type == "TripletLoss" else "MNR"
        model_dir = (
            f"tuned_model"
            f"_{loss_abbr}"
            f"_ep{int(epochs)}"
            f"_bs{int(batch_size)}"
            f"_n{len(examples)}"
        )
        save_path = os.path.join(OUTPUT_DIR, model_dir)
        ckpt_path = save_path + "_ckpt"
        ckpt_steps = max(100, len(loader) // 2)

        progress(0.1, desc=f"학습 시작... ({loss_type}, {epochs} epochs, {len(examples)}개)")
        tuned_model.fit(
            train_objectives=[(loader, loss_fn)],
            epochs=int(epochs),
            warmup_steps=warmup_steps,
            show_progress_bar=False,
            output_path=save_path,
            checkpoint_path=ckpt_path,
            checkpoint_save_steps=ckpt_steps,
            checkpoint_save_total_limit=2,
        )

        progress(0.85, desc="파인튜닝 모델로 FAISS 재구축 중...")
        tuned_index, tuned_imap = encode_and_index(tuned_model, corpus)
        save_index(tuned_index, tuned_imap, model_dir)

        progress(1.0, desc="완료")
        return (
            f"✅ 파인튜닝 완료\n"
            f"  sentence-transformers 버전: {st_version}\n"
            f"  Loss: {loss_type}"
            + (" (distance=COSINE, margin=0.5)" if loss_type == "TripletLoss" else " (In-batch negative)")
            + f"\n  Epochs: {epochs} | Batch: {batch_size} | Warmup steps: {warmup_steps}\n"
            f"  학습 샘플: {len(examples)}개\n"
            f"  저장: {save_path}\n"
            f"  인덱스: {INDEX_DIR}/{model_dir}.faiss"
            + sanity
        )
    except Exception as e:
        return f"❌ 파인튜닝 오류: {e}"


def load_tuned_model(path: str, progress=gr.Progress()) -> str:
    """저장된 파인튜닝 모델을 불러와 FAISS 인덱스를 재구축합니다."""
    global tuned_model, tuned_index

    if not corpus:
        return "❌ 먼저 Corpus를 로딩하세요."
    if not path or not os.path.exists(path):
        return "❌ 경로를 확인하세요."
    try:
        progress(0.2, desc="모델 로딩 중...")
        tuned_model = SentenceTransformer(path)

        progress(0.5, desc="corpus 임베딩 생성 중 + FAISS 구축...")
        tuned_index, tuned_imap = encode_and_index(tuned_model, corpus)

        # 인덱스를 INDEX_DIR/{모델폴더명}.faiss 로 저장
        progress(0.9, desc="FAISS 인덱스 파일 저장 중...")
        model_key = os.path.basename(path.rstrip("/\\"))
        save_index(tuned_index, tuned_imap, model_key)

        progress(1.0, desc="완료")
        return (f"✅ 파인튜닝 모델 로딩 + 색인 완료\n"
                f"  모델: {path}\n"
                f"  단락: {tuned_index.ntotal}개\n"
                f"  인덱스 저장: {INDEX_DIR}/{model_key}.faiss")
    except Exception as e:
        return f"❌ 오류: {e}"


# ══════════════════════════════════════════════
# Step 3. 검색 결과 비교
# ══════════════════════════════════════════════

def compare_search(query: str, top_k: int):
    """
    동일 질문으로 베이스/파인튜닝 모델 검색 결과를 나란히 반환합니다.
    결과 레이블: "[순위] score=X.XXX | para_id | 제목\n   본문 앞 100자..."
    """
    if not query.strip():
        return gr.CheckboxGroup(choices=[]), gr.CheckboxGroup(choices=[]), "", ""
    if base_model is None or base_index is None:
        return gr.CheckboxGroup(choices=[]), gr.CheckboxGroup(choices=[]), \
               "❌ 베이스 모델 미로딩", ""

    # para_id → corpus 조회용 dict
    corpus_by_id = {c["para_id"]: c for c in corpus}

    def fmt_results(results):
        labels = []
        for rank, (pid, score) in enumerate(results, 1):
            entry = corpus_by_id.get(pid, {})
            title   = entry.get("title", "?")
            preview = entry.get("context", "")[:100].replace("\n", " ")
            labels.append(f"[{rank}] score={score:.3f} | {pid} | {title}\n   {preview}...")
        return labels

    top_k = int(top_k)
    base_results  = search(query, base_model, base_index, id_map, top_k)
    base_labels   = fmt_results(base_results)
    base_status   = f"✅ 검색 완료 ({len(base_labels)}건)"

    if tuned_model is not None and tuned_index is not None:
        tuned_results = search(query, tuned_model, tuned_index, id_map, top_k)
        tuned_labels  = fmt_results(tuned_results)
        tuned_status  = f"✅ 검색 완료 ({len(tuned_labels)}건)"
    else:
        tuned_labels = ["⚠️ 파인튜닝 모델 없음 — Step 2를 먼저 실행하세요."]
        tuned_status = "❌ 파인튜닝 모델 없음"

    return (
        gr.CheckboxGroup(choices=base_labels,  value=base_labels),
        gr.CheckboxGroup(choices=tuned_labels, value=tuned_labels),
        base_status,
        tuned_status,
    )


# ══════════════════════════════════════════════
# Step 4. 정량 평가 (MRR@K / Recall@K)
# ══════════════════════════════════════════════

def load_eval_questions_file(path: str) -> str:
    """이미 생성된 eval_questions.json을 불러옵니다."""
    global eval_questions
    if not path or not os.path.exists(path):
        return "❌ 파일 경로를 확인하세요."
    with open(path, "r", encoding="utf-8") as f:
        eval_questions = json.load(f)
    return f"✅ 평가 질문 로딩 완료: {len(eval_questions)}개"


def scan_tuned_models() -> list:
    """
    OUTPUT_DIR 안에서 파인튜닝된 모델 폴더를 스캔합니다.
    config.json이 있는 tuned_model_* 폴더를 유효한 모델로 판단합니다.
    Step 2의 "불러오기" 드롭다운에서 사용합니다.
    """
    if not os.path.isdir(OUTPUT_DIR):
        return []
    found = []
    for entry in sorted(os.scandir(OUTPUT_DIR), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        if not entry.name.startswith("tuned_model"):
            continue
        if os.path.isfile(os.path.join(entry.path, "config.json")):
            found.append(entry.path)
    return found


def get_loaded_model_choices() -> list:
    """
    현재 메모리에 로딩된 모델 목록을 드롭다운용 리스트로 반환합니다.
    베이스 모델과 파인튜닝 모델이 로딩된 경우에만 포함됩니다.
    """
    choices = []
    if base_model is not None:
        choices.append("베이스 모델 (base_model)")
    if tuned_model is not None:
        choices.append("파인튜닝 모델 (tuned_model)")
    return choices if choices else ["(로딩된 모델 없음)"]


def evaluate_single_model(eval_q_path: str, model_choice: str,
                          custom_model_path: str, top_k: int, max_q: int,
                          progress=gr.Progress()) -> str:
    """
    선택한 모델 하나로 MRR@K / Recall@K를 계산합니다.

    model_choice:
      "베이스 모델 (base_model)"   → 메모리의 base_model 사용
      "파인튜닝 모델 (tuned_model)" → 메모리의 tuned_model 사용
      "직접 입력"                   → custom_model_path 경로에서 로딩

    결과는 eval_history_{모델명}.json 에 누적 저장됩니다.
    모델명이 파일명에 포함되므로 여러 모델 결과를 파일별로 구분할 수 있습니다.
    """
    if not corpus:
        return "❌ Corpus가 없습니다. Step 1을 먼저 실행하세요."
    if not eval_q_path or not os.path.exists(eval_q_path):
        return "❌ eval_questions.json 경로를 확인하세요. Step 1을 먼저 실행하세요."

    # ── 평가할 모델 결정 ──────────────────────────────
    if model_choice == "베이스 모델 (base_model)":
        if base_model is None or base_index is None:
            return "❌ 베이스 모델이 로딩되지 않았습니다. Step 2를 먼저 실행하세요."
        eval_model  = base_model
        eval_index  = base_index
        eval_imap   = id_map        # load_base_model에서 세팅된 전역 id_map
        model_label = "base_model"

    elif model_choice == "파인튜닝 모델 (tuned_model)":
        if tuned_model is None or tuned_index is None:
            return "❌ 파인튜닝 모델이 없습니다. Step 2 Fine-tuning을 먼저 실행하세요."
        eval_model  = tuned_model
        eval_index  = tuned_index
        eval_imap   = id_map        # 동일 corpus 기준이므로 id_map 공유
        model_label = "tuned_model"

    else:  # 직접 입력
        path = custom_model_path.strip()
        if not path or not os.path.exists(path):
            return "❌ 모델 경로를 확인하세요."
        abs_path    = os.path.abspath(path)
        model_label = os.path.basename(path.rstrip("/\\"))

        if abs_path in _eval_model_cache:
            # 이미 로딩+인덱싱된 경우 캐시 재사용
            eval_model, eval_index, eval_imap = _eval_model_cache[abs_path]
            progress(0.2, desc=f"[캐시] {model_label} 재사용")
        else:
            try:
                progress(0.0, desc=f"모델 로딩 중... {model_label}")
                eval_model = SentenceTransformer(abs_path)

                # INDEX_DIR에 저장된 인덱스 파일이 있으면 재사용
                # 없으면 임베딩을 새로 생성하고 저장
                progress(0.15, desc="FAISS 인덱스 확인 중...")
                cached_idx, cached_imap = load_index(model_label)
                if cached_idx is not None and len(cached_imap) == len(corpus):
                    eval_index, eval_imap = cached_idx, cached_imap
                    progress(0.5, desc=f"저장된 인덱스 로딩: {INDEX_DIR}/{model_label}.faiss")
                else:
                    progress(0.15, desc="FAISS 인덱스 새로 구축 중...")
                    eval_index, eval_imap = encode_and_index(eval_model, corpus)
                    save_index(eval_index, eval_imap, model_label)

                _eval_model_cache[abs_path] = (eval_model, eval_index, eval_imap)
            except Exception as e:
                return f"❌ 모델 로딩 오류: {e}"

    # ── imap vs corpus 불일치 감지 ────────────────────────────────────
    # corpus를 재생성한 뒤 모델을 재로딩하지 않으면
    # imap에 이전 para_id(train-XXXX, dev-XXXX)가 남아
    # 평가 시 정답 para_id(all-XXXXX)와 불일치해 MRR/Recall이 0.0 이 됩니다.
    corpus_ids = {c["para_id"] for c in corpus}
    imap_sample = set(eval_imap[:min(10, len(eval_imap))])  # 앞 10개만 샘플 확인
    if imap_sample and not imap_sample.issubset(corpus_ids):
        return (
            "❌ FAISS 인덱스의 para_id가 현재 corpus와 불일치합니다.\n\n"
            f"  인덱스 para_id 샘플: {sorted(imap_sample)[:5]}\n"
            f"  corpus para_id 샘플: {sorted(list(corpus_ids))[:5]}\n\n"
            "  [해결 방법] corpus를 재생성하면 자동으로 재구축됩니다.\n"
            "  또는 Step 2에서 베이스 모델을 다시 로딩하세요.\n"
            "  '직접 입력' 모드라면 Step 4 평가 버튼을 다시 누르세요 (캐시 재구축)."
        )

    # ── eval_questions 로딩 ────────────────────────────
    with open(eval_q_path, "r", encoding="utf-8") as f:
        all_questions = json.load(f)

    if max_q and int(max_q) < len(all_questions):
        eval_qs = random.sample(all_questions, int(max_q))
    else:
        eval_qs = all_questions

    n     = len(eval_qs)
    top_k = int(top_k)

    # ── MRR / Recall 계산 + 상세 기록 수집 ───────────
    mrr_list, rec_list = [], []
    details = []   # 질문별 상세 결과 (파일로 저장)
    corpus_by_id = {c["para_id"]: c for c in corpus}
    for i, item in enumerate(eval_qs):
        if i % 200 == 0:
            progress(0.2 + 0.75 * i / n,
                     desc=f"[{model_label}] 평가 중... ({i}/{n})")
        results       = search(item["question"], eval_model, eval_index, eval_imap, top_k)
        retrieved_ids = [pid for pid, _ in results]
        pos_id        = item["pos_para_id"]
        hit  = pos_id in retrieved_ids
        rank = retrieved_ids.index(pos_id) + 1 if hit else None
        rec_list.append(1 if hit else 0)
        mrr_list.append(1.0 / rank if hit else 0.0)

        # 상세 기록: 질문 / 정답 단락 / 검색된 상위 결과
        details.append({
            "qa_id":      item["qa_id"],
            "question":   item["question"],
            "pos_para_id": pos_id,
            "pos_context": corpus_by_id.get(pos_id, {}).get("context", "")[:200],
            "hit":        hit,
            "rank":       rank,
            "retrieved": [
                {
                    "rank":    r + 1,
                    "para_id": pid,
                    "score":   round(score, 4),
                    "context": corpus_by_id.get(pid, {}).get("context", "")[:200],
                }
                for r, (pid, score) in enumerate(results)
            ],
        })

    mrr = float(np.mean(mrr_list))
    rec = float(np.mean(rec_list))
    progress(1.0, desc="완료")

    # ── 인덱스 진단 정보 ─────────────────────────────────
    sep  = "─" * 50
    sep2 = "━" * 50

    # 인덱스 파일 존재 여부
    faiss_file        = os.path.join(INDEX_DIR, f"{model_label}.faiss")
    imap_file         = os.path.join(INDEX_DIR, f"{model_label}_imap.json")
    index_file_exists = os.path.exists(faiss_file) and os.path.exists(imap_file)

    # imap 샘플
    imap_head   = eval_imap[:3]
    imap_tail   = eval_imap[-3:]
    corpus_head = [c["para_id"] for c in corpus[:3]]

    # ── pos_id → corpus 존재 여부 검사 (핵심) ──────────
    # pos_id가 corpus에 없으면 정답을 찾을 수 없어 항상 MRR=0이 됩니다.
    # corpus 재생성 후 eval_questions 파일을 재생성하지 않은 경우 발생합니다.
    sample_pos_ids  = [d["pos_para_id"] for d in details[:20]]
    missing_pos_ids = [pid for pid in sample_pos_ids if pid not in corpus_by_id]
    pos_in_corpus_ok = len(missing_pos_ids) == 0

    # ── 샘플 5개 검색 결과 ─────────────────────────────
    sample_lines = []
    for d in details[:5]:
        q_short  = d["question"][:50]
        pos_id   = d["pos_para_id"]
        pos_ctx  = corpus_by_id.get(pos_id, {}).get("context", "")[:50]
        in_corpus = "✅" if pos_id in corpus_by_id else "❌없음"
        hit_str  = f"✅ rank={d['rank']}" if d["hit"] else "❌ miss"
        top1_pid = d["retrieved"][0]["para_id"] if d["retrieved"] else "없음"
        top1_ctx = corpus_by_id.get(top1_pid, {}).get("context", "")[:50]
        sample_lines += [
            f"  Q: {q_short}",
            f"    정답ID: {pos_id} [corpus:{in_corpus}] ctx: {pos_ctx}",
            f"    hit: {hit_str}  |  Top1: {top1_pid}  ctx: {top1_ctx}",
        ]

    diag_lines = [
        sep2,
        f"🔍 인덱스 진단 — {model_label}",
        sep2,
        f"  인덱스 파일 : {faiss_file}",
        f"  파일 존재   : {'✅' if index_file_exists else '❌ 없음(메모리 사용)'}",
        f"  벡터 수     : {eval_index.ntotal}개",
        f"  imap 크기   : {len(eval_imap)}개  (corpus: {len(corpus)}개)",
        f"  크기 일치   : {'✅' if len(eval_imap)==len(corpus) else '❌ 불일치!'}",
        sep,
        f"  imap 앞3    : {imap_head}",
        f"  imap 뒤3    : {imap_tail}",
        f"  corpus 앞3  : {corpus_head}",
        sep,
        f"  ※ pos_id → corpus 존재 확인 (앞 20개 샘플)",
        f"  결과        : {'✅ 모두 존재' if pos_in_corpus_ok else f'❌ {len(missing_pos_ids)}개 없음 → eval_questions 재생성 필요!'}",
        *([ f"  없는 ID 샘플: {missing_pos_ids[:3]}"] if missing_pos_ids else []),
        sep,
        "  [샘플 5문항 검색 결과]",
        *sample_lines,
        sep2,
    ]

    # ── 결과 포매팅 ───────────────────────────────────
    lines = [
        f"📊 평가 결과 — {model_label}",
        f"  평가 질문: {n}개 | Top-K: {top_k}",
        sep,
        f"  {'MRR@'+str(top_k):<12} {mrr:>10.4f}",
        f"  {'Recall@'+str(top_k):<12} {rec:>10.4f}",
        sep,
    ]

    # ── 결과 누적 저장 (모델명이 파일명에 포함) ─────────
    result = {
        "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S"),
        "model":       model_label,
        "n_questions": n,
        "top_k":       top_k,
        "MRR":         round(mrr, 4),
        "Recall":      round(rec, 4),
    }

    # 파일명: eval_history_{모델명}.json
    # 파일시스템에 사용할 수 없는 문자 제거
    safe_label = model_label.replace("/", "_").replace("\\", "_").replace(":", "_")
    result_path = os.path.join(OUTPUT_DIR, f"eval_history_{safe_label}.json")
    history = []
    if os.path.exists(result_path):
        with open(result_path, "r", encoding="utf-8") as f:
            try:
                history = json.load(f)
            except Exception:
                history = []
    history.append(result)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # 상세 결과 파일 저장
    # 파일명: eval_detail_{모델명}_{타임스탬프}.json
    # 질문별로 정답 단락과 검색 결과 상위 K개를 함께 기록합니다.
    ts_str = time.strftime("%Y%m%d_%H%M%S")
    detail_path = os.path.join(
        OUTPUT_DIR, f"eval_detail_{safe_label}_{ts_str}.json"
    )
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump({
            "model":       model_label,
            "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_questions": n,
            "top_k":       top_k,
            "MRR":         round(mrr, 4),
            "Recall":      round(rec, 4),
            "details":     details,
        }, f, ensure_ascii=False, indent=2)

    lines.append(f"요약 저장: {result_path}  (누적 {len(history)}회)")
    lines.append(f"상세 저장: {detail_path}  (질문별 정답/검색결과 포함)")
    return "\n".join(diag_lines) + "\n\n" + "\n".join(lines)



# ══════════════════════════════════════════════
# Gradio UI
# ══════════════════════════════════════════════

def build_ui():
    with gr.Blocks(title="KorQuAD RAG 실습", theme=gr.themes.Soft()) as demo:

        gr.Markdown(
            "# KorQuAD v1.0 RAG 실습\n"
            "Bi-Encoder Fine-tuning — 데이터 준비 → 학습 → 검색 비교 → 정량 평가"
        )

        # 상태 표시 (5초마다 갱신)
        status_box = gr.Textbox(
            label="현재 상태", value=get_status,
            every=5, interactive=False, lines=4,
        )

        gr.Markdown("---")

        # ════════════════════════════════════════
        # Step 1. 데이터 준비
        # ════════════════════════════════════════
        with gr.Tab("📂 Step 1. 데이터 준비"):

            gr.Markdown(
                "### 1-A. KorQuAD JSON → Corpus + 트리플렛 생성\n"
                "처음 실행하거나 새 데이터로 다시 만들 때 사용합니다."
            )

            with gr.Row():
                train_path = gr.Textbox(
                    label="train JSON 경로",
                    value="./data/KorQuAD_v1_0_train.json",
                    scale=4,
                )
                dev_path = gr.Textbox(
                    label="dev JSON 경로",
                    value="./data/KorQuAD_v1_0_dev.json",
                    scale=4,
                )

            corpus_btn = gr.Button("🗂 Corpus 생성", variant="primary")
            corpus_out = gr.Textbox(label="Corpus 생성 결과", lines=4, interactive=False)
            corpus_btn.click(
                fn=build_corpus,
                inputs=[train_path, dev_path],
                outputs=[corpus_out],
            )

            gr.Markdown("---")
            with gr.Row():
                n_neg_sl = gr.Slider(
                    1, 5, value=2, step=1,
                    label="Negative 수 / 질문",
                    info="질문 1개당 부정 단락 수 (1~5)",
                    scale=2,
                )
                triplet_btn = gr.Button("🔗 트리플렛 생성", variant="primary", scale=1)

            triplet_out  = gr.Textbox(label="트리플렛 생성 결과", lines=3, interactive=False)
            preview_out  = gr.Textbox(label="미리보기 (상위 3개)", lines=10, interactive=False)

            triplet_btn.click(
                fn=build_triplets,
                inputs=[train_path, n_neg_sl],
                outputs=[triplet_out],
            ).then(fn=preview_triplets, outputs=[preview_out])

            gr.Markdown("---")
            gr.Markdown(
                "### 1-B. 기존 파일 불러오기\n"
                "이미 생성된 corpus/트리플렛 JSON이 있을 때 사용합니다."
            )
            with gr.Row():
                corpus_load_path = gr.Textbox(
                    label="corpus.json 경로",
                    value=os.path.join(OUTPUT_DIR, "corpus.json"),
                    scale=4,
                )
                corpus_load_btn = gr.Button("📥 Corpus 로딩", scale=1)
            corpus_load_out = gr.Textbox(label="로딩 결과", lines=2, interactive=False)
            corpus_load_btn.click(
                fn=load_corpus_file,
                inputs=[corpus_load_path],
                outputs=[corpus_load_out],
            )

            with gr.Row():
                triplet_load_path = gr.Textbox(
                    label="triplets JSON 경로",
                    value=os.path.join(OUTPUT_DIR, "triplets_neg2.json"),
                    scale=4,
                )
                triplet_load_btn = gr.Button("📥 트리플렛 로딩", scale=1)
            triplet_load_out = gr.Textbox(label="로딩 결과", lines=2, interactive=False)
            triplet_load_btn.click(
                fn=load_triplets_file,
                inputs=[triplet_load_path],
                outputs=[triplet_load_out],
            ).then(fn=preview_triplets, outputs=[preview_out])

            with gr.Row():
                eval_load_path = gr.Textbox(
                    label="eval_questions JSON 경로",
                    value=os.path.join(OUTPUT_DIR, "eval_questions_train.json"),
                    scale=4,
                    info="train: 학습 도메인 성능 확인 | dev: 일반화 성능 확인",
                )
                eval_load_btn = gr.Button("📥 평가 질문 로딩", scale=1)
            eval_load_out = gr.Textbox(label="로딩 결과", lines=2, interactive=False)
            eval_load_btn.click(
                fn=load_eval_questions_file,
                inputs=[eval_load_path],
                outputs=[eval_load_out],
            )

        # ════════════════════════════════════════
        # Step 2. 모델 로딩 & Fine-tuning
        # ════════════════════════════════════════
        with gr.Tab("🏋️ Step 2. 모델 로딩 & Fine-tuning"):

            gr.Markdown("### 2-A. 베이스 모델 로딩")
            gr.Markdown(
                "파인튜닝 전 원본 모델을 로딩합니다. "
                "Corpus 전체를 임베딩하고 FAISS 인덱스를 구축합니다."
            )
            with gr.Row():
                model_name_box = gr.Textbox(
                    label="임베딩 모델 (HuggingFace ID 또는 로컬 경로)",
                    value=DEFAULT_MODEL,
                    scale=4,
                    info="한국어 지원 SBERT 계열 권장",
                )
                base_load_btn = gr.Button("📥 베이스 모델 로딩", variant="primary", scale=1)
            base_load_out = gr.Textbox(label="로딩 결과", lines=4, interactive=False)
            base_load_btn.click(
                fn=load_base_model,
                inputs=[model_name_box],
                outputs=[base_load_out],
            )

            gr.Markdown("---")
            gr.Markdown(
                "### 2-B. Fine-tuning 설정\n\n"
                "| Loss | 입력 형태 | 특징 |\n"
                "|---|---|---|\n"
                "| **MultipleNegativesRankingLoss** (권장) | (query, pos) | 배치 내 자동 hard negative, 효과적 |\n"
                "| TripletLoss | (query, pos, neg) | negative 명시, distance=COSINE으로 고정 |\n\n"
                "> ⚠️ TripletLoss 기본값은 유클리드 거리인데 검색은 코사인 유사도를 씁니다.  \n"
                "> 여기서는 `distance_metric=COSINE`으로 강제 설정해 불일치를 방지합니다."
            )

            with gr.Row():
                loss_radio  = gr.Radio(
                    ["MultipleNegativesRankingLoss", "TripletLoss"],
                    value="MultipleNegativesRankingLoss", label="Loss 함수", scale=2,
                )
                epochs_sl   = gr.Slider(1, 10, value=3, step=1,  label="Epochs", scale=1)
                batch_sl    = gr.Slider(4, 64, value=16, step=4,  label="Batch Size", scale=1)
                warmup_sl   = gr.Slider(0.0, 0.3, value=0.1, step=0.05,
                                        label="Warmup Ratio", scale=1)

            max_samples_sl = gr.Slider(
                0, 10000, value=0, step=500,
                label="최대 학습 샘플 수 (0 = 전체)",
                info="0이면 트리플렛 전체 사용. 빠른 테스트용으로 1000~2000 권장",
            )

            train_btn = gr.Button("🚀 Fine-tuning 시작", variant="primary", size="lg")
            train_out = gr.Textbox(label="학습 결과", lines=5, interactive=False)
            train_btn.click(
                fn=run_finetuning,
                inputs=[loss_radio, epochs_sl, batch_sl, warmup_sl, max_samples_sl],
                outputs=[train_out],
            )

            gr.Markdown("---")
            gr.Markdown(
                "### 2-C. 저장된 파인튜닝 모델 불러오기\n"
                f"`{OUTPUT_DIR}/tuned_model_*/` 폴더를 스캔하여 목록을 보여줍니다."
            )
            with gr.Row():
                tuned_model_dd = gr.Dropdown(
                    label="파인튜닝 모델 선택",
                    choices=scan_tuned_models(),
                    value=None,
                    allow_custom_value=True,
                    info="목록에 없으면 경로를 직접 입력하세요",
                    scale=4,
                )
                tuned_scan_btn  = gr.Button("🔄 스캔", scale=1, min_width=60)
                tuned_load_btn  = gr.Button("📥 로딩", variant="primary", scale=1, min_width=60)
            tuned_load_out = gr.Textbox(label="로딩 결과", lines=2, interactive=False)

            tuned_scan_btn.click(
                fn=lambda: gr.Dropdown(choices=scan_tuned_models()),
                outputs=[tuned_model_dd],
            )
            tuned_load_btn.click(
                fn=load_tuned_model,
                inputs=[tuned_model_dd],
                outputs=[tuned_load_out],
            )

        # ════════════════════════════════════════
        # Step 3. 검색 결과 비교
        # ════════════════════════════════════════
        with gr.Tab("🔍 Step 3. 검색 결과 비교"):

            gr.Markdown(
                "동일한 질문으로 **베이스 모델**과 **파인튜닝 모델**의 "
                "검색 결과를 나란히 비교합니다. 순위 변화에 주목하세요."
            )
            with gr.Row():
                cmp_query = gr.Textbox(
                    label="검색 질문",
                    placeholder="예: 바그너는 괴테의 파우스트를 읽고 무엇을 쓰고자 했는가?",
                    lines=2, scale=4,
                )
                with gr.Column(scale=1):
                    topk_sl  = gr.Slider(1, 20, value=5, step=1, label="Top-K")
                    cmp_btn  = gr.Button("🔍 비교 검색", variant="primary")

            with gr.Row():
                base_status_box  = gr.Textbox(label="베이스 상태",      interactive=False, scale=1)
                tuned_status_box = gr.Textbox(label="파인튜닝 상태",    interactive=False, scale=1)

            with gr.Row():
                base_checks  = gr.CheckboxGroup(
                    label="📌 베이스 모델 결과 (학습 전)",
                    choices=[], interactive=True, scale=1,
                )
                tuned_checks = gr.CheckboxGroup(
                    label="🎯 파인튜닝 모델 결과 (학습 후)",
                    choices=[], interactive=True, scale=1,
                )

            cmp_btn.click(
                fn=compare_search,
                inputs=[cmp_query, topk_sl],
                outputs=[base_checks, tuned_checks,
                         base_status_box, tuned_status_box],
            )

        # ════════════════════════════════════════
        # Step 4. 정량 평가
        # ════════════════════════════════════════
        with gr.Tab("📊 Step 4. 정량 평가 (MRR / Recall)"):

            gr.Markdown(
                "평가할 모델을 선택하고 **MRR@K**와 **Recall@K**를 계산합니다.\n\n"
                "- **MRR@K**: 정답 단락이 처음 나타난 순위의 역수 평균 (1위=1.0, 2위=0.5, ...)\n"
                "- **Recall@K**: 정답 단락이 상위 K개 안에 있는 비율\n\n"
                "**평가 파일 선택 가이드 (9:1 분할 기준):**\n"
                "- `eval_questions_train.json` — 전체의 **90%** 단락에 속한 질문. "
                "검색기 학습에 사용한 단락이므로 파인튜닝 효과를 직접 확인할 수 있습니다.\n"
                "- `eval_questions_dev.json` — 전체의 **10%** 단락에 속한 질문 (홀드아웃). "
                "학습에서 본 적 없는 단락을 찾는 일반화 성능을 측정합니다.\n\n"
                "> corpus를 재생성하면 반드시 eval 파일도 함께 재생성해야 para_id 매핑이 유지됩니다.\n"
                "> corpus 재생성 시 로딩된 모델의 FAISS 인덱스가 **자동으로 재구축**됩니다."
            )

            with gr.Row():
                eval_set_radio = gr.Radio(
                    choices=["train (학습 90%)", "dev (홀드아웃 10%)"],
                    value="train (학습 90%)",
                    label="평가 데이터셋 선택",
                    info="train: 파인튜닝 효과 확인 (90% 단락) | dev: 홀드아웃 평가 (10% 단락)",
                    scale=3,
                )
                eval_topk = gr.Slider(1, 20, value=10, step=1, label="Top-K", scale=1)

            eval_q_path = gr.Textbox(
                label="eval_questions 파일 경로 (자동 설정 또는 직접 수정)",
                value=os.path.join(OUTPUT_DIR, "eval_questions_train.json"),
                info="위 라디오 선택 시 자동으로 바뀝니다. 직접 수정도 가능합니다.",
            )

            # 라디오 선택 → 경로 자동 변경
            eval_set_radio.change(
                fn=lambda s: (
                    os.path.join(OUTPUT_DIR, "eval_questions_train.json")
                    if "train" in s
                    else os.path.join(OUTPUT_DIR, "eval_questions_dev.json")
                ),
                inputs=[eval_set_radio],
                outputs=[eval_q_path],
            )

            gr.Markdown("### 평가 모델 선택")
            with gr.Row():
                eval_model_dd = gr.Dropdown(
                    label="로딩된 모델 선택",
                    choices=get_loaded_model_choices(),
                    value=None,
                    allow_custom_value=False,
                    scale=3,
                    info="Step 2에서 로딩한 모델 목록",
                )
                eval_refresh_btn = gr.Button("🔄 목록 새로고침", scale=1, min_width=80)

            eval_refresh_btn.click(
                fn=lambda: gr.Dropdown(choices=get_loaded_model_choices()),
                outputs=[eval_model_dd],
            )

            with gr.Row():
                eval_custom_path = gr.Textbox(
                    label="직접 입력 (로딩된 모델 대신 경로 직접 지정)",
                    placeholder="예: ./data/korquad_output/tuned_model_TL_ep3_bs16_n120807",
                    info="입력하면 위 드롭다운 선택보다 우선합니다. 비워두면 드롭다운 모델 사용",
                    scale=4,
                )

            eval_maxq = gr.Slider(
                0, 60407, value=0, step=200,
                label="최대 평가 질문 수 (0 = 전체 사용)",
                info="0이면 선택된 파일 전체 사용. 빠른 테스트용으로 500~1000 권장",
            )

            eval_btn = gr.Button("📊 평가 실행", variant="primary", size="lg")
            eval_out = gr.Textbox(label="평가 결과", lines=12, interactive=False)

            def run_eval_with_choice(eval_q_path, model_choice, custom_path,
                                     top_k, max_q, progress=gr.Progress()):
                """
                직접 입력 경로가 있으면 '직접 입력' 모드로,
                없으면 드롭다운 선택 모델로 평가를 실행합니다.
                """
                if custom_path and custom_path.strip():
                    return evaluate_single_model(
                        eval_q_path, "직접 입력", custom_path.strip(),
                        top_k, max_q, progress,
                    )
                if not model_choice or model_choice == "(로딩된 모델 없음)":
                    return "❌ 평가할 모델을 선택하거나 경로를 직접 입력하세요."
                return evaluate_single_model(
                    eval_q_path, model_choice, "",
                    top_k, max_q, progress,
                )

            eval_btn.click(
                fn=run_eval_with_choice,
                inputs=[eval_q_path, eval_model_dd, eval_custom_path,
                        eval_topk, eval_maxq],
                outputs=[eval_out],
            )

    return demo


# ──────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # 시작 시 기존 파일 자동 로딩
    corpus_path  = os.path.join(OUTPUT_DIR, "corpus.json")
    triplet_path = os.path.join(OUTPUT_DIR, "triplets_neg2.json")
    eval_q_path  = os.path.join(OUTPUT_DIR, "eval_questions_train.json")
    base_path    = os.path.join(OUTPUT_DIR, "base_model")
    tuned_path   = os.path.join(OUTPUT_DIR, "tuned_model")

    if os.path.exists(corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        print(f"[AUTO] Corpus 자동 로딩: {len(corpus)}개")

    if os.path.exists(triplet_path):
        with open(triplet_path, "r", encoding="utf-8") as f:
            triplets = json.load(f)
        print(f"[AUTO] 트리플렛 자동 로딩: {len(triplets)}개")

    if os.path.exists(eval_q_path):
        with open(eval_q_path, "r", encoding="utf-8") as f:
            eval_questions = json.load(f)
        print(f"[AUTO] 평가 질문 자동 로딩: {len(eval_questions)}개")

    if os.path.exists(base_path) and corpus:
        print("[AUTO] 베이스 모델 자동 로딩 중...")
        base_model = SentenceTransformer(base_path)
        # INDEX_DIR/base.faiss 가 있으면 바로 복원 (임베딩 재생성 불필요)
        cached_idx, cached_imap = load_index("base")
        if cached_idx is not None and len(cached_imap) == len(corpus):
            base_index, id_map = cached_idx, cached_imap
            print(f"[AUTO] 베이스 인덱스 복원: {INDEX_DIR}/base.faiss")
        else:
            print("[AUTO] 인덱스 파일 없음 — Step 2 '베이스 모델 로딩' 버튼으로 색인을 생성하세요.")
        print("[AUTO] 베이스 모델 로딩 완료")

    if os.path.exists(tuned_path) and corpus:
        print("[AUTO] 파인튜닝 모델 자동 로딩 중...")
        tuned_model = SentenceTransformer(tuned_path)
        tuned_key   = os.path.basename(tuned_path.rstrip("/\\"))
        cached_idx, cached_imap = load_index(tuned_key)
        if cached_idx is not None and len(cached_imap) == len(corpus):
            tuned_index, _ = cached_idx, cached_imap
            print(f"[AUTO] 파인튜닝 인덱스 복원: {INDEX_DIR}/{tuned_key}.faiss")
        else:
            print("[AUTO] 인덱스 파일 없음 — Step 2 '로딩' 버튼으로 색인을 생성하세요.")
        print("[AUTO] 파인튜닝 모델 로딩 완료")

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7866,
        inbrowser=True,
    )
