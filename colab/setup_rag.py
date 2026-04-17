"""
setup_rag.py  ―  RAG 실습 환경 설치 스크립트
==============================================
참조 파일: env/env_rag_cu124.yml

대상 환경
  • Google Colab  (T4 GPU, CUDA 12.x)
  • 로컬 Linux    (RTX 2080, CUDA 12.4)

[yml 대비 변경 사항 요약]
  패키지                  yml 원본          수정값              이유
  ─────────────────────────────────────────────────────────────────────
  sentence-transformers  (latest)          ==5.4.1             3.x/4.x: transformers<5.0 강제 → 5.2.0 충돌
                                                               5.4.1: transformers<6.0,>=4.41 → 5.2.0 호환
  gradio_client          ==1.5.1           (고정 제거)          gradio 6.8.0은 gradio_client==2.2.0 요구
                                                               → 1.5.1과 동시 설치 불가
  huggingface_hub        (latest)          (고정 제거)          transformers 5.2.0이 >=1.3.0 요구
                                                               → 버전 고정 시 충돌
  tokenizers             (latest)          (고정 제거)          transformers 5.2.0 의존성에 맡김
  fastapi                (latest)          (고정 제거)          Colab google-adk가 >=0.124.1 요구
  faiss                  faiss-gpu         faiss-cpu           Colab T4에 faiss-gpu 바이너리 미배포
  llama-index            (메타패키지)       core 직접 지정       메타패키지가 core 버전 강제로 충돌 유발
"""

import os
import subprocess
import importlib


# ──────────────────────────────────────────────
# 헬퍼: 명령어 실행 + 실시간 출력
# ──────────────────────────────────────────────
def run(cmd: str) -> int:
    """
    셸 명령어를 실행하고 stdout/stderr를 실시간 출력한다.
    반환값: 프로세스 종료 코드 (0 = 성공)
    """
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in process.stdout:
        print(line, end="")
    process.wait()
    return process.returncode


def is_colab() -> bool:
    """Google Colab 환경 여부 확인"""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


# ──────────────────────────────────────────────
# 설치 시작
# ──────────────────────────────────────────────
print("=" * 60)
print("  RAG 실습 환경 설치 시작")
print(f"  실행 환경: {'Google Colab' if is_colab() else '로컬 환경'}")
print("  참조: env/env_rag_cu124.yml")
print("=" * 60)


# ── STEP 1. PyTorch (CUDA 12.4) ────────────────────────────
# yml: torch / torchvision / torchaudio (cu124 wheels)
print("\n[1/7] PyTorch (CUDA 12.4) 설치 중...")
run(
    "pip install -q torch torchvision torchaudio "
    "--extra-index-url https://download.pytorch.org/whl/cu124"
)


# ── STEP 2. HuggingFace 핵심 패키지 ────────────────────────
# yml: transformers==5.2.0, accelerate, safetensors,
#      huggingface_hub, tokenizers, bitsandbytes>=0.46.1, datasets
#
# [수정] huggingface_hub, tokenizers 버전 고정 제거
#   이전 에러: huggingface_hub==0.28.1 고정 시
#              transformers 5.2.0의 요구(>=1.3.0)와 충돌
#   해결: 버전 고정 없이 pip이 transformers 5.2.0 의존성에 맞춰 자동 선택
print("\n[2/7] HuggingFace 패키지 설치 중...")
run(
    "pip install -q "
    "transformers==5.2.0 "    # yml 기준 고정
    "accelerate "             # yml 기준 (버전 고정 없음)
    "safetensors "            # yml 기준 (버전 고정 없음)
    "huggingface_hub "        # [수정] 버전 고정 제거 → pip 자동 선택
    "tokenizers "             # [수정] 버전 고정 제거 → pip 자동 선택
    "datasets "               # yml 기준 (버전 고정 없음)
    "'bitsandbytes>=0.46.1'"  # yml 기준 고정 (RTX 2080 / T4 sm_75 지원)
)


# ── STEP 3. Gradio UI ──────────────────────────────────────
# yml: gradio==6.8.0, gradio_client==1.5.1, fastapi, uvicorn
#
# [수정 1] gradio_client 버전 고정 제거
#   이전 에러: gradio==6.8.0은 내부적으로 gradio_client==2.2.0을 요구하는데
#              gradio_client==1.5.1을 명시하면 동시 설치 불가 → ResolutionImpossible
#   해결: gradio_client 버전 고정 제거, gradio가 알아서 2.2.0 선택
#
# [수정 2] fastapi, uvicorn 버전 고정 제거
#   이전 에러: fastapi==0.115.12 고정 시
#              Colab google-adk 1.29.0의 요구(>=0.124.1)와 충돌
#   해결: 버전 고정 없이 pip 자동 해결
print("\n[3/7] Gradio 설치 중...")
run(
    "pip install -q "
    "gradio==6.8.0 "   # yml 기준 고정
    # gradio_client: [수정] 버전 고정 제거 → gradio 6.8.0이 2.2.0 자동 선택
    "fastapi "         # [수정] 버전 고정 제거 → google-adk 충돌 자동 회피
    "uvicorn"          # [수정] 버전 고정 제거
)


# ── STEP 4. 외부 검색 라이브러리 ───────────────────────────
# yml: ddgs, wikipedia-api
print("\n[4/7] 외부 검색 패키지 설치 중...")
run(
    "pip install -q "
    "ddgs "            # yml 기준: DuckDuckGo 검색 (외부 RAG 검색기 1)
    "wikipedia-api"    # yml 기준: Wikipedia 검색 (외부 RAG 검색기 2)
)


# ── STEP 5. 한국어 NLP / 내부 검색 ────────────────────────
# yml: konlpy, rank_bm25, sentence-transformers, numpy==1.26.4
#
# [수정] sentence-transformers 버전을 5.4.1로 고정
#   이전 에러: 3.x / 4.x 계열은 transformers<5.0.0 강제
#              → transformers==5.2.0과 충돌
#   해결: 5.4.1은 transformers<6.0,>=4.41.0 허용 → 5.2.0 호환 ✅
print("\n[5/7] 한국어 NLP + 내부 검색 패키지 설치 중...")

# konlpy는 Java(JDK)에 의존 → apt로 먼저 설치
run("apt-get install -qq default-jdk")

# JAVA_HOME 환경변수 설정 (konlpy 내부 JPype에서 필요)
java_home = "/usr/lib/jvm/java-11-openjdk-amd64"
os.environ["JAVA_HOME"] = java_home
print(f"  JAVA_HOME={java_home}")

run(
    "pip install -q "
    # numpy 버전 고정 제거:
    #   yml 원본은 1.26.4 고정이었으나, Colab 기본 환경의 pandas/jax/opencv 등이
    #   numpy 2.x 바이너리로 빌드돼 있어 1.26.4로 다운그레이드 시
    #   "numpy.dtype size changed" 바이너리 충돌 발생 → 고정 제거
    "konlpy==0.6.0 "                  # yml 기준 (한국어 형태소 분석, numpy>=1.6 요구)
    "rank_bm25 "                      # yml 기준 (BM25 희소 검색, numpy 무관)
    "sentence-transformers==5.4.1"    # [수정] yml: latest → 5.4.1 고정
                                      #   3.x/4.x: transformers<5.0.0 강제 → 5.2.0 충돌
                                      #   5.4.1: transformers<6.0,>=4.41.0 → 5.2.0 호환 ✅
)


# ── STEP 6. FAISS ──────────────────────────────────────────
# yml 원본: faiss-gpu
# [수정] faiss-cpu로 대체
#   이유: Colab T4에 faiss-gpu 공식 바이너리 미배포
#         RTX 2080 로컬에서도 실습 수준 속도는 faiss-cpu로 충분
print("\n[6/7] FAISS 설치 중...")
print("  (yml 원본: faiss-gpu → Colab 미지원으로 faiss-cpu 대체)")
run("pip install -q faiss-cpu")


# ── STEP 7. LlamaIndex ─────────────────────────────────────
# yml 원본: llama-index, llama-index-vector-stores-faiss,
#           llama-index-embeddings-huggingface
#
# [수정] 메타패키지 llama-index 제거, core 컴포넌트 직접 지정
#   이전 에러: llama-index 메타패키지(0.14.20)가 core<0.15,>=0.14.20 강제
#              → core 0.12.x 등 다른 버전 설치 시 충돌
#   해결: core==0.14.20 + vector-stores + embeddings를 동일 계열로 직접 명시
print("\n[7/7] LlamaIndex 설치 중...")
print("  (yml 원본: llama-index 메타패키지 → core + 컴포넌트 직접 지정)")
run(
    "pip install -q --upgrade "       # --upgrade: Colab 기존 llama 패키지가 core를
                                      #   0.12.x로 다운그레이드하는 것을 방지
    "llama-index-core==0.14.20 "      # [수정] 메타패키지 대신 core 직접 지정
    "llama-index-vector-stores-faiss==0.6.0 "   # core 0.14.x 호환 확인
    "llama-index-embeddings-huggingface==0.7.0" # core 0.14.x 호환 확인
)


# ── STEP 부록. colab/ 폴더 및 gradio_compat.py 생성 ────────
# setup_rag.py 자신이 ./colab/ 안에 있으므로
# 실습 파일들이 있는 상위 폴더 기준으로 경로를 계산
# gradio 6.x에서 theme 파라미터 위치: Blocks() → launch()로 이동
# gradio_compat.py: 버전 자동 감지 → 실습 코드가 어느 환경에서도 동작
print("\n[부록] colab/gradio_compat.py (Gradio 버전 호환 유틸) 생성 중...")

compat_code = '''"""
gradio_compat.py
================
Gradio 버전 간 호환성 유틸리티

Gradio 버전별 theme 파라미터 위치 변화:
  - Gradio 3.x / 4.x / 5.x : gr.Blocks(theme=...)
  - Gradio 6.x+             : launch(theme=...)

사용법 (실습 파일 공통):
    from gradio_compat import make_blocks, safe_launch
"""
import gradio as gr


def get_gradio_major() -> int:
    """설치된 Gradio의 메이저 버전 번호 반환 (예: 5, 6)"""
    try:
        return int(gr.__version__.split(".")[0])
    except Exception:
        return 6  # 파싱 실패 시 안전한 기본값 (현재 설치 버전 기준)


def make_blocks(title: str = "", theme=None):
    """
    Gradio 버전에 따라 theme 위치를 자동 조정하여 Blocks 객체 생성.
      - Gradio < 6 : gr.Blocks(title=..., theme=...)
      - Gradio >= 6: gr.Blocks(title=...)  ← theme은 launch()에서 전달
    """
    if theme is None:
        return gr.Blocks(title=title)
    if get_gradio_major() >= 6:
        # Gradio 6+: Blocks 생성자에서 theme 파라미터 제거됨
        return gr.Blocks(title=title)
    else:
        # Gradio 3~5: Blocks 생성자에서 theme 전달
        return gr.Blocks(title=title, theme=theme)


def is_colab() -> bool:
    """Google Colab 환경 여부 확인"""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def safe_launch(demo, theme=None, server_port: int = 7860, **kwargs):
    """
    Gradio 버전 및 실행 환경(Colab / 로컬)에 따라 안전하게 launch() 호출.

    처리 내용:
      - Colab 환경 : share=True 자동 설정, inbrowser/server_name 제거
      - Gradio >= 6: theme을 launch()에 전달
      - Gradio < 6 : theme은 이미 Blocks()에 전달됐으므로 launch()에서 제외
    """
    major = get_gradio_major()

    # Colab 환경 자동 보정
    if is_colab():
        kwargs["share"] = True           # Colab UI를 외부 터널로 노출
        kwargs.pop("inbrowser", None)    # Colab에서 불필요
        kwargs.pop("server_name", None)  # Colab에서 0.0.0.0 불필요

    # theme 처리: Gradio 6+에서만 launch()에 전달
    if theme is not None and major >= 6:
        kwargs["theme"] = theme
    else:
        kwargs.pop("theme", None)        # 구버전 launch()에서는 theme 인자 제거

    demo.launch(server_port=server_port, **kwargs)
'''

# setup_rag.py 위치: ./colab/setup_rag.py
# 실습 파일 위치:    ./llm_model_manager.py 등 (상위 폴더)
# colab 패키지 경로: ./colab/  (setup_rag.py 기준 현재 폴더)
colab_dir = os.path.dirname(os.path.abspath(__file__))  # ./colab/ 절대경로
os.makedirs(colab_dir, exist_ok=True)

# __init__.py: colab/을 Python 패키지로 인식시켜 `from colab.gradio_compat import ...` 가능하게 함
init_path = os.path.join(colab_dir, "__init__.py")
if not os.path.exists(init_path):
    with open(init_path, "w", encoding="utf-8") as f:
        f.write("# colab/ 폴더를 Python 패키지로 인식시키기 위한 파일\n")

compat_path = os.path.join(colab_dir, "gradio_compat.py")
with open(compat_path, "w", encoding="utf-8") as f:
    f.write(compat_code)
print(f"  → {compat_path} 생성 완료")


# ──────────────────────────────────────────────
# 설치 검증
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("  설치 검증")
print("=" * 60)

packages = [
    ("torch",                  "torch"),
    ("transformers",           "transformers"),
    ("accelerate",             "accelerate"),
    ("bitsandbytes",           "bitsandbytes"),
    ("datasets",               "datasets"),
    ("gradio",                 "gradio"),
    ("ddgs",                   "ddgs"),
    ("wikipedia-api",          "wikipediaapi"),
    ("numpy",                  "numpy"),
    ("konlpy",                 "konlpy"),
    ("rank_bm25",              "rank_bm25"),
    ("sentence-transformers",  "sentence_transformers"),
    ("faiss",                  "faiss"),
    ("llama-index-core",       "llama_index"),
]

all_ok = True
for pkg_name, import_name in packages:
    try:
        importlib.invalidate_caches()
        mod = importlib.import_module(import_name)
        ver = getattr(mod, "__version__", "N/A")
        print(f"  ✅ {pkg_name:<30} {ver}")
    except ImportError as e:
        print(f"  ❌ {pkg_name:<30} 실패: {e}")
        all_ok = False

# GPU 상태 확인
print()
try:
    import torch
    if torch.cuda.is_available():
        gpu_name            = torch.cuda.get_device_name(0)
        vram                = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        sm_major, sm_minor  = torch.cuda.get_device_capability(0)
        print(f"  🖥️  GPU  : {gpu_name}  ({vram:.1f} GB)")
        print(f"  📐 Compute Capability: sm_{sm_major}{sm_minor}")
        if sm_major < 7:
            print("  ⚠️  sm_75 미만: bitsandbytes 4/8bit 양자화 미지원 가능성 있음")
        else:
            print("  ✅ bitsandbytes 4/8bit 양자화 지원 가능 (sm_75 이상)")
    else:
        print("  ⚠️  GPU 미인식 — 런타임 유형을 T4 GPU로 변경하세요")
        all_ok = False
except Exception as e:
    print(f"  ⚠️  GPU 확인 실패: {e}")

# colab/gradio_compat.py 존재 확인
print()
compat_check = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gradio_compat.py")
if os.path.exists(compat_check):
    print("  ✅ colab/gradio_compat.py       존재 확인")
else:
    print("  ❌ colab/gradio_compat.py       없음")
    all_ok = False

print()
print("=" * 60)
if all_ok:
    print("  🎉 환경 설정 완료! 아래 순서로 실습을 시작하세요.")
    print()
    print("  1단계  python llm_model_manager.py   # LLM 로딩 / 추론")
    print("  2단계  python external_rag.py        # 외부 검색 + LLM 생성")
    print("  3단계  python internal_rag.py        # 내부 검색 + LLM 생성")
    print("  4단계  python korquad_rag.py         # 검색기 학습 (KorQuAD)")
else:
    print("  ⚠️  위 ❌ 항목을 확인 후 재실행하세요.")
print("=" * 60)
