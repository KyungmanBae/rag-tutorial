#!/bin/bash
# =============================================================================
# env/setup_rag_uv.sh
# rag-tutorial uv 환경 자동 설치 + 검증 스크립트
# 대상: Ubuntu 24.x / RTX 2080
#
# 저장소 구조:
#   rag-tutorial/
#       ├── env/
#       │   ├── env_rag_cu124.yml   ← conda 환경 파일
#       │   └── setup_rag_uv.sh     ← 이 파일
#       ├── .venv/                  ← 스크립트가 여기에 생성
#       ├── llm_model_manager.py
#       └── ...
#
# 사용법:
#   1) 저장소를 클론합니다:
#      git clone https://github.com/KyungmanBae/rag-tutorial.git
#      cd rag-tutorial
#
#   2) 이 스크립트를 실행합니다:
#      bash env/setup_rag_uv.sh          # 시스템 Python 자동 감지
#      bash env/setup_rag_uv.sh 3.10     # Python 버전 직접 지정
#
# ※ uv 가상환경(.venv)은 base Python과 완전히 격리됩니다.
#   base에 설치된 다른 Python/패키지에 영향을 주거나 받지 않습니다.
# =============================================================================

set -e  # 오류 발생 시 즉시 중단

# ── 색상 출력 헬퍼 ────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
step()    { echo -e "\n${BOLD}${CYAN}══════════════════════════════════════${NC}"; \
            echo -e "${BOLD}${CYAN} $*${NC}"; \
            echo -e "${BOLD}${CYAN}══════════════════════════════════════${NC}"; }

# =============================================================================
# STEP 0: 시스템 사전 확인
# =============================================================================
step "STEP 0: 시스템 환경 사전 확인"

info "※ uv 가상환경은 base Python과 완전히 격리됩니다."
info "  base에 어떤 Python이 설치돼 있어도 충돌하지 않습니다."
echo ""

# ── 저장소 루트 경로 자동 계산 ───────────────────────────────────────────────
# 이 스크립트는 env/ 폴더 안에 있으므로 상위 폴더가 저장소 루트
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 저장소 루트인지 검증 (주요 소스파일 존재 여부 확인)
if [ ! -f "$REPO_ROOT/llm_model_manager.py" ] || [ ! -f "$REPO_ROOT/external_rag.py" ]; then
    error "rag-tutorial 저장소를 찾을 수 없습니다.\n\
  올바른 실행 순서:\n\
    git clone https://github.com/KyungmanBae/rag-tutorial.git\n\
    cd rag-tutorial\n\
    bash env/setup_rag_uv.sh"
fi

# 이후 모든 작업은 저장소 루트 기준
cd "$REPO_ROOT"
success "저장소 루트: $REPO_ROOT"
success "스크립트 위치: $SCRIPT_DIR"

# ── OS 확인 ──────────────────────────────────────────────────────────────────
if [ -f /etc/os-release ]; then
    . /etc/os-release
    info "OS: $PRETTY_NAME"
else
    warn "OS 정보를 확인할 수 없습니다."
fi

# ── Python 버전 선택 ──────────────────────────────────────────────────────────
# 우선순위:
#   1) 스크립트 인자로 직접 지정: bash env/setup_rag_uv.sh 3.11
#   2) 시스템에 설치된 3.9~3.12 중 호환 버전 자동 감지 (3.10 최우선)
#   3) uv가 자동으로 3.10 다운로드 (시스템에 없어도 됨)

REQUESTED_PY="${1:-}"
SELECTED_PY=""
SELECTED_PY_VER=""
COMPAT_VERSIONS=("3.9" "3.11" "3.12" "3.10")  # 3.10 마지막 = 가장 우선

if [ -n "$REQUESTED_PY" ]; then
    if command -v "python${REQUESTED_PY}" &>/dev/null; then
        SELECTED_PY="python${REQUESTED_PY}"
        SELECTED_PY_VER="$REQUESTED_PY"
        success "지정한 Python ${REQUESTED_PY} 발견: $($SELECTED_PY --version)"
    else
        warn "Python ${REQUESTED_PY}이 시스템에 없습니다."
        info "uv가 Python ${REQUESTED_PY}을 자동으로 다운로드합니다."
        SELECTED_PY_VER="$REQUESTED_PY"
        SELECTED_PY=""
    fi
else
    info "시스템에 설치된 Python 버전 감지 중..."
    for ver in "${COMPAT_VERSIONS[@]}"; do
        if command -v "python${ver}" &>/dev/null; then
            SELECTED_PY="python${ver}"
            SELECTED_PY_VER="$ver"
        fi
    done

    if [ -n "$SELECTED_PY" ]; then
        success "사용할 Python: $SELECTED_PY_VER ($($SELECTED_PY --version))"
        if [ "$SELECTED_PY_VER" != "3.10" ]; then
            warn "권장 버전은 3.10입니다. 지정하려면: bash env/setup_rag_uv.sh 3.10"
        fi
    else
        warn "호환 Python(3.9~3.12)을 찾지 못했습니다."
        info "uv가 Python 3.10을 자동으로 다운로드합니다. (시스템 Python에 영향 없음)"
        SELECTED_PY_VER="3.10"
        SELECTED_PY=""
    fi
fi

# ── NVIDIA GPU / CUDA 확인 ────────────────────────────────────────────────────
GPU_FOUND=false
CUDA_VER=""

if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    GPU_FOUND=true
    success "GPU 감지: $GPU_NAME"
    info  "드라이버: $DRIVER_VER"

    # CUDA 버전 파싱 (4가지 방법 순차 시도)
    CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version:\s*\K[0-9]+\.[0-9]+" | head -1)

    if [ -z "$CUDA_VER" ] && command -v nvcc &>/dev/null; then
        CUDA_VER=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+")
    fi
    if [ -z "$CUDA_VER" ] && [ -f /usr/local/cuda/version.txt ]; then
        CUDA_VER=$(grep -oP "[0-9]+\.[0-9]+" /usr/local/cuda/version.txt | head -1)
    fi
    if [ -z "$CUDA_VER" ] && [ -f /usr/local/cuda/version.json ]; then
        CUDA_VER=$(grep -oP '"cuda"\s*:\s*"\K[0-9]+\.[0-9]+' /usr/local/cuda/version.json | head -1)
    fi

    if [ -n "$CUDA_VER" ]; then
        success "CUDA 버전 감지: $CUDA_VER"
    else
        warn "CUDA 버전을 자동으로 감지하지 못했습니다."
        info "nvidia-smi 출력 확인:"
        nvidia-smi | grep -i cuda || true
        echo ""
        echo -e "${YELLOW}CUDA 버전을 직접 입력하세요 (예: 11.8 또는 12.4):${NC}"
        read -r CUDA_VER
        if [ -z "$CUDA_VER" ]; then
            CUDA_VER="11.8"
            warn "입력 없음 → RTX 2080 기본값 11.8 사용"
        fi
        success "CUDA 버전 설정: $CUDA_VER"
    fi
else
    warn "nvidia-smi를 찾을 수 없습니다."
    GPU_FOUND=false
fi

# ── PyTorch wheel 선택 ────────────────────────────────────────────────────────
# GPU 있으면 절대 cpu 빌드 사용 안 함
CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d'.' -f1)

if [ "$GPU_FOUND" = true ]; then
    if [ "$CUDA_MAJOR" = "12" ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu124"
        TORCH_TAG="cu124"
    elif [ "$CUDA_MAJOR" = "11" ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        TORCH_TAG="cu118"
    else
        warn "CUDA ${CUDA_VER}는 미지원 버전입니다. cu124로 설치를 시도합니다."
        TORCH_INDEX="https://download.pytorch.org/whl/cu124"
        TORCH_TAG="cu124 (fallback)"
    fi
    success "PyTorch GPU 빌드 선택: ${TORCH_TAG}"
else
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    TORCH_TAG="cpu"
    warn "GPU 없음 → PyTorch CPU 빌드 설치 (LLM 추론이 매우 느릴 수 있습니다)"
fi
info "index-url: $TORCH_INDEX"

# ── Java 설치 여부 확인 (KoNLPy 의존) ────────────────────────────────────────
if command -v java &>/dev/null; then
    success "Java 감지: $(java -version 2>&1 | head -1)"
else
    warn "Java가 없습니다. KoNLPy를 위해 JDK를 설치합니다..."
    sudo apt update -y && sudo apt install -y default-jdk
    success "JDK 설치 완료: $(java -version 2>&1 | head -1)"
fi

# =============================================================================
# STEP 1: uv 설치
# =============================================================================
step "STEP 1: uv 설치"

if command -v uv &>/dev/null; then
    success "uv 이미 설치됨: $(uv --version)"
else
    info "uv 설치 중..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # 현재 서브셸 세션에 경로 즉시 반영
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    source "$HOME/.cargo/env" 2>/dev/null || true

    # .bashrc에 PATH 영구 등록
    # conda init 블록 뒤에 위치해야 conda가 덮어쓰지 않음
    BASHRC="$HOME/.bashrc"
    UV_MARKER="# uv PATH (rag-tutorial)"
    UV_PATH_LINE='export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"'

    if grep -qF "$UV_MARKER" "$BASHRC" 2>/dev/null; then
        info ".bashrc에 uv PATH가 이미 등록되어 있습니다."
    else
        if grep -q "conda initialize" "$BASHRC" 2>/dev/null; then
            sed -i "/# <<< conda initialize <<</a\\
\\
$UV_MARKER\\
$UV_PATH_LINE" "$BASHRC"
            info "conda init 블록 뒤에 uv PATH 등록 완료"
        else
            echo "" >> "$BASHRC"
            echo "$UV_MARKER" >> "$BASHRC"
            echo "$UV_PATH_LINE" >> "$BASHRC"
            info ".bashrc 끝에 uv PATH 등록 완료"
        fi
        success "uv PATH 영구 등록 완료 → 새 터미널부터 자동 적용"
    fi

    if command -v uv &>/dev/null; then
        success "uv 설치 완료: $(uv --version)"
    else
        error "uv 설치 후에도 명령어를 찾을 수 없습니다.\n  수동으로 실행하세요: export PATH=\"\$HOME/.local/bin:\$PATH\" && uv --version"
    fi
fi

# =============================================================================
# STEP 2: uv 가상환경 생성
# =============================================================================
step "STEP 2: Python ${SELECTED_PY_VER} 가상환경 생성"

# .venv는 저장소 루트(REPO_ROOT)에 생성 — env/ 폴더가 아님
VENV_DIR="$REPO_ROOT/.venv"

if [ -d "$VENV_DIR" ]; then
    EXISTING_VER=$("$VENV_DIR/bin/python" --version 2>&1 | awk '{print $2}' | cut -d'.' -f1,2)
    if [ "$EXISTING_VER" = "$SELECTED_PY_VER" ]; then
        warn ".venv(Python ${EXISTING_VER}) 이미 존재합니다. 재사용합니다."
    else
        warn ".venv가 있지만 Python 버전이 다릅니다 (기존: ${EXISTING_VER}, 선택: ${SELECTED_PY_VER})."
        warn "기존 .venv를 삭제하고 새로 생성합니다..."
        rm -rf "$VENV_DIR"
        uv venv --python "$SELECTED_PY_VER" "$VENV_DIR"
        success "가상환경 생성 완료: Python ${SELECTED_PY_VER}"
    fi
else
    uv venv --python "$SELECTED_PY_VER" "$VENV_DIR"
    success "가상환경 생성 완료: $VENV_DIR  (Python ${SELECTED_PY_VER})"
fi

# .venv/bin/python 경로를 변수로 고정
# → source activate 없이도 항상 이 환경에 설치·실행됨이 보장됨
VENV_PYTHON="$VENV_DIR/bin/python"
success "가상환경 경로 고정: $VENV_DIR"
info  "Python  → $VENV_PYTHON ($(${VENV_PYTHON} --version))"

# =============================================================================
# STEP 3: PyTorch 설치 (CUDA 버전 자동 선택)
# =============================================================================
step "STEP 3: PyTorch 설치 (${TORCH_TAG})"

info "설치 대상: $VENV_DIR"
uv pip install --python "$VENV_PYTHON" \
    torch torchvision torchaudio \
    --index-url "$TORCH_INDEX"
success "PyTorch 설치 완료 → $VENV_DIR"

# =============================================================================
# STEP 4: 핵심 RAG 패키지 설치
# =============================================================================
step "STEP 4: RAG 핵심 패키지 설치"

info "[1/4] LLM 패키지 설치 중..."
uv pip install --python "$VENV_PYTHON" \
    transformers \
    accelerate \
    bitsandbytes \
    huggingface_hub
success "LLM 패키지 설치 완료"

info "[2/4] 검색기 패키지 설치 중..."
uv pip install --python "$VENV_PYTHON" \
    sentence-transformers \
    rank_bm25 \
    datasets \
    ddgs \
    wikipedia-api
success "검색기 패키지 설치 완료"

info "[3/4] FAISS 설치 중..."
if uv pip install --python "$VENV_PYTHON" faiss-gpu 2>/dev/null; then
    success "faiss-gpu 설치 완료"
else
    warn "faiss-gpu 설치 실패 → faiss-cpu로 대체 설치합니다."
    uv pip install --python "$VENV_PYTHON" faiss-cpu
    success "faiss-cpu 설치 완료 (GPU 가속 없음)"
fi

info "[4/4] Gradio 설치 중..."
# 원본 저장소(rag-tutorial) conda 환경 기준 버전으로 고정
uv pip install --python "$VENV_PYTHON" "gradio==6.8.0"
success "Gradio 6.8.0 설치 완료"

# =============================================================================
# STEP 5: 한국어 NLP (KoNLPy)
# =============================================================================
step "STEP 5: 한국어 NLP 패키지 설치 (KoNLPy)"

uv pip install --python "$VENV_PYTHON" konlpy
success "KoNLPy 설치 완료"

# =============================================================================
# STEP 6: KorFactScore 저장소 클론
# =============================================================================
step "STEP 6: KorFactScore 저장소 클론"

KORF_DIR="$REPO_ROOT/KorFactScore"
KORF_URL="https://github.com/ETRI-XAINLP/KorFactScore.git"

if [ -d "$KORF_DIR/.git" ]; then
    warn "KorFactScore 폴더가 이미 존재합니다. git pull로 최신화합니다."
    cd "$KORF_DIR" && git pull && cd "$REPO_ROOT"
else
    git clone "$KORF_URL" "$KORF_DIR"
    success "KorFactScore 클론 완료: $KORF_DIR"
fi

if [ -f "$KORF_DIR/requirements.txt" ]; then
    info "KorFactScore requirements.txt 설치 중..."
    uv pip install --python "$VENV_PYTHON" -r "$KORF_DIR/requirements.txt"
    success "KorFactScore 의존 패키지 설치 완료"
else
    warn "KorFactScore/requirements.txt 없음 — 건너뜁니다."
fi

# =============================================================================
# STEP 7: 설치 검증
# =============================================================================
step "STEP 7: 설치 검증"

info "검증 대상 Python: $VENV_PYTHON"
"$VENV_PYTHON" - <<'PYCHECK'
import sys

results = []

def check(label, fn):
    try:
        result = fn()
        print(f"  \033[32m[OK]\033[0m  {label}: {result}")
        results.append((label, True, result))
    except Exception as e:
        print(f"  \033[31m[NG]\033[0m  {label}: {e}")
        results.append((label, False, str(e)))

# ── Python 버전 ──────────────────────────────
check("Python 버전",
    lambda: sys.version.split()[0])

# ── PyTorch + CUDA ───────────────────────────
import torch
check("torch 버전",
    lambda: torch.__version__)
check("CUDA 버전 (torch)",
    lambda: torch.version.cuda or "N/A")
check("GPU 사용 가능",
    lambda: f"{'YES' if torch.cuda.is_available() else 'NO (CPU 모드)'}")
check("GPU 이름",
    lambda: torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
check("GPU VRAM",
    lambda: f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            if torch.cuda.is_available() else "N/A")

# ── 양자화 ───────────────────────────────────
import bitsandbytes as bnb
check("bitsandbytes",
    lambda: bnb.__version__)

# ── Transformers ─────────────────────────────
import transformers
check("transformers",
    lambda: transformers.__version__)

# ── Hugging Face Hub ─────────────────────────
import huggingface_hub
check("huggingface_hub",
    lambda: huggingface_hub.__version__)

# ── Sentence Transformers ────────────────────
import sentence_transformers
check("sentence-transformers",
    lambda: sentence_transformers.__version__)

# ── FAISS ────────────────────────────────────
import faiss
check("faiss",
    lambda: f"ver {faiss.__version__}  |  GPU 지원: {faiss.get_num_gpus()} 개")

# ── BM25 ─────────────────────────────────────
import rank_bm25
check("rank_bm25",
    lambda: rank_bm25.__version__ if hasattr(rank_bm25, '__version__') else "설치됨")

# ── Datasets ─────────────────────────────────
import datasets
check("datasets",
    lambda: datasets.__version__)

# ── DDGS ─────────────────────────────────────
from duckduckgo_search import DDGS
check("ddgs",
    lambda: "설치됨")

# ── Wikipedia API ─────────────────────────────
import wikipediaapi
check("wikipedia-api",
    lambda: "설치됨")

# ── Gradio ───────────────────────────────────
import gradio as gr
check("gradio",
    lambda: gr.__version__)

# ── KoNLPy ───────────────────────────────────
try:
    from konlpy.tag import Okt
    okt = Okt()
    tokens = okt.morphs("안녕하세요 RAG 실습입니다")
    check("konlpy (Okt 형태소 분석)",
        lambda: f"설치됨  |  테스트: {tokens}")
except Exception as e:
    print(f"  \033[31m[NG]\033[0m  konlpy: {e}")
    results.append(("konlpy", False, str(e)))

# ── 최종 요약 ────────────────────────────────
ok  = [r for r in results if r[1]]
ng  = [r for r in results if not r[1]]
print()
print("─" * 50)
print(f"  검증 결과: {len(ok)} 성공 / {len(ng)} 실패")
if ng:
    print("\n  ⚠️  실패 항목:")
    for name, _, msg in ng:
        print(f"    - {name}: {msg}")
print("─" * 50)
PYCHECK

# =============================================================================
# 완료 메시지
# =============================================================================
step "설치 완료!"

echo ""
info "저장소 루트  : $REPO_ROOT"
info "가상환경 위치: $VENV_DIR"
info "Python 버전  : ${SELECTED_PY_VER}"
echo ""

cat <<MSG

  다음 명령으로 실습을 시작하세요:

    cd $REPO_ROOT
    source .venv/bin/activate

    python llm_model_manager.py   # Step1: LLM 기반 생성
    python external_rag.py        # Step2: 외부 검색 + LLM 생성
    python internal_rag.py        # Step3: 내부 검색 + LLM 생성
    python korquad_rag.py         # KorQuAD 검색기 학습
    python korfactscore_lab.py --korfs_path ./KorFactScore

  ⚠️  주의:
    - 새 터미널을 열 때마다 source .venv/bin/activate 실행 필요
    - uv 명령이 안 되면: export PATH="\$HOME/.local/bin:\$PATH"
    - KorQuAD 데이터는 별도 다운로드 필요 (README 4번 참고)
    - RTX 2080 (8~11GB VRAM) → 모델 로딩 시 4bit 양자화 권장

MSG
