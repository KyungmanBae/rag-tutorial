# RAG Tutorial

RAG tutorial, Gradio, KorQuAD, KorFactScore, FAISS, BM25, Korean retrieval

> 이 저장소를 사용하기 전에 먼저 [`TERMS.md`](TERMS.md)를 확인하세요.  
> Please read [`TERMS.md`](TERMS.md) before using this repository.

검색증강생성(RAG) 실습을 위한 예제 코드, 환경 설정 파일, 발표 자료를 모아둔 저장소입니다.  
This repository contains example code, environment setup files, and presentation materials for hands-on Retrieval-Augmented Generation (RAG) practice.

이 저장소는 표준 오픈소스 라이선스로 배포되지 않습니다. [`LICENSE.md`](LICENSE.md)와 [`TERMS.md`](TERMS.md)를 함께 확인하세요.

> This repository is not distributed under a standard open source license. Please refer to [`LICENSE.md`](LICENSE.md) and [`TERMS.md`](TERMS.md).

---

## 1. 저장소 소개

이 저장소는 RAG의 기본 흐름을 단계별로 실습할 수 있도록 구성되어 있습니다.

### 1-1. 실습 스크립트

실습 파일은 환경에 관계없이 동일합니다. conda, Colab, uv 중 어느 환경을 사용하더라도 동일한 스크립트를 실행합니다.

* `llm_model_manager.py` : Step 1. LLM 기반 생성 실습
* `external_rag.py` : Step 2. 외부 검색 + LLM 생성
* `internal_rag.py` : Step 3. 내부 검색(FAISS/BM25) + LLM 생성
* `korquad_rag.py` : KorQuAD 기반 검색 학습 및 평가 실습
* `korfactscore_lab.py` : KorFactScore 기반 사실 검증 실습

이 저장소의 주요 실행 스크립트는 대부분 **Gradio 기반 웹 UI**를 실행합니다.  
실행 후 터미널에 표시되는 로컬 주소를 웹 브라우저에서 열어 사용하세요.

Most executable scripts in this repository launch **Gradio-based web UIs**.  
After running a script, open the local URL shown in the terminal in your web browser.

### 1-2. 환경 설정 파일

환경은 3가지 버전으로 제공됩니다. **uv 환경을 권장합니다.**

| 환경 | 파일 위치 | 비고 |
|------|----------|------|
| **uv** ⭐ 권장 | `./env/setup_rag_uv.sh` | 완전 무료, 빠른 설치 |
| conda | `./env/env_rag_cu124.yml` | conda-forge 채널 사용 시 무료 |
| Colab | `./colab/` | Google Colab 노트북 |

> ⚠️ **conda 사용 시 주의:** Anaconda `defaults` 채널은 상업적 이용 시 유료입니다.  
> 반드시 `conda-forge` 채널을 사용하거나 uv 환경을 사용하세요.

### 1-3. 발표 자료 PDF

| 파일 | 환경 | 내용 |
|------|------|------|
| `RAG(Retrieval Augmented Generation) 소개 및 실습.pdf` | conda | RAG 소개 + conda 환경 기반 실습 안내 |
| `RAG(Retrieval Augmented Generation) 실습 - colab.pdf` | Colab | Google Colab 환경 기반 실습 안내 |
| `RAG(Retrieval Augmented Generation)_환경셋팅_uv환경.pdf` | uv | uv 환경 설정 + 실습 안내 |

실습 내용 자체는 3개 PDF 모두 동일하며, 환경 설정 방법만 다릅니다.

---

## 2. 환경 설정

검증 기준 환경:

* OS: Ubuntu 24.x
* Python: 3.10
* CUDA: 12.4
* GPU Driver: NVIDIA Driver 550.xx 계열

---

## 2-1. uv 환경 ⭐ 권장

**완전 무료(MIT 라이선스)** 이며, 기존 Python/conda 환경에 영향을 주지 않습니다.

### 설치

```bash
# 1. 저장소 클론
git clone https://github.com/KyungmanBae/rag-tutorial.git
cd rag-tutorial

# 2. 환경 설치 스크립트 실행 (Python 버전 자동 감지)
bash env/setup_rag_uv.sh

# Python 버전을 직접 지정하려면
bash env/setup_rag_uv.sh 3.10
```

스크립트가 자동으로 처리하는 항목:

* uv 설치 및 PATH 등록
* Python 3.10 가상환경 생성 (`.venv/`)
* CUDA 버전 자동 감지 후 PyTorch 설치
* 전체 RAG 패키지 설치
* KorFactScore 저장소 클론
* 설치 검증

### 실행

```bash
# 가상환경 활성화 (새 터미널마다 필요)
source .venv/bin/activate

python llm_model_manager.py
python external_rag.py
python internal_rag.py
```

### PyTorch CUDA 확인

```bash
.venv/bin/python - <<'PY'
import torch
print("torch version:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY
```

### PyTorch 삭제 후 재설치 (uv)

```bash
# 삭제
uv pip uninstall --python .venv/bin/python torch torchvision torchaudio

# CUDA 버전에 맞게 재설치
# CUDA 12.x
uv pip install --python .venv/bin/python torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# CUDA 11.x
uv pip install --python .venv/bin/python torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118
```

---

## 2-2. conda 환경

> ⚠️ **주의:** Anaconda `defaults` 채널은 상업적 이용 시 유료입니다.  
> 아래 설치 순서에 따라 반드시 `conda-forge` 채널만 사용하세요.

### 설치

```bash
# 1. 현재 채널 확인
conda config --show channels

# 2. defaults 채널 제거 후 conda-forge 우선 사용 설정
conda config --remove channels defaults
conda config --add channels conda-forge
conda config --set channel_priority strict

# 3. 환경 생성
conda env create -f ./env/env_rag_cu124.yml

# 4. 환경 활성화
conda activate rag_env_261_cu124
```

참고:

* `defaults`가 등록되어 있지 않은 경우 제거 명령은 무시해도 됩니다.
* CUDA 버전, GPU 드라이버 버전이 다르면 패키지 버전 조정이 필요할 수 있습니다.

### 실행

```bash
conda activate rag_env_261_cu124

python llm_model_manager.py
python external_rag.py
python internal_rag.py
```

### PyTorch CUDA 확인

```bash
python - <<'PY'
import torch
print("torch version:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY
```

### PyTorch 삭제 후 재설치 (conda)

```bash
# 삭제
pip uninstall -y torch torchvision torchaudio
pip cache purge

# CUDA 버전에 맞게 재설치
# CUDA 12.x
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# CUDA 11.x
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118
```

---

## 2-3. Colab 환경

Google Colab에서 실습할 경우 `./colab/` 폴더의 노트북 파일을 사용합니다.

* Colab 환경은 노트북 첫 셀에서 필요한 패키지를 자동으로 설치합니다.
* GPU 런타임(T4 이상)을 선택해서 사용하세요.
* 자세한 내용은 `RAG(Retrieval Augmented Generation) 실습 - colab.pdf`를 참고하세요.

---

## 2-4. 검증 환경 / Tested Environment

| 항목 | 값 |
|------|-----|
| OS | Ubuntu 24.x |
| Python | 3.10 |
| CUDA | 12.4 |
| GPU Driver | NVIDIA Driver 550.xx 계열 |

주의:

* 다른 OS(Ubuntu 22.x, Windows, WSL, macOS 등) 또는 다른 CUDA 버전에서는 일부 패키지 버전 조정이 필요할 수 있습니다.
* **PyTorch, torchvision, torchaudio, bitsandbytes, faiss, konlpy** 등은 환경 차이에 민감합니다.
* GPU 환경에서는 CUDA 버전과 PyTorch 호환성을 반드시 함께 확인하세요.

---

## 2-5. PyTorch / CUDA 문제 해결 (공통)

GPU 인식 문제가 발생하면 다음 순서로 확인하세요.

### CUDA 상태 확인

```bash
# CUDA 버전 확인
nvidia-smi

# PyTorch GPU 인식 확인
python - <<'PY'
import torch
print("torch version:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY
```

### 대표적인 문제

CUDA 12.4 환경에서 `torch 2.x.x+cu130`처럼 **CUDA 13.0 빌드**가 설치되면 GPU를 인식하지 못할 수 있습니다.

증상:

* `cuda available: False`
* `The NVIDIA driver on your system is too old`
* Gradio UI는 실행되지만 GPU 가속이 동작하지 않음

이 경우 위의 각 환경별 **"PyTorch 삭제 후 재설치"** 항목을 참고하세요.

---

## 2-6. 학습에 사용한 베이스 임베딩 모델 / Base Embedding Model Used for Training

KorQuAD 기반 검색 실습에서는 다음 한국어 문장 임베딩 모델을 베이스 모델로 사용했습니다.

* `snunlp/KR-SBERT-V40K-klueNLI-augSTS`

모델 개요:

* 한국어 문장 임베딩을 위한 SBERT 계열 모델
* `sentence-transformers` 방식으로 바로 사용할 수 있음
* 문장 및 문단을 768차원 dense vector로 변환
* 의미 기반 검색, 문장 유사도, semantic search, clustering 등에 활용 가능

공식 모델 페이지: <https://huggingface.co/snunlp/KR-SBERT-V40K-klueNLI-augSTS>  
관련 GitHub 저장소: <https://github.com/snunlp/KR-SBERT>

### 라이선스 및 인용 / License and Citation

* License: MIT

```bibtex
@misc{kr-sbert,
  author = {Park, Suzi and Hyopil Shin},
  title = {KR-SBERT: A Pre-trained Korean-specific Sentence-BERT model},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/snunlp/KR-SBERT}}
}
```

---

## 3. 실행 방법

환경 활성화 후 아래 스크립트를 실행합니다. 실행 후 터미널에 표시되는 주소를 브라우저에서 열어 사용하세요.

### Step 1. LLM 기반 생성

```bash
python llm_model_manager.py
```

### Step 2. 외부 검색 + LLM 생성

```bash
python external_rag.py
```

### Step 3. 내부 검색 + LLM 생성

```bash
python internal_rag.py
```

### KorQuAD 기반 검색 실습 (뉴럴 모델 학습)

```bash
python korquad_rag.py
```

### KorFactScore 기반 사실 검증 실습

```bash
python korfactscore_lab.py --korfs_path ./KorFactScore
```

---

## 4. KorQuAD 사용 안내

KorQuAD 원본 데이터 파일은 이 저장소에 포함하지 않습니다.  
공식 페이지에서 직접 다운로드한 뒤, 로컬 환경의 `./data/` 폴더에 복사해서 사용하세요.

공식 페이지: <https://korquad.github.io/category/1.0_KOR.html>

```bash
mkdir -p ./data
wget -O ./data/KorQuAD_v1_0_train.json https://korquad.github.io/dataset/KorQuAD_v1.0_train.json
wget -O ./data/KorQuAD_v1_0_dev.json https://korquad.github.io/dataset/KorQuAD_v1.0_dev.json
```

파일명을 로컬 편의상 변경하더라도 원본 데이터의 라이선스나 배포 조건은 변경되지 않습니다.

KorQuAD original dataset files are not included in this repository.  
Please download them from the official source and copy them into the local `./data/` directory.

---

## 5. KorFactScore 사용 안내

KorFactScore는 별도 저장소를 clone하여 사용합니다.  
uv 환경의 경우 `env/setup_rag_uv.sh` 실행 시 자동으로 클론됩니다.

공식 저장소: <https://github.com/ETRI-XAINLP/KorFactScore>

수동으로 클론하려면:

```bash
git clone https://github.com/ETRI-XAINLP/KorFactScore.git
```

이 실습에서는 위키피디아 덤프를 기반으로 SQLite DB를 생성한 뒤 사용합니다.

```bash
python build_kowiki_db.py
```

생성 완료 후 DB 파일 위치: `./KorFactScore/downloaded_files/`

```bash
python korfactscore_lab.py --korfs_path ./KorFactScore
```

---

## 6. 포함하지 않는 항목

이 저장소는 다음 항목을 기본적으로 포함하지 않습니다.

* 원본 데이터셋 파일
* 대용량 데이터베이스 파일
* 모델 weight 파일
* 외부 저장소 전체 코드
* 로컬 캐시 및 생성 산출물

---

## 7. 이용 조건

이 저장소의 사용 조건은 [`TERMS.md`](TERMS.md)에 정리되어 있습니다.

핵심 요약:

* 비상업적 교육·실습 목적 사용 허용
* 출처 표시 필요
* 수정본 공개 및 재배포 금지
* 상업적 사용 금지
* 외부 리소스는 각 원 이용 조건을 따름

---

## 8. 외부 리소스 안내

이 저장소는 실습과 연구 편의를 위해 외부 저장소, 데이터셋, 모델, 패키지, 문서 등을 참고하거나 연동할 수 있습니다.  
이러한 제3자 리소스의 저작권과 이용 조건은 각 원저작자 및 원 라이선스를 따릅니다.

* KorQuAD: <https://korquad.github.io/category/1.0_KOR.html>
* KorFactScore: <https://github.com/ETRI-XAINLP/KorFactScore>
* Hugging Face: <https://huggingface.co/>
* Gradio: <https://www.gradio.app/>
* Sentence Transformers: <https://www.sbert.net/>
* FAISS: <https://github.com/facebookresearch/faiss>

---

## 9. AI 활용 고지

이 저장소의 일부 코드는 AI 도구의 도움을 받아 초안 작성, 테스트, 수정되었을 수 있습니다.  
최종 검토와 공개 책임은 저장소 관리자에게 있습니다.

Some parts of this repository may have been drafted, tested, or revised with the help of AI tools.  
Final review and release responsibility remains with the repository maintainer.

---

## 10. 참고 자료

* KorQuAD 공식 페이지
* KorFactScore 공식 저장소
* Hugging Face 모델 저장소
* 발표자료 PDF (저장소 루트)

---

## 11. 연락 및 문의

재배포, 수정본 공개, 상업적 사용 등 별도 허가가 필요한 경우 저장소 관리자에게 문의하세요.

For permissions beyond the scope of [`TERMS.md`](TERMS.md), please contact the repository maintainer.
