# RAG Tutorial

> 이 저장소를 사용하기 전에 먼저 [`TERMS.md`](./TERMS.md)를 확인하세요.  
> Please read [`TERMS.md`](./TERMS.md) before using this repository.

검색증강생성(RAG) 실습을 위한 예제 코드, 환경 설정 파일, 발표 자료를 모아둔 저장소입니다.  
This repository contains example code, environment setup files, and presentation materials for hands-on Retrieval-Augmented Generation (RAG) practice.

이 저장소는 표준 오픈소스 라이선스로 배포되지 않습니다. [`LICENSE.md`](./LICENSE.md)와 [`TERMS.md`](./TERMS.md)를 함께 확인하세요.  
> This repository is not distributed under a standard open source license. Please refer to [`LICENSE.md`](./LICENSE.md) and [`TERMS.md`](./TERMS.md).
---

## 1. 저장소 소개

이 저장소는 RAG의 기본 흐름을 단계별로 실습할 수 있도록 구성되어 있습니다.

주요 구성:
- `llm_model_manager.py` : Step 1. LLM 기반 생성 실습
- `external_rag.py` : Step 2. 외부 검색 + LLM 생성
- `internal_rag.py` : Step 3. 내부 검색(FAISS/BM25) + LLM 생성
- `korquad_rag.py` : KorQuAD 기반 검색 학습 및 평가 실습
- `korfactscore_lab.py` : KorFactScore 기반 사실 검증 실습
- `./env/env_rag_cu124.yml` : conda 환경 설정 파일
- 발표자료 PDF : 실습 설명 자료

이 저장소의 주요 실행 스크립트는 대부분 **Gradio 기반 웹 UI**를 실행합니다.  
실행 후 터미널에 표시되는 로컬 주소를 웹 브라우저에서 열어 사용하세요.

Most executable scripts in this repository launch **Gradio-based web UIs**.  
After running a script, open the local URL shown in the terminal in your web browser.

---

## 2. 환경 설정

이 저장소는 `./env/env_rag_cu124.yml` 환경 파일을 사용합니다.

```bash
conda env create -f ./env/env_rag_cu124.yml
conda activate rag_env_261_cu124
```

참고:
- 이 저장소의 예제 환경은 `conda-forge` 기준으로 작성되어 있습니다.
- 다른 채널 또는 다른 패키지 배포 경로를 사용하는 경우, 패키지 호환성과 조직 내 사용 정책을 별도로 검토하세요.
- 조직 또는 기업 환경에서는 채널 선택, 패키지 배포 정책, 라이선스 검토 기준이 다를 수 있습니다.
- GPU 환경에서는 CUDA 버전과 PyTorch 호환성을 함께 확인하세요.

Notes:
- The example environment for this repository is written with `conda-forge` in mind.
- If you use other channels or package distribution paths, review compatibility and your organization's internal policy separately.
- In institutional or commercial environments, package channel policy and license review requirements may differ.
- For GPU environments, also verify CUDA and PyTorch compatibility.

---

## 3. 실행 방법

아래 스크립트들은 실행 후 **Gradio 웹 UI**를 띄우는 형태로 구성되어 있습니다.  
터미널에 표시되는 주소를 브라우저에서 열어 사용하세요.

The scripts below are designed to launch **Gradio web UIs**.  
After running each script, open the local address shown in the terminal.

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

### KorQuAD 기반 검색 실습

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

공식 페이지:
- https://korquad.github.io/category/1.0_KOR.html

예시 원본 파일명:
- `KorQuAD_v1.0_train.json`
- `KorQuAD_v1.0_dev.json`

로컬 사용 편의를 위해 파일명을 아래처럼 바꾸어 사용할 수 있습니다.

- `KorQuAD_v1_0_train.json`
- `KorQuAD_v1_0_dev.json`

예시:

```bash
mkdir -p ./data
cp /download/path/KorQuAD_v1.0_train.json ./data/KorQuAD_v1_0_train.json
cp /download/path/KorQuAD_v1.0_dev.json ./data/KorQuAD_v1_0_dev.json
```

단, 위 파일명 변경은 로컬 사용 편의를 위한 예시일 뿐이며, 원본 데이터 파일의 라이선스나 배포 조건이 변경되는 것은 아닙니다.

KorQuAD original dataset files are not included in this repository.  
Please download them from the official source and copy them into the local `./data/` directory.

Renaming the files locally for convenience does not change the original licensing or distribution conditions of the dataset.

---

## 5. KorFactScore 사용 안내

KorFactScore는 별도 저장소를 clone하여 사용합니다.

공식 저장소:
- https://github.com/ETRI-XAINLP/KorFactScore

예시:

```bash
git clone https://github.com/ETRI-XAINLP/KorFactScore.git
python korfactscore_lab.py --korfs_path ./KorFactScore
```

이후 KorFactScore 저장소 안내에 따라 필요한 Wikipedia DB를 준비한 뒤 실행하세요.

KorFactScore must be cloned separately.  
Please follow the original repository instructions for the required Wikipedia DB and related setup.

---

## 6. 포함하지 않는 항목

이 저장소는 다음 항목을 기본적으로 포함하지 않습니다.

- 원본 데이터셋 파일
- 대용량 데이터베이스 파일
- 모델 weight 파일
- 외부 저장소 전체 코드
- 로컬 캐시 및 생성 산출물 일부

This repository does not normally include:
- original dataset files
- large database files
- model weights
- full external repositories
- local caches or generated outputs

---

## 7. 이용 조건

이 저장소의 사용 조건은 [`TERMS.md`](./TERMS.md)에 정리되어 있습니다.

핵심 요약:
- 비상업적 교육·실습 목적 사용 허용
- 출처 표시 필요
- 수정본 공개 및 재배포 금지
- 상업적 사용 금지
- 외부 리소스는 각 원 이용 조건을 따름

Usage conditions are described in [`TERMS.md`](./TERMS.md).

Summary:
- non-commercial educational/practice use is allowed
- attribution is required
- redistribution and public release of modified versions are not allowed
- commercial use is not allowed
- third-party resources remain subject to their own original terms

---

## 8. 외부 리소스 안내

이 저장소는 실습과 연구 편의를 위해 외부 저장소, 데이터셋, 모델, 패키지, 문서 등을 참고하거나 연동할 수 있습니다.  
이러한 제3자 리소스의 저작권과 이용 조건은 각 원저작자 및 원 라이선스를 따릅니다.

예시:
- KorQuAD  
  https://korquad.github.io/category/1.0_KOR.html
- KorFactScore  
  https://github.com/ETRI-XAINLP/KorFactScore
- Hugging Face  
  https://huggingface.co/
- Gradio  
  https://www.gradio.app/
- Sentence Transformers  
  https://www.sbert.net/
- FAISS  
  https://github.com/facebookresearch/faiss

사용자는 본 저장소를 이용할 때, 이러한 외부 리소스에 대해서는 각 공식 배포처의 라이선스와 이용 조건을 별도로 확인하고 준수해야 합니다.

This repository may reference or interoperate with external repositories, datasets, models, packages, and documents for research and educational practice.  
Such third-party resources remain subject to their own original licenses, copyright notices, and terms of use.

Examples:
- KorQuAD  
  https://korquad.github.io/category/1.0_KOR.html
- KorFactScore  
  https://github.com/ETRI-XAINLP/KorFactScore
- Hugging Face  
  https://huggingface.co/
- Gradio  
  https://www.gradio.app/
- Sentence Transformers  
  https://www.sbert.net/
- FAISS  
  https://github.com/facebookresearch/faiss

Users are responsible for reviewing and complying with the original licenses and terms of any third-party resource used with this repository.

---

## 9. AI 활용 고지

이 저장소의 일부 코드는 AI 도구의 도움을 받아 초안 작성, 테스트, 수정되었을 수 있습니다.  
최종 검토와 공개 책임은 저장소 관리자에게 있습니다.

Some parts of this repository may have been drafted, tested, or revised with the help of AI tools.  
Final review and release responsibility remains with the repository maintainer.

---

## 10. 참고 자료

- KorQuAD 공식 페이지
- KorFactScore 공식 저장소
- Hugging Face 모델 저장소
- 발표자료 PDF

---

## 11. 연락 및 문의

재배포, 수정본 공개, 상업적 사용 등 별도 허가가 필요한 경우 저장소 관리자에게 문의하세요.

For permissions beyond the scope of [`TERMS.md`](./TERMS.md), please contact the repository maintainer.
