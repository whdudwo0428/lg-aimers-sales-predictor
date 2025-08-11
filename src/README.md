# LG Aimers Forecasting Pipeline (`src/`)

이 디렉터리는 LG Aimers 매출 예측 해커톤을 위해 설계된
엔드‑투‑엔드 파이프라인의 소스코드를 담고 있습니다.  외부
모델 구현(`lg‑project/models`에 있는 Autoformer, PatchTST, TimesFM
등)과는 별도로 **우리만의 래퍼와 데이터 처리 코드**가 담겨
있으며, 각 모델을 간편하게 학습·추론할 수 있도록 구조화돼
있습니다.  여기서 정의된 코드는 skeleton 형태로 제공되므로
실제 대회 참가 시에는 모델과 데이터 처리 로직을 적절히
확장해야 합니다.

## 📂 폴더 구조

```
project-root/
├─ src/
│  ├─ config.py                 # 공통 설정 (MODEL_NAME 등)
│  ├─ train_any.py              # 학습 엔트리
│  ├─ predict_any.py            # 예측 엔트리 (모델 공통)
│  ├─ core/                     # DataModule, LightningModule, utils, feature_engineer, holidays ...
│  ├─ models/                   # 래퍼들 + 외부 원본 라이브러리(models/FEDformer, Autoformer, PatchTST ...)
│  └─ optuna/
│     ├─ runner.py              # Optuna 실행 엔트리(튜닝 스터디 생성/재개/저장)
│     ├─ objective.py           # “학습 1회”를 수행하는 objective
│     ├─ spaces.py              # 모델별 탐색공간 정의(넓게, 조건부 포함)
│     └─ utils.py               # 설정 오버라이드/리소스 정리/저장 헬퍼
├─ dataset/
│  ├─ train.csv
│  ├─ test/TEST_*.csv           # 테스트 분할들
│  └─ sample_submission.csv
├─ results/
│  ├─ checkpoints/              # 학습 시 자동 저장되는 ckpt
│  ├─ optuna/<model>/           # 튠 결과(best_params.json, trials.csv 등)
│  └─ submission_*.csv          # 예측 산출물
├─ models/                      # 깃허브 원본 배치(폴더 이름/위치 유지)
│  ├─ FEDformer/...
│  ├─ Autoformer/...
│  └─ PatchTST/...
├─ requirements.txt             # 의존성 명세(아래 참조)
└─ README.md                    # (이 문서)

```

## 🔧 사용 방법

### 1. 환경 준비

* Python 3.10 이상, PyTorch 2.x가 설치된 가상환경을 사용하세요.
* 프로젝트 루트에서 외부 모델을 editable 모드로 설치할 수 있습니다. 예:

```bash
pip install -e models/Autoformer
pip install -e models/PatchTST
pip install -e models/timesfm
```

### 2. 데이터 배치

`dataset/train/train.csv`, `dataset/test/TEST_00.csv`–`TEST_09.csv`, `dataset/sample_submission.csv`를 적절한 경로에 배치하세요.  기본 로더는 컬럼 이름을 자동으로 정규화합니다(`영업일자`→`date`, `영업장명_메뉴명`→`store_item`, `매출수량`→`sales`).

### 3. 모델 학습

각 모델은 ``train_<model>.py``를 통해 학습합니다.  예를 들어 FedFormer 학습은 다음과 같이 실행합니다:

```bash
python -m src.train_fedformer
```

학습 로그는 콘솔에 출력되며, 마지막 epoch 후 모델 가중치는 `checkpoint/<model_name>/best.ckpt` 파일로 저장됩니다.  현재는 간단한 선형 모델을 사용하므로 성능이 낮지만, `src/models/<model>/model.py`를 수정하여 실제 모델을 사용하면 됩니다.

### 4. 예측 및 제출

학습이 완료된 후 각 모델의 `predict_<model>.py`를 실행하면 `results/<model>_submission.csv`가 생성됩니다.  기본 구현은 모든 예측을 0으로 설정하므로 반드시 `predict.py`를 수정하여 모델 추론을 수행하세요.

```bash
python -m src.predict_fedformer
```

### 5. 코드 확장하기

* **모델 교체**: `src/models/<model>/model.py` 내부의 `ForecastModel`을 외부 구현으로 교체하고, `build_model` 함수에서 적절히 초기화하세요.
* **하이퍼파라미터 조정**: `src/models/<model>/config.py`의 dataclass를 수정해 입력 길이, 배치 크기 등을 조정하세요.
* **데이터 전처리 강화**: `src/core/feature_engineer.py`에서 lag 주기나 이동평균 윈도우, 날짜 피처를 자유롭게 추가할 수 있습니다.
* **평가 로직 추가**: `src/evaluate.py`를 참고하여 교차검증 등을 구현할 수 있습니다.

## 📢 주의 사항

현재 제공된 코드는 **학습과 추론의 전체 파이프라인을 보여주는 구조적 예시**이며, 실제 대회에서 높은 성능을 얻으려면 모델 정의와 학습 로직을 반드시 보완해야 합니다.  특히 FedFormer, PatchTST, TimesFM, Autoformer의 원본 구현을 import하여 `ForecastModel`을 교체하고, `predict.py`의 예측 로직을 완성해야 합니다.


---
```
# 1) 가상환경 만들고 필수 설치
python -m venv .venv && source .venv/bin/activate         # (Windows: .venv\Scripts\Activate.ps1)
pip install -U pip wheel setuptools
pip install -r requirements.txt

# 2) (CUDA) GPU용 PyTorch 설치 — 예시(CUDA 12.x)
# Windows/Linux 공통: 본인 CUDA 버전에 맞춰 torch/torchvision/torchaudio 설치
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# 3) 학습 (Config.MODEL_NAME으로 모델 선택)
python -m src.train_any

---
# 확인용 3순회 옵튜나 
mkdir -p results/optuna/fedformer_quick
python -m src.optuna.runner \
  --model fedformer \
  --trials 3 \
  --storage sqlite:///$PWD/results/optuna/fedformer_quick/study.sqlite3 \
  --study-name optuna_fedformer_quick


# 4) 튠(학습과 동시에 Optuna)
python -m src.optuna.runner \
  --model fedformer \
  --trials 100 \
  --storage sqlite:///$PWD/results/optuna/fedformer/study.sqlite3 \
  --study-name optuna_fedformer

python -m src.optuna.runner \
  --model autoformer \
  --trials 100 \
  --storage sqlite:///$PWD/results/optuna/autoformer/study.sqlite3 \
  --study-name optuna_autoformer

python -m src.optuna.runner \
  --model patchtst \
  --trials 100 \
  --storage sqlite:///$PWD/results/optuna/patchtst/study.sqlite3 \
  --study-name optuna_patchtst


# 5) 베스트 파라미터로 재학습(긴 에폭)
python -m src.train_any --override results/optuna/fedformer/best_config_overrides.json

# 6) 예측 CSV 생성
python -m src.predict_any --model fedformer

```

