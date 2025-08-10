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
src/
├── core/                  # 공통 기능 모음
│   ├── data_loader.py         # CSV 읽기/병합/기본 전처리 함수
│   ├── data_module.py         # Lightning DataModule + 슬라이딩 윈도우
│   ├── feature_engineer.py    # Lag, MA, 요일·월·연중일·공휴일 피처
│   ├── loss.py                # Weighted SMAPE 등 손실 함수
│   ├── evaluation.py          # SMAPE 등 평가 메트릭
│   ├── holidays.py            # 공휴일 리스트 정의
│   └── utils.py               # seed 고정, config 베이스 등 유틸
│
├── models/                # 모델별 래퍼와 학습/추론 스크립트
│   ├── fedformer/
│   │   ├── model.py           # FedFormer 래퍼 (placeholder)
│   │   ├── config.py          # FedFormer 하이퍼파라미터 정의
│   │   ├── train.py           # FedFormer 학습 루프
│   │   └── predict.py         # FedFormer 추론 & 제출 파일 생성
│   ├── patchtst/              # PatchTST 래퍼 (placeholder)
│   │   ├── model.py
│   │   ├── config.py
│   │   ├── train.py
│   │   └── predict.py
│   ├── timesfm/               # TimesFM 래퍼 (placeholder)
│   │   ├── model.py
│   │   ├── config.py
│   │   ├── train.py
│   │   └── predict.py
│   └── autoformer/            # Autoformer 래퍼 (placeholder)
│       ├── model.py
│       ├── config.py
│       ├── train.py
│       └── predict.py
│
├── train_fedformer.py     # 진입점: FedFormer 학습
├── train_patchtst.py      # 진입점: PatchTST 학습
├── train_timesfm.py       # 진입점: TimesFM 학습
├── train_autoformer.py    # 진입점: Autoformer 학습
├── predict_fedformer.py   # 진입점: FedFormer 예측
├── predict_patchtst.py    # 진입점: PatchTST 예측
├── predict_timesfm.py     # 진입점: TimesFM 예측
├── predict_autoformer.py  # 진입점: Autoformer 예측
└── evaluate.py            # 간단한 평가/검증 스크립트
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
# 프로젝트 루트
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 공통 의존성
pip install -U pip wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # GPU일 때; CPU면 일반 pip install torch ...
pip install pytorch-lightning pandas numpy scikit-learn tqdm optuna

# FEDformer 원본이 요구하는 추가 패키지(필요 시)
pip install -r models/FEDformer/requirements.txt

# 학습 실행
python -m src.train_fedformer
```

