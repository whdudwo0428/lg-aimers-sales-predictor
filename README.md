# LG Aimers Sales Predictor

This repository contains a complete experimental framework for the LG
Aimers Gonjiam Resort demand forecasting competition.  The goal is to
predict menu sales for multiple food & beverage outlets over a 7‑day
horizon given the past 28 days of sales.  A diverse set of forecasting
models is implemented along with unified data loading, feature
engineering and evaluation utilities.

## Directory Structure

```
lg-aimers-sales-predictor/
├── dataset/
│   ├── train/
│   │   ├── train.csv              # Training data (copied from root)
│   ├── test/
│   │   ├── TEST_00.csv            # Past 28‑day sequences for test split 00
│   │   ├── TEST_01.csv            # Test files 01–09 (provided by organisers)
│   │   └── …
│   └── sample_submission.csv      # Submission template
├── results/
│   ├── baseline_submission.csv     # Baseline predictions produced by the original notebook (copied here for convenience)
│   └── experiment_summary.csv      # Log of experiment runs and validation SMAPE (generated at runtime)
├── baseline_submission.csv         # Baseline submission produced by baseline.ipynb
├── baseline.ipynb                  # Original baseline notebook for reference
├── EDA/
│   ├── eda_analysis.py             # Script to perform exploratory data analysis
│   └── figures/                    # Directory where EDA plots are saved
├── models/
│   ├── base.py                     # Abstract base model & factory
│   ├── common.py                   # DataLoader, feature engineering & SMAPE
│   ├── tft.py                      # CatBoost approximation of TFT
│   ├── nbeats.py                   # MLP approximation of N‑BEATS
│   ├── dlinear.py                  # Linear regression baseline
│   ├── autoformer.py               # LightGBM approximation of Autoformer
│   ├── fedformer.py                # Random forest approximation of FEDformer
│   ├── patchtst.py                 # ExtraTrees approximation of PatchTST
│   ├── deepar.py                   # ARIMA approximation of DeepAR
│   ├── gbt.py                      # XGBoost gradient boosted trees
│   ├── sliding_transformer.py      # LightGBM with lag differences
│   └── hybrid.py                   # LightGBM + MLP hybrid model
├── scripts/
│   └── run_experiment.py           # Command line driver for training & inference
├── evaluate.py                     # SMAPE evaluation tool
├── util.py                         # Date feature augmentation and basic SMAPE
└── README.md                       # This file
```

## Installation
All dependencies are specified in the `package.json`.  To install
Python packages manually, you can create a virtual environment and
install the minimal requirements:

```bash
python -m venv .venv
source .venv/bin/activate

pip install \
  numpy==1.26.4 \
  pandas==2.2.2 \
  scikit-learn==1.4.2 \
  lightgbm==4.3.0 \
  catboost==1.2.2 \
  xgboost==2.0.3 \
  pmdarima==2.0.4
```

Alternatively you can use the existing environment if it already has
the required libraries installed (numpy, pandas, scikit‑learn,
lightgbm, catboost, xgboost, pmdarima).

> **Note on `pmdarima` compatibility**
>
> The `deepar` model uses the optional `pmdarima` library to fit ARIMA
> models.  Recent versions of NumPy (2.0+) are not yet fully supported
> by the latest `pmdarima` releases on some platforms.  If `pmdarima`
> fails to install or import due to a binary compatibility error (for
> example, when using Python 3.9 with NumPy 2.0), the `deepar` module
> will automatically fall back to a simple naive forecast and emit a
> warning.  Other models are unaffected.  You can also choose to pin
> an older NumPy version (e.g. `numpy<2.0`) or omit installing
> `pmdarima` entirely if you do not intend to run the `deepar` model.

## Running an Experiment

Use the `run_experiment.py` script to train a model, evaluate it on a
time‑aware validation split and generate predictions for the test
files.  Below is an example using the gradient boosted tree model:

```bash
python scripts/run_experiment.py \
    --model gbt \
    --params n_estimators=500,learning_rate=0.05,max_depth=6 \
    --lookback 28 \
    --horizon 7 \
    --val_ratio 0.2 \
    --train_path dataset/train/train.csv \
    --test_dir dataset/test \
    --output_dir results
```

This command will:

1. Load the training data from `train.csv` and generate sliding
   windows of length 28 with horizon 7.
2. Randomly split 20% of the windows into a validation set (seeded
   for reproducibility).
3. Train an XGBoost regressor per forecast horizon on the remaining
   windows.
4. Evaluate the model on the validation set by computing SMAPE per
   item, per store and overall (simple mean across stores).  The
   validation SMAPE is printed to the console.
5. Load each `TEST_XX.csv` in `dataset/test`, generate features and
   produce a 7‑day forecast for each item.  The predictions are saved
   as `results/submission_gbt_<params>.csv` following the
   `sample_submission.csv` format.
6. Append an entry to `results/experiment_summary.csv` containing the
   timestamp, model name, hyperparameters and validation SMAPE.

Replace `gbt` with any of the following model names to run a
different algorithm:

* `tft` – CatBoost approximation of Temporal Fusion Transformer
* `nbeats` – Multi‑layer perceptron approximation of N‑BEATS
* `dlinear` – Ridge regression baseline
* `autoformer` – LightGBM approximation of Autoformer
* `fedformer` – Random forest approximation of FEDformer
* `patchtst` – ExtraTrees approximation of PatchTST
* `deepar` – ARIMA approximation of DeepAR (requires fitting per item)
* `gbt` – XGBoost gradient boosted trees
* `sliding_transformer` – LightGBM with lag difference features
* `hybrid` – LightGBM + MLP hybrid model

### Example: Running the Hybrid Model

```bash
python scripts/run_experiment.py \
    --model hybrid \
    --params base_params.n_estimators=200,base_params.learning_rate=0.05,res_params.hidden_layer_sizes=64_32 \
    --lookback 28 --horizon 7 \
    --output_dir results
```

## Evaluation

The script `evaluate.py` can be used to compute the SMAPE between
ground truth values and a submission file.  For local validation you
can split the training data and treat the hold‑out horizon as test.

```bash
python evaluate.py --truth path/to/ground_truth.csv --pred path/to/submission.csv --output breakdown.csv
```

The SMAPE is averaged per item and then per store.  Because the
competition’s official leaderboard uses undisclosed store weights, the
overall SMAPE reported here may differ from the leaderboard.  Still,
relative model performance should remain comparable.

## Baseline Notebook

The repository ships with the original `baseline.ipynb` notebook used to
generate a simple benchmark.  Executing this notebook will produce
`baseline_submission.csv`, which is saved under the `results/` directory.
This file is included for reference so that you can compare your model
outputs against a known starting point.  The baseline notebook also
demonstrates how the data were loaded and how the official
`sample_submission.csv` format is constructed.

## Notes

* This codebase avoids deep learning dependencies in order to run
  autonomously in offline environments.  Nevertheless the modelling
  strategies mimic many ideas from state‑of‑the‑art architectures
  through engineered features and ensemble methods.
* Feel free to extend the feature engineering (for example adding
  more statistical summaries over the history) or replace the model
  implementations with more sophisticated ones if you have access to
  PyTorch or TensorFlow.

---

결론
아래 10개 모델/기법이 **지금 주신 도메인 제약**(외부 데이터 직접 사용 불가, 날짜 기반 파생변수 가능), **평가 방식(SMAPE)**, **입력 28일 → 예측 7일 범위 / 윈도윙 평가**에 가장 잘 맞는 SOTA 후보들
각각 적합한 이유, 핵심 변형/적용 포인트, 장단점, SMAPE 정합성에 대한 메모 명시

---

## 1. Temporal Fusion Transformer (TFT)

**왜 적합한가:** 시계열에 있는 시점별 중요도, 카테고리(업장/메뉴) 임베딩, 장기/단기 패턴을 동시에 다루며 해석성도 있음.
**핵심 변형:**

* 업장\_메뉴명 embedding
* 윈도우 입력(28일) → 출력 7일을 multi-horizon 예측 구조로
* Static + time-varying known (요일/월/휴일) + time-varying unknown(과거 수요) 구분
  **장점:** 강력한 표현력, 불확실성 감지 가능(예: quantile output)
  **단점:** 비교적 무겁고 튜닝 복잡
  **SMAPE 정합성:** 마지막 출력에 대해 SMAPE를 근사하는 custom loss 사용 (예: differentiable 형태로 $\frac{2|y-\hat y|}{|y|+|\hat y|+\epsilon}$ 평균).

---

## 2. N-BEATS / N-BEATSx

**왜 적합한가:** 시계열 자체만으로 트렌드/계절성/잔차를 분해해서 예측, 외부 피처 없이도 잘 작동.
**핵심 변형:**

* Multi-horizon(7일) 출력 설정
* Residual stacking depth 조절
  **장점:** 단순 구조 대비 강력, 일반화 잘됨
  **단점:** 카테고리 효과(업장/메뉴) 통합하려면 입력 분리 or embedding 후 앙상블 필요
  **SMAPE 정합성:** loss를 SMAPE 기반으로 직접 쓸 수 있게 수정 (원래는 MSE 계열이지만 custom block으로 교체).

---

## 3. DLinear (and variants like SVD-Lin)

**왜 적합한가:** 시계열을 trend/season decomposition 없이 직접 선형적 패턴으로 분리해서 예측, 특히 최근 논문에서 단순한 구조로 SOTA 성능 보여줌.
**핵심 변형:**

* 업장별로 독립적 파라미터 또는 그룹화
* 윈도우 분할(슬라이딩) + ensembled output
  **장점:** 매우 가볍고 학습/추론 빠름, 오버핏 덜함
  **단점:** 비선형성이 강한 경우 한계 (하지만 윈도윙 + 앙상블으로 커버 가능)
  **SMAPE 정합성:** 출력에 SMAPE 직접 적용.

---

## 4. Autoformer

**왜 적합한가:** 장기 시계열에서의 자기상호작용을 효율적으로 처리하는 decomposition-based transformer, trend/seasonality 분리 구조.
**핵심 변형:**

* 입력 28 → 예측 7 multi-horizon
* 업장/메뉴 구분은 prefix embedding 또는 separate encoder
  **장점:** 장기 의존성도 잡고, 노이즈에 강함
  **단점:** 구조가 TFT보다 덜 해석적, 튜닝 필요
  **SMAPE 정합성:** 예측 output layer에 SMAPE loss 적용 + residual 연결로 안정화.

---

## 5. FEDformer

**왜 적합한가:** Frequency Enhanced Decomposition Transformer. 계절/주기성(주말, 월별 패턴)을 주파수 수준에서 분해해서 잡아주는 방식.
**핵심 변형:**

* FFT 기반 주기성 분해
* 도메인 지식으로 “주말/월말” 주기 반영 filter design 가능
  **장점:** 계절성 강한 시계열에 성능 좋아짐
  **단점:** frequency component selection 튜닝이 필요
  **SMAPE 정합성:** reconstruction 단계 이후 SMAPE로 직접 정렬.

---

## 6. PatchTST (Transformer with patching + channel modeling)

**왜 적합한가:** 윈도우를 일정 길이 패치로 나누어 transformer에 공급, local 패턴과 global 패턴 모두 활용.
**핵심 변형:**

* 업장별 시계열을 하나의 “channel”로 보고 multi-channel 확장도 가능
* 28일을 여러 패치로 분할해 input sequence 생성
  **장점:** 데이터 효율 좋고 노이즈 완화
  **단점:** 패치 크기/stride 튜닝 필요
  **SMAPE 정합성:** 예측된 패치 합에 SMAPE loss 적용.

---

## 7. DeepAR (probabilistic RNN)

**왜 적합한가:** 카테고리(업장/메뉴)별로 조건부 확률 모델링, 불확실성 있는 multi-step 예측에 강점.
**핵심 변형:**

* 조건부 input으로 업장/메뉴 embedding + 과거 28일
* 예측 분포를 얻고, 중앙값 혹은 예측치로 SMAPE 평가
  **장점:** 불확실성 표현, 시계열간 공유 학습
  **단점:** 전통적인 RNN 기반이라 트렌드 캡처에서 transformer보다 약할 수 있음
  **SMAPE 정합성:** point estimate로 변환한 뒤 SMAPE 사용, 또는 expected SMAPE 근사 고려 (추정임).

---

## 8. Gradient Boosted Trees (LightGBM / CatBoost) with lag/rolling features

**왜 적합한가:** 비딥러닝 baseline이지만 date-derived, lag, rolling statistics, 업장/메뉴 캡슐화하면 강력하고 해석 가능.
**핵심 변형:**

* lookback 기반 lag (1,7,14), moving average, day-of-week interaction
* 업장\_메뉴 one-hot 또는 target encoding
  **장점:** 학습/튜닝 빠름, 과적합 제어 쉬움
  **단점:** 시계열 구조를 sequence로 직접 모델링하지 않음 (feature engineering에 의존)
  **SMAPE 정합성:** 예측값에 SMAPE 그대로 계산; loss로 쓰려면 목적함수 커스터마이징(예: LightGBM의 사용자 정의 objective) 가능.

---

## 9. Windowed / Sliding Ensemble of Transformers (e.g., LogTrans / Informer style variant)

**왜 적합한가:** 짧은 입력 윈도우를 여러 방식으로 겹쳐 예측하고 앙상블해서 7일 horizon을 보강.
**핵심 변형:**

* 여러 overlapping 28-day windows → 각각 예측 → 가중 평균
* Informer의 ProbSparse attention 아이디어 도입해 효율화
  **장점:** 안정적인 multi-horizon 예측, 윈도윙 평가와 직결
  **단점:** 복잡도 상승, 앙상블 정합성 관리 필요
  **SMAPE 정합성:** 각 윈도우 출력에 SMAPE 기반 calibration (예: 학습 시 윈도우별 weight 학습).

---

## 10. Hybrid / Residual Stack: LightGBM + Neural Residual Correction

**왜 적합한가:** 트렌드를 tree-based가 잡고, residual(비선형/미세 패턴)을 transformer/N-BEATS 등이 보정.
**핵심 변형:**

* 1단계: LightGBM으로 baseline 예측
* 2단계: 그 residual을 작게 transformer/N-BEATS로 학습
  **장점:** 강한 편향-분산 균형, 실용적
  **단점:** 파이프라인 복잡도, 두 단계 튜닝
  **SMAPE 정합성:** 최종 합산 출력에 SMAPE loss로 end-to-end 조정 (residual scaling 포함).

---

## SMAPE-aligned Custom Loss (공통 적용법)

* 기본 형태:

  $$
  \text{SMAPE}(y,\hat y) = \frac{2|y-\hat y|}{|y| + |\hat y| + \epsilon}
  $$
* differentiable smoothing: $|\cdot|$을 $\sqrt{(y-\hat y)^2 + \delta}$로 근사하거나 Huber 스타일 혼합.
* zero값 안정화 위해 denominator에 $\epsilon$ 추가.
* multi-horizon 예측은 horizon-wise로 평균 내되, 중요도 가중치(예: 가까운 날에 더 가중)도 고려 가능.

---

## 제안된 순서 (우선순위 예시)

1. **N-BEATS / DLinear**로 빠른 baseline 구축 (경량, SMAPE loss 적용 쉬움)
2. **TFT 또는 Autoformer**로 고차원 표현력 확보
3. \*\*Hybrid (LightGBM + residual)\*\*으로 안정성과 성능 조합
4. **PatchTST / FEDformer**로 계절성/局部 패턴 보강
5. **Ensemble / 윈도윙 구조**로 최종 스태킹

---

## 다음 액션 제안 (즉시 가능한 것)

* A. SMAPE 기반 custom loss 함수 코드 템플릿 먼저 만들어서 모든 후보에 붙일 수 있게 해두기
* B. 경량 baseline: DLinear / N-BEATS에 현재 피처(업장/메뉴 구분 + 날짜 파생) 넣고 28→7 실험
* C. 모델별 validation split (예: time-blocked rolling validation) 설계
* D. 업장/메뉴별 embedding 방식 결정 (shared vs separate)

-
