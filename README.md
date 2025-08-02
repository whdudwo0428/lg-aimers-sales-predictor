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
