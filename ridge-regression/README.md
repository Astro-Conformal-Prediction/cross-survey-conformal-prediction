# Ridge Regression Baseline

Ridge Regression baseline for the **Cross-Survey Conformal Prediction**. 

## Directory Structure

```text
ridge-regression/
│
├── cross-survey-translation/
│   ├── train_kfold_ridge.py
│   └── results.csv
│
├── conformal-prediction-analysis/
│   ├── conformal_prediction.py
│   └── results/
│       ├── thresh_80/
│       ├── thresh_90/
│       └── thresh_95/
│
├── src/
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```
## Run Experiments

### Cross-Survey Translation

```bash
python3 ridge-regression/cross-survey-translation/train_kfold_ridge.py
```
- `train_kfold_ridge.py` – Ridge Regression translator
- `results.csv` – Translation results

### Conformal Prediction Analysis

```bash
python3 ridge-regression/conformal-prediction-analysis/conformal_prediction.py
```

- `conformal_prediction.py` – Conformal evaluation pipeline
- `results/` – Coverage statistics and uncovered objects for different confidence thresholds

### Shared Utilities
- `src/` – Common helper modules
