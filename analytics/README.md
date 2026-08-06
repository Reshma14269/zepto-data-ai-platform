# Analytics

Full end-to-end ML pipeline: EDA → feature engineering → model training → evaluation.
This is the largest module (50/100 marks) — give it the most build time.

## What this module does *(fill in as you build)*
- Exploratory data analysis on a customer/passenger-style dataset
- Feature engineering: encoding, scaling, handling missing values
- Train/validation/test split, baseline model
- Train and compare 2-3 models
- Evaluate with appropriate metrics (accuracy/precision/recall/F1 for classification,
  or MAE/RMSE/R² for regression)
- Written interpretation of results and model choice

## Files
- `eda.ipynb` — exploratory analysis *(TBD)*
- `model_pipeline.py` / `.ipynb` — feature engineering + training + evaluation *(TBD)*

## Setup

```bash
cd analytics
pip install -r requirements.txt
```

## How to run

```bash
jupyter notebook eda.ipynb
# or
python model_pipeline.py
```

## Design decisions

*TBD — explain dataset choice, feature engineering decisions, which models you
compared and why, and how you picked the final model, once built.*
