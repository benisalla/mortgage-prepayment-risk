# Examples

Short scripts that use the `mbs_risk` package API in the order you would use it. They run from
the repository root on the 5,000-row stratified sample shipped in `data/sample/`, on a laptop
CPU, in well under a minute each. Outputs go to `examples/output/` (git-ignored).

```bash
pip install -e .
python examples/01_load_and_clean.py
```

| Script | What it shows | Output |
|---|---|---|
| [`01_load_and_clean.py`](01_load_and_clean.py) | Read the export with an explicit schema, map `"X"` / impossible zeros to missing, missing-value report | `01_clean_sample.csv` |
| [`02_feature_engineering.py`](02_feature_engineering.py) | Annuity payment, scheduled balance, implied income, `RateSpread`, bins, the `Prepaid36` target | console |
| [`03_eda_figures.py`](03_eda_figures.py) | Class balance, distributions, Spearman correlations, loan survival curves | `03_*.png` |
| [`04_test_hypotheses.py`](04_test_hypotheses.py) | Check the original hypotheses H1-H6 with effect sizes and chi-square tests | `04_hypotheses.png` |
| [`05_train_delinquency_model.py`](05_train_delinquency_model.py) | Majority vs logistic regression vs gradient boosting on `EverDelinquent`, ROC/PR and before/after figure | `05_*.png` |
| [`06_prepayment_model_and_scoring.py`](06_prepayment_model_and_scoring.py) | Prepayment model, permutation importance, save with joblib, score two new loan applications | `06_importance.png`, `06_prepay_model.joblib` |

Numbers printed by the examples come from 5,000 loans (1,000 test loans) and are noisier than
the full-data results in the main README; use `python scripts/train.py` for those.
