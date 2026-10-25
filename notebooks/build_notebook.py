"""Generate notebooks/mortgage_prepayment_risk.ipynb from the mbs_risk package.

The notebook is a thin, readable driver around the package (no copied code), so it stays in
sync with src/. It runs on Kaggle or Colab (it clones the repo and pip-installs it when the
package is not importable) and locally from the repo.

    python notebooks/build_notebook.py              # write the notebook
    python notebooks/build_notebook.py --execute    # write and execute it in place
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
OUT = HERE / "mortgage_prepayment_risk.ipynb"

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

CELLS = [
    md("""# Mortgage prepayment and delinquency risk

End-to-end walkthrough of the [`mbs_risk`](https://github.com/benisalla/mortgage-prepayment-risk)
package on 291,451 Freddie Mac fixed-rate mortgages (first payments 1999-2013):

1. load and clean the loan export,
2. engineer origination-time features,
3. explore the data and test the original hypotheses H1-H6,
4. model **delinquency** (`EverDelinquent`) and **prepayment within 36 months** (`Prepaid36`),
5. evaluate (ROC-AUC, PR-AUC, tuned-threshold F1, calibration) and explain the models.

Every number below is computed in this notebook; the README quotes the same pipeline."""),
    code("""# Setup: works locally (repo checkout) and on Kaggle / Colab (clones the repo).
import os, subprocess, sys
try:
    import mbs_risk
except ImportError:
    if not os.path.exists("mortgage-prepayment-risk"):
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/benisalla/mortgage-prepayment-risk.git"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "mortgage-prepayment-risk"], check=True)
    sys.path.insert(0, os.path.abspath("mortgage-prepayment-risk/src"))
    import mbs_risk
import pandas as pd
from IPython.display import Image, display
from mbs_risk import config, plotting
pd.set_option("display.width", 160, "display.max_columns", 40)
FIG = config.ROOT / "outputs" / "notebook"
FIG.mkdir(parents=True, exist_ok=True)
def show(fn, *args, name, **kw):
    fn(*args, path=FIG / name, **kw)
    display(Image(filename=str(FIG / name)))
print("mbs_risk", mbs_risk.__version__)"""),
    md("""## 1. Load and clean

`download_data()` fetches the full export (37 MB) into `data/raw/` once; `load_loans()`
reads it. Set `USE_SAMPLE = True` to run everything on the 5,000-row stratified sample
shipped with the repo, in a minute."""),
    code("""USE_SAMPLE = False
raw = mbs_risk.load_loans(config.SAMPLE_CSV if USE_SAMPLE else mbs_risk.download_data())
print(raw.shape)
raw.head()"""),
    code("""from mbs_risk.cleaning import missing_report
df = mbs_risk.prepare(raw)          # clean + features + bins + targets
missing_report(df).round(2)"""),
    md("""Missing-value codes: `"X"` in code columns (MSA, first-time buyer, PPM, number of
borrowers), and impossible zeros (credit score, LTV, DTI). They become real missing values;
the model pipeline imputes them (median / constant) and adds missing-indicator columns."""),
    md("""## 2. Targets and leakage

* `EverDelinquent` is given. `MonthsDelinquent > 0` is *exactly* `EverDelinquent == 1`, so it is
  dropped from the inputs (the original notebook kept it, which makes the task trivial).
* `MonthsInRepayment` is the observed life of the loan, known only after the fact; it defines the
  prepayment target and is never a model input.
* `Prepaid36 = 1` when a never-delinquent loan left the pool within 36 months of its first
  payment (voluntary payoff: refinance or sale). Ever-delinquent loans are excluded from this
  task, because an early exit after delinquency may be a default."""),
    code("""from mbs_risk.features import model_frame
for t in [config.DELINQ_TARGET, config.PREPAY_TARGET]:
    X, y = model_frame(df, t)
    print(f"{t:15s} rows={len(y):,} positive rate={y.mean():.3f} inputs={X.shape[1]}")
pd.crosstab(df.EverDelinquent, df.MonthsDelinquent > 0)"""),
    md("## 3. Exploratory analysis"),
    code("""show(plotting.plot_target_balance, df, name="target_balance.png")
show(plotting.plot_distributions, df, name="distributions.png")"""),
    code("""show(plotting.plot_correlation, df, name="correlation.png")"""),
    code("""show(plotting.plot_survival, df, name="survival.png")"""),
    md("""## 4. Hypotheses H1-H6

All six original hypotheses are about prepayment, so they are tested with `Prepaid36` on
never-delinquent loans. Verdicts are based on effect size (at least 2 percentage points between
the extreme groups); with ~230k loans every p-value is tiny."""),
    code("""hyp = mbs_risk.check_hypotheses(df)
show(plotting.plot_hypotheses, hyp, name="hypotheses.png")
hyp.drop(columns="rates")"""),
    md("""## 5. Delinquency model (`EverDelinquent`)

Stratified 64/16/20 train/validation/test split. Each model is a single scikit-learn pipeline
(impute, one-hot, cross-fitted target encoding for state/MSA/seller), so no statistic is fitted
on validation or test rows. The F1 threshold is tuned on validation."""),
    code("""MODELS = ["majority", "logreg", "hist_gb", "xgboost"]   # random_forest is in scripts/train.py
deq = mbs_risk.run_task(df, config.DELINQ_TARGET, MODELS)
deq.metrics.round(4)"""),
    code("""show(plotting.plot_roc_pr, deq, name="roc_pr_delinq.png")
show(plotting.plot_before_after, deq, "logreg", deq.best_model(), name="before_after_delinq.png")
show(plotting.plot_calibration, deq, name="calibration_delinq.png")"""),
    code("""from mbs_risk.evaluation import permutation_importance_table
imp_deq = permutation_importance_table(deq.models[deq.best_model()], deq.X_test, deq.y_test)
show(plotting.plot_importance, imp_deq, "Delinquency: permutation importance", name="importance_delinq.png")
imp_deq.head(10).round(4)"""),
    md("## 6. Prepayment model (`Prepaid36`)"),
    code("""pre = mbs_risk.run_task(df, config.PREPAY_TARGET, MODELS)
pre.metrics.round(4)"""),
    code("""show(plotting.plot_roc_pr, pre, name="roc_pr_prepay.png")
show(plotting.plot_before_after, pre, "logreg", pre.best_model(), name="before_after_prepay.png")
imp_pre = permutation_importance_table(pre.models[pre.best_model()], pre.X_test, pre.y_test)
show(plotting.plot_importance, imp_pre, "Prepayment: permutation importance", name="importance_prepay.png")
imp_pre.head(10).round(4)"""),
    md("""## 7. Take-aways

* Both targets are only moderately predictable from origination data (test ROC-AUC about 0.72
  for delinquency and 0.70 for prepayment): what happens after origination (market rates,
  house prices, unemployment) drives most of the outcome.
* Gradient boosting beats logistic regression clearly on prepayment (non-linear seller and rate
  effects) and only marginally on delinquency, where credit score dominates.
* The interest rate, LTV and the seller drive prepayment; higher-rate loans prepay much faster,
  which contradicts H6 as written and matches refinance-incentive theory.
* See the README and `docs/results.md` for the full-data numbers and the hypothesis verdicts."""),
]


def build(execute: bool = False) -> Path:
    nb = nbf.v4.new_notebook()
    nb["cells"] = CELLS
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python"}}
    if execute:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=3600, kernel_name="python3",
                       resources={"metadata": {"path": str(HERE)}}).execute()
    nbf.write(nb, OUT)
    return OUT


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    print("wrote", build(ap.parse_args().execute))
