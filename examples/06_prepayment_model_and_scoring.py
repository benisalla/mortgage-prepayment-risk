"""06 - Prepayment model, permutation importance, and scoring new loans.

Trains a logistic regression and a histogram gradient boosting model on Prepaid36 (sample),
explains the better one with permutation importance, saves it with joblib, and scores two
hand-written loan applications.

    python examples/06_prepayment_model_and_scoring.py
"""
from pathlib import Path

import joblib
import pandas as pd

from mbs_risk import add_features, clean_loans, config, load_loans, plotting, prepare, run_task
from mbs_risk.evaluation import permutation_importance_table

OUT = Path(__file__).parent / "output"
df = prepare(load_loans(config.SAMPLE_CSV))
res = run_task(df, config.PREPAY_TARGET, ["logreg", "hist_gb"])
best = res.best_model()
model = res.models[best]

imp = permutation_importance_table(model, res.X_test, res.y_test, n_repeats=3)
print("\ntop features:\n", imp.head(8).round(4).to_string(index=False))
plotting.plot_importance(imp, f"Permutation importance ({best}, sample)", OUT / "06_importance.png")
joblib.dump(model, OUT / "06_prepay_model.joblib")

# Two new applications, identical except for the interest rate (6.5% vs 7.75%).
base = dict(CreditScore=720, FirstPaymentDate=199904, FirstTimeHomebuyer="N", MaturityDate=202903,
            MSA="31084", MIP=0, Units=1, Occupancy="O", OCLTV=80, DTI=35, OrigUPB=150_000, LTV=80,
            OrigInterestRate=6.5, Channel="R", PPM="N", ProductType="FRM", PropertyState="CA",
            PropertyType="SF", PostalCode="90000", LoanSeqNum="NEW1", LoanPurpose="P",
            OrigLoanTerm=360, NumBorrowers="2", SellerName="Ot", ServicerName="Other servicers")
apps = pd.DataFrame([base, {**base, "OrigInterestRate": 7.75, "LoanSeqNum": "NEW2"}])
# reuse the vintage rate table from the training data so RateSpread matches training
ref = df.groupby("VintageMonth")["OrigInterestRate"].median()
X_new = add_features(clean_loans(apps), rate_reference=ref)[config.MODEL_FEATURES]
for rate, p in zip(apps.OrigInterestRate, model.predict_proba(X_new)[:, 1]):
    print(f"rate {rate:.2f}% -> P(prepay within 36 months) = {p:.1%}")
