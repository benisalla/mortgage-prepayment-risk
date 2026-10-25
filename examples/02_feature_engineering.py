"""02 - Engineer origination-time features and the two targets.

Shows: the annuity payment formula, the implied income from DTI, the refinance-incentive proxy
RateSpread, the readable bins from the original notebook, and how Prepaid36 is defined.

    python examples/02_feature_engineering.py
"""
from mbs_risk import config, load_loans, monthly_payment, prepare
from mbs_risk.features import scheduled_balance

# A single loan by hand: 117,000 at 6.75% over 360 months
pay = monthly_payment(117_000, 6.75, 360)
print(f"monthly payment: {pay:,.2f}")
print(f"scheduled balance after 52 payments: {scheduled_balance(117_000, 6.75, 360, 52):,.0f}")

df = prepare(load_loans(config.SAMPLE_CSV))
cols = ["OrigUPB", "OrigInterestRate", "DTI", "MonthlyPayment", "EstMonthlyIncome", "RateSpread",
        "CreditRange", "LTV_Range", "MonthsInRepayment", "EverDelinquent", "Prepaid36"]
print("\n", df[cols].head(8).round(2).to_string())

print(f"\nEverDelinquent rate: {df.EverDelinquent.mean():.1%}")
print(f"Prepaid36 rate among never-delinquent loans: {df.Prepaid36.mean():.1%} "
      f"(defined for {df.Prepaid36.notna().sum():,} of {len(df):,} loans)")
