"""01 - Load the loan export and clean it.

Shows: reading the shipped 5k-row sample with an explicit schema, turning the "X" and impossible
zero codes into real missing values, and the missing-value report.

    python examples/01_load_and_clean.py
"""
from pathlib import Path

from mbs_risk import clean_loans, config, load_loans
from mbs_risk.cleaning import missing_report

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

raw = load_loans(config.SAMPLE_CSV)
print(f"raw sample: {raw.shape[0]:,} rows x {raw.shape[1]} columns")
print("raw codes before cleaning:", raw["MSA"].head(3).tolist(), raw["NumBorrowers"].unique().tolist())

clean = clean_loans(raw)
print("\nmissing values after cleaning:")
print(missing_report(clean).round(2).to_string())

clean.to_csv(OUT / "01_clean_sample.csv", index=False)
print(f"\nwrote {OUT / '01_clean_sample.csv'}")
