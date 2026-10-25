"""Project-wide constants: paths, column groups and the random seed.

Everything that decides *which* columns a model may see lives here, so the leakage rules
(what is known at origination and what is not) are written down in exactly one place.
"""
from __future__ import annotations

from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "LoanExport.csv"
RAW_ZIP = DATA_DIR / "LoanExport.zip"
SAMPLE_CSV = DATA_DIR / "sample" / "loans_sample_5k.csv"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

# A public copy of the export on GitHub (37 MB CSV); the data itself is not redistributed here.
DATA_URL = (
    "https://raw.githubusercontent.com/tech-girl-lead/Predicting-Mortgage-Backed-Securities-Prepayment-Risk/master/Django%20Deployment/Notebooks/LoanExport.csv"
)

# --------------------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------------------
DELINQ_TARGET = "EverDelinquent"
PREPAY_TARGET = "Prepaid36"
PREPAY_HORIZON_MONTHS = 36

# --------------------------------------------------------------------------------------
# Columns that must never be used as model inputs
# --------------------------------------------------------------------------------------
ID_COLUMNS = ["LoanSeqNum", "PostalCode"]
# Observed *after* origination. MonthsDelinquent > 0 is literally EverDelinquent == 1, and
# MonthsInRepayment is the observed life of the loan (it defines the prepayment target).
POST_ORIGINATION_COLUMNS = ["MonthsDelinquent", "MonthsInRepayment"]
# Redundant or not reliably origination-time information.
REDUNDANT_COLUMNS = [
    "ProductType",     # constant: every loan is a fixed-rate mortgage (FRM)
    "MaturityDate",    # = FirstPaymentDate + OrigLoanTerm
    "FirstPaymentDate",  # replaced by the engineered VintageMonth
    "ServicerName",    # servicing can be transferred after origination
]

# --------------------------------------------------------------------------------------
# Model input groups (after cleaning + feature engineering)
# --------------------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "CreditScore",
    "MIP",
    "Units",
    "OCLTV",
    "DTI",
    "OrigUPB",
    "LTV",
    "OrigInterestRate",
    "OrigLoanTerm",
    "VintageMonth",
    "MonthlyPayment",
    "TotalInterest",
    "EstMonthlyIncome",
    "RateSpread",
]
ONEHOT_FEATURES = [
    "FirstTimeHomebuyer",
    "Occupancy",
    "Channel",
    "PPM",
    "PropertyType",
    "LoanPurpose",
    "NumBorrowers",
]
TARGET_ENCODED_FEATURES = ["PropertyState", "MSA", "SellerName"]

MODEL_FEATURES = NUMERIC_FEATURES + ONEHOT_FEATURES + TARGET_ENCODED_FEATURES
