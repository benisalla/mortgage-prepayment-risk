"""Loading the Freddie Mac loan-level export, plus download and sampling helpers.

The raw file (``LoanExport.csv``, 291,451 loans x 28 columns) is a 1999-2013 origination
extract of the Freddie Mac Single-Family Loan-Level dataset. The full file is not stored in this
repository: :func:`download_data` fetches it from a public copy (``config.DATA_URL``) into
``data/raw/``. A 5,000-row stratified sample under ``data/sample/`` lets tests and examples run
offline.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from . import config

# Column -> dtype used when reading. Codes such as MSA, PostalCode and NumBorrowers are read as
# strings because they contain the placeholder "X" for "not available".
RAW_DTYPES: dict[str, str] = {
    "CreditScore": "int64",
    "FirstPaymentDate": "int64",
    "FirstTimeHomebuyer": "string",
    "MaturityDate": "int64",
    "MSA": "string",
    "MIP": "int64",
    "Units": "int64",
    "Occupancy": "string",
    "OCLTV": "int64",
    "DTI": "int64",
    "OrigUPB": "int64",
    "LTV": "int64",
    "OrigInterestRate": "float64",
    "Channel": "string",
    "PPM": "string",
    "ProductType": "string",
    "PropertyState": "string",
    "PropertyType": "string",
    "PostalCode": "string",
    "LoanSeqNum": "string",
    "LoanPurpose": "string",
    "OrigLoanTerm": "int64",
    "NumBorrowers": "string",
    "SellerName": "string",
    "ServicerName": "string",
    "EverDelinquent": "int64",
    "MonthsDelinquent": "int64",
    "MonthsInRepayment": "int64",
}
RAW_COLUMNS = list(RAW_DTYPES)


def default_data_path() -> Path:
    """Return the best available copy of the data: extracted CSV, else the shipped zip, else the sample."""
    for p in (config.RAW_CSV, config.RAW_ZIP, config.SAMPLE_CSV):
        if p.exists():
            return p
    raise FileNotFoundError(
        "No data found. Run `python scripts/download_data.py` or pass an explicit path."
    )


def load_loans(path: str | Path | None = None, nrows: int | None = None) -> pd.DataFrame:
    """Read the loan export (CSV or zipped CSV) with an explicit schema.

    Args:
        path: CSV or ZIP file. ``None`` picks :func:`default_data_path`.
        nrows: optional row limit, handy for quick experiments.

    Returns:
        Raw DataFrame with the 28 original columns (strings untouched, no cleaning).
    """
    path = Path(path) if path is not None else default_data_path()
    df = pd.read_csv(path, dtype=RAW_DTYPES, nrows=nrows, encoding="utf-8-sig")
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df[RAW_COLUMNS]


def download_data(url: str = config.DATA_URL, dest_dir: str | Path | None = None) -> Path:
    """Download ``LoanExport.csv`` (a CSV or a zip holding it) into ``data/raw/``.

    Returns the path of the CSV. Skips the download if the CSV already exists, and extracts a
    local ``data/LoanExport.zip`` instead of downloading when one is present.
    """
    dest_dir = Path(dest_dir) if dest_dir is not None else config.RAW_CSV.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / "LoanExport.csv"
    if out.exists():
        return out
    if config.RAW_ZIP.exists():
        blob = config.RAW_ZIP.read_bytes()
    else:
        with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 (fixed https URL)
            blob = resp.read()
    if zipfile.is_zipfile(io.BytesIO(blob)):
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
            blob = zf.read(name)
    out.write_bytes(blob)
    return out


def stratified_sample(
    df: pd.DataFrame, n: int = 5000, by: str = config.DELINQ_TARGET, seed: int = config.SEED
) -> pd.DataFrame:
    """Draw ``n`` rows keeping the class ratio of column ``by``."""
    frac = n / len(df)
    out = df.groupby(by, group_keys=False).sample(frac=frac, random_state=seed)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
