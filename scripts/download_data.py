"""Download the full loan export (37 MB) into data/raw/LoanExport.csv.

The full data is not stored in this repository (only a 5,000-row sample in data/sample/);
this script fetches it from a public copy on GitHub (mbs_risk.config.DATA_URL).

    python scripts/download_data.py
"""
from mbs_risk import config
from mbs_risk.data import download_data

if __name__ == "__main__":
    path = download_data()
    print(f"data ready: {path.relative_to(config.ROOT)} ({path.stat().st_size / 1e6:.1f} MB)")
