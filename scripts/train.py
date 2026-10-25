"""Train every model on both targets and save the fitted results to models/.

    python scripts/train.py                    # full data (291k loans), all five models
    python scripts/train.py --sample           # 5k-row sample, seconds
    python scripts/train.py --models logreg xgboost --cv logreg xgboost

Each task is saved as models/<target>.joblib (a mbs_risk.pipeline.TaskResult holding the fitted
pipelines, test-set probabilities and metrics). Run scripts/evaluate.py afterwards.
"""
import argparse
import time

from pathlib import Path

import joblib

from mbs_risk import MODEL_NAMES, config, load_loans, prepare, run_task


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="CSV/ZIP path (default: data/raw/LoanExport.csv, else the 5k sample)")
    ap.add_argument("--sample", action="store_true", help="use the 5k-row sample shipped in data/sample/")
    ap.add_argument("--models", nargs="+", default=MODEL_NAMES)
    ap.add_argument("--cv", nargs="*", default=["logreg", "hist_gb", "xgboost"],
                    help="models that also get 3-fold stratified CV on train+val")
    ap.add_argument("--tasks", nargs="+", default=[config.DELINQ_TARGET, config.PREPAY_TARGET])
    ap.add_argument("--out", default=str(config.MODELS_DIR))
    args = ap.parse_args()

    path = config.SAMPLE_CSV if args.sample else args.data
    t0 = time.time()
    df = prepare(load_loans(path))
    print(f"loaded + prepared {len(df):,} loans in {time.time() - t0:.1f}s")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        res = run_task(df, task, args.models, cv_models=tuple(m for m in args.cv if m in args.models))
        joblib.dump(res, out / f"{task}.joblib", compress=3)
        print(res.metrics.round(4).to_string(), "\n")
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
