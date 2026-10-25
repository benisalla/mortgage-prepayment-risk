"""EDA and hypothesis figures (no model training needed).

    python scripts/make_figures.py             # full data
    python scripts/make_figures.py --sample    # 5k-row sample

Writes reports/figures/eda_*.png, reports/figures/hypotheses.png and reports/hypotheses.csv.
"""
import argparse
from pathlib import Path

from mbs_risk import check_hypotheses, config, load_loans, plotting, prepare
from mbs_risk.cleaning import missing_report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--reports", default=str(config.REPORTS_DIR))
    args = ap.parse_args()
    rep = Path(args.reports)
    fig = rep / "figures"
    fig.mkdir(parents=True, exist_ok=True)

    df = prepare(load_loans(config.SAMPLE_CSV if args.sample else args.data))
    print(f"{len(df):,} loans")
    print(missing_report(df).round(2).to_string())
    plotting.plot_target_balance(df, fig / "eda_target_balance.png")
    plotting.plot_distributions(df, fig / "eda_distributions.png")
    plotting.plot_correlation(df, fig / "eda_correlation.png")
    plotting.plot_survival(df, fig / "eda_survival_curves.png")

    hyp = check_hypotheses(df)
    plotting.plot_hypotheses(hyp, fig / "hypotheses.png")
    hyp.drop(columns="rates").to_csv(rep / "hypotheses.csv")
    print(hyp.drop(columns="rates").to_string())
    print(f"figures -> {fig}")


if __name__ == "__main__":
    main()
