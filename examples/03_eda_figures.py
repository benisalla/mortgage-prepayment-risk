"""03 - Exploratory figures: class balance, distributions, correlations, loan survival.

    python examples/03_eda_figures.py      # writes PNGs to examples/output/
"""
from pathlib import Path

from mbs_risk import config, load_loans, plotting, prepare

OUT = Path(__file__).parent / "output"
df = prepare(load_loans(config.SAMPLE_CSV))

plotting.plot_target_balance(df, OUT / "03_target_balance.png")
plotting.plot_distributions(df, OUT / "03_distributions.png")
plotting.plot_correlation(df, OUT / "03_correlation.png")
plotting.plot_survival(df, OUT / "03_survival_curves.png")
print("wrote", *sorted(p.name for p in OUT.glob("03_*.png")))
