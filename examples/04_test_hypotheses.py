"""04 - Check the original hypotheses H1-H6 against the data.

Each hypothesis is about prepayment and is tested on never-delinquent loans with Prepaid36.
The verdict uses effect size (at least 2 percentage points between extreme groups), not only
p-values. On the 5k sample the numbers are noisier than the full-data figures in the README.

    python examples/04_test_hypotheses.py
"""
from pathlib import Path

from mbs_risk import check_hypotheses, config, load_loans, plotting, prepare

OUT = Path(__file__).parent / "output"
df = prepare(load_loans(config.SAMPLE_CSV))
h = check_hypotheses(df)
print(h[["statement", "verdict", "detail", "effect_pp", "p_value"]].round(4).to_string())
plotting.plot_hypotheses(h, OUT / "04_hypotheses.png")
print(f"\nwrote {OUT / '04_hypotheses.png'}")
