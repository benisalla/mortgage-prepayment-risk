import numpy as np
import pytest

from mbs_risk import config, model_frame, monthly_payment
from mbs_risk.features import scheduled_balance, vintage_month


def test_monthly_payment_known_value():
    # 200k at 6% over 30 years -> 1199.10 (standard amortisation table)
    assert monthly_payment(200_000, 6.0, 360) == pytest.approx(1199.10, abs=0.01)
    assert monthly_payment(1200, 0.0, 12) == pytest.approx(100.0)


def test_scheduled_balance_amortises_to_zero():
    assert scheduled_balance(100_000, 7.0, 360, 0) == pytest.approx(100_000)
    assert scheduled_balance(100_000, 7.0, 360, 360) == pytest.approx(0, abs=1e-4)
    assert 0 < scheduled_balance(100_000, 7.0, 360, 120) < 100_000


def test_vintage_month():
    import pandas as pd
    assert list(vintage_month(pd.Series([199901, 199903, 200001]))) == [0, 2, 12]


def test_prepay_target_definition(sample):
    t = sample[config.PREPAY_TARGET]
    delinquent = sample[config.DELINQ_TARGET] == 1
    assert t[delinquent].isna().all()
    ok = ~delinquent
    assert ((sample.loc[ok, "MonthsInRepayment"] <= 36) == (t[ok] == 1)).all()


@pytest.mark.parametrize("target", [config.DELINQ_TARGET, config.PREPAY_TARGET])
def test_model_frame_has_no_leakage(sample, target):
    X, y = model_frame(sample, target)
    banned = set(config.POST_ORIGINATION_COLUMNS + config.ID_COLUMNS) | {
        config.DELINQ_TARGET, config.PREPAY_TARGET}
    assert not banned & set(X.columns)
    assert list(X.columns) == config.MODEL_FEATURES
    assert len(X) == len(y) and set(np.unique(y)) == {0, 1}


def test_engineered_features_are_finite(sample):
    for c in ["MonthlyPayment", "TotalInterest", "VintageMonth", "RateSpread"]:
        assert np.isfinite(sample[c]).all(), c
