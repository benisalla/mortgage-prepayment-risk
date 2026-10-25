import numpy as np
import pytest

from mbs_risk import MODEL_NAMES, config, make_model, model_frame, run_task
from mbs_risk.evaluation import classification_metrics, tune_threshold
from mbs_risk.hypotheses import check_all


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_fits_and_predicts(sample, name):
    X, y = model_frame(sample, config.DELINQ_TARGET)
    m = make_model(name).fit(X.iloc[:1500], y.iloc[:1500])
    p = m.predict_proba(X.iloc[1500:2000])[:, 1]
    assert p.shape == (500,) and np.all((p >= 0) & (p <= 1))


def test_run_task_beats_majority(sample):
    res = run_task(sample, config.DELINQ_TARGET, ["majority", "logreg"], verbose=False)
    m = res.metrics
    assert m.loc["majority", "roc_auc"] == pytest.approx(0.5)
    assert m.loc["logreg", "roc_auc"] > 0.62
    assert m.loc["logreg", "pr_auc"] > m.loc["majority", "pr_auc"]
    assert res.best_model() == "logreg"


def test_threshold_and_metrics():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    thr = tune_threshold(y, p)
    met = classification_metrics(y, p, thr)
    assert met["roc_auc"] == pytest.approx(0.75)
    assert met["tp"] + met["fn"] == 2 and met["f1"] > 0


def test_hypotheses_table(sample):
    h = check_all(sample)
    assert list(h.index) == ["H1", "H2", "H3", "H4", "H5", "H6"]
    assert h.loc["H4", "verdict"] == "not testable"
    assert set(h["verdict"]) <= {"supported", "contradicted", "no material effect", "not testable"}
