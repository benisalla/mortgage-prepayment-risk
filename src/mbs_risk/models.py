"""Model zoo: one shared preprocessing block and a set of classifiers.

Every model is a single scikit-learn :class:`~sklearn.pipeline.Pipeline` so that imputation,
one-hot encoding and target encoding are fitted on the training fold only. This fixes the
original notebook's leakage, where an M-estimate target encoder was fitted on a 20% split that
later overlapped the validation data.

High-cardinality codes (state, MSA, seller) use :class:`sklearn.preprocessing.TargetEncoder`,
which cross-fits its encoding inside ``fit_transform`` (each training row is encoded with
statistics computed without that row's fold), so tree models cannot memorise the target
through the encoding.
"""
from __future__ import annotations

import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from . import config

N_JOBS = 2  # the reference machine is shared; keep CPU usage modest


def _target_encoder(seed: int) -> TargetEncoder:
    """Cross-fitted target encoder; scikit-learn >= 1.9 takes the CV splitter directly."""
    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 9):
        return TargetEncoder(target_type="binary", cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    return TargetEncoder(target_type="binary", cv=5, shuffle=True, random_state=seed)


def build_preprocessor(seed: int = config.SEED) -> ColumnTransformer:
    """Column transformer: numeric impute+scale, one-hot for low cardinality, target encoding for high."""
    numeric = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    onehot = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=False),
    )
    target = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        _target_encoder(seed),
        StandardScaler(),
    )
    return ColumnTransformer(
        [
            ("num", numeric, config.NUMERIC_FEATURES),
            ("ohe", onehot, config.ONEHOT_FEATURES),
            ("te", target, config.TARGET_ENCODED_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def _xgb(seed: int, scale_pos_weight: float = 1.0):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=N_JOBS,
        random_state=seed,
        scale_pos_weight=scale_pos_weight,
    )


def make_model(name: str, seed: int = config.SEED) -> Pipeline:
    """Build one named model pipeline.

    Names: ``majority``, ``logreg``, ``random_forest``, ``hist_gb``, ``xgboost``.
    """
    if name == "majority":
        clf = DummyClassifier(strategy="prior")
    elif name == "logreg":
        clf = LogisticRegression(max_iter=2000, C=1.0)
    elif name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=200, min_samples_leaf=20, max_features="sqrt", max_samples=0.5,
            n_jobs=N_JOBS, random_state=seed,
        )
    elif name == "hist_gb":
        clf = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=30, random_state=seed,
        )
    elif name == "xgboost":
        clf = _xgb(seed)
    else:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
    return Pipeline([("prep", build_preprocessor(seed)), ("clf", clf)])


MODEL_NAMES = ["majority", "logreg", "random_forest", "hist_gb", "xgboost"]
BASELINES = ["majority", "logreg"]
