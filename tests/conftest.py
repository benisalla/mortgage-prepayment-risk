import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import pytest  # noqa: E402

from mbs_risk import config, load_loans, prepare  # noqa: E402


@pytest.fixture(scope="session")
def raw_sample():
    return load_loans(config.SAMPLE_CSV)


@pytest.fixture(scope="session")
def sample(raw_sample):
    return prepare(raw_sample)
