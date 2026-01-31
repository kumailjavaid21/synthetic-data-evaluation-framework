import numpy as np
from sklearn.metrics import roc_auc_score

from debug_mia_evaluation import debug_mia_evaluation


def test_auc_flip_property():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    auc = roc_auc_score(y, scores)
    auc_flip = roc_auc_score(1 - y, scores)
    assert abs((auc + auc_flip) - 1.0) < 1e-6


def test_random_labels_auc_near_half():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, size=100)
    scores = rng.rand(100)
    auc = roc_auc_score(y, scores)
    assert abs(auc - 0.5) < 0.2  # loose sanity bound


def test_mia_with_members_nonmembers_returns_sanity():
    rng = np.random.RandomState(2)
    members = rng.rand(50, 4)
    nonmembers = rng.rand(50, 4)
    synth = rng.rand(50, 4)
    res = debug_mia_evaluation(
        real_data=members,
        synth_data=synth,
        member_data=members,
        nonmember_data=nonmembers,
        verbose=False,
    )
    assert 0.0 <= res["sanity_split_auc"] <= 1.0
    assert "mia_auc_raw" in res
