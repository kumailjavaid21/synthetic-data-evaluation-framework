import numpy as np

from mia_attacks import knn_distance_attack


def test_mia_knn_attack_sanity_close_vs_far():
    rng = np.random.RandomState(0)
    members = rng.rand(20, 3)
    # nonmembers far away
    nonmembers = rng.rand(20, 3) + 5.0
    synth = members.copy()  # synth identical to members

    res = knn_distance_attack(members, nonmembers, synth, k=1)
    assert res["mia_auc_star"] > 0.6
    assert res["mia_advantage_abs"] >= 0.1


def test_mia_knn_attack_random_noise():
    rng = np.random.RandomState(1)
    members = rng.rand(20, 2)
    nonmembers = rng.rand(20, 2)
    synth = rng.rand(40, 2)  # unrelated noise
    res = knn_distance_attack(members, nonmembers, synth, k=1)
    assert 0.0 <= res["mia_auc_star"] <= 1.0
    assert 0.0 <= res["mia_advantage_abs"] <= 0.5
