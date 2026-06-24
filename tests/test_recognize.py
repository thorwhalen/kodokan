"""Tests for recognition descriptors and LOO classifiers."""

import numpy as np


def test_pooled_descriptor_fixed_length():
    from kodokan.recognize import pooled_descriptor

    a = np.random.RandomState(0).rand(30, 8)
    b = np.random.RandomState(1).rand(55, 8)
    assert pooled_descriptor(a).shape == pooled_descriptor(b).shape
    assert pooled_descriptor(a).shape[0] == 8 * (1 + 2 + 4) * 2  # mean+std over 7 segments


def test_demo_feature_modes(synth_seq):
    from kodokan.recognize import demo_feature

    for mode, dim in [("primary_angles", 8), ("tori_angles", 8), ("tori_angles_pos", 42)]:
        f = demo_feature(synth_seq, 0.0, 2.0, mode=mode)
        assert f.shape[1] == dim and len(f) > 0


def test_tori_index_valid(synth_seq):
    from kodokan.recognize import tori_index

    assert tori_index(synth_seq, 0.0, 2.0) in (0, 1)


def test_pool_classifiers_separable():
    from kodokan.recognize import pool_centroid_accuracy, pool_knn_accuracy

    rng = np.random.RandomState(0)
    X = np.vstack([rng.rand(6, 20), rng.rand(6, 20) + 5.0])  # two well-separated classes
    y = np.array([0] * 6 + [1] * 6)
    assert pool_centroid_accuracy(X, y) > 0.9
    assert pool_knn_accuracy(X, y, k=3) > 0.9
