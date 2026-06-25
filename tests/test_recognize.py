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


def test_loo_group_predict_excludes_group():
    """Leave-one-CLIP-out must train on OTHER groups; separable 2-source data classifies."""
    from kodokan.recognize import classification_metrics, loo_pooled_predict

    rng = np.random.RandomState(0)
    # 2 techniques, 2 sources (groups) each, a few demos per (technique, source)
    X = np.vstack([
        rng.rand(4, 12), rng.rand(4, 12),            # technique 0, sources A,B
        rng.rand(4, 12) + 6, rng.rand(4, 12) + 6,    # technique 1, sources C,D
    ])
    y = np.array([0] * 8 + [1] * 8)
    groups = np.array(["A"] * 4 + ["B"] * 4 + ["C"] * 4 + ["D"] * 4)
    preds = loo_pooled_predict(X, y, method="centroid", groups=groups)
    assert classification_metrics(y, preds)["top1"] > 0.9
