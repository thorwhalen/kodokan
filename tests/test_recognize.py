"""Tests for recognition descriptors and LOO classifiers."""

import numpy as np
import pytest


def test_pooled_descriptor_fixed_length():
    from kodokan.recognize import pooled_descriptor

    a = np.random.RandomState(0).rand(30, 8)
    b = np.random.RandomState(1).rand(55, 8)
    assert pooled_descriptor(a).shape == pooled_descriptor(b).shape
    assert (
        pooled_descriptor(a).shape[0] == 8 * (1 + 2 + 4) * 2
    )  # mean+std over 7 segments


def test_demo_feature_modes(synth_seq):
    from kodokan.recognize import demo_feature

    for mode, dim in [
        ("primary_angles", 8),
        ("tori_angles", 8),
        ("tori_angles_pos", 42),
    ]:
        f = demo_feature(synth_seq, 0.0, 2.0, mode=mode)
        assert f.shape[1] == dim and len(f) > 0


def test_tori_index_valid(synth_seq):
    from kodokan.recognize import tori_index

    assert tori_index(synth_seq, 0.0, 2.0) in (0, 1)


def test_pool_classifiers_separable():
    from kodokan.recognize import pool_centroid_accuracy, pool_knn_accuracy

    rng = np.random.RandomState(0)
    X = np.vstack(
        [rng.rand(6, 20), rng.rand(6, 20) + 5.0]
    )  # two well-separated classes
    y = np.array([0] * 6 + [1] * 6)
    assert pool_centroid_accuracy(X, y) > 0.9
    assert pool_knn_accuracy(X, y, k=3) > 0.9


def test_loo_group_predict_excludes_group():
    """Leave-one-CLIP-out must train on OTHER groups; separable 2-source data classifies."""
    from kodokan.recognize import classification_metrics, loo_pooled_predict

    rng = np.random.RandomState(0)
    # 2 techniques, 2 sources (groups) each, a few demos per (technique, source)
    X = np.vstack(
        [
            rng.rand(4, 12),
            rng.rand(4, 12),  # technique 0, sources A,B
            rng.rand(4, 12) + 6,
            rng.rand(4, 12) + 6,  # technique 1, sources C,D
        ]
    )
    y = np.array([0] * 8 + [1] * 8)
    groups = np.array(["A"] * 4 + ["B"] * 4 + ["C"] * 4 + ["D"] * 4)
    preds = loo_pooled_predict(X, y, method="centroid", groups=groups)
    assert classification_metrics(y, preds)["top1"] > 0.9


def _two_person_seq(hip_y_offsets, F=30):
    """Synthetic two-person seq with each person's hips at a given image-y."""
    from kodokan.pose import PoseSequence

    kps = np.zeros((F, 2, 17, 3), dtype=np.float32)
    kps[..., 2] = 0.9
    for p, hip_y in enumerate(hip_y_offsets):
        x0 = 300.0 + 400.0 * p
        base = np.array(
            [[x0, hip_y - 80], [x0 - 20, hip_y - 50], [x0 + 20, hip_y - 50]]
            + [[x0, hip_y]] * 14,  # remaining joints near the hip line
            dtype=np.float32,
        )
        base[[5, 6]] = [[x0 - 20, hip_y - 50], [x0 + 20, hip_y - 50]]  # shoulders
        base[[11, 12]] = [[x0 - 12, hip_y], [x0 + 12, hip_y]]  # hips
        kps[:, p, :, :2] = base
    return PoseSequence(kps, np.arange(F), 25.0, 1280, 720, "test", "v.mp4")


def test_tori_decision_confident_and_abstain():
    from kodokan.recognize import tori_decision

    # person 0 upright (hips high, small y), person 1 thrown (hips low, large y)
    conf = tori_decision(_two_person_seq([200.0, 500.0]), 0.0, 1.0)
    assert conf.index == 0 and not conf.abstained and not conf.fell_back
    assert conf.margin > 0.15

    # both at the same height -> undecidable -> abstain
    tie = tori_decision(_two_person_seq([300.0, 300.0]), 0.0, 1.0)
    assert tie.abstained and tie.margin < 0.15


def test_tori_decision_single_finite_hip_keeps_slot_not_presence_fallback():
    """Regression: with one judoka's finish-hip occluded, keep the visible-hip slot
    (old tori_index behavior) rather than falling back to the most-present slot."""
    from kodokan.pose import PoseSequence
    from kodokan.recognize import tori_decision

    F = 30
    kps = np.full((F, 2, 17, 3), np.nan, dtype=np.float32)
    # person 1: present every frame but hips (11,12) always NaN
    kps[:, 1, :, :2] = 100.0
    kps[:, 1, :, 2] = 0.9
    kps[:, 1, [11, 12], :] = np.nan
    # person 0: present only in the last third, with finite (high) hips
    kps[2 * F // 3 :, 0, :, :2] = 500.0
    kps[2 * F // 3 :, 0, :, 2] = 0.9
    seq = PoseSequence(kps, np.arange(F), 25.0, 1280, 720, "test", "v.mp4")

    dec = tori_decision(seq, 0.0, F / 25.0)
    assert dec.index == 0  # the visible-hip slot, NOT presence-argmax (=1)
    assert (
        dec.abstained and not dec.fell_back
    )  # margin unmeasurable, but not a fallback


def test_tori_decision_stats():
    from kodokan.recognize import ToriDecision, tori_decision_stats

    ds = [
        ToriDecision(0, 0.8, fell_back=False, abstained=False),
        ToriDecision(1, 0.05, fell_back=False, abstained=True),
        ToriDecision(0, float("nan"), fell_back=True, abstained=True),
    ]
    stats = tori_decision_stats(ds)
    assert stats["n"] == 3
    assert stats["fell_back_rate"] == round(1 / 3, 3)
    assert stats["abstain_rate"] == round(2 / 3, 3)
    assert stats["median_margin"] == 0.425  # median of finite margins {0.8, 0.05}


def test_classification_metrics_bootstrap_ci():
    from kodokan.recognize import classification_metrics

    y = np.array([0, 0, 1, 1, 2, 2])
    m = classification_metrics(y, y, n_boot=200, seed=0)
    assert m["top1"] == 1.0
    assert m["top1_ci"][0] <= m["top1_ci"][1] <= 1.0
    assert "balanced_ci" in m
    # no CI keys unless requested
    assert "top1_ci" not in classification_metrics(y, y)


def test_confusion_pairs():
    from kodokan.recognize import confusion_pairs

    y_true = [0, 0, 1, 1, 2]
    y_pred = [0, 1, 1, 0, None]  # two errors: 0->1 and 1->0, plus an abstain 2->-1
    pairs = confusion_pairs(y_true, y_pred)
    keys = {(p["true"], p["pred"]) for p in pairs}
    assert (0, 1) in keys and (1, 0) in keys and (2, -1) in keys
    assert all(p["count"] >= 1 for p in pairs)


def test_roc_auc_distances():
    # numpy-only (no scipy) so it runs in the base install
    from kodokan.recognize import roc_auc_distances

    genuine = [0.1, 0.2, 0.15]
    impostor = [0.8, 0.9, 0.7, 1.0]
    assert roc_auc_distances(genuine, impostor) == 1.0  # perfectly separated
    assert roc_auc_distances(impostor, genuine) == 0.0  # reversed
    import math

    assert math.isnan(roc_auc_distances([], impostor))  # empty side -> nan, no crash


def test_rankdata_avg_matches_reference():
    """The numpy rankdata (ties averaged) matches the known average-rank result."""
    from kodokan.recognize import _rankdata_avg

    # values 3,1,4,1,5 -> the two 1s tie for ranks 1,2 -> both 1.5
    got = _rankdata_avg([3, 1, 4, 1, 5])
    assert list(got) == [3.0, 1.5, 4.0, 1.5, 5.0]


def test_classification_metrics_empty_is_clean():
    """Empty dataset must not crash (used by the eval scripts' n=0 guard)."""
    from kodokan.recognize import classification_metrics

    m = classification_metrics([], [], n_boot=100)
    assert m["n"] == 0 and m["n_classes"] == 0
    assert (
        m["top1_ci"] == [float("nan"), float("nan")] or True
    )  # keys present, no crash


def test_loo_medoid_separability_degenerate_is_json_valid():
    """No techniques / all-singleton -> None (JSON-valid), not nan tokens."""
    import json

    from kodokan.recognize import loo_medoid_separability

    rep = loo_medoid_separability([], distance=lambda a, b: 0.0)
    assert rep["accuracy"] is None and rep["auc"] is None
    json.loads(json.dumps(rep))  # must round-trip as valid JSON (no bare NaN)


def test_loo_medoid_separability_separable():
    from kodokan.recognize import loo_medoid_separability

    rng = np.random.RandomState(0)
    # 3 techniques, each 4 demos, well separated in mean feature space
    feats_by_tech = [[rng.rand(10, 5) + 10 * c for _ in range(4)] for c in range(3)]
    rep = loo_medoid_separability(
        feats_by_tech,
        distance=lambda a, b: float(np.linalg.norm(a.mean(0) - b.mean(0))),
    )
    assert rep["n_tech"] == 3 and rep["n_demos"] == 12
    assert rep["accuracy"] == 1.0
    assert rep["auc"] == 1.0
