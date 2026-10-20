"""The application recognizer: top-k with scores, offline segments, live buffer (#3)."""

import numpy as np
import pytest

from kodokan.recognizer import (
    LiveRecognizer,
    Recognition,
    TechniqueRecognizer,
    recognize_segments,
)
from kodokan.segment import Segment

TECHNIQUES = ["seoi-nage", "o-goshi", "uchi-mata"]


def _fitted(method="centroid"):
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(c, 0.3, size=(6, 5)) for c in (0.0, 3.0, 6.0)])
    y = np.repeat(TECHNIQUES, 6)
    return TechniqueRecognizer(method=method).fit(X, y)


@pytest.mark.parametrize("method", ["centroid", "lda"])
def test_top_k_is_ranked_and_every_score_is_exposed(method):
    rec = _fitted(method)
    r = rec.recognize(np.full(5, 6.1), k=2)
    assert isinstance(r, Recognition) and r.best.technique == "uchi-mata"
    assert [g.score for g in r.guesses] == sorted((g.score for g in r.guesses), reverse=True)
    assert set(r.scores) == set(TECHNIQUES) and sum(r.scores.values()) == pytest.approx(1)
    assert r.score_of("uchi-mata") == r.best.score and r.score_of("tomoe-nage") == 0.0


def test_temperature_spreads_the_scores():
    sharp = TechniqueRecognizer(method="centroid", temperature=0.5)
    soft = TechniqueRecognizer(method="centroid", temperature=5.0)
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(c, 0.3, size=(4, 3)) for c in (0.0, 3.0)])
    y = np.repeat(TECHNIQUES[:2], 4)
    x = np.full(3, 0.5)
    assert sharp.fit(X, y).recognize(x).best.score > soft.fit(X, y).recognize(x).best.score


def test_bad_use_is_refused():
    with pytest.raises(ValueError, match="method"):
        TechniqueRecognizer(method="svm")
    with pytest.raises(ValueError, match="temperature"):
        TechniqueRecognizer(temperature=0)
    with pytest.raises(RuntimeError, match="fit"):
        TechniqueRecognizer().recognize(np.zeros(3))


def _seg(i, a, b, fps=30):
    return Segment(index=i, start_s=a / fps, end_s=b / fps, start_frame=a, end_frame=b, peak_activity=1.0)


def test_offline_segments_are_each_recognized():
    rec = _fitted()
    pose_seq = object()  # the stub segmenter and featurizer never look inside
    segs = [_seg(0, 0, 30), _seg(1, 60, 90)]
    out = recognize_segments(
        pose_seq,
        rec,
        k=1,
        segmenter=lambda ps: segs,
        featurize=lambda ps, a, b: np.full(5, 0.0 if a < 1 else 3.0),
    )
    assert [r.best.technique for _, r in out] == ["seoi-nage", "o-goshi"]


def _frames(fps=30, bursts=((60, 120), (200, 260)), n=340):
    """One person standing still, moving fast during each burst."""
    rng = np.random.default_rng(2)
    base = rng.uniform(100, 300, size=(17, 2))
    for f in range(n):
        moving = any(a <= f < b for a, b in bursts)
        offset = 40 * np.sin(f / 2.0) if moving else 0.0
        kp = np.concatenate([base + offset + rng.normal(0, 0.2, (17, 2)), np.ones((17, 1))], axis=1)
        yield f, kp[None]


def test_live_recognizes_each_throw_once_after_it_closes():
    rec = _fitted()
    which = iter([0.0, 6.0])
    live = LiveRecognizer(
        rec, fps=30, k=2, close_after_s=0.5, featurize=lambda ps, a, b: np.full(5, next(which))
    )
    found = []
    for f, kp in _frames():
        for seg, r in live.push(kp, f):
            found.append((f, seg, r.best.technique))
    assert [t for _, _, t in found] == ["seoi-nage", "uchi-mata"]
    first_close, first_seg, _ = found[0]
    assert first_close >= first_seg.end_frame and first_seg.start_frame >= 50


def test_live_buffer_is_bounded_and_frames_are_checked():
    live = LiveRecognizer(_fitted(), fps=10, buffer_s=1.0)
    for f, kp in _frames(n=30):
        live.push(kp, f)
    assert live.buffer().n_frames == 10
    with pytest.raises(ValueError, match="17, 3"):
        live.push(np.zeros((17, 3)))


def test_a_constant_feature_does_not_flatten_the_scores():
    """Review of #56: a near-zero std must not blow a tiny query deviation up."""
    rng = np.random.default_rng(3)
    X = np.vstack([rng.normal(c, 0.3, size=(5, 3)) for c in (0.0, 4.0)])
    X[:, 2] = 1.0  # constant across every demo
    rec = TechniqueRecognizer(method="centroid").fit(X, np.repeat(TECHNIQUES[:2], 5))
    x = np.array([4.0, 4.0, 1.0 + 1e-6])
    assert rec.recognize(x).best.score > 0.9


def test_one_demo_per_technique_still_gives_graded_scores():
    X = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]])
    rec = TechniqueRecognizer(method="centroid").fit(X, TECHNIQUES)
    assert 0.4 < rec.recognize(np.array([1.0, 0.0])).best.score < 0.95


def test_a_descriptor_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="shape"):
        _fitted().recognize(np.zeros(1))


def test_live_idle_jitter_is_not_a_throw():
    """Review of #56: the default segmenter's quantile thresholds need an absolute floor."""
    rng = np.random.default_rng(4)
    base = rng.uniform(100, 300, size=(17, 2))
    live = LiveRecognizer(_fitted(), fps=30, featurize=lambda ps, a, b: np.zeros(5))
    found = []
    for f in range(900):
        kp = np.concatenate([base + rng.normal(0, 0.5, (17, 2)), np.ones((17, 1))], axis=1)
        found += live.push(kp[None], f)
    assert found == []
