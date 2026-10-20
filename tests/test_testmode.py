"""Technique test mode: random prompt from a pool, grade, next prompt (#5)."""

import numpy as np
import pytest

from kodokan.recognizer import TechniqueRecognizer
from kodokan.testmode import LiveTechniqueTest, TechniqueTest

TECHNIQUES = ["seoi-nage", "o-goshi", "uchi-mata"]
CENTRES = dict(zip(TECHNIQUES, (0.0, 3.0, 6.0)))


def _recognizer():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(CENTRES[t], 0.3, size=(6, 5)) for t in TECHNIQUES])
    return TechniqueRecognizer(method="centroid").fit(X, np.repeat(TECHNIQUES, 6))


def test_prompts_come_from_the_pool_and_never_repeat_back_to_back():
    test = TechniqueTest(_recognizer(), pool=["seoi-nage", "uchi-mata"], seed=3)
    prompts = [test.next_prompt() for _ in range(20)]
    assert set(prompts) == {"seoi-nage", "uchi-mata"}
    assert all(a != b for a, b in zip(prompts, prompts[1:]))


def test_performing_the_target_scores_high_and_another_scores_low():
    test = TechniqueTest(_recognizer(), seed=0)
    target = test.next_prompt()
    good = test.grade(np.full(5, CENTRES[target]))
    other = next(t for t in TECHNIQUES if t != target)
    bad = test.grade(np.full(5, CENTRES[other]))
    assert good.passed and not bad.passed and good.score > 80 > bad.score
    assert good.coaching is None  # no reference given: confidence only


def test_bad_setups_are_refused():
    with pytest.raises(ValueError, match="unknown"):
        TechniqueTest(_recognizer(), pool=["tomoe-nage"])
    with pytest.raises(ValueError, match="fit"):
        TechniqueTest(TechniqueRecognizer())
    with pytest.raises(RuntimeError, match="next_prompt"):
        TechniqueTest(_recognizer()).grade(np.zeros(5))


def test_a_reference_adds_closeness_and_advice():
    pytest.importorskip("dtaidistance")
    from kodokan.score import build_reference

    t = np.linspace(0, 1, 30)[:, None]
    rng = np.random.default_rng(1)
    base = np.hstack([np.sin(t * 3)] * 8)
    ref = build_reference([base + rng.normal(0, 0.15, base.shape) for _ in range(6)])
    test = TechniqueTest(_recognizer(), pool=["o-goshi"], references={"o-goshi": ref})
    test.next_prompt()
    result = test.grade(np.full(5, 3.0), base)
    assert result.coaching is not None and result.coaching.tips
    w = test.confidence_weight
    expected = w * 100 * result.target_confidence + (1 - w) * result.coaching.score
    assert result.score == pytest.approx(expected, abs=0.1)


def test_live_loop_grades_each_throw_and_prompts_again():
    rec = _recognizer()
    prompts = []
    test = TechniqueTest(rec, seed=2)
    live = LiveTechniqueTest(
        test,
        fps=30,
        close_after_s=0.5,
        on_prompt=prompts.append,
        featurize=lambda ps, a, b: np.full(5, CENTRES[test.target]),  # always on target
    )
    rng = np.random.default_rng(6)
    base = rng.uniform(100, 300, size=(17, 2))
    results = []
    for f in range(340):
        moving = any(a <= f < b for a, b in ((60, 120), (200, 260)))
        offset = 40 * np.sin(f / 2.0) if moving else 0.0
        kp = np.concatenate(
            [base + offset + rng.normal(0, 0.2, (17, 2)), np.ones((17, 1))], axis=1
        )
        results += live.push(kp[None], f)
    assert len(results) == 2 and all(r.passed for _, r in results)
    assert len(prompts) == 3 and [r.target for _, r in results] == prompts[:2]
