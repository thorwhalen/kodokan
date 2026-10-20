"""Targeted coaching: multi-dimensional score and EN/FR advice (#4)."""

import numpy as np
import pytest

pytest.importorskip("dtaidistance")

from kodokan.coach import ADVICE, PHASES, coach  # noqa: E402
from kodokan.compare import ANGLE_NAMES  # noqa: E402
from kodokan.score import build_reference  # noqa: E402


def _reference():
    t = np.linspace(0, 1, 30)[:, None]
    rng = np.random.default_rng(0)
    base = np.hstack([np.sin(t * 3)] * 8)
    return build_reference([base + rng.normal(0, 0.15, base.shape) for _ in range(6)])


def _off(joint, *, phase=None, by=0.6):
    t = np.linspace(0, 1, 30)[:, None]
    q = np.hstack([np.sin(t * 3)] * 8)
    rows = slice(None) if phase is None else slice(phase * 10, phase * 10 + 10)
    q[rows, ANGLE_NAMES.index(joint)] += by
    return q


@pytest.mark.parametrize("lang, words", [("en", "right hip"), ("fr", "hanche droite")])
def test_the_worst_joint_gets_advice_in_the_chosen_language(lang, words):
    c = coach(_off("r_hip"), _reference(), technique="o-goshi", lang=lang)
    assert c.worst_joint == "r_hip" and any(words in tip for tip in c.tips)
    assert c.tips[0].startswith("o-goshi") and c.score is not None
    assert set(c.per_joint_deg) == set(ANGLE_NAMES) and set(c.per_phase_deg) == set(
        PHASES
    )


def test_the_worst_phase_is_named():
    c = coach(_off("l_knee", phase=2), _reference())
    assert c.worst_phase == "kake" and ADVICE["en"]["kake"] in c.tips


def test_a_close_performance_scores_higher_than_a_far_one():
    ref = _reference()
    near, far = (
        coach(_off("l_elbow", by=0.05), ref),
        coach(_off("l_elbow", by=0.9), ref),
    )
    assert near.score > far.score


def test_unknown_language_and_empty_reference_are_refused():
    with pytest.raises(ValueError, match="available"):
        coach(_off("l_knee"), _reference(), lang="de")
    with pytest.raises(ValueError, match="reference"):
        coach(_off("l_knee"), build_reference([]))


def test_live_coach_coaches_each_detected_throw():
    from kodokan.coach import LiveCoach

    rng = np.random.default_rng(5)
    base = rng.uniform(100, 300, size=(17, 2))
    live = LiveCoach(
        "seoi-nage",
        _reference(),
        fps=30,
        lang="fr",
        close_after_s=0.5,
        featurize=lambda ps, a, b: _off("l_knee"),
    )
    found = []
    for f in range(340):
        moving = any(a <= f < b for a, b in ((60, 120), (200, 260)))
        offset = 40 * np.sin(f / 2.0) if moving else 0.0
        kp = np.concatenate(
            [base + offset + rng.normal(0, 0.2, (17, 2)), np.ones((17, 1))], axis=1
        )
        found += live.push(kp[None], f)
    assert len(found) == 2
    assert all(c.technique == "seoi-nage" and c.lang == "fr" for _, c in found)
    assert any("genou gauche" in t for t in found[0][1].tips)
