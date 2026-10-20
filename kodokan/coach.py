"""Targeted coaching: score a performed throw and say how to improve, in EN or FR (#4).

:func:`coach` compares one performance of a target technique with that technique's
reference (:func:`kodokan.score.build_reference`), scores it on several dimensions
(overall 0-100, per joint angle, per phase: kuzushi / tsukuri / kake), and turns the
worst dimensions into short advice in the chosen language. The advice is template-
based (:data:`ADVICE`, one table per language), so it needs no model and no network;
pass ``advice=`` to use other wording or add a language. :class:`LiveCoach` does the
same on a live pose stream: it detects each throw (:class:`~kodokan.recognizer.
ThrowDetector`) and coaches it against the chosen target.

>>> import numpy as np
>>> from kodokan.score import build_reference
>>> t = np.linspace(0, 1, 30)[:, None]
>>> demos = [np.hstack([np.sin(t * 3 + k * 0.01)] * 8) for k in range(4)]
>>> ref = build_reference(demos)
>>> off = demos[0].copy(); off[:, 6] += 0.6        # the left knee is way off
>>> c = coach(off, ref, technique="seoi-nage", lang="fr")
>>> c.worst_joint, c.lang
('l_knee', 'fr')
>>> any("genou gauche" in tip for tip in c.tips)
True
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kodokan.compare import ANGLE_NAMES
from kodokan.pose import PoseSequence
from kodokan.recognizer import ThrowDetector
from kodokan.score import demo_features, feedback, score
from kodokan.segment import Segment

#: ``(pose_seq, start_s, end_s) -> per-frame joint angles`` of one throw window.
AngleFeaturizer = Callable[[PoseSequence, float, float], np.ndarray]

#: The three phases of a throw, in order (``feedback``'s ``per_phase_deg``).
PHASES = ("kuzushi", "tsukuri", "kake")

#: Side words per language: (masculine, feminine) for ``{side}`` / ``{side_f}``.
_SIDES = {
    "en": {"l": ("left", "left"), "r": ("right", "right")},
    "fr": {"l": ("gauche", "gauche"), "r": ("droit", "droite")},
}

#: Advice templates per language: one per joint kind (``{side}`` or ``{side_f}``,
#: ``{deg}``), one per phase, ``headline`` and ``close`` (nothing worth a tip). A table
#: for another language may carry its own side words as ``"sides"``:
#: ``{"l": (masculine, feminine), "r": (...)}``. One per
#: phase, plus the headline. Keys must cover every joint kind and every phase.
ADVICE: dict[str, dict[str, str]] = {
    "en": {
        "headline": "{technique}: {score}/100.",
        "elbow": "Your {side} elbow is {deg:.0f}° off the reference: keep that arm's "
        "pull firm and close to your body.",
        "shoulder": "Your {side} shoulder is {deg:.0f}° off: lift and turn that arm "
        "with the reference timing to take your partner's balance.",
        "hip": "Your {side} hip is {deg:.0f}° off: bring your hips lower and closer "
        "to your partner.",
        "knee": "Your {side} knee is {deg:.0f}° off: bend your knees more as you "
        "enter, then drive up.",
        "kuzushi": "Most of the difference is in kuzushi: break your partner's "
        "balance before you turn in.",
        "tsukuri": "Most of the difference is in tsukuri: step in faster and closer.",
        "kake": "Most of the difference is in kake: commit through the finish.",
        "close": "Close to the reference: keep it up.",
    },
    "fr": {
        "headline": "{technique} : {score}/100.",
        "elbow": "Votre coude {side} s'écarte de {deg:.0f}° de la référence : gardez "
        "la traction de ce bras ferme et près du corps.",
        "shoulder": "Votre épaule {side_f} s'écarte de {deg:.0f}° : levez et tournez ce "
        "bras au rythme de la référence pour déséquilibrer votre partenaire.",
        "hip": "Votre hanche {side_f} s'écarte de {deg:.0f}° : placez vos hanches plus "
        "bas et plus près de votre partenaire.",
        "knee": "Votre genou {side} s'écarte de {deg:.0f}° : fléchissez davantage les "
        "genoux en entrant, puis poussez vers le haut.",
        "kuzushi": "L'essentiel de l'écart est dans le kuzushi : déséquilibrez votre "
        "partenaire avant d'entrer.",
        "tsukuri": "L'essentiel de l'écart est dans le tsukuri : entrez plus vite et "
        "plus près.",
        "kake": "L'essentiel de l'écart est dans le kake : allez au bout de la "
        "projection.",
        "close": "Proche de la référence : continuez ainsi.",
    },
}


@dataclass(frozen=True)
class Coaching:
    """A scored performance and the advice drawn from its worst dimensions."""

    technique: str | None
    lang: str
    score: float | None  # 0-100, None without a genuine-demo baseline
    closer_than_pct: float | None
    per_joint_deg: dict[str, float] = field(default_factory=dict)
    per_phase_deg: dict[str, float] = field(default_factory=dict)
    worst_joint: str | None = None
    worst_phase: str | None = None
    tips: tuple[str, ...] = ()
    #: Why there is no score or tip: ``"too_short"`` when the throw window had fewer
    #: than two usable frames to compare.
    reason: str | None = None

    @property
    def text(self) -> str:
        """The advice as one paragraph."""
        return " ".join(self.tips)


def _sides(lang: str, table: Mapping[str, Any]) -> Mapping[str, tuple[str, str]]:
    sides = table.get("sides") or _SIDES.get(lang)
    if sides is None:
        raise ValueError(
            f"no joint-side words for {lang!r}: give its advice table a "
            '"sides" entry, {"l": (masculine, feminine), "r": (masculine, feminine)}'
        )
    return sides


def _check_lang(lang: str, advice: Mapping[str, Mapping[str, Any]]) -> None:
    """Fail at once, not at the first throw, on a language that cannot be advised in."""
    if lang not in advice:
        raise ValueError(f"no advice in {lang!r}; available: {sorted(advice)}")
    _sides(lang, advice[lang])


def _joint_tip(joint: str, deg: float, table: Mapping[str, Any], lang: str) -> str:
    side, kind = joint.split("_", 1)
    masculine, feminine = _sides(lang, table)[side]
    return table[kind].format(side=masculine, side_f=feminine, deg=deg)


def coach(
    query_features: np.ndarray,
    reference: dict,
    *,
    technique: str | None = None,
    lang: str = "en",
    n_tips: int = 2,
    min_deviation_deg: float = 10.0,
    advice: Mapping[str, Mapping[str, Any]] = ADVICE,
) -> Coaching:
    """Score ``query_features`` against ``reference`` and advise on the worst dimensions.

    :param query_features: per-frame joint angles of the performance
        (:func:`kodokan.score.demo_features`).
    :param reference: the target technique's :func:`~kodokan.score.build_reference`.
    :param lang: a key of ``advice`` (``"en"`` or ``"fr"`` built in).
    :param n_tips: joint tips to give, worst first, after the phase tip.
    :param min_deviation_deg: a phase or joint closer than this gets no tip; with
        none left, the advice says the performance is close to the reference.

    A window with fewer than two usable frames cannot be compared: it gets a
    :class:`Coaching` with no score and ``reason="too_short"`` rather than an error,
    so a live loop goes on.
    """
    _check_lang(lang, advice)
    if reference.get("reference") is None:
        raise ValueError("the reference has no demonstration to compare with")
    table = advice[lang]
    if len(query_features) < 2 or not np.isfinite(query_features).any():
        return Coaching(technique, lang, None, None, reason="too_short")
    graded = score(query_features, reference)
    if graded["score"] is not None and not np.isfinite(graded["score"]):
        graded = {**graded, "score": None}
    try:
        fb = feedback(query_features, reference, n_phases=len(PHASES))
    except ValueError:  # nothing aligned (an all-NaN comparison)
        return Coaching(technique, lang, None, None, reason="too_short")
    per_joint = {k: v for k, v in fb["per_angle_deg"].items() if np.isfinite(v)}
    per_phase = {p: v for p, v in zip(PHASES, fb["per_phase_deg"]) if np.isfinite(v)}
    worst_phase = max(per_phase, key=per_phase.get) if per_phase else None
    ranked = sorted(per_joint, key=lambda j: -per_joint[j])
    tips = []
    if technique is not None and graded["score"] is not None:
        tips.append(
            table["headline"].format(technique=technique, score=round(graded["score"]))
        )
    off = [j for j in ranked if per_joint[j] >= min_deviation_deg][:n_tips]
    corrections = [_joint_tip(j, per_joint[j], table, lang) for j in off]
    # A phase averages every joint, so it is named when some joint is off, or when the
    # whole phase is.
    if worst_phase is not None and (off or per_phase[worst_phase] >= min_deviation_deg):
        corrections.insert(0, table[worst_phase])
    tips += corrections or [table["close"]]
    return Coaching(
        technique=technique,
        lang=lang,
        score=graded["score"],
        closer_than_pct=graded["closer_than_pct"],
        per_joint_deg=per_joint,
        per_phase_deg=per_phase,
        worst_joint=ranked[0] if ranked else None,
        worst_phase=worst_phase,
        tips=tuple(tips),
    )


class LiveCoach:
    """Coach a target technique on a live pose stream, one throw at a time.

    :meth:`push` takes one frame (``(P, 17, 3)`` keypoints) and returns
    ``(segment, coaching)`` for each throw that closed with it. The target can be
    changed between throws (:attr:`target`, :attr:`reference`). Detector options
    (``buffer_s``, ``close_after_s``, ``segmenter``, ...) pass through.
    """

    def __init__(
        self,
        target: str,
        reference: dict,
        *,
        fps: float,
        lang: str = "en",
        n_tips: int = 2,
        advice: Mapping[str, Mapping[str, Any]] = ADVICE,
        featurize: AngleFeaturizer = demo_features,
        **detector_kwargs,
    ):
        _check_lang(lang, advice)
        if reference.get("reference") is None:
            raise ValueError("the reference has no demonstration to compare with")
        self.target, self.reference = target, reference
        self.lang, self.n_tips, self.advice, self.featurize = (
            lang,
            n_tips,
            advice,
            featurize,
        )
        self.detector = ThrowDetector(fps=fps, **detector_kwargs)

    def push(
        self, keypoints, frame_index: int | None = None
    ) -> list[tuple[Segment, Coaching]]:
        """Add one frame; return each throw it closed, coached against the target."""
        return [
            (
                seg,
                coach(
                    self.featurize(pose_seq, seg.start_s, seg.end_s),
                    self.reference,
                    technique=self.target,
                    lang=self.lang,
                    n_tips=self.n_tips,
                    advice=self.advice,
                ),
            )
            for pose_seq, seg in self.detector.push(keypoints, frame_index)
        ]


def _check_advice_tables() -> None:
    kinds = {name.split("_", 1)[1] for name in ANGLE_NAMES}
    for lang, table in ADVICE.items():
        missing = (kinds | set(PHASES) | {"headline", "close"}) - set(table)
        assert not missing, f"ADVICE[{lang!r}] lacks {sorted(missing)}"


_check_advice_tables()
