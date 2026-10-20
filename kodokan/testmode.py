"""Technique test mode: perform the technique the system names, get scored (#5).

:class:`TechniqueTest` draws a technique at random from a focus pool (all the
recognizer knows by default), names it, and grades the next performance on two
things: how confidently the recognizer sees the *target* in it, and, when a reference
for the target is given, how close it is to that reference (:func:`kodokan.coach.coach`,
with its advice). :meth:`TechniqueTest.push` runs the same loop on a live pose stream.

>>> import numpy as np
>>> from kodokan.recognizer import TechniqueRecognizer
>>> X = np.vstack([np.full((3, 4), c) + np.arange(3)[:, None] * 0.1 for c in (0.0, 3.0)])
>>> rec = TechniqueRecognizer(method="centroid").fit(X, ["o-goshi"] * 3 + ["uchi-mata"] * 3)
>>> test = TechniqueTest(rec, pool=["uchi-mata"], seed=1)
>>> test.next_prompt()
'uchi-mata'
>>> result = test.grade(np.full(4, 3.1))
>>> result.passed, result.score > 50
(True, True)
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from kodokan.coach import ADVICE, AngleFeaturizer, Coaching, coach
from kodokan.pose import PoseSequence
from kodokan.recognizer import (
    Featurizer,
    Recognition,
    TechniqueRecognizer,
    ThrowDetector,
    pooled_demo_descriptor,
)
from kodokan.score import demo_features
from kodokan.segment import Segment


@dataclass(frozen=True)
class TestResult:
    """The grade of one performance of the prompted technique."""

    __test__ = False  # not a pytest test class, despite the name

    target: str
    recognition: Recognition
    target_confidence: float  # the recognizer's score for the target, 0-1
    coaching: Coaching | None  # None without a reference for the target
    score: float  # 0-100: confidence and closeness blended by ``confidence_weight``

    @property
    def passed(self) -> bool:
        """Whether the recognizer's best guess is the target."""
        best = self.recognition.best
        return best is not None and best.technique == self.target


class TechniqueTest:
    """Prompt a random technique from a focus pool; grade the performance.

    :param recognizer: a fitted :class:`~kodokan.recognizer.TechniqueRecognizer`.
    :param pool: the techniques to draw from (default: all the recognizer knows).
    :param references: ``{technique: build_reference(...)}``; techniques with one are
        also graded on closeness, with advice.
    :param confidence_weight: share of the score from the target's confidence (the
        rest from closeness to the reference, when there is one).
    :param seed: or ``rng``, for a reproducible sequence of prompts.
    """

    def __init__(
        self,
        recognizer: TechniqueRecognizer,
        *,
        pool: Iterable[str] | None = None,
        references: Mapping[str, dict] | None = None,
        lang: str = "en",
        confidence_weight: float = 0.5,
        seed: int | None = None,
        rng: random.Random | None = None,
        advice: Mapping[str, Mapping[str, str]] = ADVICE,
    ):
        if recognizer.classes_ is None:
            raise ValueError("fit the recognizer before testing with it")
        known = [str(c) for c in recognizer.classes_]
        if len(known) < 2:
            raise ValueError(
                "a test needs a recognizer that knows at least 2 techniques"
            )
        self.pool = list(dict.fromkeys(pool)) if pool is not None else known
        unknown = sorted(set(self.pool) - set(known))
        if not self.pool or unknown:
            raise ValueError(
                f"the pool must be non-empty techniques the recognizer knows; "
                f"unknown: {unknown}"
                if unknown
                else "the pool is empty"
            )
        if not 0 <= confidence_weight <= 1:
            raise ValueError(
                f"confidence_weight must be in [0, 1], got {confidence_weight}"
            )
        empty = sorted(
            t for t, r in (references or {}).items() if r.get("reference") is None
        )
        if empty:
            raise ValueError(
                f"references with no demonstration to compare with: {empty}"
            )
        self.recognizer, self.references = recognizer, dict(references or {})
        self.lang, self.advice, self.confidence_weight = lang, advice, confidence_weight
        self.rng = rng or random.Random(seed)
        self.target: str | None = None

    def next_prompt(self) -> str:
        """Draw the next technique to perform (never the same one twice in a row)."""
        choices = [t for t in self.pool if t != self.target] or self.pool
        self.target = self.rng.choice(choices)
        return self.target

    def grade(self, descriptor, angle_features: np.ndarray | None = None) -> TestResult:
        """Grade one performance: its pooled ``descriptor`` and, for closeness, its
        per-frame ``angle_features`` (:func:`kodokan.score.demo_features`)."""
        if self.target is None:
            raise RuntimeError("call next_prompt() before grading a performance")
        recognition = self.recognizer.recognize(
            descriptor, k=len(self.recognizer.classes_)
        )
        confidence = recognition.score_of(self.target)
        reference = self.references.get(self.target)
        coaching = None
        if reference is not None and angle_features is not None:
            coaching = coach(
                angle_features,
                reference,
                technique=self.target,
                lang=self.lang,
                advice=self.advice,
            )
        score = 100.0 * confidence
        if coaching is not None and coaching.score is not None:
            w = self.confidence_weight
            score = w * score + (1 - w) * coaching.score
        return TestResult(
            self.target, recognition, confidence, coaching, round(score, 1)
        )

    def grade_clip(
        self,
        pose_seq: PoseSequence,
        start_s: float,
        end_s: float,
        *,
        featurize: Featurizer = pooled_demo_descriptor,
        angle_featurize: AngleFeaturizer = demo_features,
    ) -> TestResult:
        """Grade the performance in ``pose_seq`` between ``start_s`` and ``end_s``."""
        angles = (
            angle_featurize(pose_seq, start_s, end_s)
            if self.target in self.references
            else None
        )
        return self.grade(featurize(pose_seq, start_s, end_s), angles)


class LiveTechniqueTest:
    """The test loop on a live stream: prompt, detect the throw, grade, prompt again.

    :meth:`push` takes one frame and returns ``(segment, result)`` for each throw that
    closed with it; after each graded throw the next technique is drawn
    (:attr:`prompt`). Detector options pass through.

    A throw is graded once it has *closed* (``close_after_s`` after it ends), so the
    performer should wait for the next prompt before the next throw; any motion the
    detector takes for a throw also uses up a prompt.
    """

    def __init__(
        self,
        test: TechniqueTest,
        *,
        fps: float,
        featurize: Featurizer = pooled_demo_descriptor,
        angle_featurize: AngleFeaturizer = demo_features,
        on_prompt: Callable[[str], None] | None = None,
        **detector_kwargs,
    ):
        self.test, self.featurize, self.angle_featurize = (
            test,
            featurize,
            angle_featurize,
        )
        self.on_prompt = on_prompt
        self.detector = ThrowDetector(fps=fps, **detector_kwargs)
        self._prompt()

    @property
    def prompt(self) -> str:
        return self.test.target

    def _prompt(self) -> None:
        target = self.test.next_prompt()
        if self.on_prompt is not None:
            self.on_prompt(target)

    def push(
        self, keypoints, frame_index: int | None = None
    ) -> list[tuple[Segment, TestResult]]:
        """Add one frame; grade each throw it closed and draw the next prompt."""
        graded = []
        for pose_seq, seg in self.detector.push(keypoints, frame_index):
            result = self.test.grade_clip(
                pose_seq,
                seg.start_s,
                seg.end_s,
                featurize=self.featurize,
                angle_featurize=self.angle_featurize,
            )
            graded.append((seg, result))
            self._prompt()
        return graded
