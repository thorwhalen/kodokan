"""Technique recognition for applications: top-k guesses with inspectable scores (#3).

:mod:`kodokan.recognize` holds the research tools (features, LOO evaluation, metrics).
This module is the application facade over them:

- :class:`TechniqueRecognizer` — fit on pooled demo descriptors, then
  :meth:`~TechniqueRecognizer.recognize` returns a :class:`Recognition`: the top-k
  :class:`Guess` es *and* the score of every technique (the intermediate vector a UI
  can show).
- :func:`recognize_segments` — offline video: detect the throws in a whole pose
  sequence and recognize each.
- :class:`LiveRecognizer` — live stream: push pose frames into a rolling buffer; each
  throw is recognized once its segment has closed.

Scores are a softmax over negative distances to each technique's centroid, in the
train-standardized (and, when scikit-learn is installed, LDA-projected) descriptor
space, with distances in units of the typical within-technique spread (measured
leave-one-out, as a held-out demo would see it). They rank techniques and sum to 1;
they are not calibrated probabilities. The LDA projection is still fit on every
training demo, so with few demos per technique the spread stays optimistic.

>>> rng = np.random.default_rng(0)
>>> X = np.vstack([rng.normal(c, 0.3, size=(5, 4)) for c in (0.0, 3.0, 6.0)])
>>> y = np.repeat(["seoi-nage", "o-goshi", "uchi-mata"], 5)
>>> rec = TechniqueRecognizer(method="centroid").fit(X, y)
>>> r = rec.recognize(np.full(4, 3.1), k=2)
>>> r.guesses[0].technique, len(r.guesses), round(sum(r.scores.values()), 6)
('o-goshi', 2, 1.0)
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from kodokan.pose import PoseSequence
from kodokan.recognize import demo_feature, pooled_descriptor
from kodokan.segment import Segment, find_segments, pose_motion_energy

#: ``(pose_seq, start_s, end_s) -> fixed-length descriptor`` of one throw window.
Featurizer = Callable[[PoseSequence, float, float], np.ndarray]

DFLT_FEATURE_MODE = "tori_angles_pos"  # the LOO baseline's best mode (#7)
DFLT_LEVELS = (1, 2, 4)
#: Smoothed motion energy (keypoint speed, px per analysed frame) a throw must reach;
#: below it a "segment" is pose jitter, which quantile thresholds alone would accept.
DFLT_MIN_PEAK_ACTIVITY = 3.0
_MIN_SD = 1e-6  # a near-constant feature is centred, not divided by ~0


def pooled_demo_descriptor(
    pose_seq: PoseSequence,
    start_s: float,
    end_s: float,
    *,
    mode: str = DFLT_FEATURE_MODE,
    levels: Sequence[int] = DFLT_LEVELS,
) -> np.ndarray:
    """The default :data:`Featurizer`: per-frame demo features, temporal-pyramid pooled."""
    return pooled_descriptor(demo_feature(pose_seq, start_s, end_s, mode=mode), levels=levels)


@dataclass(frozen=True)
class Guess:
    """One candidate technique and its score (higher is more likely)."""

    technique: str
    score: float


@dataclass(frozen=True)
class Recognition:
    """The top-k guesses, best first, and every technique's score."""

    guesses: tuple[Guess, ...]
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def best(self) -> Guess | None:
        return self.guesses[0] if self.guesses else None

    def score_of(self, technique: str) -> float:
        """The score of ``technique`` (0 for a technique the recognizer never saw)."""
        return self.scores.get(technique, 0.0)


class TechniqueRecognizer:
    """Fit on pooled descriptors; recognize with top-k guesses and the full score vector.

    :param method: ``"lda"`` (LDA projection when scikit-learn is installed, else plain
        standardized space) or ``"centroid"`` (standardized space only).
    :param temperature: softmax temperature; higher spreads the scores out.
    """

    def __init__(self, *, method: str = "lda", temperature: float = 1.0):
        if method not in ("lda", "centroid"):
            raise ValueError(f"unknown method {method!r}; use 'lda' or 'centroid'")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.method = method
        self.temperature = temperature
        self.classes_: np.ndarray | None = None

    # -- fitting -------------------------------------------------------------------

    def fit(self, X, y) -> TechniqueRecognizer:
        """Fit on pooled descriptors ``X`` (one row per demo) and technique labels ``y``."""
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        if X.ndim != 2 or len(X) != len(y) or len(X) == 0:
            raise ValueError("X must be a non-empty (n_demos, n_features) array matching y")
        self.classes_ = np.unique(y)
        sd = X.std(0)
        self._mu, self._sd = X.mean(0), np.where(sd > _MIN_SD, sd, 1.0)
        Z = (X - self._mu) / self._sd
        self._lda = None
        if self.method == "lda" and len(self.classes_) > 1:
            try:
                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            except ImportError:  # optional: the standardized space is the fallback
                LinearDiscriminantAnalysis = None
            if LinearDiscriminantAnalysis is not None:
                try:
                    self._lda = LinearDiscriminantAnalysis().fit(Z, y)
                    Z = self._lda.transform(Z)
                except (ValueError, np.linalg.LinAlgError) as e:
                    warnings.warn(f"LDA failed ({e}); using the standardized space")
                    self._lda = None
        self._centroids = np.stack([Z[y == c].mean(0) for c in self.classes_])
        self._scale = self._held_out_spread(Z, y)
        return self

    def _held_out_spread(self, Z: np.ndarray, y: np.ndarray) -> float:
        """Median distance of a demo to its technique's centroid computed without it.

        With one demo per technique there is no within spread: half the median
        distance between centroids stands in for it.
        """
        spread = []
        for c, m in zip(self.classes_, self._centroids):
            rows = Z[y == c]
            n = len(rows)
            if n >= 2:
                others = (n * m - rows) / (n - 1)  # each row's centroid without it
                spread.append(np.linalg.norm(rows - others, axis=1))
        if spread and np.median(np.concatenate(spread)) > 0:
            return float(np.median(np.concatenate(spread)))
        if len(self._centroids) > 1:
            gaps = np.linalg.norm(self._centroids[:, None] - self._centroids[None], axis=-1)
            between = gaps[np.triu_indices(len(gaps), 1)]
            if np.median(between) > 0:
                return float(np.median(between)) / 2
        return 1.0

    def fit_demos(self, feature_seqs, labels, *, levels=DFLT_LEVELS) -> TechniqueRecognizer:
        """Fit on per-frame feature sequences (e.g. :func:`kodokan.recognize.demo_feature`)."""
        X = np.stack([pooled_descriptor(f, levels=levels) for f in feature_seqs])
        return self.fit(X, labels)

    # -- recognizing ---------------------------------------------------------------

    def _project(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape != self._mu.shape:
            raise ValueError(
                f"descriptor has shape {x.shape}, the recognizer was fit on {self._mu.shape}"
                " (an empty throw window pools to a 1-element descriptor)"
            )
        z = (x - self._mu) / self._sd
        return self._lda.transform(z[None])[0] if self._lda is not None else z

    def scores(self, x) -> dict[str, float]:
        """Every technique's score for one pooled descriptor (they sum to 1)."""
        if self.classes_ is None:
            raise RuntimeError("fit the recognizer before recognizing")
        d = np.linalg.norm(self._centroids - self._project(x), axis=1) / self._scale
        logits = -d / self.temperature
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return {str(c): float(s) for c, s in zip(self.classes_, p)}

    def recognize(self, x, *, k: int = 3) -> Recognition:
        """Top-``k`` guesses (best first) and the full score vector for one descriptor."""
        scores = self.scores(x)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[: max(k, 0)]
        return Recognition(tuple(Guess(t, s) for t, s in ranked), scores)

    def recognize_clip(
        self,
        pose_seq: PoseSequence,
        start_s: float,
        end_s: float,
        *,
        k: int = 3,
        featurize: Featurizer = pooled_demo_descriptor,
    ) -> Recognition:
        """Recognize the throw in ``pose_seq`` between ``start_s`` and ``end_s``."""
        return self.recognize(featurize(pose_seq, start_s, end_s), k=k)


def motion_segments(
    pose_seq: PoseSequence,
    *,
    min_peak_activity: float = DFLT_MIN_PEAK_ACTIVITY,
    **segment_kwargs,
) -> list[Segment]:
    """Throw windows by motion energy (:func:`kodokan.segment.find_segments`), keeping
    only those whose peak energy reaches ``min_peak_activity`` (an absolute floor:
    the hysteresis thresholds are quantiles, so an idle buffer still has "segments")."""
    energy = pose_motion_energy(pose_seq)
    found = find_segments(energy, pose_seq.frame_indices, pose_seq.fps, **segment_kwargs)
    return [s for s in found if s.peak_activity >= min_peak_activity]


def recognize_segments(
    pose_seq: PoseSequence,
    recognizer: TechniqueRecognizer,
    *,
    k: int = 3,
    featurize: Featurizer = pooled_demo_descriptor,
    segmenter: Callable[[PoseSequence], list[Segment]] = motion_segments,
) -> list[tuple[Segment, Recognition]]:
    """Offline video mode: every detected throw in ``pose_seq``, recognized."""
    return [
        (seg, recognizer.recognize_clip(pose_seq, seg.start_s, seg.end_s, k=k, featurize=featurize))
        for seg in segmenter(pose_seq)
    ]


class LiveRecognizer:
    """Live mode: a rolling pose buffer; each throw is recognized once its segment closes.

    Push one frame at a time (``(P, 17, 3)`` keypoints, as in :class:`PoseSequence`).
    :meth:`push` returns the throws that closed with that frame, each recognized once.
    A segment counts as closed when it ends at least ``close_after_s`` before the newest
    frame; the buffer keeps the last ``buffer_s`` seconds of pushed frames (at ``fps``
    frames a second: push every analysed frame, or pass the analysed rate as ``fps``).
    A segment that mostly overlaps a throw already recognized is that throw.
    """

    def __init__(
        self,
        recognizer: TechniqueRecognizer,
        *,
        fps: float,
        width: int = 0,
        height: int = 0,
        k: int = 3,
        buffer_s: float = 20.0,
        close_after_s: float = 1.0,
        featurize: Featurizer = pooled_demo_descriptor,
        segmenter: Callable[[PoseSequence], list[Segment]] = motion_segments,
    ):
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        self.recognizer, self.fps, self.k = recognizer, fps, k
        self.size = (width, height)
        self.max_frames = max(int(buffer_s * fps), 2)
        self.close_after_s = close_after_s
        self.featurize, self.segmenter = featurize, segmenter
        self._frames: list[np.ndarray] = []
        self._indices: list[int] = []
        self._emitted: list[tuple[float, float]] = []  # throws already recognized

    def buffer(self) -> PoseSequence:
        """The buffered frames as a :class:`PoseSequence` (people padded with NaN)."""
        n_people = max(f.shape[0] for f in self._frames)
        kp = np.full((len(self._frames), n_people, 17, 3), np.nan)
        for i, f in enumerate(self._frames):
            kp[i, : f.shape[0]] = f
        w, h = self.size
        return PoseSequence(
            keypoints=kp,
            frame_indices=np.asarray(self._indices),
            fps=self.fps,
            width=w,
            height=h,
            backend="live",
            video_path="",
        )

    def push(self, keypoints, frame_index: int | None = None) -> list[tuple[Segment, Recognition]]:
        """Add one frame; return the throws that closed with it, recognized."""
        kp = np.asarray(keypoints, dtype=float)
        if kp.ndim != 3 or kp.shape[1:] != (17, 3):
            raise ValueError(f"a frame is (persons, 17, 3) keypoints, got {kp.shape}")
        index = frame_index if frame_index is not None else (self._indices[-1] + 1 if self._indices else 0)
        self._frames.append(kp)
        self._indices.append(int(index))
        if len(self._frames) > self.max_frames:
            del self._frames[0], self._indices[0]
        if len(self._frames) < 2:
            return []
        pose_seq = self.buffer()
        newest_s = self._indices[-1] / self.fps
        oldest_s = self._indices[0] / self.fps
        self._emitted = [(s, e) for s, e in self._emitted if e >= oldest_s]
        closed = []
        for seg in self.segmenter(pose_seq):
            if seg.end_s > newest_s - self.close_after_s:
                continue  # still open: the throw may go on
            # Re-segmenting a sliding buffer can move a throw's edges a little: one
            # that mostly overlaps a throw already recognized is that throw.
            length = max(seg.end_s - seg.start_s, 1e-9)
            if any(
                (min(seg.end_s, e) - max(seg.start_s, s)) / length > 0.5
                for s, e in self._emitted
            ):
                continue
            self._emitted.append((seg.start_s, seg.end_s))
            rec = self.recognizer.recognize_clip(
                pose_seq, seg.start_s, seg.end_s, k=self.k, featurize=self.featurize
            )
            closed.append((seg, rec))
        return closed
