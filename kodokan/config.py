"""Where kodokan keeps its data (downloaded clips + derived artifacts).

Large/churny artifacts (videos, per-frame keypoints, rendered overlays) live
*outside* the repo by default — under ``~/kodokan_data`` — so they don't bloat
the (Dropbox-synced) source tree. Override the root with the
``KODOKAN_DATA_DIR`` environment variable.

On the dev Mac ``~/kodokan_data`` is a symlink into an off-machine-backed data
area; code addresses it transparently through the symlink either way. None of
this data is precious — every subfolder is downloaded or pipeline-derived; see
``misc/docs/regenerate-data.md`` for how to rebuild it from scratch.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Env var overriding the data root (default ``~/kodokan_data``).
KODOKAN_DATA_ENV = "KODOKAN_DATA_DIR"

# ---------------------------------------------------------------------------
# Two-person coverage thresholds (single source of truth).
#
# A demo's ``two_person_frac`` is the fraction of its frames in which *both*
# judoka are tracked. Two distinct gates use it, on purpose, at different
# strictness — they were previously scattered as bare ``0.3``/``0.4`` literals
# across the pipeline and eval scripts (adversarial-review item), which made the
# discrepancy look like a bug. They are centralized (and their intent
# documented) here so every call site imports the same named constant.
# ---------------------------------------------------------------------------

#: Inclusion gate used when *segmenting* a clip: keep a detected demonstration
#: only if both judoka are present for at least this fraction of its frames.
#: Deliberately lenient — we would rather store a borderline demo (and filter it
#: later) than silently drop a real one. Used by the batch/segmentation step.
SEGMENT_MIN_TWO_PERSON_FRAC: float = 0.3

#: Quality gate used when *consuming* stored demos for recognition/scoring/
#: catalog building: only trust demos where both judoka are clearly visible.
#: Stricter than the segmentation gate on purpose. Used by the eval harnesses,
#: :func:`kodokan.flashcards.build_catalog`, and scoring.
EVAL_MIN_TWO_PERSON_FRAC: float = 0.4


def data_dir() -> Path:
    """Resolve the kodokan data root (``$KODOKAN_DATA_DIR`` or ``~/kodokan_data``)."""
    override = os.environ.get(KODOKAN_DATA_ENV)
    return Path(override).expanduser() if override else Path.home() / "kodokan_data"


def _subdir(name: str) -> Path:
    d = data_dir() / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def clips_dir() -> Path:
    """Directory holding downloaded video clips + their ``*.info.json`` sidecars."""
    return _subdir("clips")


def pose_dir() -> Path:
    """Directory holding extracted per-clip keypoint sequences (NPZ / Parquet)."""
    return _subdir("pose")


def viz_dir() -> Path:
    """Directory holding rendered overlays (MP4) and Rerun recordings (RRD)."""
    return _subdir("viz")


def models_dir() -> Path:
    """Directory for downloaded model weights (kept out of the repo)."""
    return _subdir("models")
