"""Where kodokan keeps its data (downloaded clips + derived artifacts).

Large/churny artifacts (videos, per-frame keypoints, rendered overlays) live
*outside* the repo by default — under ``~/kodokan_data`` — so they don't bloat
the (Dropbox-synced) source tree. Override the root with the
``KODOKAN_DATA_DIR`` environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Env var overriding the data root (default ``~/kodokan_data``).
KODOKAN_DATA_ENV = "KODOKAN_DATA_DIR"


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
