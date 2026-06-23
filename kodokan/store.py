"""dol-backed stores for the pipeline's artifacts (the analysis SSOT).

Two dict-like stores keyed by ``video_id``:

- :func:`pose_store` — ``video_id -> PoseSequence``, each persisted as a tidy/long
  **Parquet** file (columns ``fidx, frame, t_sec, person, keypoint, x, y, conf``)
  with sequence metadata (fps, dims, backend, source_url, frame indices) carried in
  the Parquet schema metadata. Tidy/long Parquet is the analysis SSOT — columnar,
  compressed, and pandas/polars/DuckDB-native (research §6).
- :func:`segments_store` — ``video_id -> list[demo dicts]`` as JSON.

:func:`load_all_tidy` concatenates every clip's table (adding a ``video_id`` column)
into one DataFrame for cross-clip analysis.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from kodokan import config
from kodokan.pose import COCO17_KEYPOINTS, PoseSequence

_META_KEY = b"kodokan_meta"
_TIDY_COLUMNS = ("fidx", "frame", "t_sec", "person", "keypoint", "x", "y", "conf")


def sequence_to_tidy_df(seq: PoseSequence, *, video_id: str | None = None, drop_missing: bool = True):
    """Convert a :class:`PoseSequence` to a tidy/long DataFrame.

    One row per (frame, person, keypoint). ``fidx`` is the 0-based analyzed-frame
    position; ``frame`` is the source frame index. A ``video_id`` column is added
    when given (used by :func:`load_all_tidy`).
    """
    import pandas as pd

    kp = seq.keypoints  # (F, P, K, 3)
    F, P, K, _ = kp.shape
    fidx = np.repeat(np.arange(F), P * K)
    person = np.tile(np.repeat(np.arange(P), K), F)
    kpi = np.tile(np.arange(K), F * P)
    flat = kp.reshape(F * P * K, 3)
    frame = seq.frame_indices[fidx]
    df = pd.DataFrame(
        {
            "fidx": fidx,
            "frame": frame,
            "t_sec": frame / float(seq.fps),
            "person": person,
            "keypoint": kpi,
            "x": flat[:, 0],
            "y": flat[:, 1],
            "conf": flat[:, 2],
        }
    )
    if drop_missing:
        df = df[np.isfinite(df["x"].to_numpy())].reset_index(drop=True)
    if video_id is not None:
        df.insert(0, "video_id", video_id)
    return df


def _sequence_meta(seq: PoseSequence) -> dict:
    return {
        "fps": seq.fps,
        "width": seq.width,
        "height": seq.height,
        "backend": seq.backend,
        "video_path": seq.video_path,
        "source_url": seq.source_url or "",
        "n_persons": seq.n_persons,
        "frame_indices": [int(i) for i in seq.frame_indices],
    }


def sequence_to_parquet_bytes(seq: PoseSequence) -> bytes:
    """Serialize a :class:`PoseSequence` to Parquet bytes (tidy + schema metadata)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    df = sequence_to_tidy_df(seq, drop_missing=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    md = dict(table.schema.metadata or {})
    md[_META_KEY] = json.dumps(_sequence_meta(seq)).encode()
    table = table.replace_schema_metadata(md)
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd")
    return buf.getvalue()


def parquet_bytes_to_sequence(data: bytes) -> PoseSequence:
    """Reconstruct a :class:`PoseSequence` from Parquet bytes written above."""
    import pyarrow.parquet as pq

    table = pq.read_table(io.BytesIO(data))
    meta = json.loads(table.schema.metadata[_META_KEY])
    frame_indices = np.asarray(meta["frame_indices"], dtype=int)
    F, P, K = len(frame_indices), int(meta["n_persons"]), len(COCO17_KEYPOINTS)
    out = np.full((F, P, K, 3), np.nan, dtype=np.float32)
    df = table.to_pandas()
    fi = df["fidx"].to_numpy()
    pr = df["person"].to_numpy()
    kpi = df["keypoint"].to_numpy()
    out[fi, pr, kpi, 0] = df["x"].to_numpy()
    out[fi, pr, kpi, 1] = df["y"].to_numpy()
    out[fi, pr, kpi, 2] = df["conf"].to_numpy()
    return PoseSequence(
        keypoints=out,
        frame_indices=frame_indices,
        fps=float(meta["fps"]),
        width=int(meta["width"]),
        height=int(meta["height"]),
        backend=str(meta["backend"]),
        video_path=str(meta["video_path"]),
        source_url=meta["source_url"] or None,
    )


def pose_store(directory: str | Path | None = None):
    """A dict-like ``{video_id: PoseSequence}`` store backed by per-clip Parquet."""
    from dol import Files, filt_iter, wrap_kvs

    directory = str(directory or config.pose_dir())
    Path(directory).mkdir(parents=True, exist_ok=True)
    backend = filt_iter(Files(directory), filt=lambda k: k.endswith(".parquet"))
    return wrap_kvs(
        backend,
        key_of_id=lambda _id: _id[: -len(".parquet")],
        id_of_key=lambda k: f"{k}.parquet",
        obj_of_data=parquet_bytes_to_sequence,
        data_of_obj=sequence_to_parquet_bytes,
    )


def segments_store(directory: str | Path | None = None):
    """A dict-like ``{video_id: <json>}`` store for demo segments (and metadata)."""
    from dol import JsonFiles, filt_iter, wrap_kvs

    directory = str(directory or (config.pose_dir() / "segments"))
    Path(directory).mkdir(parents=True, exist_ok=True)
    backend = filt_iter(JsonFiles(directory), filt=lambda k: k.endswith(".json"))
    return wrap_kvs(
        backend,
        key_of_id=lambda _id: _id[: -len(".json")],
        id_of_key=lambda k: f"{k}.json",
    )


def load_all_tidy(store=None):
    """Concatenate every clip's tidy table (with a ``video_id`` column) into one DataFrame."""
    import pandas as pd

    store = store or pose_store()
    frames = [sequence_to_tidy_df(store[k], video_id=k) for k in store]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
