# kodokan.store

dol-backed stores for the pipeline’s artifacts (the analysis SSOT).

Two dict-like stores keyed by `video_id`:

- [`pose_store()`](#kodokan.store.pose_store) — `video_id -> PoseSequence`, each persisted as a tidy/long
  **Parquet** file (columns `fidx, frame, t_sec, person, keypoint, x, y, conf`)
  with sequence metadata (fps, dims, backend, source_url, frame indices) carried in
  the Parquet schema metadata. Tidy/long Parquet is the analysis SSOT — columnar,
  compressed, and pandas/polars/DuckDB-native (research §6).
- [`segments_store()`](#kodokan.store.segments_store) — `video_id -> list[demo dicts]` as JSON.

[`load_all_tidy()`](#kodokan.store.load_all_tidy) concatenates every clip’s table (adding a `video_id` column)
into one DataFrame for cross-clip analysis.

### Functions

| [`check_sequence_integrity`](#kodokan.store.check_sequence_integrity)(seq)                      | Integrity report for a single `PoseSequence` (via its tidy table).                                               |
|-----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| [`check_tidy_integrity`](#kodokan.store.check_tidy_integrity)(df, \*[, width, height, ...]) | Row-level integrity checks on a tidy pose DataFrame.                                                             |
| [`load_all_tidy`](#kodokan.store.load_all_tidy)([store])                             | Concatenate every clip's tidy table (with a `video_id` column) into one DataFrame.                               |
| [`parquet_bytes_to_sequence`](#kodokan.store.parquet_bytes_to_sequence)(data)                    | Reconstruct a `PoseSequence` from Parquet bytes written above.                                                   |
| [`pose_store`](#kodokan.store.pose_store)([directory])                            | A dict-like `{video_id: PoseSequence}` store backed by per-clip Parquet.                                         |
| [`segments_store`](#kodokan.store.segments_store)([directory])                        | A dict-like `{video_id: <json>}` store for demo segments (and metadata).                                         |
| [`sequence_to_parquet_bytes`](#kodokan.store.sequence_to_parquet_bytes)(seq)                     | Serialize a `PoseSequence` to Parquet bytes (tidy + schema metadata).                                            |
| [`sequence_to_tidy_df`](#kodokan.store.sequence_to_tidy_df)(seq, \*[, video_id, ...])      | Convert a `PoseSequence` to a tidy/long DataFrame.                                                               |
| [`store_integrity_report`](#kodokan.store.store_integrity_report)([store])                    | Run [`check_sequence_integrity()`](#kodokan.store.check_sequence_integrity) over every clip in a pose store. |

### kodokan.store.check_sequence_integrity(seq)

Integrity report for a single `PoseSequence` (via its tidy table).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### kodokan.store.check_tidy_integrity(df, , width=None, height=None, oob_margin_frac=0.05)

Row-level integrity checks on a tidy pose DataFrame.

Returns counts of anomalies (all should be `0` for a clean store):

- `n_duplicate_rows`: rows sharing a `(fidx, person, keypoint)` key (each
  such triple must be unique per clip).
- `n_oob_xy`: keypoints falling outside the frame by more than
  `oob_margin_frac` of its size (only checked when `width`/`height` are
  given; a small margin is tolerated since limbs can extend just past frame).
- `n_bad_conf`: confidences outside `[0, 1]`.
- `n_nan_xy`: non-finite coordinates (occluded rows should have been dropped).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### kodokan.store.load_all_tidy(store=None)

Concatenate every clip’s tidy table (with a `video_id` column) into one DataFrame.

### kodokan.store.parquet_bytes_to_sequence(data)

Reconstruct a `PoseSequence` from Parquet bytes written above.

* **Return type:**
  [`PoseSequence`](kodokan.pose.html.md#kodokan.pose.PoseSequence)

### kodokan.store.pose_store(directory=None)

A dict-like `{video_id: PoseSequence}` store backed by per-clip Parquet.

### kodokan.store.segments_store(directory=None)

A dict-like `{video_id: <json>}` store for demo segments (and metadata).

### kodokan.store.sequence_to_parquet_bytes(seq)

Serialize a `PoseSequence` to Parquet bytes (tidy + schema metadata).

* **Return type:**
  [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)

### kodokan.store.sequence_to_tidy_df(seq, , video_id=None, drop_missing=True)

Convert a `PoseSequence` to a tidy/long DataFrame.

One row per (frame, person, keypoint). `fidx` is the 0-based analyzed-frame
position; `frame` is the source frame index. A `video_id` column is added
when given (used by [`load_all_tidy()`](#kodokan.store.load_all_tidy)).

### kodokan.store.store_integrity_report(store=None)

Run [`check_sequence_integrity()`](#kodokan.store.check_sequence_integrity) over every clip in a pose store.

Returns `{video_id: report}`; a clip that fails to load records
`{"error": ...}` rather than aborting the whole sweep.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
