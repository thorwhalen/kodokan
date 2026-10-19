# kodokan.config

Where kodokan keeps its data (downloaded clips + derived artifacts).

Large/churny artifacts (videos, per-frame keypoints, rendered overlays) live
*outside* the repo by default — under `~/kodokan_data` — so they don’t bloat
the (Dropbox-synced) source tree. Override the root with the
`KODOKAN_DATA_DIR` environment variable.

On the dev Mac `~/kodokan_data` is a symlink into an off-machine-backed data
area; code addresses it transparently through the symlink either way. None of
this data is precious — every subfolder is downloaded or pipeline-derived; see
`misc/docs/regenerate-data.md` for how to rebuild it from scratch.

### Module Attributes

| [`KODOKAN_DATA_ENV`](#kodokan.config.KODOKAN_DATA_ENV)            | Env var overriding the data root (default `~/kodokan_data`).                                                                                       |
|------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| [`SEGMENT_MIN_TWO_PERSON_FRAC`](#kodokan.config.SEGMENT_MIN_TWO_PERSON_FRAC) | keep a detected demonstration only if both judoka are present for at least this fraction of its frames.                                            |
| [`EVAL_MIN_TWO_PERSON_FRAC`](#kodokan.config.EVAL_MIN_TWO_PERSON_FRAC)    | Quality gate used when *consuming* stored demos for recognition/scoring/ catalog building: only trust demos where both judoka are clearly visible. |

### Functions

| [`clips_dir`](#kodokan.config.clips_dir)()   | Directory holding downloaded video clips + their `*.info.json` sidecars.   |
|----------------------------------------------------------------|----------------------------------------------------------------------------|
| [`data_dir`](#kodokan.config.data_dir)()    | Resolve the kodokan data root (`$KODOKAN_DATA_DIR` or `~/kodokan_data`).   |
| [`models_dir`](#kodokan.config.models_dir)()  | Directory for downloaded model weights (kept out of the repo).             |
| [`pose_dir`](#kodokan.config.pose_dir)()    | Directory holding extracted per-clip keypoint sequences (NPZ / Parquet).   |
| [`viz_dir`](#kodokan.config.viz_dir)()     | Directory holding rendered overlays (MP4) and Rerun recordings (RRD).      |

### kodokan.config.EVAL_MIN_TWO_PERSON_FRAC *: [float](https://docs.python.org/3/builtins/functions.html#float)* *= 0.4*

Quality gate used when *consuming* stored demos for recognition/scoring/
catalog building: only trust demos where both judoka are clearly visible.
Stricter than the segmentation gate on purpose. Used by the eval harnesses,
[`kodokan.flashcards.build_catalog()`](kodokan.flashcards.html.md#kodokan.flashcards.build_catalog), and scoring.

### kodokan.config.KODOKAN_DATA_ENV *= 'KODOKAN_DATA_DIR'*

Env var overriding the data root (default `~/kodokan_data`).

### kodokan.config.SEGMENT_MIN_TWO_PERSON_FRAC *: [float](https://docs.python.org/3/builtins/functions.html#float)* *= 0.3*

keep a detected demonstration
only if both judoka are present for at least this fraction of its frames.
Deliberately lenient — we would rather store a borderline demo (and filter it
later) than silently drop a real one. Used by the batch/segmentation step.

* **Type:**
  Inclusion gate used when *segmenting* a clip

### kodokan.config.clips_dir()

Directory holding downloaded video clips + their `*.info.json` sidecars.

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### kodokan.config.data_dir()

Resolve the kodokan data root (`$KODOKAN_DATA_DIR` or `~/kodokan_data`).

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### kodokan.config.models_dir()

Directory for downloaded model weights (kept out of the repo).

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### kodokan.config.pose_dir()

Directory holding extracted per-clip keypoint sequences (NPZ / Parquet).

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### kodokan.config.viz_dir()

Directory holding rendered overlays (MP4) and Rerun recordings (RRD).

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
