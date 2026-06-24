# kodokan

Study Kodokan Judo throws from video via body-pose analysis: download technique
demonstrations, extract two-person (tori/uke) skeletons, split each clip into its
repeated demonstrations, visualize them, and compare/score demonstrations.

```python
from kodokan.acquire import download_techniques
from kodokan.track import estimate_poses_tracked
from kodokan.segment import segment_demonstrations
from kodokan.viz import render_skeleton_video

res = download_techniques(playlist_items="2")[0]          # Seoi-nage (#002), with metadata
seq = estimate_poses_tracked(res.path, source_url=res.info["webpage_url"])  # tracked tori/uke COCO-17
demos = segment_demonstrations(seq, min_two_person_frac=0.3)                # per-demo (start_s, end_s)
render_skeleton_video(seq, out_path="overlay.mp4", source_video=res.path)  # skeletons on the video
render_skeleton_video(seq, out_path="skeleton.mp4", blank_canvas=True)      # skeletons on blank canvas
```

## What it does

A functional pipeline over the official *Kodokan 100 Techniques* YouTube playlist:

```
acquire (yb) ─► pose (rtmlib / YOLO, tracked tori/uke) ─► segment (motion-energy)
       └─► dol stores (Parquet pose + JSON segments) ─► visualize (overlay / blank / Rerun)
       └─► compare two demos (joint-angle DTW) ─► score + eval harness
```

YouTube acquisition lives in the [`yb`](https://github.com/thorwhalen/yb) package
(`download_youtube_playlist`); `kodokan` is the analysis layer on top.

## Install

`import kodokan` needs only **numpy**; everything heavy is an optional extra (imported
lazily on first use), so the import never fails for a missing one:

```bash
pip install -e '.[all]'      # or pick extras: .[pose,viz,analysis,storage,acquire]
```

| extra | for | brings |
|---|---|---|
| `pose` | pose estimation | rtmlib, onnxruntime, ultralytics |
| `viz` | rendering | opencv-python, rerun-sdk, supervision, matplotlib |
| `analysis` | segment / compare / score | scipy, dtaidistance, pandas, pyarrow |
| `storage` | dol stores | dol |
| `acquire` | YouTube download | yb |

You also need **ffmpeg** on PATH (acquisition/merge). The optional **3D lift**
(`scripts/lift_3d_mediapipe.py`) runs in a **separate venv**, because MediaPipe is
ABI-incompatible with numpy 2.x:

```bash
python -m venv ~/.kodokan_mp
~/.kodokan_mp/bin/pip install 'mediapipe==0.10.18' 'numpy<2' 'opencv-python-headless==4.10.0.84'
```

Data (videos, keypoints, renders, weights) lives **outside the repo** under
`~/kodokan_data` (override with `KODOKAN_DATA_DIR`).

## The pipeline

| module | purpose |
|---|---|
| `kodokan.acquire` | download techniques (wraps `yb`), skip the PV, keep source URLs |
| `kodokan.pose` | `estimate_poses` facade (rtmlib / ultralytics backends), COCO-17, `PoseSequence` |
| `kodokan.track` | `estimate_poses_tracked` — stable tori/uke identity (BoT-SORT + spatial continuity) |
| `kodokan.segment` | hysteresis motion-energy segmentation + two-person gate + self-similarity |
| `kodokan.store` | `pose_store` (tidy Parquet) / `segments_store` (JSON), the analysis SSOT |
| `kodokan.viz` | overlay / blank-canvas MP4 + Rerun logging |
| `kodokan.compare` | joint-angle (soft-)DTW comparison of two demonstrations |
| `kodokan.score` | reference-based 0–100 scoring + per-joint/per-phase feedback |
| `kodokan.descriptors` | experimental feature descriptors (for the eval harness) |

Runnable end-to-end examples live in `examples/` (`warmup_seoinage.py`,
`batch_pipeline.py`, `segment_review.py`, `compare_demos.py`, `score_demos.py`,
`eval_features.py`).

## Dataset

`examples/batch_pipeline.py` builds a small dataset (10 techniques · 84
demonstrations · 18.3k frames) into the `dol` stores. Load it:

```python
from kodokan.store import pose_store, segments_store, load_all_tidy
seq = pose_store()["zIq0xI0ogxk"]          # (F, 2, 17, 3) COCO-17 (x, y, conf)
demos = segments_store()["zIq0xI0ogxk"]    # demo intervals + source_url
df = load_all_tidy()                       # tidy DataFrame across all clips
```

See [`misc/docs/dataset.md`](misc/docs/dataset.md).

## Status & honest limits

Works well: acquisition, tracked two-person pose, demo segmentation, the dol stores,
visualization, and same-technique demo comparison (joint-angle DTW is speed-invariant)
with interpretable per-joint/per-phase feedback.

Does **not** work yet — and this is *measured, not assumed*: **technique recognition /
cross-demo scoring**. A feature bake-off ([`misc/docs/feature-bakeoff.md`](misc/docs/feature-bakeoff.md))
shows every 2D descriptor *and* MediaPipe 3D joint angles sit at chance (separation
AUC ≈ 0.49–0.56). The blockers are noisy monocular 3D under grappling occlusion,
tori/uke role inconsistency, and the weakness of hand-crafted angle-DTW for few
examples — not viewpoint alone. Recognition needs a **learned** skeleton representation
(few-shot JEANIE, or trained PoseC3D/CTR-GCN) and/or cleaner multi-person 3D with
role-consistent features. The eval harness (`examples/eval_features*.py`) is ready to
validate those.

## Background & rationale

- [`misc/docs/research-architecture.md`](misc/docs/research-architecture.md) — cited
  tools/architecture research (75 refs).
- [`misc/docs/dataset.md`](misc/docs/dataset.md) — dataset card.
- [`misc/docs/feature-bakeoff.md`](misc/docs/feature-bakeoff.md) — why hand-crafted
  features don't discriminate techniques (the empirical finding).
