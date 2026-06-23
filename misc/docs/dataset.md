# Kodokan pose dataset (warm-up build)

A small dataset built end-to-end by the pipeline from the *Kodokan 100 Techniques*
playlist: download → tracked two-person pose → demo segmentation → `dol` stores.

## Build it

```bash
# downloads #002–#011 (idempotent), runs tracked pose + segmentation, writes the stores
PYTHONPATH=<repo> python examples/batch_pipeline.py --items 2:11 --device mps
PYTHONPATH=<repo> python examples/dataset_overview.py    # stats + contact sheet
```

Data lives **outside the repo** under `~/kodokan_data` (override `KODOKAN_DATA_DIR`):
`clips/` (mp4 + `.info.json`), `pose/*.parquet` (pose store), `pose/segments/*.json` (segments store).

## Contents (this build)

10 techniques · **86 demonstrations** · 18,323 analyzed frames · 7.7 MB Parquet.

| video_id | technique | frames | both-present | demos |
|---|---|---|---|---|
| zIq0xI0ogxk | 背負投 / Seoi-nage | 1997 | 61% | 11 |
| FQnOlCxo4oI | 一本背負投 / Ippon-seoi-nage | 1338 | 51% | 4 |
| vu1TMVNnq34 | 背負落 / Seoi-otoshi | 2018 | 67% | 11 |
| 4x6S3Q-Ktv8 | 体落 / Tai-otoshi | 1891 | 55% | 9 |
| cnHRhSy8yi4 | 肩車 / Kata-guruma | 1741 | 68% | 8 |
| vU6aJ2kFxoI | 掬投 / Sukui-nage | 2667 | 71% | 13 |
| ff8U2TVZIYI | 帯落 / Obi-otoshi | 1864 | 66% | 7 |
| 6H5tmncOY4Q | 浮落 / Uki-otoshi | 1398 | 52% | 6 |
| lLU9wv52ni0 | 隅落 / Sumi-otoshi | 1284 | 66% | 7 |
| MGlyKmSuzdc | 山嵐 / Yama-arashi | 2125 | 49% | 10 |

`both-present` = fraction of analyzed frames with both tori and uke tracked. ~50–70%
reflects genuine single-person framing (intros/close-ups) and apex occlusion — not a
bug (the stored skeletons are clean; missing frames are `NaN`, not garbage).

## Loading

```python
from kodokan.store import pose_store, segments_store, load_all_tidy

ps = pose_store()                  # {video_id: PoseSequence}  (tidy Parquet under the hood)
seq = ps["zIq0xI0ogxk"]            # (F, 2, 17, 3) COCO-17 (x, y, conf); NaN where a slot is empty
segs = segments_store()["zIq0xI0ogxk"]   # {technique, source_url, n_demos, demos:[{start_s,end_s,...}]}
df = load_all_tidy()               # one tidy DataFrame across all clips (video_id, frame, person, keypoint, x, y, conf)
```

## Schema

**Pose** — per-clip Parquet, tidy/long: `fidx, frame, t_sec, person, keypoint, x, y, conf`
(missing keypoints dropped, not stored as NaN); sequence metadata
(`fps, width, height, backend, video_path, source_url, n_persons, frame_indices`) in the
Parquet schema metadata. **Segments** — per-clip JSON with the demo intervals + provenance
(includes the canonical `source_url`).

## Caveats / next

- Pose backend here is YOLO11-pose + BoT-SORT (id) + spatial-continuity tracking; rtmlib/RTMO
  is the Apache-licensed alternative.
- Demo segmentation is motion-energy valleys (training-free); slow-motion reps may over/under-split
  pending the TSM/RepNet cross-checks.
- Identity is best-effort through the apex; comparison across camera angles is speed-invariant but
  not yet viewpoint-invariant. See `research-architecture.md`.
