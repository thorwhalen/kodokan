# Regenerating `kodokan_data` (everything outside the repo is reproducible)

All of `kodokan`'s bulk data lives **outside the repo** at the root returned by
`kodokan.config.data_dir()` — `$KODOKAN_DATA_DIR`, or `~/kodokan_data` by default.
On the dev Mac that default path is a **symlink** into
`~/Dropbox/_odata/misc/kodokan_data/` (moved there for off-machine backup); code and
docs still address it as `~/kodokan_data` and resolve transparently through the symlink.

**None of it is precious.** Every subfolder is either downloaded from a public source or
derived by a pipeline script in `examples/`. This doc is the map from *artifact → how to
recreate it*, so the tree can be safely lost, pruned, moved to object storage, or rebuilt
on a fresh machine. Sizes are indicative (a ~1.3 GB build).

## Prerequisites

- Repo installed with the relevant extras (`pip install -e '.[pose,viz,analysis,storage,acquire]'`)
  and **ffmpeg** on PATH — see the README.
- `KODOKAN_DATA_DIR` pointing where you want the data (or rely on the `~/kodokan_data` default).
- Network access (YouTube + model-weight downloads).

## Artifact map

| Subfolder | Size | Kind | Regenerate with |
|---|---|---|---|
| `clips/` | ~708M | downloaded | `examples/batch_pipeline.py` (downloads the Kodokan-IJF playlist clips + `*.info.json`, idempotent via a yt-dlp archive) |
| `clips_efficient_judo/` | ~332M | downloaded | second source set — the *Efficient Judo* playlist `PLwd8pJWYTk07K6hDg2_N-9xd31gUCsew7` (see `data-sources.md`); acquired via `kodokan.acquire.download_source(source="efficient_judo")` |
| `pose/` | ~102M | **derived** | `examples/batch_pipeline.py` — tracked two-person pose + demo segmentation → Parquet pose store + JSON segments store. Pure function of `clips/`. |
| `viz/` | ~73M | **derived** | rendered overlays / Rerun recordings from `kodokan.viz` (e.g. `examples/warmup_seoinage.py`, `segment_review.py`). Pure function of `clips/` + `pose/`. |
| `models/yolo11n-pose.pt` | ~6M | downloaded | auto-fetched by `ultralytics` on first pose run (`kodokan.pose`); no manual step. |
| `style_models/face_paint_512_v2_0.onnx` | ~8M | downloaded | AnimeGANv2 `face_paint_512_v2` weights (upstream: `bryandlee/animegan2-pytorch`) exported to ONNX. Used only by `examples/generate_stylized_clips.py`. **This is the one weight without an automated fetch** — keep a copy or re-export from upstream if lost. |
| `clips_prestyle_backup/`, `clips_styl_anime_only_backup/`, `clips_styl_oldtest/` | ~105M | **snapshots** | manual backups / experiments from the issue #39 stylization rollout — *not* consumed by any pipeline. Safe to delete; regenerate a fresh stylized set with `examples/generate_stylized_clips.py`. |

## Full rebuild from zero

```bash
export KODOKAN_DATA_DIR=~/kodokan_data          # or wherever you want it

# 1. clips/ + pose/ + pose/segments/  (downloads + tracked pose + segmentation)
PYTHONPATH=. python examples/batch_pipeline.py --items 2:31 --device mps

# 2. (optional) second source for cross-clip recognition — see data-sources.md
#    kodokan.acquire.download_source(source="efficient_judo") into clips_efficient_judo/

# 3. (optional) renders / Rerun recordings
PYTHONPATH=. python examples/warmup_seoinage.py
PYTHONPATH=. python examples/segment_review.py

# 4. (optional) web-app stylized clips — needs style_models/face_paint_512_v2_0.onnx
KODOKAN_DATA_DIR=~/kodokan_data PYTHONPATH=. python examples/generate_stylized_clips.py
```

`models/yolo11n-pose.pt` downloads itself on the first pose run. Everything is
skip-existing / idempotent, so a partial tree just tops itself up.

## Provenance

- **Playlists / sources:** `misc/docs/data-sources.md`, `misc/docs/playlist_index.txt`.
- **Dataset shape & schema:** `misc/docs/dataset.md`.
- **Stylization / face-privacy pipeline:** `misc/docs/adr-video-face-privacy.md`,
  `examples/generate_stylized_clips.py`.
