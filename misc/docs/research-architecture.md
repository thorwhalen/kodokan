# Kodokan Judo Video Pose-Analysis — Tools & Architecture Research

> Deep research (2024–2026 sources) to choose tools and architecture for: downloading
> the *Kodokan Judo 100 Techniques* playlist + metadata, body-pose estimation to skeletal
> keypoint sequences, segmenting each demo clip into its repeated demonstrations, visualizing
> skeletons (overlay-on-video and overlay-on-blank-canvas), and eventually recognizing,
> comparing, and **scoring** judo throws (incl. live webcam).
>
> Method: a multi-agent workflow — 4 agents read the existing local code (`yb`, `theremin`,
> `thoremin`, the package ecosystem), 7 web-research specialists swept the SOTA, and every
> load-bearing technical claim was adversarially fact-checked against primary sources. Where
> a fact-check corrected a claim, the correction is reflected here.
>
> Constraints assumed throughout: **Apple-Silicon Mac (CPU + Metal, no NVIDIA GPU)**; Python-first;
> functional design + `dol`-style storage + reuse of the local package ecosystem; offline-batch
> now, live-webcam later; **two people in close contact (tori + uke) ⇒ heavy mutual occlusion**.

---

## 1. Executive summary

1. **The playlist is ideal and already mapped.** 100 entries: #001 is the all-techniques *PV*
   (the reference — skip it in the main line), #002–#100 are 99 single-technique demos, each
   ~76–86 s at **1920×1080 @ 25 fps**. Skip the PV with `yt-dlp` `playlist_items="2:"`.
2. **Don't reuse MediaPipe for the analysis.** MediaPipe Pose is **single-person by design**;
   it cleanly separates people only when they're well apart and collapses/mis-attributes limbs
   under the close contact intrinsic to a throw [1][3][4]. Keep MediaPipe **only** for the later
   *single-person* live-webcam learner scorer (where you already know its API from `theremin`/`thoremin`).
3. **Primary pose engine: `rtmlib`** (pip, pure ONNXRuntime, no mmcv/mmpose install pain) running
   **RTMO** (one-stage, multi-person, degrades better under overlap) or **RTMPose**; true
   multi-person, Apache-2.0, runs on Apple-Silicon CPU/MPS [6][7][8]. **Fallback / cross-check:
   Ultralytics YOLO11-pose** — best Apple-Silicon (MPS/CoreML) story and a built-in tracker, but
   **AGPL-3.0** [9][10]. This is independently the choice of the two mature sports pipelines,
   **Sports2D** and **Pose2Sim**, which both build on `rtmlib` [68][69].
4. **The two-person clinch is the hard wall — be honest about it.** With *monocular, single-camera*
   tools in 2026, reliable two-person skeletons **through the apex of a throw are not solved**.
   The only in-the-wild close-contact dataset (Harmony4D) needed **20+ synchronized cameras**, and
   even a model fine-tuned on it still has ~59 mm vertex error in severe-contact frames [18]. Plan for
   ID-swaps and dropouts at the apex; build the reference from the cleaner entry/exit phases and
   **average across the many reps** in each clip.
5. **Segment reps by rhythm, not by cuts.** Scene-cut detectors (PySceneDetect) find nothing in a
   continuous take [29]. The robust, training-free signal is **whole-body motion energy** (optical
   flow + summed keypoint velocity) → low-motion "reset" valleys = rep boundaries; validate/group
   variable-speed reps with a **self-similarity matrix + DTW**; cross-check the count and detect
   slow-motion with **RepNet** [30][31][35]. Expect ~90 % boundary recall when reps have visible
   resets; ship a tiny review/nudge UI from day one.
6. **Comparison & scoring start training-free.** Normalize skeletons (hip-center, torso-scale,
   rotation), convert to **joint-angle features** (scale/occlusion-robust), align with **(soft-)DTW**,
   and read off per-joint / per-phase deviation [39][53][55]. This is the consensus "compare-to-reference"
   method and needs **zero labels** — it's the right first deliverable for "compare two demonstrations."
7. **Store raw, normalize on read.** Canonical schema `(person, keypoint, channel)` per frame in
   **COCO-17** layout with image-2D + normalized-2D + (optional) 3D + confidence + `role`(tori/uke) + `track_id`.
   SSOT = **per-clip Parquet (tidy/long) in a `dol` store**; **NPZ** dense-array cache for the numeric
   inner loop; rep boundaries as a **minimal JSON-interval store** (mirrors to WebVTT/OTIO via `lacing`) [61][67].
   *(Note: Parquet is chosen for columnar scans + ecosystem fit, **not** raw read speed — the fact-check
   refuted a "5× faster than HDF5" claim; HDF5 is actually faster per-record.)*
8. **Visualize with Rerun.** One `rerun.io` recording gives overlay-on-video **and**
   skeleton-on-blank-canvas **and** side-by-side compare on a scrubbable timeline [65]; bake shareable
   MP4s with `supervision` + OpenCV. The best *live* UX (per a user study) is the learner's **own
   skeleton over their own webcam feed**, smoothed with a **OneEuro** filter [72][73].
9. **You already own most of the spine.** `yb` (acquisition — extend with a playlist function),
   `mixing` (lazy video frames), `dol` (storage + `cache_this`), `lacing` (interval/segment store with
   WebVTT/OTIO/ELAN adapters), `meshed`/`i2` (pipeline composition), `theremin` (MediaPipe pattern for
   the live path), `an`/`artful` (animation). The only essential **new external** deps are `rtmlib`
   (+`onnxruntime`), `rerun-sdk`, `supervision`, and a DTW lib (`dtaidistance`/`tslearn`).
10. **It's greenfield.** There is **no public judo throw/pose dataset or pipeline** [51]; the nearest
    neighbor is a BJJ auto-scorer that fine-tuned ViTPose on Harmony4D and documents exactly the pitfalls
    you'll hit [71]. The strategy follows from this: alignment-first + **few-shot** (not big-model training).

---

## 2. Problem framing & the central difficulty

A judo throw is a **two-body, contact-heavy, fast-rotational** event. Every downstream goal inherits one
hard constraint: from a single camera, when tori and uke clinch, their bounding boxes overlap, limbs
intertwine, and any per-person pose estimator must guess "whose limb is whose" with little evidence.

**How hard, concretely?** The field's own reality checks:
- The only in-the-wild close-contact video dataset, **Harmony4D** (wrestling/MMA/grappling), was built
  with **20+ synchronized cameras** precisely because monocular capture fails here; even **HMR2.0
  fine-tuned on it** only reduces vertex error to **~59 mm** in severe-occlusion/contact frames [18].
  *(Fact-check correction: an earlier draft read "54.8 % PVE" as an accuracy score — it is actually a
  54.8 % **reduction** in vertex error from fine-tuning; the honest residual is ~59 mm.)*
- Apple's **CoMotion** — the best monocular multi-person 3D-tracking model for a Mac — names this exact
  failure: *"tracks that collapse together into a single identity"* [17].
- Generic 3D pose models degrade **+150 % to +350 % MPJPE** under heavy occlusion; distal joints
  (wrists/feet) suffer most, core joints (hips/shoulders/elbows) stay most robust [25].

**Operating principle (the honest one):** treat the **apex as expected-degraded**. The entry (kuzushi/tsukuri)
and finish phases are tractable; the collision peak is not. Because every clip repeats *one* throw several
times at different speeds/angles, **aggregate across reps** and lean on the clean phases to build references
and to anchor identity. If contact-grade 3D ground truth ever becomes a hard requirement, the only thing
that currently works is **multi-view capture** (even 2–3 phones for a handful of gold references) [18][28].

---

## 3. Pose estimation — comparison & recommendation

### Comparison (realistic options for this project)

| Tool | Multi-person | 3D | Apple-Silicon | License | Fit | Role |
|---|---|---|---|---|---|---|
| **rtmlib → RTMO / RTMPose** [6][7][8] | **Yes** (RTMO one-stage best under overlap) | 2D (lift w/ MotionBERT) | CPU/MPS via ONNXRuntime; CoreML fork exists (unproven) | **Apache-2.0** | ★★★★★ | **Primary, offline analysis** |
| **Ultralytics YOLO11-pose** [9][10] | Yes (top-down) | No | **Best** (MPS + CoreML), built-in tracker | **AGPL-3.0** | ★★★★ | **Fallback / cross-check; live w/ partner** |
| **MediaPipe PoseLandmarker** [1][2] | **No** (single-person) | **Yes** (metric world landmarks) | CPU-only in practice on Mac | Apache-2.0 | ★★ | **Single-person live-webcam scorer only** |
| **MoveNet (MultiPose)** | up to 6 | No | TF/TFLite (Metal hit-or-miss) | Apache-2.0 | ★★★ | lightweight alt; weaker than RTM/YOLO |
| **MotionBERT (2D→3D lift)** [15] | per-track | **Yes** (skeleton) | **Yes** (CPU/MPS; consumes keypoints) | Apache-2.0 | ★★★★ | **cheap 3D on the Mac** from 2D tracks |
| **CoMotion (Apple)** [17] | **Yes** (3D + IDs) | **Yes** (SMPL) | **Yes** (ships CoreML detection) | research | ★★★★ | best monocular 3D+tracking; pilot it |
| **NLF / 4D-Humans (SMPL mesh)** [13][14] | Yes (video) | **Yes** (mesh) | **No** (effectively CUDA) | code MIT / weights **non-commercial** | ★★★ | offline **GPU batch** for gold 3D refs |
| **Meta Sapiens** [12] | top-down (needs detector) | dense 2D | **No** (A100-class) | per-checkpoint | ★★ | offline part-seg to disambiguate limbs |

Three precise facts the fact-checks **confirmed**:
- MediaPipe `num_poses` defaults to 1 and its bundled BlazePose model is single-person; raising it finds
  *well-separated* extra people but **fails in close contact** — disqualifying for the demo clips [1][3][4].
- MediaPipe **does** output genuine *model-estimated* metric 3D world landmarks (33 pts, meters, hip-origin) —
  valuable for the *solo* learner scorer [2].
- On macOS Apple-Silicon the MediaPipe Python GPU/Metal delegate is **unreliable/no-benefit** across
  0.10.15–0.10.32 (crashes with segmentation, shader errors, no speedup, memory swap); **budget for CPU** [5].

### Recommendation

- **Offline judo-video analysis — PRIMARY:** `rtmlib` with **RTMO** (one-stage, handles intertwined bodies
  best) and **RTMPose-m**/whole-body (DWPose/RTMW, 133 kpts) when you need grips/feet detail. Wrap behind a
  single `estimate_poses(...)` facade with a **pluggable backend strategy** [6][7][8].
- **Offline — FALLBACK / cross-check:** **YOLO11-pose** (validates rtmlib; its built-in ByteTrack/BoT-SORT
  solves ID assignment for free). Mind **AGPL-3.0** if you ever distribute [9][10][24].
- **3D without a GPU:** lift 2D tracks with **MotionBERT** on the Mac (it consumes keypoints, not video) [15].
- **Gold-standard 3D mesh (hard apex case):** **NLF + nlf-pipeline** or **4D-Humans** as an **offline batch
  on a rented CUDA GPU** — respect the non-commercial weight licenses for an educational project [13][14].
- **Live webcam (later), single learner:** **MediaPipe PoseLandmarker** (single person ⇒ its limitation is
  moot; metric 3D for free; you know the API) [1][2]; YOLO11-pose on MPS if a partner is in frame.

---

## 4. Two-person tracking & identity (tori vs uke)

Recommended layered pipeline (top-down + strong tracking + temporal repair):

1. **Per-person 2D pose:** RTMO/RTMPose (or YOLO11-pose) — top-down gives best per-crop accuracy [7][8][9].
2. **Identity through occlusion — use appearance-based tracking, not motion-only.** **BoT-SORT** (or
   **Deep OC-SORT**) adds ReID embeddings + camera-motion compensation; plain **ByteTrack** swaps IDs when
   two judogi-clad bodies overlap [22][23][24]. *(Fact-check nuance: BoT-SORT's edge over ByteTrack is real
   but modest (~3 IDF1 pts) and partly from CMC, not ReID alone — so pair it with the heuristic below.)*
   `boxmot` integrates these with YOLO detectors.
3. **Seed tori/uke from clean pre-contact frames** (relative position, who initiates kuzushi), propagate via
   tracking, and **re-resolve after the throw** with ReID continuity. Persist `role` as metadata next to the
   required source URL.
4. **Disambiguate intertwined limbs** with body-part segmentation (Sapiens 28-class, or a lighter YOLO-seg)
   to assign overlapping keypoints to the correct person [12].
5. **Temporal repair:** confidence-gate to flag collapsed/merged apex frames; reconstruct with
   **energy-minimization gap-filling** (beats naive interpolation) and **SmoothNet/OneEuro** jitter
   smoothing [26][73]. Trust core joints over wrists/feet [25].
6. **Pilot the all-in-one option:** Apple **CoMotion** gives monocular multi-person 3D + persistent IDs with a
   CoreML detection stage — test it on 2–3 real clips before locking architecture [17].

The **BJJ auto-scoring** precedent reports ID-swaps and missing keypoints in the clinch as the *dominant*
practical failure, and that **fine-tuning ViTPose on Harmony4D** (close-contact data) transfers far better
than stock COCO models — the realistic path if accuracy proves insufficient [71][18].

---

## 5. Segmenting a clip into per-demo start/end times

**Do not lead with scene-cut detection.** PySceneDetect's content/adaptive detectors fire on editorial
discontinuities; two reps of the same throw in one take produce none — confirmed at the source-code level [29].
Use it only as a cheap *pre-pass* to split clips that contain real edits (title cards, spliced slow-mo replays).

**Recommended recipe (classical, unsupervised, CPU-fast):**
0. **PySceneDetect pre-pass** → edit-bounded shots (often just one) [29].
1. **Motion-energy valley detection (source of truth):** per frame, fuse dense **Farneback optical-flow**
   magnitude (restricted to the actors' bbox) with the **confidence-weighted sum of all keypoint
   velocities**; z-normalize, smooth, and find **low-motion valleys** (`scipy.signal.find_peaks` on the
   negated signal) = the walk-back/re-grip/bow "resets" between reps. Speed/angle/position-invariant by
   construction [35][38].
2. **Validate & group with TSM + DTW:** a frame×frame **self-similarity matrix** on per-frame features
   reveals repeats as off-diagonal stripes; pairwise **DTW** confirms "same throw, different speed" and
   yields warp paths you reuse later for scoring [30][39][40].
3. **Cross-check count + detect slow-mo with RepNet** (materight PyTorch port, CPU-runnable): its per-frame
   period-length doubles as a local playback-speed map; agreement with the valley count ⇒ high confidence [30][31].
4. **Ship a review/nudge UI from day one** — render the activity signal + proposed boundaries on a scrubber.
   Persist segments to a `dol`/`lacing` interval store: `{(video_id, rep_idx): {start_s, end_s, speed,
   angle, position, confidence, source_url}}`.

**Accuracy expectation:** ~90 %+ boundary recall within ~0.3–1.0 s when reps have visible resets (the standard
Kodokan teaching format); back-to-back reps or spliced slow-mo need the TSM/DTW + RepNet cross-checks and the
occasional manual nudge. For robustness later, **ASOT** (optimal-transport unsupervised segmentation, CVPR 2024)
is a drop-in upgrade [37]. Avoid heavy deep counters (RACnet/ESCounts) first — strong but slow on CPU and
out-of-distribution on two intertwined bodies [33][34].

---

## 6. Data model, storage & visualization

### Representation (store raw; normalize on read)
- Per frame: `(P, K, C)` — `P` = person slot (tori/uke), `K` = keypoints (**COCO-17** canonical; BlazePose-33
  or WholeBody-133 optional), `C` = channels: image-`(x,y)`, normalized-`(x,y)`, optional world-`(x,y,z)`,
  plus `confidence`. Add explicit `track_id` + `role` [67].
- **Normalization as a pure function at analysis time:** hip-midpoint root-centering (translation), torso-length
  scale (scale), hip-vector rotation alignment (view) → then **joint-angle features** (scale/translation/
  occlusion-robust). This is the converged skeleton-AR recipe [67][53].

### Storage (two `dol` stores)
- **Pose SSOT:** per-clip **Parquet (tidy/long)** — columns `clip_id, frame, t_sec, person, role, keypoint,
  x, y, z, conf, space` — in a `dol` `{clip_id: parquet_bytes}` store. Chosen for **columnar scans + pandas/
  polars/DuckDB-native tooling + compression** (not raw read latency — fact-check refuted the "5× HDF5" claim;
  HDF5 is faster per-record, Parquet wins on analytical scans) [61].
- **Numeric cache:** **NPZ** `(F,P,K,C)` per clip for the DTW/model inner loop (derived, not SSOT).
- **Scale-up later:** **Zarr** for whole-corpus chunked access; not needed at playlist scale [62].
- **Do not** use BVH/FBX/glTF as the analysis store — they're rotational rig/animation formats, useful only as
  a downstream **avatar export** for blank-canvas 3D playback. Skip legacy OpenPose JSON; standardize on COCO.

### Segmentation/annotation
- SSOT = **minimal JSON intervals** in a `dol` store (captures speed/angle/position) — and you already own
  **`lacing`**, a `TimeInterval`-keyed annotation store with **round-trip adapters to WebVTT, OTIO, ELAN/EAF,
  JAMS** and Allen's interval algebra. Use `lacing` as the segment store; generate WebVTT for browser review;
  use **Label Studio** (video TimelineLabels + KeyPointLabels + JSON export + MMPose→COCO bridge) as the
  human-in-the-loop correction UI for boundaries and ID-swap fixes; normalize its exports down to the SSOT [64].

### Visualization
- **Primary: Rerun (`rerun-sdk`).** One recording → overlay-on-video **and** skeleton-on-blank-3D-canvas
  **and** side-by-side compare on a scrubbable timeline, via `rr.Image` + `rr.Points2D/Points3D(keypoint_ids=…)`
  + an `AnnotationContext(ClassDescription(keypoint_connections=…))`; log tori/uke as separate entity paths.
  **Pin the version** (pre-1.0, breaking changes) [65].
- **Baked MP4 deliverables:** `supervision` `VertexAnnotator`/`EdgeAnnotator` + OpenCV `VideoWriter` (overlay
  and blank-canvas) [66]. `matplotlib` for one-off figures only.
- **Web viewer (future):** `thoremin`'s `canvas_overlay.ts` (mirror/normalize transforms, per-element toggles)
  and its NDJSON landmark schema are directly adaptable; switch `HandLandmarker` → `PoseLandmarker`.

---

## 7. The eventual goals — a staged path

| Stage | Goal | Approach | Data needed |
|---|---|---|---|
| **1** | **Compare two demonstrations** | Normalize → joint-angle features → **(soft-)DTW** → per-joint/per-phase deviation curves | **none (training-free)** [39][53][55] |
| **2** | **Segment reps** | Motion-energy valleys + TSM/DTW + RepNet (§5) | none [30][35] |
| **3** | **Recognize a *specific* throw from few clips** | **Few-shot / metric learning**, not softmax. **JEANIE** (DTW that jointly warps **time + camera viewpoint** — absorbs the "reps vary by angle" problem) ; or embed via MotionBERT/PoseC3D + nearest-neighbor | a few labeled exemplars [46][15] |
| **4** | **Full throw classifier (more data)** | **PoseConv3D** (noise-robust, multi-person-native, skeleton-only) or **CTR-GCN + CHASE** (lighter, two-person geometry). Train in the cloud, infer on Mac | small labeled set [41][44][45] |
| **5** | **Score a live attempt vs reference** | **AQA contrastive regression** (CoRe/MCoRe: query-vs-exemplar relative score); decompose throw into phases (kuzushi/tsukuri/kake) **FineDiving-style**; reuse Stage-1 DTW deviations as interpretable feedback | labeled quality pairs [47][48][49] |

Caveats the fact-checks flagged: PoseC3D's "multi-person at no extra cost" is **unverified for heavy
occlusion** — benchmark it against CTR-GCN+CHASE on real clips rather than assuming [41]. DTW handles the
**speed** axis well but **not viewpoint/position** — that's what JEANIE's joint warping and the §6
normalization address [46]. And **scoring needs a defined target** (judo has no judge-score dataset like
diving) — likely "similarity-to-reference" plus an expert rubric on balance/kuzushi.

Live UX (Stage 5): the highest-value feedback is the learner's **own skeleton over their own webcam**,
smoothed with **OneEuro**, committing feedback only after a few stable frames; run pose live but defer DTW
scoring to throw-completion to keep the loop responsive [72][73].

---

## 8. YouTube acquisition — how to extend `yb`

`yb.download.youtube` is a clean, keyword-only, yt-dlp-based **single-video** module that already keeps
`webpage_url` in its `_INFO_FIELDS` and supports `write_info_json`/subtitles. It hardcodes `noplaylist=True`
and has **no playlist iteration** — that's the gap. Recommended (fact-checked) extension:

- Add `download_youtube_playlist(playlist_url, *, download_dir=None, playlist_items=None,
  download_archive=None, fmt='bestvideo+bestaudio/best', merge_to='mp4', write_info_json=True,
  cookies_from_browser=None, extractor_args=None, extra_opts=None) -> list[DownloadResult]`. Internally:
  reuse the existing opts/helpers but **drop `noplaylist`**, set `playlist_items`, `download_archive`,
  `ignoreerrors='only_download'`; run `extract_info(download=True)`; iterate `result['entries']` [57][58].
- **Skip the PV:** `playlist_items="2:"` (yt-dlp is **1-based**; colon = slice-to-end). `"1,"` would download
  *only* item 1 — the opposite. More robust: a `match_filter` rejecting titles containing "PV" so ordering
  changes don't matter [58]. *(Ground truth confirmed: PV is index 1; #002–#100 are the techniques.)*
- **Metadata:** `webpage_url` is the canonical SSOT source URL (`original_url` only echoes user input — add it
  too). Add `fps, width, height` to `_INFO_FIELDS` so the pose stage gets native frame rate without re-probing.
  Persist `webpage_url` with **every** clip's keypoint output (your hard requirement) [57].
- **Resolution:** keep `bestvideo+bestaudio/best` → mp4; **don't chase 4K** — pose models downscale internally
  (~256×256) and the occlusion problem is a *model* problem, not a resolution one. fps ≥ 24 matters [60].
- **2025–2026 reliability:** wire **off-by-default** knobs for bot-detection/SABR/PO-token issues —
  `cookies_from_browser=('safari',)`, `extractor_args={'youtube': {'player_client': ['web_safari']}}`, retries;
  document `bgutil-ytdlp-pot-provider` as the escalation. A one-shot batch from a residential Mac IP usually
  just works [59]. Add a `check_requirements` for **ffmpeg** (needed for the merge).
- **Legal:** yt-dlp is legal; downloading these third-party demos conflicts with YouTube ToS, but personal,
  offline **educational/research** use of public material is the low-risk case — store `webpage_url`, don't redistribute.

---

## 9. Recommended end-to-end architecture

**Layering / dependency direction** (everything points *into* the spine you already own):

```
                         ┌────────────────────────────── kodokan (orchestrator + judo domain) ──────────────────────────────┐
 yb ──► acquire ─►  clips+meta store (dol)  ─►  pose (rtmlib/yolo/mediapipe facade)  ─►  track/role (boxmot+heuristic)
 (playlist dl)          │  webpage_url SSOT          │  per-frame (P,K,C)                  │  tori/uke ids
                         ▼                            ▼                                    ▼
 mixing (frames) ──► pose store: Parquet SSOT + NPZ cache (dol, cache_this) ──► normalize (pure fns: center/scale/rot/angles)
                                                      │                                    │
 lacing (intervals) ◄── segment (motion-energy + TSM/DTW + RepNet) ──► rep store          ▼
                                                      │                          compare (soft-DTW) ─► viz (Rerun + supervision)
 meshed/i2 (compose) ─────────────────────────────────┘                          recognize (few-shot/JEANIE) · score (AQA)  [later]
```

**Module breakdown (proposed `kodokan` package):**
- `kodokan.acquire` — thin wrapper over the new `yb.download_youtube_playlist`; builds the clips+metadata
  `dol` store keyed by `video_id` (media + info.json + `webpage_url`), skipping the PV.
- `kodokan.pose` — `estimate_poses(video, *, backend='rtmo', ...)` **facade** with a pluggable backend
  **strategy** (`rtmlib` | `ultralytics` | `mediapipe`), sensible defaults (rtmlib+RTMO offline, mediapipe live).
- `kodokan.track` — tori/uke identity (boxmot BoT-SORT or YOLO built-in) + spatial-heuristic seeding + role labels.
- `kodokan.store` — `dol` stores: Parquet pose SSOT + NPZ cache; `cache_this` on the expensive pose stage.
- `kodokan.normalize` — pure functions (root-center, torso-scale, rotation-align, joint-angles).
- `kodokan.segment` — motion-energy valleys + TSM/DTW + RepNet → `lacing` interval store; review-UI hook.
- `kodokan.viz` — Rerun logging (overlay / blank-canvas / side-by-side) + `supervision`+OpenCV baked MP4.
- `kodokan.compare` — (soft-)DTW alignment + per-joint/per-phase deviation. *(Stage 1 deliverable.)*
- `kodokan.recognize` / `kodokan.score` — later (few-shot JEANIE; AQA contrastive regression).

**Why this shape:** it mirrors the proven **Pose2Sim/Sports2D** reference architecture (config-driven, discrete
**individually-cached stages**, each writing its own artifacts so stages are independently re-runnable) [68][69],
expressed in *your* idioms — functional, keyword-only, `dol`-backed, `meshed`-composed. Reuse: **`yb`**
(acquisition), **`mixing`** (lazy `Video`/`VideoFrames`), **`dol`** (`Store`/`cache_this`), **`lacing`** (segments
+ format adapters), **`theremin`** (MediaPipe pattern for the live path), **`an`/`artful`** (avatar/animation later).

---

## 10. Warm-up plan (validate on 1–2 videos first)

**Targets:** skip #001 (PV). Use **Seoi-nage** (#002, 80 s, `zIq0xI0ogxk`) and optionally **Tai-otoshi**
(#005, 76 s, `4x6S3Q-Ktv8`) or **O-goshi** (#020, 86 s, `yhu1mfy2vJ4`). All 1080p25. `yt-dlp` is installed in
the `p12` pyenv (`2026.03.17`). Full index saved at `misc/docs/playlist_index.txt`.

Ordered steps (each is a small, independently verifiable deliverable):
0. **Acquire:** extend `yb` with `download_youtube_playlist`; download #002 (and #005), `write_info_json=True`,
   confirm `webpage_url` is captured. ✅ = two clips + metadata (with URL) in a `dol` store.
1. **Frames:** iterate via `mixing.Video`; sanity-check one frame round-trips.
2. **Pose bake-off:** run **rtmlib RTMO** (multi-person) **and** **YOLO11-pose** on the two-body clip, **and**
   MediaPipe for contrast (show it collapses tori+uke). Save per-frame keypoints → Parquet + NPZ. ✅ = two
   skeletons tracked across most frames; documented apex degradation.
3. **Track tori/uke:** YOLO built-in ByteTrack/BoT-SORT (or boxmot); eyeball ID stability; note apex swaps.
4. **Visualize:** a **Rerun** recording (overlay-on-video + skeleton-on-blank-canvas + side-by-side) **and** a
   baked MP4 via `supervision`/OpenCV. ✅ = both overlay modes render.
5. **Segment:** motion-energy valley detection → rep intervals; cross-check count with RepNet; store in
   `lacing`; render boundaries on the Rerun timeline; nudge manually. ✅ = per-demo start/end times you can verify.
6. **Compare:** pick a fast rep vs a slow rep of the *same* throw; normalize → joint-angles → **soft-DTW** →
   plot per-joint deviation. ✅ = end-to-end "compare two demonstrations" working on one throw.

**Apple-Silicon install notes:** `pip install rtmlib onnxruntime` (CPU; try `device='mps'`, benchmark vs CPU);
`pip install ultralytics` (MPS auto); `pip install rerun-sdk supervision opencv-python`;
`pip install dtaidistance tslearn`; `ffmpeg` on PATH (Homebrew); `mediapipe` only for the later live path.
Benchmark RTMO/RTMPose/YOLO FPS on *your* chip before committing — the open question the research can't settle remotely.

---

## 11. Risks, open questions & decisions for you

**Risks (designed-around, not eliminated):**
- **Apex occlusion** — two-person monocular capture through the throw is partially reliable at best; set
  expectations in the README and aggregate across reps [18][17][25].
- **Viewpoint sensitivity** — 2D-angle features are unreliable off the sagittal/frontal plane; YouTube angles
  vary widely. Mitigate with view-invariant features, JEANIE-style viewpoint warping, or (fragile) 3D lift [68][46].
- **Greenfield data** — no judo dataset; "dataset quality > quantity" was the explicit BJJ lesson [71][51].

**Decisions I'd like your call on (collected in the question prompt):**
1. **Pose backend default** — `rtmlib`/RTMO (Apache, best occlusion behavior) vs YOLO11-pose (best Mac speed,
   AGPL). Affects licensing if you ever distribute.
2. **3D scope now** — stay 2D (+ optional MotionBERT lift on Mac), or stand up an offline **GPU batch**
   (NLF/CoMotion) for gold 3D references, or accept a small **multi-camera** rig for a few reference demos.
3. **Build target** — do you want me to start the **warm-up implementation** (extend `yb` + Stages 0–2), or
   keep researching/repo-scaffolding first?
4. **Whole-body keypoints** — COCO-17 body-only, or 133-kpt WholeBody for grips/feet (kumi-kata/kuzushi)?

---

## 12. Appendix — datasets & notable repos
- **Harmony4D** — close-contact (wrestling/MMA) video dataset, 20+ cams; the asset for fine-tuning/eval on
  two-person contact [18]. **Hi4D** — close-interaction benchmark (2 actors, 8 views, GT SMPL-X) [21].
- **Sports2D** (BSD-3, active) — closest single-clip multi-person analog; borrow its stage decomposition [68].
  **Pose2Sim** (BSD-3) — canonical staged/cached layout [69]. **rtmlib** (Apache-2.0) — the pose engine [6].
- **BJJ auto-scoring** (Kevin Patel) — only public grappling pipeline; documents the real pitfalls [71].
- **mmaction2 PoseC3D** — reference recipe for the later skeleton classifier (extract the recipe, infer on rtmlib) [42].
- **No judo-specific public pipeline/dataset exists** — kodokan's opportunity [51].

---

## References

[1] MediaPipe Pose Landmarker guide. Google AI Edge. https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker
[2] MediaPipe `pose.md` (world landmarks: meters, hip-origin). https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md
[3] MediaPipe issue #4681 — Improving PoseLandmarker multipose. https://github.com/google-ai-edge/mediapipe/issues/4681
[4] MediaPipe issue #4894 — how does `num_poses` work (returns 1; MoveNet keeps adjacent people separate). https://github.com/google/mediapipe/issues/4894
[5] MediaPipe macOS Apple-Silicon GPU issues (#5788, #6216, #6223, #5568, #5674). https://github.com/google-ai-edge/mediapipe/issues/6216
[6] rtmlib — RTMPose/RTMO/RTMW/DWPose/ViTPose without mmcv/mmpose/mmdet. https://github.com/Tau-J/rtmlib
[7] RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose. arXiv:2303.07399. https://arxiv.org/abs/2303.07399
[8] RTMO: One-Stage Real-Time Multi-Person Pose Estimation. CVPR 2024. https://arxiv.org/pdf/2312.07526
[9] Ultralytics YOLO11 Pose docs. https://docs.ultralytics.com/tasks/pose
[10] Ultralytics YOLO11 model (AGPL-3.0; Apple MPS). https://docs.ultralytics.com/models/yolo11
[11] MoveNet (Lightning/Thunder/MultiPose). TensorFlow Hub. https://www.tensorflow.org/hub/tutorials/movenet
[12] Sapiens: Foundation for Human Vision Models. arXiv:2408.12569. https://arxiv.org/abs/2408.12569
[13] NLF — Neural Localizer Fields (NeurIPS'24) + nlf-pipeline. https://github.com/isarandi/nlf
[14] 4D-Humans: Reconstructing & Tracking Humans (HMR2.0). https://github.com/shubham-goel/4D-Humans
[15] MotionBERT: Unified Human Motion Representations (2D→3D lift). arXiv:2210.06551. https://arxiv.org/abs/2210.06551
[16] BlazePose GHUM (33-kpt 3D). arXiv:2206.11678. https://arxiv.org/abs/2206.11678
[17] CoMotion: Concurrent Multi-person 3D Motion (Apple). arXiv:2504.12186 ; https://github.com/apple/ml-comotion
[18] Harmony4D: In-The-Wild Close Human Interactions (NeurIPS 2024). arXiv:2410.20294 ; https://github.com/jyuntins/harmony4d
[19] Reconstructing Close Human Interaction with Appearance and Proxemics Reasoning. arXiv:2507.02565. https://arxiv.org/html/2507.02565v1
[20] Generative Proxemics (BUDDI). https://muelea.github.io/buddi/
[21] Hi4D: 4D Instance Segmentation of Close Human Interaction (CVPR 2023). https://yifeiyin04.github.io/Hi4D/
[22] BoT-SORT: Robust Associations Multi-Pedestrian Tracking. https://github.com/NirAharon/BoT-SORT
[23] Deep OC-SORT: Adaptive Re-Identification. CVPR 2023. https://arxiv.org/pdf/2302.11813
[24] Object Tracking with ByteTrack and BoT-SORT. Ultralytics docs. https://docs.ultralytics.com/modes/track
[25] Benchmarking 3D Human Pose Estimation Models under Occlusions (2025). arXiv:2504.10350. https://arxiv.org/html/2504.10350v2
[26] Temporal Smoothing for 3D HPE & Localization for Occluded People. arXiv:2011.00250. https://arxiv.org/pdf/2011.00250
[27] Monocular 3D Multi-Person Pose by Integrating Top-Down & Bottom-Up Networks. CVPR 2021. arXiv:2104.01797. https://arxiv.org/abs/2104.01797
[28] Multi-person Physics-based Pose Estimation for Combat Sports (2025). arXiv:2504.08175. https://arxiv.org/html/2504.08175v1
[29] PySceneDetect Detectors (Content/Adaptive/Threshold). https://www.scenedetect.com/docs/latest/api/detectors.html
[30] Counting Out Time (RepNet), Dwibedi et al. CVPR 2020. https://sites.google.com/view/repnet
[31] RepNet-pytorch (CPU-runnable port + weights). https://github.com/materight/RepNet-pytorch
[32] TransRAC + RepCount dataset. CVPR 2022. https://github.com/SvipRepetitionCounting/TransRAC
[33] RACnet: Rethinking Temporal Self-Similarity for Repetition Counting. ICIP 2024. arXiv:2407.09431. https://arxiv.org/html/2407.09431v1
[34] Every Shot Counts (ESCounts). ACCV 2024. arXiv:2403.18074. https://arxiv.org/html/2403.18074v1
[35] Unsupervised Temporal Segmentation of Repetitive Human Actions (Zhou et al.). arXiv:1512.04115. https://arxiv.org/abs/1512.04115
[36] Skeleton Motion Words for Unsupervised Temporal Action Segmentation. arXiv:2508.04513. https://arxiv.org/abs/2508.04513
[37] Temporally Consistent Unbalanced Optimal Transport (ASOT). CVPR 2024. https://github.com/mingu6/action_seg_ot
[38] OpenCV Optical Flow (dense Farneback). https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
[39] tslearn — Dynamic Time Warping. https://tslearn.readthedocs.io/en/stable/user_guide/dtw.html
[40] dtaidistance — fast CPU DTW. https://github.com/wannesm/dtaidistance
[41] Revisiting Skeleton-based Action Recognition (PoseConv3D). CVPR 2022. arXiv:2104.13586. https://arxiv.org/abs/2104.13586
[42] mmaction2 PoseC3D (config + custom-dataset training). https://github.com/open-mmlab/mmaction2/blob/main/configs/skeleton/posec3d/README.md
[43] PYSKL — Good Practices for Skeleton Action Recognition. https://github.com/kennymckormick/pyskl
[44] CTR-GCN: Channel-wise Topology Refinement GCN. ICCV 2021. arXiv:2107.12213. https://arxiv.org/abs/2107.12213
[45] CHASE: Convex Hull Adaptive Shift for Multi-Entity Recognition. 2024. arXiv:2410.07153. https://arxiv.org/abs/2410.07153
[46] JEANIE: Temporal–Viewpoint Aligned Few-Shot 3D-Skeleton Recognition. IJCV 2024. arXiv:2402.04599. https://arxiv.org/abs/2402.04599
[47] A Decade of Action Quality Assessment (survey). 2025. arXiv:2502.02817. https://arxiv.org/abs/2502.02817
[48] Multi-Stage Contrastive Regression for AQA (MCoRe). 2024. arXiv:2401.02841. https://arxiv.org/abs/2401.02841
[49] FineDiving: Procedure-aware AQA. CVPR 2022. arXiv:2204.03646. https://arxiv.org/abs/2204.03646
[50] FineParser: Fine-grained Spatio-temporal Action Parser. CVPR 2024. arXiv:2405.06887. https://arxiv.org/html/2405.06887v1
[51] Annotation Techniques for Judo Combat Phase Classification. 2024. arXiv:2412.07155. https://arxiv.org/html/2412.07155v1
[52] AthletePose3D benchmark. 2025. arXiv:2503.07499. https://arxiv.org/html/2503.07499v1
[53] Skeleton-Based AQA with Anomaly-Aware DTW. Sensors 2025. https://doi.org/10.3390/s25237160
[54] 3D Pose-Based Temporal Action Segmentation for Figure Skating. arXiv:2408.16638. https://arxiv.org/pdf/2408.16638
[55] Real-time action scoring for PT using pose estimation (DTW + NCC). Sci. Rep. 2025. https://www.nature.com/articles/s41598-025-29062-7
[56] Fitness exercise evaluation via improved DTW. Sci. Rep. 2025. https://www.nature.com/articles/s41598-025-02535-5
[57] yt-dlp README (output template, format selection, playlist-items, info.json). https://github.com/yt-dlp/yt-dlp
[58] yt-dlp `options.py` (1-based `playlist_items` slice semantics). https://github.com/yt-dlp/yt-dlp/blob/master/yt_dlp/options.py
[59] yt-dlp PO Token Guide (2025–2026 bot-detection/SABR). https://github.com/yt-dlp/yt-dlp/wiki/PO-Token-Guide
[60] Accuracy Evaluation of 3D Pose with MediaPipe (256×256; fps thresholds). PMC11644880. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11644880/
[61] An Empirical Evaluation of Columnar Storage Formats (Parquet). arXiv:2304.05028. https://arxiv.org/pdf/2304.05028
[62] What Is Zarr? A Cloud-Native Format for Tensor Data. Earthmover. https://www.earthmover.io/blog/what-is-zarr/
[63] OpenTimelineIO (editorial timeline interchange). https://github.com/AcademySoftwareFoundation/OpenTimelineIO
[64] Label Studio export + MMPose Label-Studio→COCO bridge. https://labelstud.io/guide/export
[65] Rerun Annotation Context + MediaPipe human-pose example (2D/3D synced timeline). https://rerun.io/docs/concepts/annotation-context
[66] Supervision Keypoint Annotators (Vertex/Edge). https://supervision.roboflow.com/develop/keypoint/annotators/
[67] 3D Skeleton-Based Action Recognition: A Review (normalization conventions). arXiv:2506.00915. https://arxiv.org/html/2506.00915v1
[68] Sports2D — 2D pose + angles from video/webcam (rtmlib backend). https://github.com/davidpagnon/Sports2D
[69] Pose2Sim — markerless kinematics; staged/cached Config.toml architecture. https://github.com/perfanalytics/pose2sim
[70] AlphaPose — multi-person pose + tracking (non-commercial; stale). https://github.com/MVIG-SJTU/AlphaPose
[71] Patel, K. Jiu-Jitsu Match Auto Scoring with Computer Vision (ViTPose fine-tuned on Harmony4D). https://www.kevinbpatel.com/work/jiu-jitsu
[72] Tharatipyakul et al. Pose Estimation for Movement Learning from Online Videos. AVI 2020. arXiv:2004.03209. https://arxiv.org/abs/2004.03209
[73] SmoothNet / 1€ (OneEuro) filter for real-time pose jitter. arXiv:2112.13715. https://arxiv.org/pdf/2112.13715
[74] A comprehensive survey on pose estimation and tracking in sports. AI Review 2025. https://link.springer.com/article/10.1007/s10462-025-11344-1
[75] fitness-trainer-pose-estimation (MediaPipe rep-count + 0–100 form score UX). https://github.com/yakupzengin/fitness-trainer-pose-estimation
