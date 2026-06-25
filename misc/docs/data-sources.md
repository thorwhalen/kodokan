# Data sources for honest (cross-clip) technique recognition

The recognition validity blocker (issue #9, `adversarial-review.md`): our Kodokan-IJF
playlist has **one video per technique**, so within-clip leave-one-demo-out can't tell
"recognizes the throw" from "recognizes the video". We need **≥2 independent source clips
per technique** and **leave-one-CLIP-out** (group CV by source video). A research sweep
(arXiv / Papers-with-Code / HF / Kaggle / Zenodo / Google Dataset Search, incl. Japanese)
found:

## Key finding
- **No ready-made public dataset** has performer-diverse, Kodokan-named, skeleton judo
  throws. (Closest: `adenhaus/judo_throws` — HF, MIT, RGB, **4 throws**, but no source
  metadata; PLOS Kinect 3 throws / 24 judoka but raw joints unreleased; Tsukuba 781-throw
  corpus unreleased; otherwise wrestling/BJJ/MMA/taekwondo.)
- **A clean second instructional series EXISTS:** **"Efficient Judo — Demos: 67 Throws of
  Kodokan Judo"** (`PLwd8pJWYTk07K6hDg2_N-9xd31gUCsew7`) — a different London dojo, one
  single-throw real-human demo per technique, consistent hyphenated romaji titles.
  **Verified overlap with our set: 66 techniques** (the full Nage-waza; the unmatched are
  Katame-waza pins/chokes + category dividers, correctly absent).
- Third sources for gap-fill / a 3rd group: **Judo Canada Gokyo** playlist; **IJF
  competition DB** (different modality, all-rights-reserved).

## Calibration (what's achievable honestly)
Under cross-performer / group-aware splits, named-technique skeleton recognition in
combat sports is ~**0.80–0.87** (taekwondo viewpoint-agnostic 0.867; Open FSW wrestling
0.829). The 0.90–0.96 headlines come from within-clip / 2-athlete random splits — exactly
the leakage we're removing. Expect a fine-grained penalty as the throw vocabulary grows.
Named-judo-throw recognition is essentially **greenfield** (the only judo-video ML paper
does combat *phase*, not throws).

## Plan
1. **Acquire Efficient Judo as `source="efficient_judo"`** into its own clips dir (kept
   separate from the Kodokan source so each clip's source/technique are unambiguous).
2. Tag every clip with a **`technique_key`** (canonical romaji, e.g. `oguruma`,
   `deashiharai`) + **`source`**, so the same throw across sources shares a label while
   the **source video is the CV group**.
3. **Leave-one-CLIP-out** (group CV by `video_id`) over technique_keys present in ≥2
   sources → the first honest cross-source accuracy; compare to the within-clip upper bound.
4. Pilot quickly on `adenhaus/judo_throws` ∩ our set (ippon-seoi-nage, o-goshi, osoto-gari,
   uchi-mata) for a same-day 3-group check.

## Licensing posture
Same as our existing Kodokan use: **internal research, pose-extraction only, store the
source URL, no redistribution**. Efficient Judo + Judo Canada are single contactable
owners (ask before any data/model release); IJF is strictest. See references in the
research transcript.
