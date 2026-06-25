# Adversarial review — methodology & code (and what it changed)

A 4-lens adversarial review (statistical validity · CV/pose methodology · code
correctness · experimental reproducibility), with each critical/high finding
independently verified against the code. 32 findings (3 critical, 10 high, …). The
honest upshot below corrects several claims made earlier in `recognition-bakeoff.md`
and `feature-bakeoff.md`.

## The one that matters most — clip-identity confound (CRITICAL, confirmed)

In this dataset **each technique class == exactly one YouTube video**. The recognition
eval is leave-one-**demo**-out, so it trains and tests on reps from the *same clip*
(same demonstrators, gi, camera, background, lighting, codec). The classifier can win
by recognizing the **video**, not the **throw**. Therefore the reported recognition
numbers (e.g. 0.242 / "6.7× chance") are an **upper bound** and do **not** establish
clip-independent technique recognition.

**The only real fix is data:** ≥2 *independent* source clips per technique (different
demonstrators/venues/cameras) and **leave-one-CLIP-out** (group CV keyed by `video_id`).
Scaling the Kodokan playlist adds classes but is still 1 clip/technique, so it does
**not** resolve this for recognition (it does serve the flashcard app and coverage).
Tracked in issue [Data] multi-source clips.

## Confirmed code leakages — FIXED

- **Feature standardization fit on the full dataset** (incl. the held-out point),
  outside the LOO loop, in all pooled classifiers (`recognize.py`). → Now standardized
  **per fold on the train rows only**; LDA likewise fit per fold on train-only features.
  Empirical effect on the headline: **negligible (0.242 → 0.242)** — real bug, immaterial here.
- **DTW "normalized" = dist / len(path)** left a ~1/√len length bias (`compare.py`,
  used by scoring, `feature-bakeoff`, dtw-1NN). → Now **mean Euclidean cost per aligned
  step**. (`feature-bakeoff` AUCs should be recomputed; its null conclusion is robust.)
- **Reporting**: added **balanced accuracy** and an **empirical majority-class baseline**
  to `eval_learned` (top-1 alone is misleading on imbalanced, many-tiny-class data).

Leak-free, within-clip, 28 classes: top-1 **0.242**, balanced **0.229**, majority
baseline **0.059**. So a within-clip signal exists — but see the confound above.

## 3D "refutes viewpoint" claim — WITHDRAWN (high/critical, partially confirmed)

`feature-bakeoff.md` concluded that 3D joint angles didn't beat 2D, "refuting" the
viewpoint hypothesis. But MediaPipe per-crop **world landmarks are camera-aligned, not
canonicalized**, and the bbox crop distorts their scale/orientation — so the 3D angles
were **not actually viewpoint-invariant**. The experiment doesn't test what it claimed.
→ Language softened; a valid test needs **pelvis-frame (Procrustes) canonicalization**
before computing angles, plus square/letterboxed crops and stride-matched, person-matched
comparison. Tracked as a research task.

## Deferred (tracked in issues), by priority

1. **Multi-source data → leave-one-clip-out** — the real validity fix.
2. **Tori/uke heuristic validation** — `tori_index` (lowest finish-frame hip-y) is
   camera-dependent and unvalidated; the "role-consistency is the dominant lever" causal
   claim is plausible but needs ground-truth tori labels + an abstain/instrumentation path.
3. **3D pelvis-frame canonicalization** + fair (stride/person-matched) 2D-vs-3D re-run.
4. **`feature-bakeoff` rigor** — leave-one-demo-out medoid (no self-zero), standard
   ROC-AUC, re-run with the fixed DTW normalization.
5. **Robust position normalization** (`descriptors._norm_positions`) — clip/floor torso
   scale to avoid outlier "fingerprints".
6. **Confidence intervals / repeated CV / per-class confusion**; **shared `MIN_2P`/
   thresholds config** (batch used 0.3, eval 0.4); tracker **swap-rate metric**;
   store **integrity checks** (duplicate/oob rows).

## Revised, honest conclusions

- There is a **within-clip** technique signal (leak-free 0.242 top-1 / 0.229 balanced vs
  0.059 majority), but it is **not yet validated as clip-independent technique recognition**.
- The **"2D/3D descriptors are at chance for separability"** null result is robust (the
  biases were optimistic, yet the result was still null); the **"3D refutes viewpoint"**
  causal claim is withdrawn pending a canonicalized test.
- **Role-consistency** and **learned-classifier** levers appear to hold after the leak fix
  (re-measured), but differences should be reported with CIs and, ultimately, under
  leave-one-clip-out.
