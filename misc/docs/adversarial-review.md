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

## Deferred items — status (issue #10)

1. **Multi-source data → leave-one-clip-out** — ✅ **DONE** (issue #9). The second source
   (*Efficient Judo*, 66 overlapping techniques) is acquired and posed; the definitive
   cross-source **leave-one-CLIP-out = 0.004 top-1 over 65 techniques** (uniform chance
   0.015). The pipeline recognizes throws **at chance across independent sources** — the
   within-clip numbers were clip identity. Harness: `examples/eval_cross_source.py`.
2. **Tori/uke heuristic validation** — ⚠️ **PARTIAL**. `recognize.tori_decision` now returns
   a confidence **margin** (finish hip-y separation in torso-length units) plus
   `abstained`/`fell_back` flags, and `tori_decision_stats` aggregates them (`eval_learned`
   reports them). Over 804 demos: **fell-back 0.0%, abstained 16.8%, median margin 0.96
   torso-lengths** — confident on ~83% of demos. **Still open:** the heuristic is
   *instrumented*, not *validated* — there is no hand-labeled tori ground truth to check it
   against.
3. **3D pelvis-frame canonicalization** + fair 2D-vs-3D re-run — ⚠️ **PARTIAL**.
   `kodokan.canon3d` (`pelvis_frame` / `canonicalize_pose`) lands the Procrustes
   canonicalization, unit-tested for invariance to camera rotation + translation + scale.
   Canonical 3D **positions** are meaningfully more separable than raw camera-frame ones
   (**0.232 acc / AUC 0.564** vs **0.049 / AUC 0.451**, 10 techniques). But joint **angles**
   are *already* invariant to rigid + isotropic-scale transforms, so canonicalization is a
   mathematical **no-op for angle features** (verified: identical numbers) — the earlier
   angle-based 3D conclusion is unaffected by it. **Still open:** canonicalization cannot
   undo the *anisotropic* bbox-crop distortion this review named, so a complete viewpoint
   re-test still needs **square/letterboxed re-lifting** plus a **stride- & person-matched**
   2D-vs-3D comparison.
4. **`feature-bakeoff` rigor** — ✅ **DONE (code)**. `recognize.loo_medoid_separability` does
   a true **leave-one-demo-out medoid** (a held-out demo never influences its own technique's
   reference, so there are no self-zeros to filter), and `recognize.roc_auc_distances` is a
   **standard ROC-AUC** (Mann–Whitney U, ties = 0.5). Both `eval_features.py` and
   `eval_features_3d.py` use it, on the fixed DTW normalization. (The full 2D DTW re-run at
   131-technique scale is O(n²·DTW) and slow; not yet re-run.)
5. **Robust position normalization** — ✅ **DONE**. `descriptors._torso_scale` floors the
   torso scale by a fraction of the person's keypoint bbox diagonal (projection-robust) and
   `_norm_positions` clips residual outliers, so an edge-on torso can no longer collapse the
   scale toward zero and explode positions into a per-clip outlier "fingerprint".
6. **CIs / per-class confusion; shared thresholds; swap-rate; store integrity** — ✅ **DONE**.
   `classification_metrics(n_boot=…)` adds percentile-bootstrap 95% CIs and `confusion_pairs`
   reports the most-frequent true→pred errors; `config.SEGMENT_MIN_TWO_PERSON_FRAC` /
   `EVAL_MIN_TWO_PERSON_FRAC` are the single source of truth for the 0.3/0.4 gates (they
   differ *on purpose* — lenient when segmenting, strict when consuming);
   `track.identity_swap_rate` is a ground-truth-free identity-discontinuity metric; and
   `store.check_tidy_integrity` / `store_integrity_report` check duplicate / out-of-bounds /
   bad-confidence rows.

With CIs, the role-consistency lever still holds at the current 131-technique scale:
`tori_angles_pos` **0.086** [0.068, 0.109] vs `primary_angles` **0.057** [0.040, 0.074]
(majority 0.021, uniform chance 0.008) — but this remains a **within-clip upper bound**,
and item 1 shows it collapses to chance across independent clips.

## Revised, honest conclusions

- There is a **within-clip** technique signal (leak-free 0.242 top-1 / 0.229 balanced vs
  0.059 majority), but it is **not yet validated as clip-independent technique recognition**.
- The **"2D/3D descriptors are at chance for separability"** null result is robust (the
  biases were optimistic, yet the result was still null); the **"3D refutes viewpoint"**
  causal claim is withdrawn pending a canonicalized test.
- **Role-consistency** and **learned-classifier** levers appear to hold after the leak fix
  (re-measured), but differences should be reported with CIs and, ultimately, under
  leave-one-clip-out.
