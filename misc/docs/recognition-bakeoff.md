# Recognition bake-off — what separates judo techniques (with validity caveats)

> ## 🚨 RESULT THAT SUPERSEDES EVERYTHING BELOW: cross-source ≈ chance
> We acquired a **second independent source** (Efficient Judo) and ran the honest
> **leave-one-CLIP-out** eval (`examples/eval_cross_source.py`) over techniques present in
> both sources (group = source video). On the **same data**, two protocols:
>
> | protocol | top-1 | notes |
> |---|---|---|
> | within-clip (leave-one-demo-out, leaky) | ~0.08–0.12 | the kind of number reported below |
> | **cross-clip (leave-one-CLIP-out, honest)** | **~0.02–0.03** | ≈ chance (uniform ~0.018, majority ~0.034) |
>
> (52–58 techniques, ~2 clips each, tori_angles_pos+LDA / tori_angles+centroid.) **Across
> independent video sources, the current monocular-2D-pose pipeline recognizes the throw at
> CHANCE.** The within-clip "signal" (incl. the earlier "0.242 / 6.7× chance") was largely
> **clip identity** (people/gi/camera/background), confirmed by the within-vs-cross gap.
> The numbers in the rest of this doc are within-clip and should be read as such.
> See `adversarial-review.md`; next steps in issue #9/#10 and the strategy note below.

> **⚠️ Validity update (adversarial review, see `adversarial-review.md`).**
> **Each technique class == one YouTube video**, so leave-one-demo-out trains/tests on
> reps from the *same clip*; the numbers below are a **within-clip upper bound** and do
> **not** prove clip-independent technique recognition (the model may key on the video,
> not the throw). Honest validation needs ≥2 independent clips/technique + leave-one-clip-out.
> Two code leakages were found and **fixed** (per-fold standardization; DTW length-bias);
> re-running leak-free left the headline numbers **unchanged** (the leakage was immaterial).
> Reporting now also includes **balanced accuracy** and an **empirical majority-class
> baseline** (top-1 alone is misleading on imbalanced, many-tiny-class data).

The feature bake-off showed the original approach (primary-person joint angles +
DTW) can't tell techniques apart. This experiment asks whether the bottleneck is the
*classifier*, the *features*, or tori/uke *role inconsistency* — by sweeping
feature modes × classifiers with **leave-one-out CV** on the dataset's two-person
demos (`examples/eval_learned.py`, `kodokan/recognize.py`).

Setup: 82 demos, 10 techniques, **chance = 0.10**. `tori` = the person left standing
at the finish (lower hip-y); `pool_*` = temporal-pyramid pooled descriptor (mean+std
over 1+2+4 segments) + a classifier.

## Results (LOO accuracy, sorted)

| feature | method | accuracy | × chance |
|---|---|---|---|
| `tori_angles_pos` | `pool_lda_knn` | **0.341** | 3.4× |
| `tori_angles` | `pool_lda_knn` | 0.305 | 3.0× |
| `primary_angles` | `pool_lda_knn` | 0.244 | 2.4× |
| `tori_angles_pos` | `pool_knn` | 0.244 | 2.4× |
| `tori_angles` | `pool_knn` | 0.232 | 2.3× |
| `tori_angles_pos` | `pool_centroid` | 0.207 | 2.1× |
| `primary_angles` | `pool_knn` | 0.195 | 2.0× |
| `tori_angles` | `pool_centroid` | 0.195 | 2.0× |
| `tori_angles_pos` | `dtw_1nn` | 0.195 | 2.0× |
| `primary_angles` | `pool_centroid` | 0.146 | 1.5× |
| `primary_angles` | `dtw_1nn` (original baseline) | **0.098** | 1.0× (chance) |

(`tori_angles`+`dtw_1nn` failed on a transient API error; not re-run — `dtw_1nn` is
the weakest method regardless.)

## Conclusion — three levers that stack

1. **Learned classifier** ≫ DTW-1NN: LDA-reduced kNN reaches 0.244 even on the
   baseline features, vs 0.098 for DTW-1NN. Pooled descriptor + supervised projection
   captures discriminative structure that pairwise DTW does not.
2. **Role consistency** (tori-only vs primary-person) adds ~0.05–0.06 consistently —
   confirming the suspected tori/uke role-mixing noise was real.
3. **Richer features** (angles + normalized positions) beat angles alone.

Best combo (`tori_angles_pos` + LDA+kNN) = **0.341, 3.4× chance** — a clear,
measured win over the original (chance). The task is far from solved (66% error on 10
classes), but there *is* learnable technique signal, and the path is evidenced:

**Next:** (a) **scale the dataset** (more techniques + more demos/technique — the
classifiers are data-starved at ~8 demos/class); (b) **better pose** (whole-body
133-kpt, cleaner multi-person 3D, better tori/uke ID); (c) then deep skeleton models
(PoseC3D / few-shot JEANIE) become viable. The same LOO harness validates each.

## Update — scaled to 28 techniques (236 demos)

We scaled the dataset to 28 usable techniques (#002–#031, 236 two-person demos) and
re-ran the LOO harness:

| feature | method | 10-tech acc (×chance) | 28-tech acc (×chance) |
|---|---|---|---|
| `primary_angles` | `pool_lda_knn` | 0.244 (2.4×) | 0.093 (2.6×) |
| `tori_angles` | `pool_lda_knn` | 0.305 (3.0×) | 0.178 (4.9×) |
| `tori_angles_pos` | `pool_lda_knn` | **0.341 (3.4×)** | **0.242 (6.7×)** |

Two clear takeaways:
1. Raw accuracy drops with 2.8× more classes (harder), but the **signal-to-chance
   ratio roughly doubles** (3.4× → 6.7×) — more data makes the learned representation
   generalize better. 24% top-1 over **28 judo techniques** from monocular video is a
   meaningful baseline.
2. **Role-consistency is the dominant lever at scale:** primary-person features barely
   clear chance (2.6×), while tori-only reaches 6.7×. Getting tori/uke identity right
   matters more than any other single choice here.

## Leak-free re-run (28 techniques) + balanced accuracy

After fixing the standardization/DTW leakages, the 28-technique numbers are **unchanged**,
now reported with balanced accuracy and the empirical majority-class baseline (0.059):

| feature | method | top-1 | balanced | note |
|---|---|---|---|---|
| primary_angles | pool_lda_knn | 0.093 | 0.088 | role-inconsistent baseline |
| tori_angles | pool_lda_knn | 0.178 | 0.164 | +role-consistency |
| **tori_angles_pos** | **pool_lda_knn** | **0.242** | **0.229** | +richer features (best) |
| primary_angles | pool_centroid | 0.097 | 0.101 | |
| tori_angles_pos | pool_knn | 0.148 | 0.135 | learned-projection (LDA) helps |

The role-consistency and learned-classifier levers survive the leak fix. **But** all of
this is the within-clip upper bound (see the validity box at top): until there are ≥2
independent clips per technique, we cannot separate "recognizes the throw" from
"recognizes the video". That is the next experiment (leave-one-clip-out).

Path: (1) acquire multi-source clips → leave-one-clip-out (the validity fix); (2) improve
pose/role quality (validated tori detection, whole-body, canonicalized 3D); (3) deep
skeleton models — each validated by this harness with balanced accuracy + CIs.

Reproduce: `PYTHONPATH=<repo> python examples/eval_learned.py --feature <f> --method <m>`
(features/methods in `kodokan/recognize.py`); full sweep via the
`kodokan-recognition-bakeoff` workflow.
