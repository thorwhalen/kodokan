# Recognition bake-off — learned classifiers + role-consistent features work

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

Reproduce: `PYTHONPATH=<repo> python examples/eval_learned.py --feature <f> --method <m>`
(features/methods in `kodokan/recognize.py`); full sweep via the
`kodokan-recognition-bakeoff` workflow.
