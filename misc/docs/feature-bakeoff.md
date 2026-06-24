# Feature descriptor bake-off — why 2D isn't enough

After the Stage-5 scoring scaffold showed poor technique discrimination, we measured
*which* 2D feature descriptor (if any) separates the 10 techniques, using the eval
harness `examples/eval_features.py` (medoid reference per technique; technique-ID =
each technique's demos → nearest reference; **AUC** = P(impostor DTW distance >
genuine DTW distance); 0.5 = chance).

## Result (10 techniques, two-person demos with ≥40% coverage)

| descriptor | dim | AUC | technique-ID acc | genuine median | impostor median |
|---|---|---|---|---|---|
| `angles_pos` (angles + norm. positions) | 42 | **0.560** | 0.30 | 0.268 | 0.285 |
| `angles_both` (both people, activity-ordered) | 16 | 0.515 | 0.10 | 0.195 | 0.198 |
| `pos_both` | 68 | 0.507 | 0.10 | 0.439 | 0.446 |
| `angles` (baseline) | 8 | 0.492 | 0.30 | 0.100 | 0.099 |
| `angles_vel` (angles + velocity) | 16 | 0.474 | 0.30 | 0.131 | 0.129 |

## Conclusion

**Every 2D descriptor is at chance** (AUC 0.47–0.56; genuine ≈ impostor distance in
all cases). Adding the two-person interaction (`angles_both`), velocity, or raw
normalized positions does **not** help. So the discrimination failure is **not** a
descriptor or role-assignment problem — it is the fundamental **viewpoint** problem:
2D joint angles/positions of the same throw filmed from different camera angles differ
more than the same angle across different throws.

**Implication:** technique recognition and meaningful scoring need **viewpoint-invariant
3D** features — lift the 2D tracks to 3D (MotionBERT, image-free, consumes the stored
2D) and re-run this exact harness. If AUC rises well above 0.5, 3D is the unlock.

Reproduce: `PYTHONPATH=<repo> python examples/eval_features.py --mode <descriptor>`
(modes in `kodokan/descriptors.py`). Run across modes via the `kodokan-feature-bakeoff`
workflow.
