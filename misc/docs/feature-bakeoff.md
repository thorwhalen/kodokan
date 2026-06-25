# Feature descriptor bake-off — why 2D isn't enough

> **⚠️ Validity caveats (adversarial review, see `adversarial-review.md`).**
> (1) The **"3D refutes the viewpoint hypothesis"** conclusion below is **withdrawn**:
> MediaPipe per-crop world landmarks are camera-aligned (not canonicalized) and the bbox
> crop distorts them, so those "3D angles" were **not** actually viewpoint-invariant — the
> experiment doesn't test what it claimed. A valid test needs pelvis-frame (Procrustes)
> canonicalization. (2) The DTW "normalized" distance had a length bias (now fixed) and the
> medoid reference was built in-sample, so the **exact AUC values are biased-optimistic**;
> the **null conclusion ("descriptors ~at chance") is robust** (the biases inflate apparent
> separability, yet it was still ~chance), but re-run with the fixes before citing exact AUCs.

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

## 3D lift — hypothesis tested and REFUTED

The hypothesis was that viewpoint-invariant **3D** would fix it. We lifted the 2D
tracks to metric 3D world landmarks per person (MediaPipe per-crop, isolated venv —
`scripts/lift_3d_mediapipe.py`), computed 3D joint angles (invariant to camera angle
by construction), and re-ran this exact harness:

| descriptor | feat_dim | AUC | technique-ID acc |
|---|---|---|---|
| 2D joint angles | 8 | 0.492 | 0.30 |
| **3D joint angles (MediaPipe world)** | 8 | **0.491** | 0.40 |

**3D did not help** — AUC stayed at chance, genuine ≈ impostor (0.137 vs 0.135).
So viewpoint is *not* the single fixable lever. The real limits are:
1. **Monocular per-crop 3D depth is too noisy** under grappling occlusion (MediaPipe
   world-landmark z is unreliable when bodies overlap).
2. **Tori/uke role inconsistency** — primary-person features compare the *thrower* in
   one demo to the *receiver* in another (the `angles_both` interaction variant also
   failed, but role-aligned features were not isolated).
3. **Hand-crafted 8-angle DTW + medoid reference** is too weak a representation for
   only ~8 demos/technique.

**Revised implication:** recognition/scoring needs a **learned** skeleton representation
(few-shot **JEANIE**, or trained **PoseC3D / CTR-GCN** — research §7 Stage 3-4) and/or
**cleaner multi-person 3D** (e.g. CoMotion) with **role-consistent** tori/uke features —
not just a 3D lift of the same coarse descriptor. The per-clip/per-demo eval harness
here is exactly what's needed to validate those next.

## Reproduce

- 2D: `PYTHONPATH=<repo> python examples/eval_features.py --mode <descriptor>` (modes in
  `kodokan/descriptors.py`); across modes via the `kodokan-feature-bakeoff` workflow.
- 3D: `python examples/export_2d_bridge.py` → `~/.kodokan_mp/bin/python scripts/lift_3d_mediapipe.py`
  → `python examples/eval_features_3d.py`. The 3D lift runs in an isolated venv
  (`~/.kodokan_mp`: mediapipe==0.10.18, numpy<2, opencv-python-headless==4.10) because
  mediapipe is ABI-incompatible with the main env's numpy 2.x.
