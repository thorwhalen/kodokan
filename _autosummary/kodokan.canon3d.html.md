# kodokan.canon3d

Pelvis-frame (Procrustes) canonicalization of 3D world landmarks.

The Stage-5  *“3D refutes viewpoint”* conclusion was **withdrawn** by the
adversarial review (`misc/docs/adversarial-review.md`): MediaPipe per-crop
world landmarks are camera-aligned — not canonicalized — and the bounding-box
crop distorts their scale/orientation, so 3D joint angles computed straight off
them are **not** actually viewpoint-invariant. Before a fair 2D-vs-3D test, each
frame’s landmarks must be expressed in a *subject-centered anatomical frame*:
origin at the pelvis, axes fixed to the body (hip line + spine), scale normalized
by torso length. This module provides that canonicalization as a pure, testable
geometric function.

After [`canonicalize_pose()`](#kodokan.canon3d.canonicalize_pose), two demonstrations of the same articulated pose
seen from different camera azimuths map to the **same** canonical coordinates
(invariant to camera rotation, translation, and — with `scale=True` — subject
size). Downstream angle/position features are then genuinely viewpoint-invariant.

Indices default to the MediaPipe **BlazePose-33** convention used by the repo’s
3D lift (`scripts/lift_3d_mediapipe.py`); pass `joints=COCO17` (or a custom
[`BodyJoints`](#kodokan.canon3d.BodyJoints)) to reuse it for any layout exposing both hips and shoulders.

### Module Attributes

| [`BLAZEPOSE33`](#kodokan.canon3d.BLAZEPOSE33)   | MediaPipe BlazePose-33 (the repo's 3D-lift output format).                   |
|----------------------------------------------------------------|------------------------------------------------------------------------------|
| [`COCO17`](#kodokan.canon3d.COCO17)        | COCO-17 (the 2D pose format), for reusing the canonicalization on 2.5D data. |

### Functions

| [`canonicalize_pose`](#kodokan.canon3d.canonicalize_pose)(landmarks, \*[, joints, scale])   | Express one frame's 3D landmarks `(K, C>=3)` in the pelvis frame `(K, 3)`.         |
|------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| [`canonicalize_sequence`](#kodokan.canon3d.canonicalize_sequence)(world, \*[, joints, scale])   | Canonicalize every frame/person of a `(..., K, C)` landmark stack → `(..., K, 3)`. |
| [`pelvis_frame`](#kodokan.canon3d.pelvis_frame)(landmarks, \*[, joints])               | Anatomical frame of a single pose: `(pelvis_origin, R)`.                           |

### Classes

| [`BodyJoints`](#kodokan.canon3d.BodyJoints)(l_hip, r_hip, l_shoulder, r_shoulder)   | Landmark indices needed to build the pelvis anatomical frame.   |
|-----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------|

### kodokan.canon3d.BLAZEPOSE33 *= BodyJoints(l_hip=23, r_hip=24, l_shoulder=11, r_shoulder=12)*

MediaPipe BlazePose-33 (the repo’s 3D-lift output format).

### *class* kodokan.canon3d.BodyJoints(l_hip, r_hip, l_shoulder, r_shoulder)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Landmark indices needed to build the pelvis anatomical frame.

### kodokan.canon3d.COCO17 *= BodyJoints(l_hip=11, r_hip=12, l_shoulder=5, r_shoulder=6)*

COCO-17 (the 2D pose format), for reusing the canonicalization on 2.5D data.

### kodokan.canon3d.canonicalize_pose(landmarks, , joints=BodyJoints(l_hip=23, r_hip=24, l_shoulder=11, r_shoulder=12), scale=True)

Express one frame’s 3D landmarks `(K, C>=3)` in the pelvis frame `(K, 3)`.

Removes camera rotation + translation (and, with `scale=True`, subject size
by dividing by torso length). Columns beyond `xyz` (e.g. visibility) are
dropped; `nan` inputs propagate (so occluded frames stay droppable).

* **Return type:**
  `ndarray`

### kodokan.canon3d.canonicalize_sequence(world, , joints=BodyJoints(l_hip=23, r_hip=24, l_shoulder=11, r_shoulder=12), scale=True)

Canonicalize every frame/person of a `(..., K, C)` landmark stack → `(..., K, 3)`.

Each frame (and person) is canonicalized to *its own* pelvis frame — the
viewpoint-invariant choice for pose-only technique features.

* **Return type:**
  `ndarray`

### kodokan.canon3d.pelvis_frame(landmarks, , joints=BodyJoints(l_hip=23, r_hip=24, l_shoulder=11, r_shoulder=12))

Anatomical frame of a single pose: `(pelvis_origin, R)`.

`R` (shape `(3, 3)`, rows are the new basis) maps world coordinates into
the subject frame: **x** = subject’s right (hip line), **y** = up
(pelvis→neck spine), **z** = forward (`y × x`). The hip line is
orthogonalized against the spine so the basis is orthonormal (Gram-Schmidt).

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`ndarray`, `ndarray`]
