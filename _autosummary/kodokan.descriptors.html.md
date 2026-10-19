# kodokan.descriptors

Experimental per-demo feature descriptors (to test what makes throws separable).

The Stage-5 eval showed plain single-person joint angles don’t discriminate
technique (dominated by viewpoint + tori/uke role inconsistency). This module
offers alternative descriptors so the eval harness can measure which lever helps:

- `angles`      — primary person’s joint angles (the current baseline).
- `angles_vel`  — angles + their frame-to-frame velocity (dynamics).
- `angles_pos`  — angles + root-centered, torso-scaled keypoint positions.
- `angles_both` — both people’s joint angles, ordered by activity (captures the
  : tori/uke *interaction*, robust to which slot is which).
- `pos_both`    — both people’s normalized positions, activity-ordered.

All are 2D (still viewpoint-variant); the point is to isolate how much of the
failure is role/interaction/descriptor vs genuinely needing 3D.

### Functions

| [`demo_descriptor`](#kodokan.descriptors.demo_descriptor)(pose_seq, start_s, end_s, \*)   | Per-frame feature matrix for one demo window under the chosen `mode`.   |
|--------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|

### kodokan.descriptors.demo_descriptor(pose_seq, start_s, end_s, , mode='angles')

Per-frame feature matrix for one demo window under the chosen `mode`.

* **Return type:**
  `ndarray`
