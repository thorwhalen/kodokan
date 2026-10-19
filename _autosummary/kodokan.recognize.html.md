# kodokan.recognize

Learned-vs-DTW technique recognition experiments (data-appropriate, no GPU).

The feature bake-off showed DTW on hand-crafted angles can’t tell techniques apart.
This module tests whether the bottleneck is the *classifier*, the *features*, or
tori/uke *role inconsistency*, with leave-one-out CV on the small dataset:

- feature modes: `primary_angles` (baseline), `tori_angles` (role-consistent —
  tori picked as the person left standing), `tori_angles_pos`.
- methods: `dtw_1nn` (DTW nearest-neighbour), `pool_centroid` / `pool_knn`
  (temporal-pyramid pooled descriptor + nearest-centroid / kNN), and `pool_lda_knn`
  (LDA-reduced + kNN, if scikit-learn is available).

Everything is numpy (sklearn optional), appropriate for ~80 samples where deep nets
would overfit; the same harness scales to more techniques once more data is collected.

### Functions

| [`classification_metrics`](#kodokan.recognize.classification_metrics)(y_true, y_pred, \*[, ...])   | Top-1, balanced (macro-recall), majority-class baseline, n_classes, n.                                    |
|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| [`confusion_pairs`](#kodokan.recognize.confusion_pairs)(y_true, y_pred, \*[, top])          | Most-frequent misclassifications as `{true, pred, count}` (off-diagonal).                                 |
| [`demo_feature`](#kodokan.recognize.demo_feature)(pose_seq, start_s, end_s, \*, mode)    | Per-frame features for one demo under a role/feature mode.                                                |
| `dtw_1nn_accuracy`(feature_seqs, labels)                                                             |                                                                                                           |
| [`dtw_1nn_predict`](#kodokan.recognize.dtw_1nn_predict)(feature_seqs, labels)               | Leave-one-out 1-NN by DTW distance; returns predicted label per item.                                     |
| [`loo_medoid_separability`](#kodokan.recognize.loo_medoid_separability)(feats_by_tech, \*, ...)     | Leak-free (leave-one-demo-out medoid) technique-separability report.                                      |
| [`loo_pooled_predict`](#kodokan.recognize.loo_pooled_predict)(X, y, \*[, method, k, groups])   | Leave-one-out (or leave-one-GROUP-out) predictions; preprocessing fit on train only.                      |
| `pool_centroid_accuracy`(X, y)                                                                       |                                                                                                           |
| `pool_knn_accuracy`(X, y[, k])                                                                       |                                                                                                           |
| `pool_lda_knn_accuracy`(X, y[, k])                                                                   |                                                                                                           |
| [`pooled_descriptor`](#kodokan.recognize.pooled_descriptor)(feat, \*[, levels])               | Temporal-pyramid pooled fixed-length descriptor (mean+std per segment).                                   |
| [`roc_auc_distances`](#kodokan.recognize.roc_auc_distances)(genuine, impostor)                | Standard ROC-AUC for a *distance* score: `P(impostor > genuine)` (ties=0.5).                              |
| [`tori_decision`](#kodokan.recognize.tori_decision)(pose_seq, start_s, end_s, \*[, ...])  | Tori (thrower) heuristic with a confidence margin and an abstain flag.                                    |
| [`tori_decision_stats`](#kodokan.recognize.tori_decision_stats)(decisions)                      | Aggregate fall-back / abstain rates and median margin over decisions.                                     |
| [`tori_index`](#kodokan.recognize.tori_index)(pose_seq, start_s, end_s)                | Heuristic tori slot (thin wrapper over [`tori_decision()`](#kodokan.recognize.tori_decision)). |

### Classes

| [`ToriDecision`](#kodokan.recognize.ToriDecision)(index, margin, fell_back, abstained)   | Outcome of the tori/uke role heuristic, with a confidence margin.   |
|------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------|

### *class* kodokan.recognize.ToriDecision(index, margin, fell_back, abstained)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Outcome of the tori/uke role heuristic, with a confidence margin.

The “role-consistency is the dominant recognition lever” claim rests on this
heuristic, so — per the adversarial review — every call is now instrumented.

#### index

Chosen tori slot (always a usable slot, even when abstaining).

* **Type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

#### margin

Finish-frame hip-y separation between the two judoka in
*torso-length* units (how much lower uke’s hips sit than tori’s).
Larger = more confident; `nan` when the cue was unusable.

* **Type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

#### fell_back

The hip-y cue was unusable, so the most-present slot was used.

* **Type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

#### abstained

Low-confidence decision (`margin` below the abstain
threshold, or a fall-back) — the role label should not be trusted.

* **Type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### kodokan.recognize.classification_metrics(y_true, y_pred, , n_boot=0, seed=0)

Top-1, balanced (macro-recall), majority-class baseline, n_classes, n.

Pass `n_boot > 0` to add 95% percentile-bootstrap confidence intervals
(`top1_ci`/`balanced_ci`) — the rigor the review asked for, so small-n
accuracy differences aren’t over-read. Default `n_boot=0` is unchanged.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### kodokan.recognize.confusion_pairs(y_true, y_pred, , top=None)

Most-frequent misclassifications as `{true, pred, count}` (off-diagonal).

Per-class confusion that stays readable at many classes: instead of a dense
N×N matrix it returns only the non-zero true→pred error cells, sorted by
count. A `pred` of `-1` means an abstain/no-prediction.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### kodokan.recognize.demo_feature(pose_seq, start_s, end_s, , mode)

Per-frame features for one demo under a role/feature mode.

* **Return type:**
  `ndarray`

### kodokan.recognize.dtw_1nn_predict(feature_seqs, labels)

Leave-one-out 1-NN by DTW distance; returns predicted label per item.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)

### kodokan.recognize.loo_medoid_separability(feats_by_tech, , distance)

Leak-free (leave-one-demo-out medoid) technique-separability report.

The earlier feature bake-off built each technique’s medoid reference from
*all* its demos, including the one being tested — so genuine distances were
optimistically low and needed an ad-hoc self-zero filter. Here a held-out
demo’s GENUINE distance is measured to the medoid of its technique’s *other*
demos (true leave-one-out, no self-zero), and its IMPOSTOR distances to every
other technique’s medoid. Reports leave-one-out nearest-medoid accuracy and
the standard ROC-AUC ([`roc_auc_distances()`](#kodokan.recognize.roc_auc_distances)).

* **Parameters:**
  * **feats_by_tech** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)) – per technique, a list of per-demo feature arrays.
  * **distance** – `(feat_a, feat_b) -> float` (e.g. normalized DTW).
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### kodokan.recognize.loo_pooled_predict(X, y, , method='lda_knn', k=3, groups=None)

Leave-one-out (or leave-one-GROUP-out) predictions; preprocessing fit on train only.

If `groups` is given (e.g. source `video_id` per sample), each fold excludes ALL
samples sharing the held-out sample’s group — i.e. **leave-one-clip-out**, which removes
within-clip leakage. Otherwise it is plain leave-one-sample-out.

* **Return type:**
  `ndarray`

### kodokan.recognize.pooled_descriptor(feat, , levels=(1, 2, 4))

Temporal-pyramid pooled fixed-length descriptor (mean+std per segment).

* **Return type:**
  `ndarray`

### kodokan.recognize.roc_auc_distances(genuine, impostor)

Standard ROC-AUC for a *distance* score: `P(impostor > genuine)` (ties=0.5).

Equals the Mann–Whitney U statistic divided by `|genuine|·|impostor|` — the
“standard ROC-AUC” the review asked for (replacing the ad-hoc self-zero-dropping
estimate). Larger = better separation (impostor demos sit farther than genuine
ones). NaNs are dropped; returns `nan` if either side is empty.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### kodokan.recognize.tori_decision(pose_seq, start_s, end_s, , abstain_margin=0.15)

Tori (thrower) heuristic with a confidence margin and an abstain flag.

Tori is the judoka whose hips stay highest (smallest image-y) over the last
third of the demo — uke has been thrown to the ground. The margin is the two
judoka’s finish hip-y separation in torso-length units; below
`abstain_margin` (or when the cue is unusable and we fall back to presence)
the role assignment is unreliable and `abstained` is set. See
[`tori_decision_stats()`](#kodokan.recognize.tori_decision_stats) for aggregate fall-back/abstain rates.

* **Return type:**
  [`ToriDecision`](#kodokan.recognize.ToriDecision)

### kodokan.recognize.tori_decision_stats(decisions)

Aggregate fall-back / abstain rates and median margin over decisions.

The instrumentation the review asked for: run this over every demo’s
[`tori_decision()`](#kodokan.recognize.tori_decision) to see how often the “dominant lever” heuristic is
actually confident vs. abstaining.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### kodokan.recognize.tori_index(pose_seq, start_s, end_s)

Heuristic tori slot (thin wrapper over [`tori_decision()`](#kodokan.recognize.tori_decision)).

Backward-compatible: returns just the chosen slot. Use [`tori_decision()`](#kodokan.recognize.tori_decision)
when you need the confidence margin / abstain flag.

* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)
