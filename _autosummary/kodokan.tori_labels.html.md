# kodokan.tori_labels

Hand-labelling of the tori heuristic (#49): sample demos, then score agreement.

`recognize.tori_index` decides which tracked judoka is tori (the thrower) from hip
heights alone. There is no ground truth to validate it against, so the owner labels a
stratified sample of demos by eye (in the web app’s owner-only labelling page) and this
module closes the loop:

* [`build_label_manifest()`](#kodokan.tori_labels.build_label_manifest) – pick ~60 demos (stratified over throw, source and the
  confident/abstain split) and emit, per demo, the two tracked people’s boxes over time,
  normalised to the frame so the page can outline them on the clip in two colours. The
  heuristic’s own answer is stored but the app never sends it to the labelling page.
* [`agreement_report()`](#kodokan.tori_labels.agreement_report) – given the owner’s labels, the agreement with the heuristic
  (Wilson CI), split by confident/abstained, plus the “can’t tell” rate in each.

Command line:

```default
python -m kodokan.tori_labels build manifest.json --n 60
python -m kodokan.tori_labels report manifest.json labels.jsonl
```

Needs the pose and segment stores (`~/kodokan_data`) for `build` only; `report`
and the pure helpers need nothing but the two files.

### Module Attributes

| [`VERDICTS`](#kodokan.tori_labels.VERDICTS)          | The verdicts the labelling page records.                                 |
|--------------------------------------------------------------------|--------------------------------------------------------------------------|
| [`MIN_KEYPOINT_CONF`](#kodokan.tori_labels.MIN_KEYPOINT_CONF) | Minimum keypoint confidence for a joint to count towards a person's box. |
| [`BOX_PAD`](#kodokan.tori_labels.BOX_PAD)           | Padding added around the joints' extent, as a fraction of the box size.  |

### Functions

| [`agreement_report`](#kodokan.tori_labels.agreement_report)(manifest, labels)                 | Agreement of `tori_index` with the owner's labels.                                 |
|-----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| [`build_label_manifest`](#kodokan.tori_labels.build_label_manifest)(\*[, n, seed, abstain_share]) | Sample the demos of the local pose/segment stores into a labelling manifest.       |
| [`labelled_slots`](#kodokan.tori_labels.labelled_slots)(pose_seq, start_s, end_s)           | The two person slots shown to the labeller: the two most present in the window.    |
| [`latest_labels`](#kodokan.tori_labels.latest_labels)(rows)                                | The most recent label per demo id (the owner may re-label).                        |
| `main`([argv])                                                                                      |                                                                                    |
| [`manifest_entry`](#kodokan.tori_labels.manifest_entry)(\*, throw, video_id, source, ...)   | One labelling item for a demo, or `None` if it does not have two trackable people. |
| [`person_boxes`](#kodokan.tori_labels.person_boxes)(pose_seq, start_s, end_s, \*, slots)  | Normalised boxes of the given person *slots* over a demo window.                   |
| [`stratified_sample`](#kodokan.tori_labels.stratified_sample)(entries[, n, ...])               | Pick *n* entries spread over throws and sources, with a guaranteed abstain share.  |
| [`wilson_interval`](#kodokan.tori_labels.wilson_interval)(k, n, \*[, z])                     | Wilson score interval for a proportion (95% by default).                           |

### kodokan.tori_labels.BOX_PAD *= 0.08*

Padding added around the joints’ extent, as a fraction of the box size.

### kodokan.tori_labels.MIN_KEYPOINT_CONF *= 0.1*

Minimum keypoint confidence for a joint to count towards a person’s box.

### kodokan.tori_labels.VERDICTS *= ('A', 'B', 'cant_tell')*

The verdicts the labelling page records.

### kodokan.tori_labels.agreement_report(manifest, labels)

Agreement of `tori_index` with the owner’s labels.

A decided label (“A”/”B”) agrees when the named slot equals the heuristic’s. “Can’t tell”
is excluded from agreement and reported as a rate, overall and per confident/abstained
group – if abstentions are the genuinely ambiguous demos, their can’t-tell rate is higher.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### kodokan.tori_labels.build_label_manifest(, n=60, seed=0, abstain_share=0.4)

Sample the demos of the local pose/segment stores into a labelling manifest.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### kodokan.tori_labels.labelled_slots(pose_seq, start_s, end_s)

The two person slots shown to the labeller: the two most present in the window.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]

### kodokan.tori_labels.latest_labels(rows)

The most recent label per demo id (the owner may re-label).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### kodokan.tori_labels.manifest_entry(, throw, video_id, source, pose_seq, start_s, end_s, clip_file)

One labelling item for a demo, or `None` if it does not have two trackable people.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict) | [`None`](https://docs.python.org/3/builtins/constants.html#None)

### kodokan.tori_labels.person_boxes(pose_seq, start_s, end_s, , slots, min_conf=0.1, pad=0.08, max_frames=120)

Normalised boxes of the given person *slots* over a demo window.

Returns `{"t": [seconds from window start], "boxes": [[ [x0,y0,x1,y1] | None per slot ] per frame]}`
with coordinates as fractions of the frame (so they scale to any rendition of the clip).
Frames are thinned to at most *max_frames*; a slot with no usable joints in a frame is `None`.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### kodokan.tori_labels.stratified_sample(entries, n=60, , abstain_share=0.4, seed=0)

Pick *n* entries spread over throws and sources, with a guaranteed abstain share.

The abstained demos are the ones whose role label is in doubt, so they get at least
`abstain_share` of the sample (fewer only if there are not that many). Within each
half, strata are `(throw, source)` cells drawn round-robin from a seeded shuffle, so
no throw or source dominates.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### kodokan.tori_labels.wilson_interval(k, n, , z=1.96)

Wilson score interval for a proportion (95% by default).

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]
