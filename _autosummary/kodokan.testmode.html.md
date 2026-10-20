# kodokan.testmode

Technique test mode: perform the technique the system names, get scored (#5).

[`TechniqueTest`](#kodokan.testmode.TechniqueTest) draws a technique at random from a focus pool (all the
recognizer knows by default), names it, and grades the next performance on two
things: how confidently the recognizer sees the *target* in it, and, when a reference
for the target is given, how close it is to that reference (`kodokan.coach.coach()`,
with its advice). `TechniqueTest.push()` runs the same loop on a live pose stream.

```pycon
>>> import numpy as np
>>> from kodokan.recognizer import TechniqueRecognizer
>>> X = np.vstack([np.full((3, 4), c) + np.arange(3)[:, None] * 0.1 for c in (0.0, 3.0)])
>>> rec = TechniqueRecognizer(method="centroid").fit(X, ["o-goshi"] * 3 + ["uchi-mata"] * 3)
>>> test = TechniqueTest(rec, pool=["uchi-mata"], seed=1)
>>> test.next_prompt()
'uchi-mata'
>>> result = test.grade(np.full(4, 3.1))
>>> result.passed, result.score > 50
(True, True)
```

### Classes

| [`LiveTechniqueTest`](#kodokan.testmode.LiveTechniqueTest)(test, \*, fps[, featurize, ...])   | The test loop on a live stream: prompt, detect the throw, grade, prompt again.   |
|-------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| [`TechniqueTest`](#kodokan.testmode.TechniqueTest)(recognizer, \*[, pool, ...])           | Prompt a random technique from a focus pool; grade the performance.              |
| [`TestResult`](#kodokan.testmode.TestResult)(target, recognition, ...)                 | The grade of one performance of the prompted technique.                          |

### *class* kodokan.testmode.LiveTechniqueTest(test, \*, fps, featurize=<function pooled_demo_descriptor>, angle_featurize=<function demo_features>, on_prompt=None, \*\*detector_kwargs)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The test loop on a live stream: prompt, detect the throw, grade, prompt again.

[`push()`](#kodokan.testmode.LiveTechniqueTest.push) takes one frame and returns `(segment, result)` for each throw that
closed with it; after each graded throw the next technique is drawn
(`prompt`). Detector options pass through.

A throw is graded once it has *closed* (`close_after_s` after it ends), so the
performer should wait for the next prompt before the next throw; any motion the
detector takes for a throw also uses up a prompt.

#### push(keypoints, frame_index=None)

Add one frame; grade each throw it closed and draw the next prompt.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`Segment`](kodokan.segment.html.md#kodokan.segment.Segment), [`TestResult`](#kodokan.testmode.TestResult)]]

### *class* kodokan.testmode.TechniqueTest(recognizer, , pool=None, references=None, lang='en', confidence_weight=0.5, seed=None, rng=None, advice={'en': {'close': 'Close to the reference: keep it up.', 'elbow': "Your {side} elbow is {deg:.0f}° off the reference: keep that arm's pull firm and close to your body.", 'headline': '{technique}: {score}/100.', 'hip': 'Your {side} hip is {deg:.0f}° off: bring your hips lower and closer to your partner.', 'kake': 'Most of the difference is in kake: commit through the finish.', 'knee': 'Your {side} knee is {deg:.0f}° off: bend your knees more as you enter, then drive up.', 'kuzushi': "Most of the difference is in kuzushi: break your partner's balance before you turn in.", 'shoulder': "Your {side} shoulder is {deg:.0f}° off: lift and turn that arm with the reference timing to take your partner's balance.", 'tsukuri': 'Most of the difference is in tsukuri: step in faster and closer.'}, 'fr': {'close': 'Proche de la référence : continuez ainsi.', 'elbow': "Votre coude {side} s'écarte de {deg:.0f}° de la référence : gardez la traction de ce bras ferme et près du corps.", 'headline': '{technique} : {score}/100.', 'hip': "Votre hanche {side_f} s'écarte de {deg:.0f}° : placez vos hanches plus bas et plus près de votre partenaire.", 'kake': "L'essentiel de l'écart est dans le kake : allez au bout de la projection.", 'knee': "Votre genou {side} s'écarte de {deg:.0f}° : fléchissez davantage les genoux en entrant, puis poussez vers le haut.", 'kuzushi': "L'essentiel de l'écart est dans le kuzushi : déséquilibrez votre partenaire avant d'entrer.", 'shoulder': "Votre épaule {side_f} s'écarte de {deg:.0f}° : levez et tournez ce bras au rythme de la référence pour déséquilibrer votre partenaire.", 'tsukuri': "L'essentiel de l'écart est dans le tsukuri : entrez plus vite et plus près."}})

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Prompt a random technique from a focus pool; grade the performance.

* **Parameters:**
  * **recognizer** ([`TechniqueRecognizer`](kodokan.recognizer.html.md#kodokan.recognizer.TechniqueRecognizer)) – a fitted [`TechniqueRecognizer`](kodokan.recognizer.html.md#kodokan.recognizer.TechniqueRecognizer).
  * **pool** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – the techniques to draw from (default: all the recognizer knows).
  * **references** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – `{technique: build_reference(...)}`; techniques with one are
    also graded on closeness, with advice.
  * **confidence_weight** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – share of the score from the target’s confidence (the
    rest from closeness to the reference, when there is one).
  * **seed** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – or `rng`, for a reproducible sequence of prompts.

#### grade(descriptor, angle_features=None)

Grade one performance: its pooled `descriptor` and, for closeness, its
per-frame `angle_features` (`kodokan.score.demo_features()`).

* **Return type:**
  [`TestResult`](#kodokan.testmode.TestResult)

#### grade_clip(pose_seq, start_s, end_s, \*, featurize=<function pooled_demo_descriptor>, angle_featurize=<function demo_features>)

Grade the performance in `pose_seq` between `start_s` and `end_s`.

* **Return type:**
  [`TestResult`](#kodokan.testmode.TestResult)

#### next_prompt()

Draw the next technique to perform (never the same one twice in a row).

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### *class* kodokan.testmode.TestResult(target, recognition, target_confidence, coaching, score)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The grade of one performance of the prompted technique.

#### *property* passed *: [bool](https://docs.python.org/3/builtins/functions.html#bool)*

Whether the recognizer’s best guess is the target.
