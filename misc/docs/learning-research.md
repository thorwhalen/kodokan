# Optimal learning for the throw-recognition quiz — research & design

Deep research informing the adaptive flashcard engine (`kodokan/learning.py`,
deployed at apps.thorwhalen.com/kodokan). It answers three questions the project
posed: **what to record**, **how to turn records into the next question**, and
**which study sets** to offer. Findings are evidence-based with references below;
the selection function is a swappable **strategy** the learner picks in settings.

## 1. What the learning science says (and how we apply it)

- **The quiz *is* the learning event (testing effect / retrieval practice).** Active
  retrieval beats restudy for long-term retention [1,2]. → The quiz is the primary
  mode; every item forces a commitment before any reveal.
- **Interleaving look-alikes beats blocking — the most directly applicable result.**
  Learning to recognise painters' *styles* was far better when examples were
  interleaved rather than blocked [4]; judo-throw kinematic signatures are the same
  perceptual-category problem. → Never drill one throw to mastery; mix confusable
  throws within a session.
- **Spacing.** Distributed retrieval >> massed [5]. → Leitner / SM-2 / FSRS schedulers
  space each throw; correct → longer interval, miss → reset.
- **Competitive distractors (and ~3–4 options are enough).** MC trains learning *only*
  when the wrong options are genuinely plausible [6]. → Distractors are drawn from
  **confusable** throws (per-learner confusion matrix, falling back to pose-shape
  similarity), never random.
- **Desirable difficulty / the ~85% sweet spot.** Slightly-too-hard practice helps
  [3]; optimal training accuracy for gradual feedback tasks is ~85% [11]. → Adaptive
  strategies aim to keep the learner near this band (lever: distractor confusability).
- **Feedback is mandatory; immediate is fine** [7]. → Every answer reveals the correct
  throw (and a "watch original" link); also the antidote to the MC false-memory risk [9].
- **Hypercorrection — confident errors are golden** [8], and **pretesting** (guess
  before being taught) helps even when the guess is wrong [10]. → Roadmap: a confidence
  toggle that routes the richest feedback to confident misses, and a pretest mode.

## 2. What we record (per answered problem)

A flat event log (one dict per answer), replayed to reconstruct all adaptive state —
no mutable server state to corrupt:

`response_id, problem_id, user, session_id, strategy_key, target_key, mode,
choice_keys, chosen_key, correct, score, timed_out, response_time_ms,
ts_presented, ts_answered`.

`choice_keys`+`chosen_key` are what make a **per-learner confusion matrix** possible
(which throw was mistaken for which); `response_time_ms` feeds SM-2's grade; timestamps
drive spacing. Derived on the fly: confusion strength (recency-decayed), running
per-item accuracy, Leitner box, SM-2 (EF, n, I), FSRS (stability/difficulty).

## 3. The selectable strategies (record → next question)

Each maps history → (target throw, distractors), both from the active set:

| key | what it does | when to use |
|---|---|---|
| `uniform_random` | equal-probability baseline (anti-repeat) | control / casual |
| `leitner` | 5-box spaced repetition (1,2,4,8,16 d); right promotes, wrong resets | simple & transparent |
| `sm2` | SuperMemo-2 ease/interval; grade from correctness + latency | classic SR |
| `confusion_weighted` *(default)* | resurfaces throws you actually confuse; distractors = your confusers; ε-explore | recognition training |
| `fsrs_lite` | memory model (stability/difficulty), schedules at ~90% recall | modern SR |

Adding a strategy = one class + registry entry → it appears in Settings (open/closed).

## 4. Study sets

No single international belt standard exists, so the **French FFJDA "Progression
française"** is used [15], plus the **Kodokan Gokyo-no-waza** (5 groups) and **Kodokan
nage-waza groups** (te/koshi/ashi/ma-sutemi/yoko-sutemi) as international references.
Mappings to our available throws were adversarially verified; one unverified placement
(seoi-otoshi at green belt) was removed. Learners can also build/save **custom sets**.

## 5. Roadmap (researched, not yet built)

Confidence rating + hypercorrection feedback [8]; pretest/"guess-first" mode [10];
explicit ~85% difficulty controller [11]; annotated side-by-side replay of the confused
pair as feedback; richer partial-credit scoring from the confusion matrix.

## REFERENCES

1. Roediger HL, Karpicke JD. Test-enhanced learning. *Psychol Sci.* 2006. [link](https://learninglab.psych.purdue.edu/downloads/2007/2007_Karpicke_Roediger_JML.pdf)
2. Karpicke JD, Roediger HL. Repeated retrieval during learning. *J Mem Lang.* 2007. [link](https://learninglab.psych.purdue.edu/downloads/2007/2007_Karpicke_Roediger_JML.pdf)
3. Bjork EL, Bjork RA. Making things hard on yourself, but in a good way. 2011. [link](https://bjorklab.psych.ucla.edu/wp-content/uploads/sites/13/2016/04/EBjork_RBjork_2011.pdf)
4. Kornell N, Bjork RA. Learning concepts and categories: is spacing the enemy of induction? *Psychol Sci.* 2008. [link](https://web.williams.edu/Psychology/Faculty/Kornell/Publications/Kornell.Bjork.2008a.pdf)
5. Cepeda NJ, Vul E, Rohrer D, Wixted JT, Pashler H. Spacing effects in learning. *Psychol Sci.* 2008. [link](https://laplab.ucsd.edu/articles/Cepeda%20et%20al%202008_psychsci.pdf)
6. Little JL, Bjork EL. Optimizing multiple-choice tests as tools for learning. *Mem Cognit.* 2014. [link](https://bjorklab.psych.ucla.edu/wp-content/uploads/sites/13/2017/01/LittleBjorkMC2014.pdf)
7. Butler AC, Roediger HL. Feedback enhances the positive effects of MC testing. *Mem Cognit.* 2008. [link](https://gwern.net/doc/psychology/spaced-repetition/2008-butler.pdf)
8. Butterfield B, Metcalfe J. The hypercorrection effect. *Psychon Bull Rev.* 2011. [link](https://link.springer.com/article/10.3758/s13423-011-0173-y)
9. Roediger HL, Marsh EJ. The positive and negative consequences of multiple-choice testing. 2005. [link](https://www.researchgate.net/publication/7517407_The_Positive_and_Negative_Consequences_of_Multiple-Choice_Testing)
10. Kornell N, Hays MJ, Bjork RA. Unsuccessful retrieval attempts enhance subsequent learning. 2009. [link](https://learninglab.uchicago.edu/Pre-Testing_files/RichlandKornellKao.pdf)
11. Wilson RC, Shenhav A, Straccia M, Cohen JD. The eighty-five percent rule for optimal learning. *Nat Commun.* 2019. [link](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6831579/)
12. SuperMemo SM-2 algorithm (original spec). [link](https://super-memory.com/english/ol/sm2.htm)
13. FSRS — Free Spaced Repetition Scheduler. [link](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm)
14. Leitner system. [link](https://en.wikipedia.org/wiki/Leitner_system)
15. FFJDA — Progression française / référentiel technique (France Judo). [link](https://www.ffjudo.com/)
