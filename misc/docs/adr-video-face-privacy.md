# ADR — Video face-privacy treatment (stylized clips)

Status: **Accepted (revisable)** · Context: issue #39 rollout

## Context

The learning app plays short looping demo clips of judo techniques. Before publishing
we stylize every clip (`examples/generate_stylized_clips.py`) to (a) get a consistent
cartoon look and (b) hide the identity of the people in the source footage. Every clip
gets: **B1 painterly body** + **flat two-color background replacement** (which also deletes
all dojo logos/text). The open question is how to treat the **faces**.

Two failure modes shaped the decision:

- **Face transform can't cover what it can't detect.** AnimeGAN needs a detected face.
  On close **ground techniques (katame-waza — pins/chokes)** two heads overlap and one is
  usually tilted down, so RetinaFace misses it and the real face would be left exposed.
- **AnimeGAN preserves facial structure.** The cartoon face is a stylized version of the
  *real* face, so identity partially leaks. A blur on top removes that residual.

## Decision

Treatment is chosen **per technique category**:

| Category | Treatment | Why |
|---|---|---|
| **Nage-waza** (standing throws) | **AnimeGAN cartoon face** | Faces are frontal enough to detect; the anime look is the intended aesthetic. |
| **Katame-waza** (pins/chokes) | **Head-band blur** (per-person YOLO mask, no detection needed) | Guarantees coverage where detection fails. |

Category is read from `data/throws.json` (`category` present → throw → animegan; empty → pin → blur).

### Open sub-decision: blur the anime faces too?

Being evaluated. There are now **two throw variants live** so we can compare:

- **`anime_only`** — AnimeGAN face, no blur (the **initial 284 throw clips**, processed first).
- **`anime_plus_blur`** — AnimeGAN face **with a soft blur on top** (throws processed **after
  2026-07-04**, once this question was raised).

Pins are `blur` in both variants. After review we'll pick one and make it uniform.

## How to change later

The per-clip record is **`misc/docs/face_privacy_manifest.json`** (keyed by clip filename →
`{throw, mode, blur_anime}`). The generator maintains it.

- **Make all throws `anime_plus_blur`** (add blur to the initial 284): a **cheap post-pass** —
  re-detect + blur the face region on the *already-stylized* clips; **no AnimeGAN re-run** needed
  (~seconds/clip). Or re-run the full batch with `BLUR_ANIME_FACES=1` after deleting the 284.
- **Revert to `anime_only` everywhere**: re-run the batch with `BLUR_ANIME_FACES=0`.
- **Change the throw/pin split**: it's driven entirely by `throws.json` category — no code change.

The knob in the generator is the module flag **`BLUR_ANIME_FACES`** (env `BLUR_ANIME_FACES=1|0`).

## Consequences

- The deployed app is temporarily **mixed** (284 throws anime-only, later throws anime+blur)
  by design, to enable side-by-side comparison. This is intentional and reversible.
- Nothing is ever left fully exposed: pins always blur, throws always get at least the anime face.
