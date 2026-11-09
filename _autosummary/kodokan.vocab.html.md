# kodokan.vocab

Word parts of technique names: which techniques share which part, and how often.

A Kodokan technique name is a compound of recurring word parts: *Harai-goshi* is
*harai* (sweep) + *goshi* (hip), *O-soto-gari* is *o* (large) + *soto* (outer) + *gari*
(reap). Learning a dozen of those parts makes a hundred names guessable, which is what the
web app’s word-part quizzes train (thorwhalen/kodokan#65).

The same part is spelled differently depending on its position: Japanese *rendaku* voices
the first consonant of a non-initial part, so *koshi* (hip) becomes *goshi* in Harai-goshi
but stays *koshi* in Koshi-guruma; likewise kari/gari, kuruma/guruma, kake/gake,
kaeshi/gaeshi. Those variants are **one** part. A part is keyed by its glossary entry, and
a name token maps to it when it is the entry’s primary romaji or one of its declared
alternate readings (the glossary’s `alts`):

```default
>>> glossary = {
...     "goshi": {"alts": ["koshi"]},
...     "guruma": {"alts": ["kuruma"]},
...     "harai": {"alts": ["barai"]},
...     "gaeshi": {"alts": []},
... }
>>> index = variant_index(glossary)
>>> [index.get(t) for t in ("koshi", "goshi", "barai", "kaeshi", "nage")]
['goshi', 'goshi', 'harai', None, None]
>>> parts_of_name("Koshi-guruma", index)
['goshi', 'guruma']
```

Variants are declared, not guessed: rendaku is regular enough to *suggest* a variant but
not to merge one silently (devoicing *de*, “advancing”, gives *te*, “hand”, a different
word). So [`unmatched_tokens()`](#kodokan.vocab.unmatched_tokens) reports the name tokens no part claims, with the part
they are a rendaku variant of, for a person to add as an `alts` entry:

```default
>>> unmatched_tokens({"kaeshiwaza": "Kaeshi-waza"}, glossary)
{'kaeshi': 'gaeshi', 'waza': None}
```

[`word_throws()`](#kodokan.vocab.word_throws) derives `part -> techniques` by tokenizing the names, rather than
trusting a hand-kept list (which had missed Koshi-guruma under goshi), and
[`part_stats()`](#kodokan.vocab.part_stats) counts the distinct techniques per part. [`MIN_TECHNIQUES`](#kodokan.vocab.MIN_TECHNIQUES) is the
threshold for a part to be offered in the quizzes; its docstring records why.

### Module Attributes

| [`MIN_TECHNIQUES`](#kodokan.vocab.MIN_TECHNIQUES)   | A part is offered in the word-part quizzes when at least this many distinct techniques use it.   |
|-------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| [`HOMOGRAPHS`](#kodokan.vocab.HOMOGRAPHS)       | technique names where a token is NOT the glossary part.                                          |

### Functions

| [`devoiced`](#kodokan.vocab.devoiced)(token)                                | The unvoiced (rendaku-undone) form of `token`, or `token` if it starts unvoiced.    |
|-------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| [`name_tokens`](#kodokan.vocab.name_tokens)(romaji)                            | The lower-cased parts of a romaji name: `"O-soto-gari"` -> `['o', 'soto', 'gari']`. |
| [`part_stats`](#kodokan.vocab.part_stats)(part_throws, \*[, min_techniques])  | One row per part, most-shared first: `{part, n_techniques, offered}`.               |
| [`parts_of_name`](#kodokan.vocab.parts_of_name)(romaji, index)                   | The distinct part keys of a name, in order of appearance.                           |
| [`threshold_table`](#kodokan.vocab.threshold_table)(part_throws, \*[, upto])       | `k -> number of parts shared by at least k techniques`, for choosing a threshold.   |
| [`unmatched_tokens`](#kodokan.vocab.unmatched_tokens)(names, glossary)              | `token -> part it is a rendaku variant of (or None)` for tokens no part claims.     |
| [`variant_index`](#kodokan.vocab.variant_index)(glossary)                        | `spelling -> part key` for every primary key and declared alternate reading.        |
| [`word_throws`](#kodokan.vocab.word_throws)(names, glossary, \*[, homographs]) | `part key -> sorted technique keys` whose name contains the part.                   |

### kodokan.vocab.HOMOGRAPHS *: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [frozenset](https://docs.python.org/3/builtins/stdtypes.html#frozenset)[[str](https://docs.python.org/3/builtins/stdtypes.html#str)]]* *= {'kata': frozenset({'katajujijime'})}*

technique names where a token is NOT the glossary part.
Kata-juji-jime’s *kata* is 片 (“one-sided”), not the glossary’s 肩 (“shoulder”).

* **Type:**
  Same spelling, different word

### kodokan.vocab.MIN_TECHNIQUES *= 3*

A part is offered in the word-part quizzes when at least this many distinct techniques
use it. Chosen from the statistics of the 68 nage-waza (2026-11-09, kodokan#65):
23 parts are shared by >=2 techniques, 20 by >=3, 16 by >=4, 13 by >=5. A part shared by
only two techniques (obi, sumi, wakare) is a pair to memorize, not a pattern to spot,
while 3 keeps *seoi* (Seoi-nage, Ippon-seoi-nage, Seoi-otoshi), one of the four parts the
owner named as the point of the exercise; 4 would drop it. So 3.

### kodokan.vocab.devoiced(token)

The unvoiced (rendaku-undone) form of `token`, or `token` if it starts unvoiced.

```pycon
>>> devoiced("gaeshi"), devoiced("jime"), devoiced("barai"), devoiced("ashi")
('kaeshi', 'shime', 'harai', 'ashi')
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### kodokan.vocab.name_tokens(romaji)

The lower-cased parts of a romaji name: `"O-soto-gari"` -> `['o', 'soto', 'gari']`.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### kodokan.vocab.part_stats(part_throws, , min_techniques=3)

One row per part, most-shared first: `{part, n_techniques, offered}`.

```pycon
>>> part_stats({"ashi": ["a", "b", "c"], "obi": ["d", "e"], "kani": ["f"]})
...
[{'part': 'ashi', 'n_techniques': 3, 'offered': True},
 {'part': 'obi', 'n_techniques': 2, 'offered': False},
 {'part': 'kani', 'n_techniques': 1, 'offered': False}]
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### kodokan.vocab.parts_of_name(romaji, index)

The distinct part keys of a name, in order of appearance.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### kodokan.vocab.threshold_table(part_throws, , upto=6)

`k -> number of parts shared by at least k techniques`, for choosing a threshold.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### kodokan.vocab.unmatched_tokens(names, glossary)

`token -> part it is a rendaku variant of (or None)` for tokens no part claims.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)]

### kodokan.vocab.variant_index(glossary)

`spelling -> part key` for every primary key and declared alternate reading.

`glossary` maps a part key (its primary romaji, lower-case) to a mapping that may hold
`alts`: alternate readings such as `["koshi"]` for `goshi`.

* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if one spelling would belong to two parts (glossary order must never
  decide which part a name token is).
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### kodokan.vocab.word_throws(names, glossary, , homographs={'kata': frozenset({'katajujijime'})})

`part key -> sorted technique keys` whose name contains the part.

* **Parameters:**
  * **names** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – `technique key -> romaji name`.
  * **glossary** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)]) – as for [`variant_index()`](#kodokan.vocab.variant_index). Every glossary part appears in the
    result, with an empty list if no name uses it.
  * **homographs** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – `part key -> technique keys` where the spelling is another word.
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]

```pycon
>>> word_throws(
...     {"kataguruma": "Kata-guruma", "katajujijime": "Kata-juji-jime"},
...     {"kata": {}, "guruma": {}},
... )
{'kata': ['kataguruma'], 'guruma': ['kataguruma']}
```
