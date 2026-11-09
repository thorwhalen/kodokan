"""Word parts of technique names: which techniques share which part, and how often.

A Kodokan technique name is a compound of recurring word parts: *Harai-goshi* is
*harai* (sweep) + *goshi* (hip), *O-soto-gari* is *o* (large) + *soto* (outer) + *gari*
(reap). Learning a dozen of those parts makes a hundred names guessable, which is what the
web app's word-part quizzes train (thorwhalen/kodokan#65).

The same part is spelled differently depending on its position: Japanese *rendaku* voices
the first consonant of a non-initial part, so *koshi* (hip) becomes *goshi* in Harai-goshi
but stays *koshi* in Koshi-guruma; likewise kari/gari, kuruma/guruma, kake/gake,
kaeshi/gaeshi. Those variants are **one** part. A part is keyed by its glossary entry, and
a name token maps to it when it is the entry's primary romaji or one of its declared
alternate readings (the glossary's ``alts``)::

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

Variants are declared, not guessed: rendaku is regular enough to *suggest* a variant but
not to merge one silently (devoicing *de*, "advancing", gives *te*, "hand", a different
word). So :func:`unmatched_tokens` reports the name tokens no part claims, with the part
they are a rendaku variant of, for a person to add as an ``alts`` entry::

    >>> unmatched_tokens({"kaeshiwaza": "Kaeshi-waza"}, glossary)
    {'kaeshi': 'gaeshi', 'waza': None}

:func:`word_throws` derives ``part -> techniques`` by tokenizing the names, rather than
trusting a hand-kept list (which had missed Koshi-guruma under goshi), and
:func:`part_stats` counts the distinct techniques per part. :data:`MIN_TECHNIQUES` is the
threshold for a part to be offered in the quizzes; its docstring records why.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

#: A part is offered in the word-part quizzes when at least this many distinct techniques
#: use it. Chosen from the statistics of the 68 nage-waza (2026-11-09, kodokan#65):
#: 23 parts are shared by >=2 techniques, 20 by >=3, 16 by >=4, 13 by >=5. A part shared by
#: only two techniques (obi, sumi, wakare) is a pair to memorize, not a pattern to spot,
#: while 3 keeps *seoi* (Seoi-nage, Ippon-seoi-nage, Seoi-otoshi), one of the four parts the
#: owner named as the point of the exercise; 4 would drop it. So 3.
MIN_TECHNIQUES = 3

#: Rendaku pairs, voiced -> unvoiced initial (``sh`` before ``z``/``j`` so it wins).
_DEVOICE = (("j", "sh"), ("z", "s"), ("g", "k"), ("d", "t"), ("b", "h"), ("p", "h"))

_SPLIT = re.compile(r"[-\s]+")


def name_tokens(romaji: str) -> list[str]:
    """The lower-cased parts of a romaji name: ``"O-soto-gari"`` -> ``['o', 'soto', 'gari']``."""
    return [t for t in _SPLIT.split(romaji.lower()) if t]


def devoiced(token: str) -> str:
    """The unvoiced (rendaku-undone) form of ``token``, or ``token`` if it starts unvoiced.

    >>> devoiced("gaeshi"), devoiced("jime"), devoiced("barai"), devoiced("ashi")
    ('kaeshi', 'shime', 'harai', 'ashi')
    """
    for voiced, plain in _DEVOICE:
        if token.startswith(voiced):
            return plain + token[len(voiced) :]
    return token


def variant_index(glossary: Mapping[str, Mapping]) -> dict[str, str]:
    """``spelling -> part key`` for every primary key and declared alternate reading.

    ``glossary`` maps a part key (its primary romaji, lower-case) to a mapping that may hold
    ``alts``: alternate readings such as ``["koshi"]`` for ``goshi``.

    :raises ValueError: if one spelling would belong to two parts (glossary order must never
        decide which part a name token is).
    """
    index: dict[str, str] = {}
    pairs = [(k, k) for k in glossary] + [
        (a.lower(), k) for k, e in glossary.items() for a in e.get("alts") or ()
    ]
    for spelling, key in pairs:
        if index.setdefault(spelling, key) != key:
            raise ValueError(
                f"spelling {spelling!r} belongs to both {index[spelling]!r} and {key!r}"
            )
    return index


def unmatched_tokens(
    names: Mapping[str, str], glossary: Mapping[str, Mapping]
) -> dict[str, str | None]:
    """``token -> part it is a rendaku variant of (or None)`` for tokens no part claims."""
    index = variant_index(glossary)
    by_plain = {}
    for spelling, key in index.items():
        by_plain.setdefault(devoiced(spelling), key)
    tokens = {t for romaji in names.values() for t in name_tokens(romaji)}
    return {
        t: by_plain.get(devoiced(t)) for t in sorted(tokens) if t not in index
    }


def parts_of_name(romaji: str, index: Mapping[str, str]) -> list[str]:
    """The distinct part keys of a name, in order of appearance."""
    out: list[str] = []
    for token in name_tokens(romaji):
        key = index.get(token)
        if key and key not in out:
            out.append(key)
    return out


#: Same spelling, different word: technique names where a token is NOT the glossary part.
#: Kata-juji-jime's *kata* is 片 ("one-sided"), not the glossary's 肩 ("shoulder").
HOMOGRAPHS: dict[str, frozenset[str]] = {"kata": frozenset({"katajujijime"})}


def word_throws(
    names: Mapping[str, str],
    glossary: Mapping[str, Mapping],
    *,
    homographs: Mapping[str, Iterable[str]] = HOMOGRAPHS,
) -> dict[str, list[str]]:
    """``part key -> sorted technique keys`` whose name contains the part.

    :param names: ``technique key -> romaji name``.
    :param glossary: as for :func:`variant_index`. Every glossary part appears in the
        result, with an empty list if no name uses it.
    :param homographs: ``part key -> technique keys`` where the spelling is another word.

    >>> word_throws(
    ...     {"kataguruma": "Kata-guruma", "katajujijime": "Kata-juji-jime"},
    ...     {"kata": {}, "guruma": {}},
    ... )
    {'kata': ['kataguruma'], 'guruma': ['kataguruma']}
    """
    index = variant_index(glossary)
    out: dict[str, set[str]] = {key: set() for key in glossary}
    for tech, romaji in names.items():
        for key in parts_of_name(romaji, index):
            if tech not in set(homographs.get(key, ())):
                out[key].add(tech)
    return {key: sorted(techs) for key, techs in out.items()}


def part_stats(
    part_throws: Mapping[str, Iterable[str]],
    *,
    min_techniques: int = MIN_TECHNIQUES,
) -> list[dict]:
    """One row per part, most-shared first: ``{part, n_techniques, offered}``.

    >>> part_stats({"ashi": ["a", "b", "c"], "obi": ["d", "e"], "kani": ["f"]})
    ... # doctest: +NORMALIZE_WHITESPACE
    [{'part': 'ashi', 'n_techniques': 3, 'offered': True},
     {'part': 'obi', 'n_techniques': 2, 'offered': False},
     {'part': 'kani', 'n_techniques': 1, 'offered': False}]
    """
    rows = [
        {"part": key, "n_techniques": n, "offered": n >= min_techniques}
        for key, n in ((k, len(set(v))) for k, v in part_throws.items())
    ]
    return sorted(rows, key=lambda r: (-r["n_techniques"], r["part"]))


def threshold_table(part_throws: Mapping[str, Iterable[str]], *, upto: int = 6) -> dict:
    """``k -> number of parts shared by at least k techniques``, for choosing a threshold."""
    counts = [len(set(v)) for v in part_throws.values()]
    return {k: sum(c >= k for c in counts) for k in range(1, upto + 1)}
