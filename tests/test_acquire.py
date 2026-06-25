"""Tests for cross-source acquisition helpers (canonical technique keys)."""

from kodokan.acquire import canonical_technique_key as ck


def test_canonical_key_cross_source_match():
    # Kodokan "kanji / Romaji" vs Efficient Judo "Romaji - Demo" must collapse to one key
    assert ck("大外刈 / O-soto-gari") == ck("Osoto-gari - Demo") == "osotogari"
    assert ck("背負投 / Seoi-nage") == ck("Seoi-nage - Demo") == "seoinage"
    # -barai / -harai spelling unified
    assert ck("De-ashi-barai - Demo") == ck("出足払 / De-ashi-harai") == "deashiharai"


def test_canonical_key_distinguishes_techniques():
    assert ck("O-goshi") != ck("O-soto-gari")
    assert ck("Uchi-mata") == "uchimata"
