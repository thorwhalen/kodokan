"""Acquire Kodokan technique clips from YouTube (thin wrapper over ``yb.download``).

YouTube interaction lives in the ``yb`` package; this module just points it at the
kodokan data dir and the *Kodokan 100 Techniques* playlist, and keeps the source
URL with every clip (a project requirement). The PV (entry #1, "all techniques
once") is skipped from the main line by default but can be fetched as a reference.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from kodokan.config import clips_dir, data_dir

#: The official "KODOKAN × IJF ACADEMY 100 Techniques" playlist.
KODOKAN_PLAYLIST_URL = (
    "https://www.youtube.com/playlist?list=PLtz539PTepc16H2iu5F3Q3D7_He1EYlIQ"
)

#: Registered technique-demo sources (for cross-source / leave-one-clip-out validation).
#: Each is an *independent* set of demonstrations (different demonstrators/cameras), so the
#: SOURCE video can be used as the cross-validation group while ``technique_key`` is the label.
SOURCES = {
    "kodokan_ijf": KODOKAN_PLAYLIST_URL,  # 1 video / technique (our original set)
    "efficient_judo": "https://www.youtube.com/playlist?list=PLwd8pJWYTk07K6hDg2_N-9xd31gUCsew7",  # "67 Throws"
}


def canonical_technique_key(title: str) -> str:
    """Normalize a clip title to a cross-source technique key.

    Handles "kanji / Romaji" (Kodokan) and "Romaji - Demo" (Efficient Judo) titles,
    the ``-barai``/``-harai`` spelling, and hyphen/space differences. E.g. both
    "大外刈 / O-soto-gari" and "Osoto-gari - Demo" -> "osotogari".
    """
    s = title.split("/")[-1].lower()
    s = re.sub(r"\bdemo\b", "", s)
    s = re.sub(r"\bescapes?\b", "", s)  # merge "<pin> Escapes" demos into the technique
    s = s.replace("barai", "harai")
    s = s.replace("siho", "shiho")  # IJF "Kami-siho-gatame" vs "shiho" spelling -> one key
    return re.sub(r"[^a-z0-9]", "", s)


def source_clips_dir(source: str) -> Path:
    """Per-source clips directory (Kodokan keeps the legacy flat ``clips/`` dir)."""
    if source == "kodokan_ijf":
        return clips_dir()
    d = data_dir() / f"clips_{source}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_source(
    source: str,
    *,
    playlist_items: str | None = None,
    download_dir: Path | None = None,
    write_info_json: bool = True,
    **kwargs,
):
    """Download a registered SOURCE playlist into its own clips dir.

    Args:
        source: a key of :data:`SOURCES` (e.g. ``"efficient_judo"``).
        playlist_items: yt-dlp 1-based selector (default: all entries).
        download_dir: override (default: :func:`source_clips_dir`).
        **kwargs: forwarded to ``yb.download.download_youtube_playlist``.
    """
    from yb.download import download_youtube_playlist

    return download_youtube_playlist(
        SOURCES[source],
        download_dir=str(download_dir or source_clips_dir(source)),
        playlist_items=playlist_items,
        write_info_json=write_info_json,
        **kwargs,
    )


def list_techniques(*, include_pv: bool = False, **kwargs):
    """List the playlist's videos (id, title, webpage_url, duration) without downloading.

    Args:
        include_pv: Include entry #1 (the all-techniques PV) if True.
        **kwargs: Forwarded to ``yb.download.youtube_playlist_info``.
    """
    from yb.download import youtube_playlist_info

    items = "1:" if include_pv else "2:"
    return youtube_playlist_info(KODOKAN_PLAYLIST_URL, playlist_items=items, **kwargs)[
        "entries"
    ]


def local_clips(directory: Path | None = None) -> list[dict]:
    """List already-downloaded clips on disk (decouples processing from download).

    Reads each ``*.info.json`` sidecar and pairs it with its ``.mp4``. Returns
    dicts with ``id``, ``title``, ``webpage_url``, ``path``. Sidecars without a
    matching video (e.g. the playlist-level info.json) are skipped.
    """
    directory = Path(directory or clips_dir())
    out: list[dict] = []
    for ij in sorted(directory.glob("*.info.json")):
        base = ij.name[: -len(".info.json")]
        mp4 = directory / f"{base}.mp4"
        if not mp4.exists():
            continue
        info = json.loads(ij.read_text())
        out.append(
            {
                "id": info.get("id"),
                "title": info.get("title"),
                "webpage_url": info.get("webpage_url"),
                "path": mp4,
            }
        )
    return out


def download_techniques(
    *,
    playlist_items: str | None = None,
    skip_pv: bool = True,
    download_dir: Path | None = None,
    write_info_json: bool = True,
    **kwargs,
):
    """Download technique clips into the kodokan clips dir (PV skipped by default).

    Args:
        playlist_items: yt-dlp 1-based selector (overrides ``skip_pv``), e.g.
            ``"2"`` (only Seoi-nage), ``"2:11"`` (the first ten throws), ``"2:"`` (all).
        skip_pv: When ``playlist_items`` is not given, skip entry #1 (the PV).
        download_dir: Destination (default: :func:`kodokan.config.clips_dir`).
        write_info_json: Save each clip's metadata sidecar (keeps the source URL).
        **kwargs: Forwarded to ``yb.download.download_youtube_playlist``.

    Returns:
        ``list[yb.download.DownloadResult]`` — media path + trimmed info + sidecars.
    """
    from yb.download import download_youtube_playlist

    return download_youtube_playlist(
        KODOKAN_PLAYLIST_URL,
        download_dir=str(download_dir or clips_dir()),
        playlist_items=playlist_items,
        skip_first=skip_pv and playlist_items is None,
        write_info_json=write_info_json,
        **kwargs,
    )
