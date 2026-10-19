# kodokan.acquire

Acquire Kodokan technique clips from YouTube (thin wrapper over `yb.download`).

YouTube interaction lives in the `yb` package; this module just points it at the
kodokan data dir and the *Kodokan 100 Techniques* playlist, and keeps the source
URL with every clip (a project requirement). The PV (entry #1, “all techniques
once”) is skipped from the main line by default but can be fetched as a reference.

### Module Attributes

| [`KODOKAN_PLAYLIST_URL`](#kodokan.acquire.KODOKAN_PLAYLIST_URL)   | The official "KODOKAN × IJF ACADEMY 100 Techniques" playlist.                         |
|-------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| [`SOURCES`](#kodokan.acquire.SOURCES)                | Registered technique-demo sources (for cross-source / leave-one-clip-out validation). |

### Functions

| [`canonical_technique_key`](#kodokan.acquire.canonical_technique_key)(title)                     | Normalize a clip title to a cross-source technique key.                            |
|-----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| [`download_source`](#kodokan.acquire.download_source)(source, \*[, playlist_items, ...]) | Download a registered SOURCE playlist into its own clips dir.                      |
| [`download_techniques`](#kodokan.acquire.download_techniques)(\*[, playlist_items, ...])     | Download technique clips into the kodokan clips dir (PV skipped by default).       |
| [`list_techniques`](#kodokan.acquire.list_techniques)(\*[, include_pv])                  | List the playlist's videos (id, title, webpage_url, duration) without downloading. |
| [`local_clips`](#kodokan.acquire.local_clips)([directory])                           | List already-downloaded clips on disk (decouples processing from download).        |
| [`source_clips_dir`](#kodokan.acquire.source_clips_dir)(source)                           | Per-source clips directory (Kodokan keeps the legacy flat `clips/` dir).           |

### kodokan.acquire.KODOKAN_PLAYLIST_URL *= 'https://www.youtube.com/playlist?list=PLtz539PTepc16H2iu5F3Q3D7_He1EYlIQ'*

The official “KODOKAN × IJF ACADEMY 100 Techniques” playlist.

### kodokan.acquire.SOURCES *= {'efficient_judo': 'https://www.youtube.com/playlist?list=PLwd8pJWYTk07K6hDg2_N-9xd31gUCsew7', 'kodokan_ijf': 'https://www.youtube.com/playlist?list=PLtz539PTepc16H2iu5F3Q3D7_He1EYlIQ'}*

Registered technique-demo sources (for cross-source / leave-one-clip-out validation).
Each is an *independent* set of demonstrations (different demonstrators/cameras), so the
SOURCE video can be used as the cross-validation group while `technique_key` is the label.

### kodokan.acquire.canonical_technique_key(title)

Normalize a clip title to a cross-source technique key.

Handles “kanji / Romaji” (Kodokan) and “Romaji - Demo” (Efficient Judo) titles,
the `-barai`/`-harai` spelling, and hyphen/space differences. E.g. both
“大外刈 / O-soto-gari” and “Osoto-gari - Demo” -> “osotogari”.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### kodokan.acquire.download_source(source, , playlist_items=None, download_dir=None, write_info_json=True, \*\*kwargs)

Download a registered SOURCE playlist into its own clips dir.

* **Parameters:**
  * **source** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – a key of [`SOURCES`](#kodokan.acquire.SOURCES) (e.g. `"efficient_judo"`).
  * **playlist_items** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – yt-dlp 1-based selector (default: all entries).
  * **download_dir** ([`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – override (default: [`source_clips_dir()`](#kodokan.acquire.source_clips_dir)).
  * **\*\*kwargs** – forwarded to `yb.download.download_youtube_playlist`.

### kodokan.acquire.download_techniques(, playlist_items=None, skip_pv=True, download_dir=None, write_info_json=True, \*\*kwargs)

Download technique clips into the kodokan clips dir (PV skipped by default).

* **Parameters:**
  * **playlist_items** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – yt-dlp 1-based selector (overrides `skip_pv`), e.g.
    `"2"` (only Seoi-nage), `"2:11"` (the first ten throws), `"2:"` (all).
  * **skip_pv** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – When `playlist_items` is not given, skip entry #1 (the PV).
  * **download_dir** ([`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Destination (default: [`kodokan.config.clips_dir()`](kodokan.config.html.md#kodokan.config.clips_dir)).
  * **write_info_json** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Save each clip’s metadata sidecar (keeps the source URL).
  * **\*\*kwargs** – Forwarded to `yb.download.download_youtube_playlist`.
* **Returns:**
  `list[yb.download.DownloadResult]` — media path + trimmed info + sidecars.

### kodokan.acquire.list_techniques(, include_pv=False, \*\*kwargs)

List the playlist’s videos (id, title, webpage_url, duration) without downloading.

* **Parameters:**
  * **include_pv** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Include entry #1 (the all-techniques PV) if True.
  * **\*\*kwargs** – Forwarded to `yb.download.youtube_playlist_info`.

### kodokan.acquire.local_clips(directory=None)

List already-downloaded clips on disk (decouples processing from download).

Reads each `*.info.json` sidecar and pairs it with its `.mp4`. Returns
dicts with `id`, `title`, `webpage_url`, `path`. Sidecars without a
matching video (e.g. the playlist-level info.json) are skipped.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### kodokan.acquire.source_clips_dir(source)

Per-source clips directory (Kodokan keeps the legacy flat `clips/` dir).

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
