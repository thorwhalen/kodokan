"""Generate self-hosted looping clips for the kodokan web app (with text blurred).

For each technique clip we cut the chosen demo segment from the *downloaded* video,
scale it down, and **gaussian-blur the region where the source burns in the technique
name** (so the flashcard answer isn't readable), producing a small muted MP4 served
by the app as a looping ``<video>`` (no YouTube chrome / pause button). Also writes
``catalog.json`` (names + clip files + YouTube attribution URL + pose-confusability).

Per-source burned-in name locations (fractions of frame), found by inspecting frames:
  - efficient_judo : top-right   (brand watermark is top-left, left alone)
  - kodokan_ijf    : bottom-left white box (Kodokan / IJF logos left alone)

Idempotent: skips clips already generated unless --force. Usage::
    PYTHONPATH=<repo> python examples/generate_webapp_clips.py [--force]
"""

import json
import re
import subprocess
import sys
from pathlib import Path

from kodokan import flashcards as fc
from kodokan.acquire import source_clips_dir

APP_DIR = Path("/Users/thorwhalen/Dropbox/py/proj/tt/papp/migrated_apps/kodokan")
CLIPS_OUT = APP_DIR / "frontend" / "clips"
CATALOG_OUT = APP_DIR / "data" / "catalog.json"
LOOP_MAX_S, LOOP_MIN_S, N_CONFUSABLE, HEIGHT = 7.0, 1.5, 8, 480

# (x, y, w, h) as fractions of the frame — region to blur per source
SOURCE_BLUR = {
    "kodokan_ijf": (0.0, 0.84, 0.34, 0.16),     # bottom-left name box
    "efficient_judo": (0.52, 0.0, 0.48, 0.19),  # top-right name + subtitle
}
SOURCE_DIRS = {  # source -> downloaded-clips dir
    "kodokan_ijf": source_clips_dir("kodokan_ijf"),
    "efficient_judo": source_clips_dir("efficient_judo"),
}
_VID_RE = re.compile(r"\(([A-Za-z0-9_-]{6,})\)\.mp4$")


def _loop_segment(demos):
    for d in demos:
        dur = d["end_s"] - d["start_s"]
        if dur >= LOOP_MIN_S:
            return float(d["start_s"]), min(dur, LOOP_MAX_S)
    if demos:
        return float(demos[0]["start_s"]), LOOP_MAX_S
    return None


def _index_videos():
    """Map videoId -> (filepath, source) by scanning the downloaded-clip dirs."""
    idx = {}
    for source, d in SOURCE_DIRS.items():
        for f in Path(d).glob("*.mp4"):
            m = _VID_RE.search(f.name)
            if m:
                idx[m.group(1)] = (f, source)
    return idx


def _blur_filter(source):
    x, y, w, h = SOURCE_BLUR[source]
    # scale to HEIGHT, then blur a sub-rectangle (fractions via iw/ih) and overlay back
    return (
        f"scale=-2:{HEIGHT},split[a][b];"
        f"[b]crop=iw*{w}:ih*{h}:iw*{x}:ih*{y},gblur=sigma=22[bl];"
        f"[a][bl]overlay=W*{x}:H*{y}"
    )


def _make_clip(src_file, source, start, dur, out_file):
    cmd = [
        "ffmpeg", "-y", "-ss", f"{start:.2f}", "-t", f"{dur:.2f}", "-i", str(src_file),
        "-an", "-vf", _blur_filter(source),
        "-c:v", "libx264", "-profile:v", "main", "-pix_fmt", "yuv420p",
        "-crf", "28", "-preset", "veryfast", "-movflags", "+faststart", str(out_file),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0


def main():
    force = "--force" in sys.argv
    CLIPS_OUT.mkdir(parents=True, exist_ok=True)
    vindex = _index_videos()
    catalog = fc.build_catalog()
    print(f"catalog: {len(catalog)} techniques; {len(vindex)} downloaded videos indexed; "
          f"computing confusability...", flush=True)
    sim = fc.build_confusability(catalog)

    techniques, made, skipped, failed = {}, 0, 0, 0
    for key, entry in catalog.items():
        clips = []
        for c in entry["clips"]:
            vid = c["video_id"]
            seg = _loop_segment(c.get("demos", []))
            if vid not in vindex or seg is None:
                continue
            src_file, source = vindex[vid]
            start, dur = seg
            out_file = CLIPS_OUT / f"{vid}.mp4"
            if force or not out_file.exists():
                if _make_clip(src_file, source, start, dur, out_file):
                    made += 1
                else:
                    failed += 1
                    continue
            else:
                skipped += 1
            clips.append({
                "file": f"clips/{vid}.mp4",
                "videoId": vid,
                "source": source,
                "url": c.get("source_url") or f"https://www.youtube.com/watch?v={vid}",
            })
        if not clips:
            continue
        confusable = [k for k, _ in sorted(sim.get(key, {}).items(), key=lambda kv: -kv[1])[:N_CONFUSABLE]]
        techniques[key] = {
            "name": entry["name"],
            "clips": clips,
            "sources": sorted({c["source"] for c in clips}),
            "confusable": confusable,
        }
        print(f"  {key}: {len(clips)} clip(s)", flush=True)

    CATALOG_OUT.write_text(json.dumps({"n_techniques": len(techniques), "techniques": techniques}, indent=1))
    total_mb = sum(f.stat().st_size for f in CLIPS_OUT.glob("*.mp4")) / 1e6
    print(f"\ndone: {made} made, {skipped} skipped, {failed} failed | "
          f"{len(techniques)} techniques | clips total {total_mb:.0f} MB | wrote {CATALOG_OUT}", flush=True)


if __name__ == "__main__":
    main()
