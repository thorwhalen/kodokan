"""Regenerate the web-app clips with the chosen STYLIZED look (issue #39).

Pipeline per clip (one demo repetition of one throw):
  cut segment from the source video (no name-blur needed)
  -> B1 stylization           (cv2.stylization — painterly)
  -> person segmentation       (YOLO11-seg, MPS) → replace background with two FLAT colours
                               (wall/floor sampled per clip) — deletes ALL logos + burned-in text
  -> face transform            (insightface RetinaFace detection, gated to the person mask +
                               short hold; AnimeGANv2 face_paint on a context-padded crop,
                               composited into a feathered ellipse) — a cartoon face, not a blur

Output goes to ``frontend/clips_styl/`` (skip-existing → resumable); after the batch, swap it
into ``frontend/clips/`` and redeploy. Filenames match the existing clips, so throws.json /
catalog.json are unchanged.

Usage::
  KODOKAN_DATA_DIR=~/kodokan_data PYTHONPATH=<repo> python examples/generate_stylized_clips.py [--limit N] [--only KEY]
"""

import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("YOLO_VERBOSE", "False")
os.environ.setdefault("GLOG_minloglevel", "3")
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))  # sibling example modules
# reuse the catalog iteration + source indexing + rep segmentation from the cutter
from generate_webapp_clips import _index_videos, _rep_segments, HEIGHT  # noqa: E402
from kodokan import flashcards as fc  # noqa: E402

APP_DIR = Path("/Users/thorwhalen/Dropbox/py/proj/tt/papp/migrated_apps/kodokan")
OUT_DIR = APP_DIR / "frontend" / "clips_new"  # batch here; swap into clips/ after the run
ANIMEGAN = Path.home() / "kodokan_data" / "style_models" / "face_paint_512_v2_0.onnx"
FF = "/opt/homebrew/bin/ffmpeg"
DEVICE = "mps"
ORT_PROVIDERS = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
# Whether to add a soft blur ON TOP of the AnimeGAN cartoon face (throws only; pins always blur).
# Off by default (initial rollout was anime-only); set BLUR_ANIME_FACES=1 to add the blur. The
# per-clip record of which treatment each clip got lives in the manifest below. See ADR
# misc/docs/adr-video-face-privacy.md.
BLUR_ANIME_FACES = os.environ.get("BLUR_ANIME_FACES", "0") == "1"
MANIFEST = Path(__file__).resolve().parents[1] / "misc" / "docs" / "face_privacy_manifest.json"


# --------------------------------------------------------------------------- #
# models (loaded once)
# --------------------------------------------------------------------------- #

def _load_models():
    from ultralytics import YOLO
    import onnxruntime as ort
    from insightface.app import FaceAnalysis

    seg = YOLO("yolo11n-seg.pt")
    anime = ort.InferenceSession(str(ANIMEGAN), providers=ORT_PROVIDERS)
    det = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"], providers=ORT_PROVIDERS)
    det.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.35)  # lower → catch small/turned faces
    return seg, anime, det


# --------------------------------------------------------------------------- #
# per-frame ops
# --------------------------------------------------------------------------- #

def _b1(frame):
    # cv2.stylization is the per-frame bottleneck after AnimeGAN; run it at half resolution
    # and upscale — the painterly/edge-aware look is smooth, so the difference is invisible.
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    return cv2.resize(cv2.stylization(small, sigma_s=60, sigma_r=0.45), (w, h), interpolation=cv2.INTER_LINEAR)


def _person_instances(fr, seg):
    """Return (union_mask, [per-judoka instance masks]) — per-instance is used for the head
    safety-net so an undetected (head-down) face still gets covered."""
    r = seg.predict(fr, classes=[0], device=DEVICE, verbose=False)[0]
    h, w = fr.shape[:2]
    if r.masks is None:
        z = np.zeros((h, w), np.uint8)
        return z, []
    inst = [cv2.resize((m > 0.5).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            for m in r.masks.data.cpu().numpy()]
    union = np.clip(np.sum(inst, axis=0), 0, 1).astype(np.uint8) if inst else np.zeros((h, w), np.uint8)
    return union, inst


def _head_band(mask):
    """The top ~28% (by row extent) of a person instance mask — where the head is."""
    ys = np.where(mask.any(axis=1))[0]
    if len(ys) == 0:
        return np.zeros_like(mask)
    y0, y1 = ys[0], ys[-1]
    cut = y0 + int((y1 - y0) * 0.28)
    band = mask.copy()
    band[cut:] = 0
    return band


def _flat_bg(frames, seg):
    """Two flat colours (wall top / floor bottom) from the non-person pixels, split at the
    row of largest vertical colour change."""
    h, w = frames[0].shape[:2]
    acc = np.zeros((h, w, 3), np.float32)
    cnt = np.zeros((h, w), np.float32)
    for i in np.linspace(0, len(frames) - 1, min(8, len(frames))).astype(int):
        bgm = _person_instances(frames[i], seg)[0] == 0
        acc[bgm] += frames[i][bgm]
        cnt[bgm] += 1
    cnt[cnt == 0] = 1
    bg = acc / cnt[..., None]
    rowcol = np.median(bg, axis=1)
    lo, hi = int(h * 0.30), int(h * 0.78)
    hy = lo + int(np.argmax(np.linalg.norm(np.diff(rowcol, axis=0), axis=1)[lo:hi]))
    wall = np.median(rowcol[int(h * 0.05):int(h * 0.25)], axis=0)
    floor = np.median(rowcol[int(h * 0.82):int(h * 0.97)], axis=0)
    flat = np.empty((h, w, 3), np.float32)
    flat[:hy] = wall
    flat[hy:] = floor
    return flat.astype(np.uint8)


def _faces(det, seg, fr, pm):
    """RetinaFace boxes, gated to the (dilated) person mask (drops empty-bg false positives)."""
    h, w = fr.shape[:2]
    big = cv2.dilate(pm, np.ones((25, 25), np.uint8))
    out = []
    for f in det.get(fr):
        x0, y0, x1, y1 = f.bbox.astype(int)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        if 0 <= cy < h and 0 <= cx < w and big[cy, cx]:
            out.append((x0, y0, x1, y1))
    return out


def _anime_face(anime):
    iname = anime.get_inputs()[0].name

    def run(bgr):
        rgb = cv2.cvtColor(cv2.resize(bgr, (512, 512)), cv2.COLOR_BGR2RGB).astype(np.float32)
        x = (rgb / 127.5 - 1.0).transpose(2, 0, 1)[None]
        y = anime.run(None, {iname: x})[0][0]
        return cv2.cvtColor(((y.transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    return run


def _ellipse(shape, box, pad=0.12, feather=0.14):
    h, w = shape
    x0, y0, x1, y1 = box
    m = np.zeros((h, w), np.float32)
    cv2.ellipse(m, ((x0 + x1) // 2, (y0 + y1) // 2),
                (int((x1 - x0) / 2 * (1 + pad)), int((y1 - y0) / 2 * (1 + pad))), 0, 0, 360, 1.0, -1)
    return cv2.GaussianBlur(m, (0, 0), max(1.0, feather * (x1 - x0)))


def _clamp(box, w, h, pad):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    return (max(0, int(x0 - bw * pad)), max(0, int(y0 - bh * pad)),
            min(w, int(x1 + bw * pad)), min(h, int(y1 + bh * pad)))


# --------------------------------------------------------------------------- #
# one clip
# --------------------------------------------------------------------------- #

def _read(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, fps


def _blur_bands(comp, bands, h, w):
    """Feathered strong blur over the given float mask region."""
    if bands.max() <= 0:
        return comp
    m = cv2.GaussianBlur(bands, (0, 0), 4)[..., None]
    k = max(9, int(min(h, w) * 0.05) | 1)
    return (comp * (1 - m) + cv2.GaussianBlur(comp, (k, k), 0) * m).astype(np.uint8)


def process_clip(models, src_video, start, dur, out_path, mode="animegan"):
    seg, anime_sess, det = models
    anime = _anime_face(anime_sess)
    tmp = str(out_path) + ".src.mp4"
    subprocess.run([FF, "-y", "-ss", f"{start:.2f}", "-t", f"{dur:.2f}", "-i", str(src_video),
                    "-an", "-vf", f"scale=-2:{HEIGHT}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", "-preset", "veryfast", tmp],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    frames, fps = _read(tmp)
    os.remove(tmp)
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    flat = _flat_bg(frames, seg)
    ker = np.ones((7, 7), np.uint8)
    proc = subprocess.Popen(
        [FF, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}",
         "-i", "-", "-an", "-c:v", "libx264", "-profile:v", "main", "-pix_fmt", "yuv420p",
         "-crf", "27", "-preset", "veryfast", "-movflags", "+faststart", str(out_path)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    # AnimeGAN is ~0.6 s/face — far too slow per-frame. Run it only every ANIME_EVERY frames per
    # face SLOT (faces sorted left→right; ≤2 judoka) and reuse the cached anime tile (resized to
    # the current tracked box) in between. Seg/detect are cheap (~40 ms) but still subsampled.
    ANIME_EVERY, SEG_EVERY = 12, 2  # AnimeGAN cached per slot; seg subsampled. Detect EVERY frame
    slot_cache = {}  # slot -> (anime_bgr_tile, age)
    pm, insts = None, []
    faces, hold = [], 0
    for fi, f in enumerate(frames):
        if pm is None or fi % SEG_EVERY == 0:
            union, insts = _person_instances(f, seg)
            pm = cv2.morphologyEx(union, cv2.MORPH_CLOSE, ker)
        pmf = cv2.GaussianBlur(pm.astype(np.float32), (0, 0), 2)[..., None]
        comp = (_b1(f) * pmf + flat * (1 - pmf)).astype(np.uint8)

        if mode == "blur":
            # Guaranteed coverage for close ground techniques (katame-waza): blur every judoka's
            # head band — no face detection needed, so nothing slips through.
            bands = np.zeros((h, w), np.float32)
            for im in insts:
                bands = np.maximum(bands, _head_band(im).astype(np.float32))
            comp = _blur_bands(comp, bands, h, w)
            proc.stdin.write(np.ascontiguousarray(comp).tobytes())
            continue

        d = _faces(det, seg, f, pm)  # every frame → best coverage of turned/small heads
        if d:
            faces, hold = d, 0
        elif hold < 6:
            hold += 1
        else:
            faces = []
        anime_cov = np.zeros((h, w), np.float32)  # where a face got AnimeGAN'd this frame
        for slot, box in enumerate(sorted(faces, key=lambda b: b[0])):
            cx0, cy0, cx1, cy1 = _clamp(box, w, h, 0.6)
            if cx1 - cx0 < 8 or cy1 - cy0 < 8:
                continue
            tile, age = slot_cache.get(slot, (None, 999))
            if tile is None or age >= ANIME_EVERY:
                tile = anime(f[cy0:cy1, cx0:cx1])
                slot_cache[slot] = (tile, 0)
            else:
                slot_cache[slot] = (tile, age + 1)
            a = cv2.resize(tile, (cx1 - cx0, cy1 - cy0))
            ell = _ellipse((h, w), box)
            m = ell[cy0:cy1, cx0:cx1, None]
            comp[cy0:cy1, cx0:cx1] = (a * m + comp[cy0:cy1, cx0:cx1] * (1 - m)).astype(np.uint8)
            anime_cov = np.maximum(anime_cov, ell)
        # Safety net: any person's head band NOT covered by an AnimeGAN face gets blurred.
        safety = np.zeros((h, w), np.float32)
        for im in insts:
            safety = np.maximum(safety, _head_band(im).astype(np.float32) * (anime_cov < 0.4))
        comp = _blur_bands(comp, safety, h, w)
        if BLUR_ANIME_FACES:  # optional soft blur ON TOP of the anime face (privacy > residual id leak)
            comp = _blur_bands(comp, anime_cov, h, w)
        proc.stdin.write(np.ascontiguousarray(comp).tobytes())
    proc.stdin.close()
    proc.wait()
    return True


# --------------------------------------------------------------------------- #
# batch
# --------------------------------------------------------------------------- #

def main():
    only = None
    limit = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    import json
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vindex = _index_videos()
    catalog = fc.build_catalog()
    # Category decides the face treatment: nage-waza (has a category) → AnimeGAN cartoon face;
    # katame-waza pins (no category, overlapping heads) → guaranteed head-band blur.
    throws = json.loads((APP_DIR / "data" / "throws.json").read_text())["throws"]
    def mode_for(key):
        return "animegan" if (throws.get(key, {}).get("category") or "").strip() else "blur"
    models = _load_models()

    # build the work list (throw, video, rep) with the SAME names as the live clips
    work = []
    for key, entry in catalog.items():
        if only and key != only:
            continue
        m = mode_for(key)
        for c in entry["clips"]:
            vid = c["video_id"]
            reps = _rep_segments(c.get("demos", []))
            if vid not in vindex or not reps:
                continue
            src_file, _ = vindex[vid]
            for i, (start, dur) in enumerate(reps, 1):
                name = f"{vid}.mp4" if i == 1 else f"{vid}_{i}.mp4"
                work.append((key, src_file, start, dur, OUT_DIR / name, m))

    if limit:
        work = work[:limit]
    total = len(work)
    print(f"{total} clips to stylize → {OUT_DIR} (blur_anime={BLUR_ANIME_FACES})", flush=True)

    # Per-clip treatment record (SSOT for 'which clip got which face treatment'). See the ADR.
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"clips": {}}
    manifest.setdefault("clips", {})
    def record(name, key, m):
        manifest["clips"][name] = {"throw": key, "mode": m,
                                   "blur_anime": bool(BLUR_ANIME_FACES and m == "animegan")}
    def flush_manifest():
        MANIFEST.write_text(json.dumps(manifest, indent=1))

    done = fail = skip = 0
    t0 = time.time()
    for n, (key, src, start, dur, out, m) in enumerate(work, 1):
        if out.exists():
            skip += 1
            continue
        try:
            ok = process_clip(models, src, start, dur, out, m)
            done += ok
            fail += (not ok)
            if ok:
                record(out.name, key, m)
        except Exception as e:  # noqa: BLE001 — keep the batch going; one bad clip shouldn't stop it
            fail += 1
            print(f"  ! {out.name} ({key}): {e}", flush=True)
        if n % 10 == 0 or n == total:
            flush_manifest()
            el = time.time() - t0
            rate = (done + fail) / el if el else 0
            print(f"  [{n}/{total}] done={done} skip={skip} fail={fail} "
                  f"| {rate:.2f} clip/s | eta {int((total - n) / rate / 60) if rate else '?'}m", flush=True)
    flush_manifest()
    print(f"\nDONE: {done} made, {skip} skipped, {fail} failed in {int((time.time()-t0)/60)}m", flush=True)


if __name__ == "__main__":
    main()
