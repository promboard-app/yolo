"""README 그림 · 움짤을 만든다(make_results.py 결과로).

사용(저장소 루트에서):
    python -I -B docs/scripts/make_visuals.py <작업 폴더(make_results 결과)> docs <ko|en> [hero,pose,objects,confidence,gifs]
언어별 그림(hero · pose · objects · confidence .jpg)은 docs/<언어>/ 에, 글자 없는 움짤(video-pose.gif · video-objects.gif)은
두 언어가 같이 쓰도록 docs/ 에 쓴다. 움짤은 언어와 상관없으니 한 번만 만들면 된다.
"""
import os
import sys

from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
work, docs, lang = sys.argv[1:4]
res_dir = os.path.join(work, "results")
out = os.path.join(docs, lang)
os.makedirs(out, exist_ok=True)

N = {
    "ko": {"photo": "사진", "pose": "포즈 뼈대", "crop": "물체 잘라내기", "cutout": "물체 오려내기", "found": "찾은 것 표시",
           "images": "사진 결과 {n}장", "conf": "찾는 기준", "loose": "넉넉하게", "normal": "보통", "strict": "엄격하게"},
    "en": {"photo": "Photo", "pose": "Pose skeleton", "crop": "Crop objects", "cutout": "Cut out objects",
           "found": "Detections", "images": "Photo results · {n}", "conf": "Detection threshold", "loose": "Loose",
           "normal": "Normal", "strict": "Strict"},
}[lang]


def img(*p):
    return Image.open(os.path.join(res_dir, *p))


def photo(name):
    return Image.open(os.path.join(ROOT, "cards", name + ".jpg")).convert("RGB")


def results(folder):
    fs = sorted(f for f in os.listdir(os.path.join(res_dir, folder)) if f[:2].isdigit() and f.endswith(".png"))
    return [img(folder, f) for f in fs]


def thumb(im, size, cutout=False):
    """잘라낸 결과를 정사각 칸 가운데에. 투명 PNG 는 체크무늬 위에."""
    tile = C.checker((size, size), cell=10) if cutout else Image.new("RGB", (size, size), C.STAGE)
    t = C.contain(im, size - 8, size - 8)
    x, y = (size - t.width) // 2, (size - t.height) // 2
    if t.mode == "RGBA":
        tile.paste(t, (x, y), t)
    else:
        tile.paste(t.convert("RGB"), (x, y))
    return tile


def save(canvas, name):
    canvas.save(os.path.join(out, name), quality=88, optimize=True, progressive=True)


# ── 대표 그림: 사진 → 포즈 뼈대, 사진 → 찾은 것 표시 + 잘라 낸 사진 ─────────────────
def hero():
    W, pad, tw, th, head = 1600, 24, 384, 576, 58
    canvas = Image.new("RGB", (W, pad + head + th + pad), C.BG)
    d = ImageDraw.Draw(canvas)
    C.pill(d, (pad, pad), N["pose"], size=24, active=True)
    a = C.label_on_image(C.fit(photo("dancer"), tw, th), N["photo"], size=20, pos="lb", margin=14)
    b = C.label_on_image(C.fit(img("dancer-pose", "pose.png").convert("RGB"), tw, th), N["pose"], size=20, pos="lb",
                         margin=14, active=True)
    y = pad + head
    C.paste_round(canvas, a, (pad, y), 12)
    C.arrow(d, pad + tw + 12, y + th // 2, pad + tw + 44, color=C.TEXT)
    C.paste_round(canvas, b, (pad + tw + 56, y), 12)
    x0 = pad + tw * 2 + 56 + 40
    rw = W - pad - x0
    C.pill(d, (x0, pad), N["crop"], size=24, active=True)
    crops = results("desk-crop")
    n, gap = len(crops), 8
    ts = (rw - gap * (n - 1)) // n
    fh = th - 44 - ts
    f = C.label_on_image(C.fit(img("desk-crop", "found.jpg").convert("RGB"), rw, fh), N["found"], size=20, pos="lb",
                         margin=14)
    C.paste_round(canvas, f, (x0, y), 12)
    d.text((x0, y + fh + 12), N["images"].format(n=n), font=C.font(20), fill=C.PURPLE_TEXT)
    for i, c in enumerate(crops):
        C.paste_round(canvas, thumb(c, ts), (x0 + i * (ts + gap), y + fh + 44), 8)
    save(canvas, "hero.jpg")


# ── 포즈 뼈대 예시 ───────────────────────────────────────────────────────────
def pose():
    pad, tw, th, W = 24, 350, 525, 1600
    canvas = Image.new("RGB", (W, pad * 2 + th), C.BG)
    d = ImageDraw.Draw(canvas)
    x = pad
    for name in ("yoga", "friends"):
        a = C.label_on_image(C.fit(photo(name), tw, th), N["photo"], size=20, pos="lb", margin=14)
        b = C.label_on_image(C.fit(img(f"{name}-pose", "pose.png").convert("RGB"), tw, th), N["pose"], size=20,
                             pos="lb", margin=14, active=True)
        C.paste_round(canvas, a, (x, pad), 12)
        C.arrow(d, x + tw + 10, pad + th // 2, x + tw + 42, color=C.TEXT)
        C.paste_round(canvas, b, (x + tw + 52, pad), 12)
        x += tw * 2 + 52 + 48
    save(canvas, "pose.jpg")


# ── 물체 잘라내기 · 오려내기 ─────────────────────────────────────────────────
def objects():
    pad, fw, fh, W = 24, 720, 480, 1600
    rx = pad + fw + 36
    rw = W - pad - rx
    rows = [("crop", results("park-crop"), False), ("cutout", results("park-cutout"), True)]
    n, gap = max(len(r[1]) for r in rows), 10
    ts = (rw - gap * (n - 1)) // n
    canvas = Image.new("RGB", (W, pad * 2 + fh), C.BG)
    d = ImageDraw.Draw(canvas)
    f = C.label_on_image(C.fit(img("park-crop", "found.jpg").convert("RGB"), fw, fh), N["found"], size=20, pos="lb",
                         margin=14)
    C.paste_round(canvas, f, (pad, pad), 12)
    C.arrow(d, pad + fw + 6, pad + fh // 2, pad + fw + 30, color=C.TEXT)
    block = 48 + ts
    y = pad + (fh - block * 2 - 28) // 2
    for mode, ims, cut in rows:
        C.pill(d, (rx, y), N[mode], size=20, active=True)
        for i, c in enumerate(ims):
            C.paste_round(canvas, thumb(c, ts, cutout=cut), (rx + i * (ts + gap), y + 48), 8)
        y += block + 28
    save(canvas, "objects.jpg")


# ── 찾는 기준 ────────────────────────────────────────────────────────────────
def confidence():
    pad, gap, W, head = 24, 16, 1600, 56
    tw = (W - pad * 2 - gap * 2) // 3
    th = tw * 2 // 3
    canvas = Image.new("RGB", (W, pad * 2 + head + th), C.BG)
    d = ImageDraw.Draw(canvas)
    for k, (c, folder) in enumerate([("loose", "desk-crop-loose"), ("normal", "desk-crop"), ("strict", "desk-crop-strict")]):
        x = pad + k * (tw + gap)
        C.pill(d, (x, pad), f"{N['conf']} · {N[c]}", size=20, active=(c == "normal"))
        lab = N["images"].format(n=len(results(folder)))
        lw, _ = C.text_size(d, lab, C.font(18, "Regular"))
        d.text((x + tw - lw, pad + 12), lab, font=C.font(18, "Regular"), fill=C.MUTED)
        C.paste_round(canvas, C.fit(img(folder, "found.jpg").convert("RGB"), tw, th), (x, pad + head), 10)
    save(canvas, "confidence.jpg")


# ── 움짤(글자 없음): 왼쪽 · 오른쪽 영상을 나란히 ──────────────────────────────────
def video_gif(name, left, right, pw, every, count, colors=64):
    a = C.read_frames(left, every=every, count=count)
    b = C.read_frames(right, every=every, count=count)
    ph = round(a[0].height * pw / a[0].width)
    pad = 12
    W, H = pad * 3 + pw * 2, pad * 2 + ph
    frames, prev = [], None
    for x, y in zip(a, b):
        fr = Image.new("RGB", (W, H), C.BG)
        # 옛 필름의 낟알(프레임마다 바뀌는 잡티)을 조금 눌러 GIF 를 작게 — 앞 프레임과 섞고 살짝 흐리게(왼쪽 영상만)
        src = x.convert("RGB").resize((pw, ph), Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.6))
        if prev is not None:
            src = Image.blend(src, prev, 0.35)
        prev = src
        fr.paste(src, (pad, pad))
        fr.paste(y.convert("RGB").resize((pw, ph), Image.LANCZOS), (pad * 2 + pw, pad))
        frames.append(fr)
    return C.save_gif(frames, os.path.join(docs, name), 24 / every, colors=colors, dither="none")


jobs = sys.argv[4].split(",") if len(sys.argv) > 4 else ["hero", "pose", "objects", "confidence", "gifs"]
if "hero" in jobs:
    hero()
if "pose" in jobs:
    pose()
if "objects" in jobs:
    objects()
if "confidence" in jobs:
    confidence()
if "gifs" in jobs:
    video = os.path.join(work, "video")
    print("video-pose.gif", video_gif("video-pose.gif", os.path.join(video, "ella.mp4"),
                                      os.path.join(res_dir, "ella-pose", "pose.mp4"), 380, 2, 96))
    print("video-objects.gif", video_gif("video-objects.gif", os.path.join(res_dir, "lumiere-crop", "found.mp4"),
                                         os.path.join(res_dir, "lumiere-crop", "mask.mp4"), 380, 3, 64))
for f in sorted(os.listdir(out)):
    print(f, os.path.getsize(os.path.join(out, f)))
