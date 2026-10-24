"""promboard.yolo — Ultralytics YOLO 포즈 뼈대 · 물체 잘라내기 노드 코드.

노드 하나(yolo)가 사진 한 장을 받아, 고른 모드에 따라 그림을 한 장 또는 여러 장 돌려준다(출력 포트 images, multiple).
  pose     포즈 뼈대     사진 속 사람들의 자세를 검은 바탕 위 색 뼈대(OpenPose 색 규칙) 한 장으로 그린다. 원본과 같은 크기.
                        이미지 생성 노드에 참조로 이어 '같은 자세로' 쓰는 용도다.
  crop     물체 잘라내기  사람 · 물건을 찾아 하나씩 네모로 잘라 여러 장으로 낸다(확실한 것부터, 64장까지). 사진 대부분을
                        덮는 것(식탁 등 — 네모로 잘라 봐야 원본)과 한 물체를 다른 종류로 두 번 찾은 것은 뺀다.
  cutout   물체 오려내기  같은 물체를 배경을 지운 투명 PNG 로 낸다(분할 마스크). 사진 대부분을 덮는 것도 낸다(배경은 지워지므로).
설정 confidence: loose · normal · strict → 찾았다고 볼 점수 기준 0.15 · 0.25 · 0.45.
찾은 것이 없으면 빈 목록을 돌려준다(실패가 아니라 '찾은 것 없음', PromBoard 04 설계 PM 결정 10).

모델(manifest requires.models, Ultralytics 공식 GitHub 릴리스 ultralytics/assets v8.4.0, AGPL-3.0)
  yolo26n-pose.onnx  사람 + 관절 17개(COCO 순서). 출력 (1, 300, 57) = [x1, y1, x2, y2, 점수, 종류, (x, y, 점수) × 17]
  yolo26n-seg.onnx   물체 80종류(COCO) + 마스크. 출력 (1, 300, 38) = [x1, y1, x2, y2, 점수, 종류, 마스크 계수 × 32]
                     · (1, 32, 160, 160) 마스크 바탕
  두 모델 모두 입력 (1, 3, 640, 640) RGB 0~1, NMS 가 모델 안에 들어 있다(end2end). 좌표는 640 레터박스 기준이다.
실행: onnxruntime CPU. 세션은 모델마다 한 번 만들어 사이드카가 떠 있는 동안 다시 쓴다.
노드 코드 약속(NODES · ctx · 입출력 모양)은 PromBoard docs/Scrum4/02_스키마-v0.md §6.
"""
import time

POSE_MODEL = "yolo26n/yolo26n-pose.onnx"
SEG_MODEL = "yolo26n/yolo26n-seg.onnx"
SIZE = 640
PAD_VALUE = 114
CONFIDENCE = {"loose": 0.15, "normal": 0.25, "strict": 0.45}
KEYPOINT_MIN = 0.4  # 이 점수보다 낮은 관절은 그리지 않는다
MAX_ITEMS = 64  # PromBoard multiple 포트 상한
CROP_MARGIN = 0.06  # 네모로 자를 때 상자 둘레에 더하는 여유(상자 긴 변 대비)
MIN_CROP_PX = 24  # 이보다 작은 물체는 내지 않는다
WHOLE_IMAGE = 0.85  # 잘라내기에서 사진 넓이의 이만큼 이상을 덮는 상자(식탁처럼 사진 전체)는 잘라 봐야 원본이라 내지 않는다
DUPLICATE_IOU = 0.7  # 한 물체를 다른 종류로 두 번 찾은 것(end2end 모델은 종류끼리 겹침을 거르지 않는다)

# COCO 80종류(모델 메타데이터 names 와 같은 순서) — 파일 이름에 쓴다.
COCO_NAMES = (
    "person bicycle car motorcycle airplane bus train truck boat traffic-light fire-hydrant stop-sign parking-meter bench "
    "bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee skis snowboard "
    "sports-ball kite baseball-bat baseball-glove skateboard surfboard tennis-racket bottle wine-glass cup fork knife spoon "
    "bowl banana apple sandwich orange broccoli carrot hot-dog pizza donut cake chair couch potted-plant bed dining-table "
    "toilet tv laptop mouse remote keyboard cell-phone microwave oven toaster sink refrigerator book clock vase scissors "
    "teddy-bear hair-drier toothbrush"
).split()

# OpenPose 18점 뼈대(목 = 두 어깨 가운데). 0 코 · 1 목 · 2~4 오른팔 · 5~7 왼팔 · 8~10 오른다리 · 11~13 왼다리 · 14~17 눈 · 귀.
# COCO 17점 → OpenPose 18점 자리(1 목은 계산).
COCO_FOR_OPENPOSE = (0, None, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3)
OPENPOSE_LIMBS = (
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
)
OPENPOSE_COLORS = (
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
    (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
    (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85),
)

_SESSIONS = {}


# ---------------------------------------------------------------------------
# 전처리 · 후처리 (numpy · Pillow)
# ---------------------------------------------------------------------------


def letterbox(rgb_image):
    """PIL RGB → (모델 입력 (1, 3, 640, 640) float32, 배율, 왼쪽 여백, 위 여백). Ultralytics LetterBox 처럼 가운데 맞춤 · 회색 114."""
    import numpy as np
    from PIL import Image

    w, h = rgb_image.size
    r = min(SIZE / w, SIZE / h)
    nw, nh = max(1, round(w * r)), max(1, round(h * r))
    canvas = Image.new("RGB", (SIZE, SIZE), (PAD_VALUE,) * 3)
    left, top = (SIZE - nw) // 2, (SIZE - nh) // 2
    canvas.paste(rgb_image.resize((nw, nh), Image.BILINEAR), (left, top))
    x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(x), r, left, top


def detections(raw, threshold, r, left, top, width, height):
    """end2end 출력 (300, N) → 점수 기준을 넘은 줄과 원본 좌표 상자 (k, 4)."""
    rows = raw[raw[:, 4] >= threshold]
    boxes = rows[:, :4].copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / r
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / r
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    return rows, boxes


def iou(a, b):
    """두 상자(x1, y1, x2, y2)의 겹친 넓이 / 합친 넓이."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _ellipse(cx, cy, length, width, angle, steps=24):
    """끝이 둥근 막대 모양(OpenPose 뼈 그리기와 같은 타원) 꼭짓점."""
    import math

    ca, sa = math.cos(angle), math.sin(angle)
    pts = []
    for i in range(steps):
        t = 2 * math.pi * i / steps
        x, y = length * math.cos(t), width * math.sin(t)
        pts.append((cx + x * ca - y * sa, cy + x * sa + y * ca))
    return pts


def draw_skeletons(people, width, height):
    """사람마다 COCO 관절 (17, 3)(원본 좌표 · 점수) → 검은 바탕 OpenPose 색 뼈대 RGB 그림."""
    import math

    from PIL import Image, ImageDraw

    stick = max(2.0, max(width, height) / 128)
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    for kpts in people:
        pts = []
        for coco in COCO_FOR_OPENPOSE:
            if coco is None:  # 목 = 두 어깨 가운데(두 어깨가 다 보일 때만)
                ls, rs = kpts[5], kpts[6]
                ok = ls[2] >= KEYPOINT_MIN and rs[2] >= KEYPOINT_MIN
                pts.append(((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2) if ok else None)
            else:
                k = kpts[coco]
                pts.append((float(k[0]), float(k[1])) if k[2] >= KEYPOINT_MIN else None)
        # 뼈는 반투명하게 겹쳐 그린 뒤 섞는다(OpenPose 그림과 같이 색이 0.6 정도로 보인다).
        limbs = Image.new("RGB", (width, height), (0, 0, 0))
        ld = ImageDraw.Draw(limbs)
        for i, (a, b) in enumerate(OPENPOSE_LIMBS):
            if pts[a] is None or pts[b] is None:
                continue
            (x1, y1), (x2, y2) = pts[a], pts[b]
            length = math.hypot(x2 - x1, y2 - y1) / 2
            angle = math.atan2(y2 - y1, x2 - x1)
            ld.polygon(_ellipse((x1 + x2) / 2, (y1 + y2) / 2, length, stick, angle), fill=OPENPOSE_COLORS[i])
        canvas = Image.blend(canvas, Image.composite(limbs, canvas, limbs.convert("L").point(lambda v: 255 if v else 0)), 0.6)
        jd = ImageDraw.Draw(canvas)
        for i, p in enumerate(pts):
            if p is not None:
                jd.ellipse((p[0] - stick, p[1] - stick, p[0] + stick, p[1] + stick), fill=OPENPOSE_COLORS[i])
    return canvas


def object_mask(coefs, protos, box640, crop_box, size):
    """마스크 계수 (32,) · 바탕 (32, 160, 160) → 자를 칸(crop_box, 원본 좌표) 크기의 0~255 마스크(PIL L).
    box640 = 640 레터박스 좌표의 물체 상자(이 밖은 0 — Ultralytics crop_mask 와 같다), size = (가로, 세로) 원본 픽셀."""
    import numpy as np
    from PIL import Image

    c, mh, mw = protos.shape
    logits = (coefs @ protos.reshape(c, -1)).reshape(mh, mw)
    prob = 1.0 / (1.0 + np.exp(-logits))
    scale = mw / SIZE
    x1, y1, x2, y2 = (v * scale for v in box640)
    ys, xs = np.mgrid[0:mh, 0:mw] + 0.5
    prob = np.where((xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2), prob, 0.0)
    img = Image.fromarray((prob * 255).astype(np.uint8), "L")
    # 0.5 를 가운데로 좁은 폭만 반투명하게 — 딱 자르면 160px 마스크를 키운 계단이 보인다.
    return img.resize(size, Image.BILINEAR, box=crop_box).point(lambda v: min(255, max(0, (v - 112) * 4)))


# ---------------------------------------------------------------------------
# 세션
# ---------------------------------------------------------------------------


def _session(ctx, name):
    path = ctx.model_path(name)
    if path in _SESSIONS:
        return _SESSIONS[path]
    import onnxruntime as ort

    ctx.progress(0.1, "load-model")
    start = time.monotonic()
    so = ort.SessionOptions()
    so.log_severity_level = 3
    session = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    _SESSIONS[path] = session
    ctx.log(f"model {name} loaded in {time.monotonic() - start:.1f}s")
    return session


def _read(path):
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im.load()
        return ImageOps.exif_transpose(im).convert("RGB")


# ---------------------------------------------------------------------------
# 노드
# ---------------------------------------------------------------------------


def _pose(ctx, image, threshold):
    session = _session(ctx, POSE_MODEL)
    x, r, left, top = letterbox(image)
    ctx.progress(0.3, "detect")
    raw = session.run(None, {session.get_inputs()[0].name: x})[0][0]
    ctx.check_cancel()
    w, h = image.size
    rows, _ = detections(raw, threshold, r, left, top, w, h)
    people = []
    for row in rows:
        k = row[6:].reshape(17, 3).copy()
        k[:, 0] = (k[:, 0] - left) / r
        k[:, 1] = (k[:, 1] - top) / r
        people.append(k)
    ctx.log(f"pose: {len(people)} people")
    if not people:
        return []
    ctx.progress(0.7, "draw")
    out = ctx.out_path("pose.png")
    draw_skeletons(people, w, h).save(out, "PNG")
    return [out]


def _objects(ctx, image, threshold, cutout):
    session = _session(ctx, SEG_MODEL)
    x, r, left, top = letterbox(image)
    ctx.progress(0.3, "detect")
    raw, protos = session.run(None, {session.get_inputs()[0].name: x})
    raw, protos = raw[0], protos[0]
    ctx.check_cancel()
    w, h = image.size
    rows, boxes = detections(raw, threshold, r, left, top, w, h)
    found = []
    for i in rows[:, 4].argsort()[::-1]:  # 점수 높은 것(확실한 것)부터 — 이 순서로 낸다
        row, box = rows[i], boxes[i]
        bw, bh = box[2] - box[0], box[3] - box[1]
        if min(bw, bh) < MIN_CROP_PX or (not cutout and bw * bh >= WHOLE_IMAGE * w * h):
            continue
        if any(iou(box, kept) >= DUPLICATE_IOU for kept in (b for _, b in found)):
            continue
        found.append((row, box))
    outs = []
    for i, (row, box) in enumerate(found[:MAX_ITEMS]):
        ctx.check_cancel()
        pad = CROP_MARGIN * max(box[2] - box[0], box[3] - box[1])
        x1, y1 = max(0, int(box[0] - pad)), max(0, int(box[1] - pad))
        x2, y2 = min(w, int(round(box[2] + pad))), min(h, int(round(box[3] + pad)))
        piece = image.crop((x1, y1, x2, y2))
        if cutout:
            crop640 = (x1 * r + left, y1 * r + top, x2 * r + left, y2 * r + top)
            scale = protos.shape[2] / SIZE
            mask = object_mask(row[6:38], protos, row[:4], tuple(v * scale for v in crop640), piece.size)
            piece = piece.convert("RGBA")
            piece.putalpha(mask)
        cls = int(row[5])
        name = COCO_NAMES[cls] if 0 <= cls < len(COCO_NAMES) else "object"
        out = ctx.out_path(f"{i:02d}-{name}.png")
        piece.save(out, "PNG")
        outs.append(out)
        ctx.progress(0.4 + 0.55 * (i + 1) / min(len(found), MAX_ITEMS), "save")
    ctx.log(f"objects: {len(found)} found, {len(outs)} saved")
    return outs


def yolo(ctx, inputs, settings):
    ctx.progress(0.02, "read")
    image = _read(inputs["image"])
    ctx.check_cancel()
    threshold = CONFIDENCE[settings["confidence"]]
    mode = settings["mode"]
    if mode == "pose":
        outs = _pose(ctx, image, threshold)
    else:
        outs = _objects(ctx, image, threshold, cutout=(mode == "cutout"))
    ctx.progress(1.0, "done")
    return {"images": outs}


NODES = {"yolo": yolo}
