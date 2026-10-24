"""promboard.yolo — Ultralytics YOLO 포즈 뼈대 · 물체 잘라내기 · 찾기와 추적 노드 코드.

노드 하나(yolo)가 사진 · 영상을 받는다(둘 다 선택 입력, 하나는 이어야 한다 — 둘 다 이으면 둘 다 처리한다).
고른 모드(pose · crop · cutout)에 따라 결과가 갈리고, 찾기 · 추적 결과는 모드와 상관없이 늘 나온다.

사진이 이어지면
  images  (여러 장)  pose   검은 바탕 위 색 뼈대(OpenPose 색 규칙) 한 장. 원본과 같은 크기. 이미지 생성 노드의 참조용.
                    crop   찾은 사람 · 물건을 하나씩 네모로 잘라 여러 장(확실한 것부터, 64장까지). 사진 대부분을 덮는 것(식탁 등)과
                           한 물체를 다른 종류로 두 번 찾은 것은 뺀다.
                    cutout 같은 물체를 배경을 지운 투명 PNG 로(분할 마스크). 사진 대부분을 덮는 것도 낸다.
                    찾은 것이 없으면 빈 목록(실패가 아니라 '찾은 것 없음', PromBoard 04 설계 PM 결정 10).
  found   (한 장)    사진에 상자 + 종류 이름 + 점수(JPEG). 포즈 모드는 사람, 물체 모드는 찾은 것 전부(크기 · 넓이로 빼지 않는다).
영상이 이어지면 — 둘 다 원본과 같은 가로세로(홀수면 오른쪽 · 아래 1px 가장자리 복제) · fps · 프레임 수의 H.264 mp4(yuv420p, 소리 없음)
  video        pose   검은 바탕 색 뼈대가 움직이는 영상. 영상 생성 노드의 참조 영상용.
               crop · cutout  흑백 마스크 영상(흰 255 = 찾은 사람 · 물체, 검정 0 = 나머지 — 마스킹 팩 '마스크 영상'과 같은 규칙).
  found-video  원본 영상에 상자 · 종류 이름 · 추적 번호 · 짧은 이동 꼬리. 포즈 모드는 사람, 물체 모드는 찾은 것 전부.
설정 confidence: loose · normal · strict → 찾았다고 볼 점수 기준 0.15 · 0.25 · 0.45.

영상 처리
  - 추적: ByteTrack(Zhang 외 2022)을 줄여 손으로 짰다. 칼만 필터(등속, 상태 cx · cy · 가로/세로 비 · 높이)로 다음 자리를 예측하고,
    점수가 기준 이상인 것과 먼저 짝짓고 남은 번호를 낮은 점수(0.1 이상)와 한 번 더 짝짓는다. 짝은 겹침(IoU)이 큰 것부터 고른다.
    새 번호는 두 프레임 연속 보인 것만(첫 프레임은 바로) 붙이고, 놓친 번호는 1초 기억했다가 다시 나타나면 같은 번호를 준다.
    supervision 패키지(MIT, ByteTrack 포함)는 opencv · scipy · matplotlib 를 끌고 와서 쓰지 않았다.
  - 깜빡임 줄이기: 번호가 붙은 것만 그린다(한 프레임만 나온 헛것을 뺀다). 한두 프레임 놓친 물체 · 관절은 직전 값을 그대로 쓰고,
    마스크 확률과 관절 위치는 이번 프레임 0.6 + 직전 0.4 로 섞는다.
  - 영상 읽기 · 쓰기는 PyAV(FFmpeg, libx264). 화면 회전 정보(휴대폰 세로 영상)는 돌려서 보이는 방향 그대로 쓴다.
  - 길이 상한: 5분(MAX_VIDEO_SEC). 그 안이어도 이 PC 속도로 어림한 처리 시간이 실행 제한(TIME_BUDGET_*)을 넘으면 처음 몇 초 만에
    멈추고 알린다. 실측(28스레드 데스크톱 CPU): 프레임당 1080p 약 75ms · 720×1280 약 45ms — 1080p 30fps 1분이 약 2분 15초.

모델(manifest requires.models, Ultralytics 공식 GitHub 릴리스 ultralytics/assets v8.4.0, AGPL-3.0)
  yolo26n-pose.onnx  사람 + 관절 17개(COCO 순서). 출력 (1, 300, 57) = [x1, y1, x2, y2, 점수, 종류, (x, y, 점수) × 17]
  yolo26n-seg.onnx   물체 80종류(COCO) + 마스크. 출력 (1, 300, 38) = [x1, y1, x2, y2, 점수, 종류, 마스크 계수 × 32]
                     · (1, 32, 160, 160) 마스크 바탕
  두 모델 모두 입력 (1, 3, 640, 640) RGB 0~1, NMS 가 모델 안에 들어 있다(end2end). 좌표는 640 레터박스 기준이다.
실행: onnxruntime CPU. 세션은 모델마다 한 번 만들어 사이드카가 떠 있는 동안 다시 쓴다.
노드 코드 약속(NODES · ctx · 입출력 모양)은 PromBoard docs/Scrum4/02_스키마-v0.md §6.
"""
import math
import time
from collections import deque

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

# 영상
MAX_VIDEO_SEC = 300  # 이보다 긴 영상은 받지 않는다(쉬운 오류) — 결과 파일 크기 · 처리 시간 상한. 근거는 맨 위 '영상 처리'
# 처리 시간 예산(초). PromBoard 실행기는 노드 한 번에 900초(runner/run.ts PLUGIN_NODE_BUDGET_SEC)를 주고, 그래픽카드가 없다고 본
# PC(ctx.device == "cpu")는 4배(PLUGIN_CPU_SLOW_FACTOR)를 준다. 이 노드는 늘 CPU 로 돌므로 그래픽카드가 있는 PC 가 가장 빠듯하다.
# 앞 프레임들의 속도로 전체 시간을 어림해 예산을 넘을 것 같으면 바로 멈추고 알린다(15분을 기다려 '시간 초과'로 끝나지 않게).
TIME_BUDGET_SEC = {"cpu": 3300}
TIME_BUDGET_DEFAULT_SEC = 780
TIME_CHECK_AFTER_SEC = 3.0  # 어림은 이만큼 돈 뒤부터(처음 몇 프레임은 느리다)
TRACK_LOW = 0.1  # 이 점수부터 추적기에 넣는다 — 기준 밑의 것은 이미 붙은 번호를 이어 가는 데만 쓴다(ByteTrack 두 번째 짝짓기)
TRACK_FIRST_IOU = 0.2  # 기준 이상 → 예측 상자와 이만큼 겹치면 같은 것(ByteTrack match_thresh 0.8 = 1 - 0.2)
TRACK_SECOND_IOU = 0.5  # 기준 밑 → 이어 가기만 하므로 더 많이 겹쳐야 한다
TRACK_NEW_IOU = 0.3  # 한 번 본 새 후보가 다음 프레임에도 있나
TRACK_KEEP_SEC = 1.0  # 놓친 번호를 기억하는 시간(ByteTrack track_buffer 30 @ 30fps)
TRAIL_SEC = 0.8  # 이동 꼬리 길이
HOLD_FRAMES = 2  # 이만큼까지 놓친 물체 · 관절은 직전 값을 그대로 쓴다
SMOOTH = 0.6  # 마스크 확률 · 관절 위치 = 이번 프레임 × 0.6 + 직전 × 0.4
CRF_DRAWING = 20  # 찾기 · 추적 영상(원본 그림 위에 그림)
CRF_FLAT = 16  # 뼈대 · 마스크 영상(넓은 단색 면 — 낮춰도 파일이 작다, 경계를 덜 뭉갠다)
PROGRESS_EVERY = 0.3  # 진행률 알림 간격(초)

# COCO 80종류(모델 메타데이터 names 와 같은 순서) — 파일 이름 · 상자 이름에 쓴다.
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
# 상자 색(Ultralytics 기본 색 20개) — 사진은 종류 번호, 영상은 추적 번호로 고른다.
BOX_COLORS = tuple(
    tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    for h in (
        "FF3838 FF9D97 FF701F FFB21D CFD231 48F90A 92CC17 3DDB86 1A9334 00D4BB "
        "2C99A8 00C2FF 344593 6473FF 0018EC 8438FF 520085 CB38FF FF95C8 FF37C7"
    ).split()
)

_SESSIONS = {}
_FONTS = {}


# ---------------------------------------------------------------------------
# 전처리 · 후처리 (numpy · Pillow)
# ---------------------------------------------------------------------------


def letterbox_geometry(width, height):
    """원본 크기 → (배율, 줄인 가로, 줄인 세로, 왼쪽 여백, 위 여백). Ultralytics LetterBox 처럼 가운데 맞춤."""
    r = min(SIZE / width, SIZE / height)
    nw, nh = max(1, round(width * r)), max(1, round(height * r))
    return r, nw, nh, (SIZE - nw) // 2, (SIZE - nh) // 2


def model_input(small, left, top):
    """줄인 RGB 배열 (nh, nw, 3) → 회색 114 바탕 가운데에 놓은 모델 입력 (1, 3, 640, 640) float32."""
    import numpy as np

    canvas = np.full((SIZE, SIZE, 3), PAD_VALUE, dtype=np.uint8)
    canvas[top:top + small.shape[0], left:left + small.shape[1]] = small
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def letterbox(rgb_image):
    """PIL RGB → (모델 입력 (1, 3, 640, 640) float32, 배율, 왼쪽 여백, 위 여백)."""
    import numpy as np
    from PIL import Image

    r, nw, nh, left, top = letterbox_geometry(*rgb_image.size)
    small = np.asarray(rgb_image.resize((nw, nh), Image.BILINEAR))
    return model_input(small, left, top), r, left, top


def detections(raw, threshold, r, left, top, width, height):
    """end2end 출력 (300, N) → 점수 기준을 넘은 줄과 원본 좌표 상자 (k, 4)."""
    rows = raw[raw[:, 4] >= threshold]
    boxes = rows[:, :4].copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / r
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / r
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    return rows, boxes


def iou_matrix(a, b):
    """상자 묶음 a (n, 4) × b (m, 4) → 겹침 (n, m)."""
    import numpy as np

    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ix = (np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])).clip(0)
    iy = (np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])).clip(0)
    inter = ix * iy
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def distinct(rows, boxes):
    """점수 높은 것부터, 앞의 것과 IoU 0.7 이상 겹치는 것(한 물체를 다른 종류로 또 찾은 것)을 뺀 줄 번호."""
    import numpy as np

    order = rows[:, 4].argsort()[::-1]
    overlaps = iou_matrix(boxes[order], boxes[order])
    dropped = np.zeros(len(order), dtype=bool)
    keep = []
    for i in range(len(order)):
        if dropped[i]:
            continue
        keep.append(int(order[i]))
        dropped |= overlaps[i] >= DUPLICATE_IOU
    return keep


def class_name(cls):
    cls = int(cls)
    return COCO_NAMES[cls] if 0 <= cls < len(COCO_NAMES) else "object"


def _ellipse(cx, cy, length, width, angle, steps=24):
    """끝이 둥근 막대 모양(OpenPose 뼈 그리기와 같은 타원) 꼭짓점."""
    ca, sa = math.cos(angle), math.sin(angle)
    pts = []
    for i in range(steps):
        t = 2 * math.pi * i / steps
        x, y = length * math.cos(t), width * math.sin(t)
        pts.append((cx + x * ca - y * sa, cy + x * sa + y * ca))
    return pts


def openpose_points(kpts):
    """COCO 관절 (17, 3)(원본 좌표 · 점수) → OpenPose 18점 자리 [(x, y) 또는 None]."""
    pts = []
    for coco in COCO_FOR_OPENPOSE:
        if coco is None:  # 목 = 두 어깨 가운데(두 어깨가 다 보일 때만)
            ls, rs = kpts[5], kpts[6]
            ok = ls[2] >= KEYPOINT_MIN and rs[2] >= KEYPOINT_MIN
            pts.append(((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2) if ok else None)
        else:
            k = kpts[coco]
            pts.append((float(k[0]), float(k[1])) if k[2] >= KEYPOINT_MIN else None)
    return pts


def draw_skeletons(people, width, height):
    """사람마다 COCO 관절 (17, 3) → 검은 바탕 OpenPose 색 뼈대 RGB 그림.
    뼈는 한 층에 그려 0.6 로 어둡게 섞고(OpenPose 그림과 같은 색), 관절 점은 그 위에 제 색으로 찍는다."""
    from PIL import Image, ImageDraw

    stick = max(2.0, max(width, height) / 128)
    all_pts = [openpose_points(k) for k in people]
    limbs = Image.new("RGB", (width, height), (0, 0, 0))
    ld = ImageDraw.Draw(limbs)
    for pts in all_pts:
        for i, (a, b) in enumerate(OPENPOSE_LIMBS):
            if pts[a] is None or pts[b] is None:
                continue
            (x1, y1), (x2, y2) = pts[a], pts[b]
            length = math.hypot(x2 - x1, y2 - y1) / 2
            angle = math.atan2(y2 - y1, x2 - x1)
            ld.polygon(_ellipse((x1 + x2) / 2, (y1 + y2) / 2, length, stick, angle), fill=OPENPOSE_COLORS[i])
    canvas = Image.blend(Image.new("RGB", (width, height), (0, 0, 0)), limbs, 0.6)
    jd = ImageDraw.Draw(canvas)
    for pts in all_pts:
        for i, p in enumerate(pts):
            if p is not None:
                jd.ellipse((p[0] - stick, p[1] - stick, p[0] + stick, p[1] + stick), fill=OPENPOSE_COLORS[i])
    return canvas


def mask_probs(coefs, protos, boxes640):
    """마스크 계수 (k, 32) · 바탕 (32, 160, 160) · 640 레터박스 상자 (k, 4) → 물체별 확률 (k, 160, 160).
    상자 밖은 0(Ultralytics crop_mask 와 같다)."""
    import numpy as np

    c, mh, mw = protos.shape
    logits = (coefs @ protos.reshape(c, -1)).reshape(-1, mh, mw)
    prob = 1.0 / (1.0 + np.exp(-logits))
    b = boxes640 * (mw / SIZE)
    xs = np.arange(mw, dtype=np.float32) + 0.5
    ys = np.arange(mh, dtype=np.float32) + 0.5
    inside_x = (xs[None, :] >= b[:, 0:1]) & (xs[None, :] <= b[:, 2:3])  # (k, mw)
    inside_y = (ys[None, :] >= b[:, 1:2]) & (ys[None, :] <= b[:, 3:4])  # (k, mh)
    return np.where(inside_y[:, :, None] & inside_x[:, None, :], prob, 0.0).astype(np.float32)


def object_mask(coefs, protos, box640, crop_box, size):
    """마스크 계수 (32,) · 바탕 (32, 160, 160) → 자를 칸(crop_box, 160 바탕 좌표) 크기의 0~255 마스크(PIL L).
    size = (가로, 세로) 원본 픽셀."""
    import numpy as np
    from PIL import Image

    prob = mask_probs(coefs[None], protos, np.asarray(box640, dtype=np.float32)[None])[0]
    img = Image.fromarray((prob * 255).astype(np.uint8))  # 2차원 uint8 → L
    # 0.5 를 가운데로 좁은 폭만 반투명하게 — 딱 자르면 160px 마스크를 키운 계단이 보인다.
    return img.resize(size, Image.BILINEAR, box=crop_box).point(lambda v: min(255, max(0, (v - 112) * 4)))


def _font(size):
    from PIL import ImageFont

    size = int(size)
    if size not in _FONTS:
        try:
            _FONTS[size] = ImageFont.load_default(size=size)
        except TypeError:  # 아주 옛 Pillow — 크기 없는 기본 글꼴
            _FONTS[size] = ImageFont.load_default()
    return _FONTS[size]


def _line_width(width, height):
    return max(2, round((width + height) / 2 * 0.004))  # Ultralytics Annotator(0.003)보다 조금 굵게 — 노드 칸처럼 작게 볼 때


def draw_box(draw, box, text, color, lw, font, width):
    """상자 + 왼쪽 위 이름표(색 바탕). 이름표가 그림 위로 넘치면 상자 안쪽에, 오른쪽으로 넘치면 왼쪽으로 당겨 붙인다."""
    x1, y1, x2, y2 = (float(v) for v in box)
    draw.rectangle((x1, y1, x2, y2), outline=color, width=lw)
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    tw, th = r - l, b - t
    pad = max(2, lw)
    lx = max(0.0, min(x1, width - tw - 2 * pad))
    ty = y1 - th - 2 * pad
    if ty < 0:
        ty = y1
    draw.rectangle((lx, ty, lx + tw + 2 * pad, ty + th + 2 * pad), fill=color)
    light = color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114 > 150
    draw.text((lx + pad - l, ty + pad - t), text, fill=(0, 0, 0) if light else (255, 255, 255), font=font)


def draw_found(image, boxes, labels, colors):
    """사진(PIL RGB)에 상자 · 이름표를 그린 새 그림."""
    from PIL import ImageDraw

    out = image.copy()
    w, h = out.size
    lw = _line_width(w, h)
    font = _font(max(12, lw * 5))
    draw = ImageDraw.Draw(out)
    for box, text, color in zip(boxes, labels, colors):
        draw_box(draw, box, text, color, lw, font, w)
    return out


# ---------------------------------------------------------------------------
# 추적 (ByteTrack 을 줄인 것 — 맨 위 '영상 처리')
# ---------------------------------------------------------------------------


class _Kalman:
    """등속 칼만 필터 — 상태 (cx, cy, 가로/세로 비 a, 높이 h, 네 값의 속도). ByteTrack KalmanFilterXYAH 와 같은 잡음 값."""

    POS = 1.0 / 20
    VEL = 1.0 / 160

    def __init__(self):
        import numpy as np

        self.np = np
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)

    def initiate(self, m):
        np = self.np
        h = m[3]
        std = [2 * self.POS * h, 2 * self.POS * h, 1e-2, 2 * self.POS * h, 10 * self.VEL * h, 10 * self.VEL * h, 1e-5, 10 * self.VEL * h]
        return np.r_[m, np.zeros(4)], np.diag(np.square(std))

    def predict(self, mean, cov):
        np = self.np
        h = mean[3]
        std = [self.POS * h, self.POS * h, 1e-2, self.POS * h, self.VEL * h, self.VEL * h, 1e-5, self.VEL * h]
        return self.F @ mean, self.F @ cov @ self.F.T + np.diag(np.square(std))

    def update(self, mean, cov, m):
        np = self.np
        h = mean[3]
        r = np.diag(np.square([self.POS * h, self.POS * h, 1e-1, self.POS * h]))
        s = self.H @ cov @ self.H.T + r
        k = cov @ self.H.T @ np.linalg.inv(s)
        return mean + k @ (m - self.H @ mean), cov - k @ s @ k.T


def _xyah(box):
    import numpy as np

    w, h = max(1.0, box[2] - box[0]), max(1.0, box[3] - box[1])
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, w / h, h], dtype=np.float64)


class Track:
    def __init__(self, kf, box, cls, frame, trail_len):
        self.mean, self.cov = kf.initiate(_xyah(box))
        self.id = 0  # 0 = 아직 번호 없음(한 번만 봤다)
        self.votes = {int(cls): 1}
        self.tracked = True
        self.seen = frame
        self.trail = deque(maxlen=trail_len)
        self.mask = None  # 마지막 마스크 확률(160 바탕) — 놓친 동안 HOLD_FRAMES 까지 그대로 쓴다
        self.kpts = None  # 부드럽게 한 관절 (17, 3)
        self.kpt_age = None  # 관절마다 직전 값을 그대로 쓴 프레임 수
        self.kpt_frame = 0  # 관절을 마지막으로 고친 프레임

    @property
    def box(self):
        cx, cy, a, h = self.mean[:4]
        h = max(1.0, h)
        w = max(1.0, a * h)
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    @property
    def cls(self):
        return max(self.votes.items(), key=lambda kv: kv[1])[0]  # 가장 많이 나온 종류 — 이름이 프레임마다 바뀌지 않게


def _greedy(ious, min_iou):
    """겹침 행렬 → 겹침이 큰 쌍부터 고른 (행, 열) 목록. 둘 다 아직 안 쓴 것만, min_iou 이상만."""
    pairs = []
    if ious.size == 0:
        return pairs
    used_r, used_c = set(), set()
    flat = ious.ravel().argsort()[::-1]
    cols = ious.shape[1]
    for f in flat:
        r, c = divmod(int(f), cols)
        if ious[r, c] < min_iou:
            break
        if r in used_r or c in used_c:
            continue
        used_r.add(r)
        used_c.add(c)
        pairs.append((r, c))
    return pairs


class Tracker:
    """프레임마다 update(상자 (k, 4), 점수 (k,), 종류 (k,)) → 이번 프레임에 보인 번호 붙은 [(Track, 줄 번호)]."""

    def __init__(self, high, fps):
        self.kf = _Kalman()
        self.high = high
        self.keep = max(1, round(TRACK_KEEP_SEC * fps))
        self.trail_len = max(2, round(TRAIL_SEC * fps))
        self.tracks = []
        self.frame = 0
        self.next_id = 1

    def _match(self, tracks, boxes, det_idx, min_iou):
        import numpy as np

        if not tracks or not det_idx:
            return [], list(range(len(tracks))), list(det_idx)
        ious = iou_matrix(np.array([t.box for t in tracks]), boxes[det_idx])
        pairs = _greedy(ious, min_iou)
        mt, md = {r for r, _ in pairs}, {c for _, c in pairs}
        return (
            [(tracks[r], det_idx[c]) for r, c in pairs],
            [i for i in range(len(tracks)) if i not in mt],
            [det_idx[c] for c in range(len(det_idx)) if c not in md],
        )

    def _hit(self, t, box, cls):
        t.mean, t.cov = self.kf.update(t.mean, t.cov, _xyah(box))
        t.votes[int(cls)] = t.votes.get(int(cls), 0) + 1
        t.tracked = True
        t.seen = self.frame

    def update(self, boxes, scores, classes):
        self.frame += 1
        for t in self.tracks:
            if not t.tracked:
                t.mean[7] = 0.0  # 놓친 것은 높이 변화를 멈춘다(ByteTrack 과 같다)
            t.mean, t.cov = self.kf.predict(t.mean, t.cov)
        high = [i for i in range(len(scores)) if scores[i] >= self.high]
        low = [i for i in range(len(scores)) if scores[i] < self.high]
        confirmed = [t for t in self.tracks if t.id]
        pending = [t for t in self.tracks if not t.id]
        out = []
        # 1) 번호 붙은 것(보이는 중 · 놓친 것) ↔ 기준 이상
        pairs, left_t, left_high = self._match(confirmed, boxes, high, TRACK_FIRST_IOU)
        for t, d in pairs:
            self._hit(t, boxes[d], classes[d])
            out.append((t, d))
        # 2) 남은 것 중 보이던 것 ↔ 기준 밑(이어 가기만)
        still = [confirmed[i] for i in left_t if confirmed[i].tracked]
        pairs, left2, _ = self._match(still, boxes, low, TRACK_SECOND_IOU)
        for t, d in pairs:
            self._hit(t, boxes[d], classes[d])
            out.append((t, d))
        for i in left2:
            still[i].tracked = False
        # 3) 한 번 본 후보 ↔ 남은 기준 이상 — 두 번째로 보이면 번호를 붙인다
        pairs, _, left_high = self._match(pending, boxes, left_high, TRACK_NEW_IOU)
        confirmed_now = set()
        for t, d in pairs:
            self._hit(t, boxes[d], classes[d])
            t.id = self.next_id
            self.next_id += 1
            confirmed_now.add(id(t))
            out.append((t, d))
        # 4) 새 후보(첫 프레임은 바로 번호), 두 번째에 안 보인 후보 · 오래 놓친 것은 버린다
        kept = [t for t in self.tracks if t.id or id(t) in confirmed_now]
        kept = [t for t in kept if self.frame - t.seen <= self.keep]
        for d in left_high:
            t = Track(self.kf, boxes[d], classes[d], self.frame, self.trail_len)
            if self.frame == 1:
                t.id = self.next_id
                self.next_id += 1
                out.append((t, d))
            kept.append(t)
        self.tracks = kept
        for t, _ in out:
            x1, y1, x2, y2 = t.box
            t.trail.append(((x1 + x2) / 2, (y1 + y2) / 2))
        return out

    def held(self):
        """번호는 있는데 이번 프레임에 놓친 것 중 HOLD_FRAMES 안의 것."""
        return [t for t in self.tracks if t.id and t.seen != self.frame and self.frame - t.seen <= HOLD_FRAMES]


def track_color(t):
    return BOX_COLORS[(t.id - 1) % len(BOX_COLORS)]  # 1번이 진한 빨강부터


def draw_tracks(image, active):
    """영상 한 프레임(PIL RGB)에 번호 붙은 것의 이동 꼬리 · 상자 · '#번호 종류' 이름표를 그린다(제자리)."""
    from PIL import ImageDraw

    w, h = image.size
    lw = _line_width(w, h)
    font = _font(max(12, lw * 5))
    draw = ImageDraw.Draw(image)
    for t in active:
        color = track_color(t)
        pts = list(t.trail)
        for i in range(1, len(pts)):
            width = max(1, round(lw * (0.5 + i / len(pts))))  # 오래된 쪽이 가늘다
            draw.line((pts[i - 1], pts[i]), fill=color, width=width)
    for t in active:
        draw_box(draw, t.box, f"#{t.id} {class_name(t.cls)}", track_color(t), lw, font, w)
    return image


def smooth_keypoints(t, kpts, frame):
    """번호 하나의 관절을 부드럽게 — 이번에 보인 관절은 직전과 섞고, 안 보인 관절은 HOLD_FRAMES 까지 직전 값.
    바로 앞 프레임에 못 본 번호(놓쳤다 다시 찾음)는 섞지 않고 새로 시작한다(옛 자리로 끌려가지 않게)."""
    import numpy as np

    kpts = kpts.copy()
    fresh = t.kpts is None or frame - t.kpt_frame > 1
    t.kpt_frame = frame
    if fresh:
        t.kpts, t.kpt_age = kpts, np.zeros(17, dtype=np.int32)
        return kpts
    now = kpts[:, 2] >= KEYPOINT_MIN
    before = t.kpts[:, 2] >= KEYPOINT_MIN
    both = now & before
    kpts[both, :2] = SMOOTH * kpts[both, :2] + (1 - SMOOTH) * t.kpts[both, :2]
    hold = ~now & before & (t.kpt_age < HOLD_FRAMES)
    kpts[hold] = t.kpts[hold]
    age = np.where(hold, t.kpt_age + 1, 0)
    t.kpts, t.kpt_age = kpts, age
    return kpts


# ---------------------------------------------------------------------------
# 세션 · 진행률
# ---------------------------------------------------------------------------


class Progress:
    """ctx.progress 를 [lo, hi] 구간에 맞춰 부른다. 같은 단계는 PROGRESS_EVERY 초에 한 번까지."""

    def __init__(self, ctx, lo=0.0, hi=1.0):
        self.ctx, self.lo, self.hi = ctx, lo, hi
        self.last = lo
        self._t = 0.0
        self._stage = None

    def __call__(self, fraction, stage="work"):
        value = max(self.last, self.lo + (self.hi - self.lo) * max(0.0, min(1.0, fraction)))  # 뒤로 가지 않는다
        now = time.monotonic()
        if stage == self._stage and now - self._t < PROGRESS_EVERY:
            return
        self._t, self._stage, self.last = now, stage, value
        self.ctx.progress(value, stage)


def _session(ctx, progress, name):
    path = ctx.model_path(name)
    if path in _SESSIONS:
        return _SESSIONS[path]
    import onnxruntime as ort

    progress(0.0, "load-model")  # 세션을 만드는 동안 파이썬이 멈춘다 — 서버가 신호 없이 더 오래 기다리게 바로 앞에 알린다
    start = time.monotonic()
    so = ort.SessionOptions()
    so.log_severity_level = 3
    session = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    _SESSIONS[path] = session
    ctx.log(f"model {name} loaded in {time.monotonic() - start:.1f}s")
    return session


def _run(session, x):
    return session.run(None, {session.get_inputs()[0].name: x})


def _read(path):
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im.load()
        return ImageOps.exif_transpose(im).convert("RGB")


# ---------------------------------------------------------------------------
# 사진
# ---------------------------------------------------------------------------


def _photo_pose(ctx, progress, image, threshold):
    session = _session(ctx, progress, POSE_MODEL)
    x, r, left, top = letterbox(image)
    progress(0.3, "detect")
    raw = _run(session, x)[0][0]
    ctx.check_cancel()
    w, h = image.size
    rows, boxes = detections(raw, threshold, r, left, top, w, h)
    keep = distinct(rows, boxes)
    people = []
    for i in keep:
        k = rows[i, 6:].reshape(17, 3).copy()
        k[:, 0] = (k[:, 0] - left) / r
        k[:, 1] = (k[:, 1] - top) / r
        people.append(k)
    ctx.log(f"pose: {len(people)} people")
    progress(0.7, "draw")
    found = ctx.out_path("found.jpg")  # 사진 위에 그린 것 — PNG 면 몇 MB 라 JPEG 로
    draw_found(image, boxes[keep], [f"person {rows[i, 4]:.0%}" for i in keep], [BOX_COLORS[0]] * len(keep)).save(found, "JPEG", quality=92)
    if not people:
        return [], found
    out = ctx.out_path("pose.png")
    draw_skeletons(people, w, h).save(out, "PNG")
    return [out], found


def _photo_objects(ctx, progress, image, threshold, cutout):
    session = _session(ctx, progress, SEG_MODEL)
    x, r, left, top = letterbox(image)
    progress(0.3, "detect")
    raw, protos = _run(session, x)
    raw, protos = raw[0], protos[0]
    ctx.check_cancel()
    w, h = image.size
    rows, boxes = detections(raw, threshold, r, left, top, w, h)
    keep = distinct(rows, boxes)  # 점수 높은 것(확실한 것)부터
    found_path = ctx.out_path("found.jpg")
    draw_found(
        image,
        boxes[keep],
        [f"{class_name(rows[i, 5])} {rows[i, 4]:.0%}" for i in keep],
        [BOX_COLORS[int(rows[i, 5]) % len(BOX_COLORS)] for i in keep],
    ).save(found_path, "JPEG", quality=92)
    found = []
    for i in keep:
        box = boxes[i]
        bw, bh = box[2] - box[0], box[3] - box[1]
        if min(bw, bh) < MIN_CROP_PX or (not cutout and bw * bh >= WHOLE_IMAGE * w * h):
            continue
        found.append((rows[i], box))
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
        out = ctx.out_path(f"{i:02d}-{class_name(row[5])}.png")
        piece.save(out, "PNG")
        outs.append(out)
        progress(0.4 + 0.55 * (i + 1) / min(len(found), MAX_ITEMS), "save")
    ctx.log(f"objects: {len(keep)} found, {len(outs)} saved")
    return outs, found_path


# ---------------------------------------------------------------------------
# 영상
# ---------------------------------------------------------------------------


class VideoWriter:
    """H.264(libx264) · yuv420p mp4. 프레임마다 번호(pts)를 0, 1, 2 … 로 붙여 원본과 프레임 수 · fps 가 같다.
    yuv420p 는 짝수 크기만 되므로 홀수면 오른쪽 · 아래에 가장자리를 1px 복제해 덧댄다."""

    def __init__(self, path, width, height, rate, crf):
        import av

        self.path = path
        self.closed = False
        self.width, self.height = width, height
        self.pad = (height & 1, width & 1)
        self.container = av.open(path, "w", options={"movflags": "+faststart"})  # 받는 중에도 재생되게 머리 정보를 앞에
        stream = self.container.add_stream("libx264", rate=rate)
        stream.width, stream.height = width + self.pad[1], height + self.pad[0]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": "veryfast"}
        stream.codec_context.time_base = 1 / rate
        self.stream = stream
        self.time_base = 1 / rate
        self.count = 0

    def write(self, array):
        """RGB (h, w, 3) 또는 흑백 (h, w) uint8."""
        import av
        import numpy as np

        if self.pad != (0, 0):
            spec = ((0, self.pad[0]), (0, self.pad[1])) + (((0, 0),) if array.ndim == 3 else ())
            array = np.pad(array, spec, mode="edge")
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(array), format="rgb24" if array.ndim == 3 else "gray")
        frame.pts = self.count
        frame.time_base = self.time_base
        self.count += 1
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self):
        """남은 프레임을 내보내고 닫는다."""
        if self.closed:
            return
        self.closed = True
        try:
            for packet in self.stream.encode(None):
                self.container.mux(packet)
        finally:
            self.container.close()

    def abandon(self):
        """실패 · 취소 때 — 파일만 닫는다(사이드카가 떠 있는 동안 손잡이가 남지 않게)."""
        if not self.closed:
            self.closed = True
            try:
                self.container.close()
            except Exception:
                pass


def _video_rate(stream):
    from fractions import Fraction

    for rate in (stream.average_rate, stream.guessed_rate, stream.base_rate):
        if rate and 0 < float(rate) <= 240:
            return Fraction(rate).limit_denominator(1001 * 1000)
    return Fraction(30)


def _video_seconds(container, stream):
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return container.duration / 1_000_000
    return 0.0


class _Frames:
    """디코드한 프레임 → 보이는 방향의 원본 RGB 배열 · 모델 입력용으로 줄인 RGB 배열. 크기는 첫 프레임으로 정한다."""

    def __init__(self, first):
        self.rotation = self._rotation(first)
        if self.rotation % 2 == 0:
            self.width, self.height = first.width, first.height
        else:
            self.width, self.height = first.height, first.width
        self.r, self.nw, self.nh, self.left, self.top = letterbox_geometry(self.width, self.height)

    @staticmethod
    def _rotation(frame):
        """화면 회전 정보(반시계 각도) → 반시계 90° 단위 0~3. 휴대폰 세로 영상은 가로로 저장되고 이 값으로 세워 보인다."""
        return int(round((getattr(frame, "rotation", 0) or 0) / 90.0)) % 4

    def arrays(self, frame):
        import numpy as np
        from PIL import Image

        rotation = self._rotation(frame)
        if rotation:
            full = np.ascontiguousarray(np.rot90(frame.to_ndarray(format="rgb24"), rotation))
            image = Image.fromarray(full)
            if image.size != (self.width, self.height):  # 중간에 크기 · 회전이 바뀐 영상 — 첫 프레임 크기에 맞춘다
                image = image.resize((self.width, self.height), Image.BILINEAR)
                full = np.asarray(image)
            small = np.asarray(image.resize((self.nw, self.nh), Image.BILINEAR))
            return full, small
        if (frame.width, frame.height) != (self.width, self.height):
            frame = frame.reformat(width=self.width, height=self.height)
        full = frame.to_ndarray(format="rgb24")
        small = frame.reformat(width=self.nw, height=self.nh, format="rgb24", interpolation="BILINEAR").to_ndarray()
        return full, small


def _video_frame(job, frame):
    """프레임 하나 → 결과 영상 · 찾기 · 추적 영상에 한 장씩."""
    import numpy as np
    from PIL import Image

    g = job["frames"]
    full, small = g.arrays(frame)
    outputs = _run(job["session"], model_input(small, g.left, g.top))
    rows, boxes = detections(outputs[0][0], job["low"], g.r, g.left, g.top, g.width, g.height)
    keep = distinct(rows, boxes)
    rows, boxes = rows[keep], boxes[keep]
    tracker = job["tracker"]
    active = tracker.update(boxes, rows[:, 4], rows[:, 5])

    if job["pose"]:
        people = []
        for t, d in active:
            k = rows[d, 6:].reshape(17, 3).copy()
            k[:, 0] = (k[:, 0] - g.left) / g.r
            k[:, 1] = (k[:, 1] - g.top) / g.r
            people.append(smooth_keypoints(t, k, tracker.frame))
        job["result"].write(np.asarray(draw_skeletons(people, g.width, g.height)))
    else:
        protos = outputs[1][0]
        union = np.zeros(protos.shape[1:], dtype=np.float32)
        if active:
            idx = [d for _, d in active]
            probs = mask_probs(rows[idx, 6:38], protos, rows[idx, :4])
            for (t, _), p in zip(active, probs):
                t.mask = p
            union = probs.max(axis=0)
        for t in tracker.held():  # 한두 프레임 놓친 것은 직전 마스크 그대로
            if t.mask is not None:
                union = np.maximum(union, t.mask)
        prev = job["mask"]
        job["mask"] = union if prev is None else (SMOOTH * union + (1 - SMOOTH) * prev).astype(np.float32)
        scale = protos.shape[2] / SIZE
        region = (g.left * scale, g.top * scale, (g.left + g.nw) * scale, (g.top + g.nh) * scale)
        up = Image.fromarray(job["mask"]).resize((g.width, g.height), Image.BILINEAR, box=region)  # 2차원 float32 → F
        job["result"].write(np.where(np.asarray(up) >= 0.5, 255, 0).astype(np.uint8))

    job["found"].write(np.asarray(draw_tracks(Image.fromarray(full), [t for t, _ in active])))


def _video(ctx, progress, path, mode, threshold):
    """영상 한 편 → {"video": 결과 영상, "found-video": 찾기 · 추적 영상}."""
    import av

    pose = mode == "pose"
    session = _session(ctx, progress, POSE_MODEL if pose else SEG_MODEL)
    try:
        container = av.open(path)
    except av.error.FFmpegError as exc:
        raise ValueError(f"영상 파일을 열지 못했습니다. mp4 · webm · mov 영상인지 확인해 주세요. ({exc})") from None
    writers = []
    try:
        if not container.streams.video:
            raise ValueError("이 파일에는 영상이 없습니다.")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        rate = _video_rate(stream)
        seconds = _video_seconds(container, stream)
        if seconds > MAX_VIDEO_SEC:
            raise ValueError(
                f"영상이 너무 깁니다({seconds:.0f}초). {MAX_VIDEO_SEC // 60}분({MAX_VIDEO_SEC}초)까지 받습니다 — 영상을 잘라서 이어 주세요."
            )
        expected = max(1, stream.frames or round(seconds * float(rate)))
        job = {
            "session": session,
            "pose": pose,
            "low": min(TRACK_LOW, threshold),
            "tracker": Tracker(threshold, float(rate)),
            "mask": None,
        }
        budget = TIME_BUDGET_SEC.get(getattr(ctx, "device", ""), TIME_BUDGET_DEFAULT_SEC)
        t0 = time.monotonic()
        n = 0
        checked = 0.0
        try:
            for frame in container.decode(stream):
                ctx.check_cancel()
                spent = time.monotonic() - t0
                if spent >= TIME_CHECK_AFTER_SEC and spent - checked >= TIME_CHECK_AFTER_SEC and n > 0:
                    checked = spent
                    total = spent / n * expected
                    if total > budget:
                        raise ValueError(
                            f"이 PC 에서는 이 영상을 처리하는 데 약 {math.ceil(total / 60)}분이 걸려 한 번에 돌 수 있는 시간"
                            f"(약 {budget // 60}분)을 넘습니다. 영상을 짧게 자르거나 해상도를 낮춰 이어 주세요."
                        )
                if n == 0:
                    g = _Frames(frame)
                    ctx.log(f"video {g.width}x{g.height} {float(rate):.3f}fps ~{expected} frames rotation {g.rotation * 90}")
                    job["frames"] = g
                    job["result"] = VideoWriter(ctx.out_path("pose.mp4" if pose else "mask.mp4"), g.width, g.height, rate, CRF_FLAT)
                    writers.append(job["result"])
                    job["found"] = VideoWriter(ctx.out_path("found.mp4"), g.width, g.height, rate, CRF_DRAWING)
                    writers.append(job["found"])
                _video_frame(job, frame)
                n += 1
                progress(min(n / expected, 0.99), "video")
        except av.error.FFmpegError as exc:
            if n == 0:
                raise ValueError(f"영상을 읽지 못했습니다. ({exc})") from None
            ctx.log(f"video decode stopped at frame {n}: {exc}", "warn")  # 끝부분이 깨진 파일 — 읽은 데까지 낸다
        if n == 0:
            raise ValueError("영상에서 프레임을 하나도 읽지 못했습니다.")
        progress(0.995, "encode")
        for writer in writers:
            writer.close()
        spent = time.monotonic() - t0
        ctx.log(f"video: {n} frames in {spent:.1f}s ({spent / n * 1000:.0f} ms/frame), {job['tracker'].next_id - 1} ids")
        return {"video": job["result"].path, "found-video": job["found"].path}
    finally:
        for writer in writers:
            writer.abandon()
        container.close()


# ---------------------------------------------------------------------------
# 노드
# ---------------------------------------------------------------------------


def yolo(ctx, inputs, settings):
    image_path, video_path = inputs.get("image"), inputs.get("video")
    if not image_path and not video_path:
        raise ValueError("사진이나 영상을 하나 이상 이어 주세요(둘 다 이으면 둘 다 처리합니다).")
    threshold = CONFIDENCE[settings["confidence"]]
    mode = settings["mode"]
    outputs = {}
    split = 0.1 if (image_path and video_path) else 1.0  # 둘 다면 진행률 앞 10% 를 사진에
    if image_path:
        progress = Progress(ctx, 0.0, split)
        progress(0.02, "read")
        image = _read(image_path)
        ctx.check_cancel()
        if mode == "pose":
            images, found = _photo_pose(ctx, progress, image, threshold)
        else:
            images, found = _photo_objects(ctx, progress, image, threshold, cutout=(mode == "cutout"))
        outputs["images"] = images
        outputs["found"] = found
    if video_path:
        progress = Progress(ctx, split if image_path else 0.0, 1.0)
        progress(0.0, "read")
        outputs.update(_video(ctx, progress, video_path, mode, threshold))
    ctx.progress(1.0, "done")
    return outputs


NODES = {"yolo": yolo}
