"""README 그림의 재료 — 이 저장소의 nodes/main.py 를 서버 없이 그대로 돌려 사진 · 영상 결과를 만든다.

사용(저장소 루트에서, nodes/requirements.in 의 패키지를 깐 파이썬으로):
    python -I -B docs/scripts/make_results.py <작업 폴더>
하는 일
  1) 모델: manifest.json requires.models 의 주소(Ultralytics 공식 릴리스)에서 받아 sha256 을 맞춰 본다 → <작업>/models/
  2) 영상: 위키미디어 공용에서 퍼블릭 도메인 영화 두 편을 받아 README 에 쓴 구간을 자른다(ffmpeg) → <작업>/video/
       ella.mp4    「Ella Lola, a la Trilby」(1898) 330 ~ 521번째 프레임(13.75초부터 8초), 원래 크기 1364×1026
       lumiere.mp4 「La Sortie de l'usine Lumière à Lyon」(1895) 4 ~ 12초, 1024×768
  3) 노드 실행 → <작업>/results/
       사진(cards/): 포즈 뼈대(dancer · yoga · friends), 물체 잘라내기 · 오려내기(desk · park), 찾는 기준 넉넉하게 · 엄격하게(desk)
       영상: ella 포즈 뼈대, lumiere 물체 잘라내기
이미 받은 모델 · 영상은 다시 받지 않는다.
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
work = os.path.abspath(sys.argv[1])
models = os.path.join(work, "models")
video = os.path.join(work, "video")
results = os.path.join(work, "results")
UA = {"User-Agent": "PromBoard-plugin-readme/1.0 (https://github.com/promboard-app)"}
FILMS = {
    "ella": ("https://upload.wikimedia.org/wikipedia/commons/e/e1/Ella_Lola%2C_a_la_Trilby.webm",
             ["-vf", "select='between(n,330,521)',setpts=N/24/TB", "-r", "24"]),
    "lumiere": ("https://upload.wikimedia.org/wikipedia/commons/0/09/La_Sortie_de_l%27Usine_Lumi%C3%A8re_%C3%A0_Lyon_I_1895.webm",
                ["-ss", "4", "-t", "8", "-vf", "scale=1024:768:flags=lanczos"]),
}


def fetch(url, dst):
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA)) as r, open(dst + ".part", "wb") as f:
            f.write(r.read())
        os.replace(dst + ".part", dst)
    return dst


# 1) 모델
manifest = json.load(open(os.path.join(ROOT, "manifest.json"), encoding="utf-8"))
for m in manifest["requires"]["models"]:
    path = fetch(m["url"], os.path.join(models, m["path"]))
    if hashlib.sha256(open(path, "rb").read()).hexdigest() != m["sha256"]:
        raise SystemExit(f"sha256 이 manifest 와 다르다: {path}")

# 2) 영상
for name, (url, args) in FILMS.items():
    dst = os.path.join(video, name + ".mp4")
    if os.path.exists(dst):
        continue
    src = fetch(url, os.path.join(video, "source", name + ".webm"))
    pre = args[:4] if args[0] == "-ss" else []
    post = args[4:] if args[0] == "-ss" else args
    subprocess.run(["ffmpeg", "-v", "error", "-y", *pre, "-i", src, "-an", *post, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "18", "-preset", "fast", dst], check=True)

# 3) 노드 실행
spec = importlib.util.spec_from_file_location("yolo_main", os.path.join(ROOT, "nodes", "main.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class Ctx:
    device = "cpu"

    def __init__(self, out):
        self.out = out
        os.makedirs(out, exist_ok=True)

    def out_path(self, name):
        return os.path.join(self.out, name)

    def model_path(self, name):
        return os.path.join(models, name)

    def progress(self, *_a, **_k):
        return True

    def check_cancel(self):
        pass

    def log(self, msg, level="info"):
        print(f"  [{level}] {msg}", flush=True)


def run(folder, inputs, mode, confidence="normal"):
    print(folder, flush=True)
    mod.NODES["yolo"](Ctx(os.path.join(results, folder)), inputs, {"mode": mode, "confidence": confidence})


cards = os.path.join(ROOT, "cards")
for name in ("dancer", "yoga", "friends"):
    run(f"{name}-pose", {"image": os.path.join(cards, name + ".jpg")}, "pose")
for name in ("desk", "park"):
    for mode in ("crop", "cutout"):
        run(f"{name}-{mode}", {"image": os.path.join(cards, name + ".jpg")}, mode)
for conf in ("loose", "strict"):
    run(f"desk-crop-{conf}", {"image": os.path.join(cards, "desk.jpg")}, "crop", conf)
run("ella-pose", {"video": os.path.join(video, "ella.mp4")}, "pose")
run("lumiere-crop", {"video": os.path.join(video, "lumiere.mp4")}, "crop")
print("done", results)
