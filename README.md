# Ultralytics YOLO — 포즈 뼈대 · 물체 잘라내기

PromBoard 커뮤니티 플러그인 예시입니다. 캔버스에 **YOLO 포즈 · 물체** 노드 하나를 더합니다. 노드의 모드 알약으로 고릅니다.

| 모드 | 내보내는 것 |
|---|---|
| 포즈 뼈대 | 사진 속 사람(여러 명도)의 자세를 검은 바탕 색 뼈대 그림 한 장으로. 이미지 생성 노드에 참조로 이어 '같은 자세로' 그릴 때 씁니다 |
| 물체 잘라내기 | 사람 · 물건을 찾아 하나씩 네모로 잘라 여러 장으로(확실한 것부터, 64장까지) |
| 물체 오려내기 | 같은 것을 배경을 지운 투명 PNG 여러 장으로 |

![사진 → 포즈 뼈대 → 같은 자세로 새 인물](workflows/pose-to-new-person.jpg)

- 플러그인 id: `promboard.yolo`
- 모델: Ultralytics 공식 **YOLO26n** 포즈 · 분할 ONNX(각 약 12MB)를 원래 출처 [ultralytics/assets v8.4.0 릴리스](https://github.com/ultralytics/assets/releases/tag/v8.4.0)에서 받습니다. 파일마다 sha256 을 고정했습니다.
- 그래픽카드 없이 CPU 로 돕니다(onnxruntime, torch 없음). 사진 한 장에 1초 안팎입니다.
- '찾는 기준' 알약: 넉넉하게 · 보통 · 엄격하게(찾았다고 볼 점수 0.15 · 0.25 · 0.45).

## 들어 있는 것

- 예제 사진 5장(참조 이미지 카드): 춤추는 사람 · 요가 자세 · 두 친구(포즈용), 책상 위 물건 · 공원 산책길(물체용). GPT Image 2.5 로 만든 사진입니다.
- 워크플로우 「사진 → 포즈 뼈대 → 같은 자세로 새 인물」: 소스 이미지 → YOLO(포즈 뼈대) → 이미지 노드(뼈대를 참조 그림으로) + 프롬프트 노드. 소스 이미지에 사진을 올리고 YOLO 노드 ▷, 이미지 노드 ▷ 를 차례로 누릅니다. 참조 그림을 받는 이미지 모델을 고르세요.

## 설치

1. PromBoard 앱 → 플러그인 매니저 → **커뮤니티 플러그인** → 둘러보기에서 설치합니다.
2. 코드가 든 플러그인이라 **제한 모드**를 끄고 플러그인을 켭니다.
3. 처음 실행할 때 실행 준비가 파이썬 패키지(onnxruntime · numpy · Pillow)와 모델 2개(약 23MB)를 받습니다(한 번만).
4. 캔버스 빈 곳을 우클릭하면 메뉴 맨 아래 플러그인 묶음에 **YOLO 포즈 · 물체**가 있습니다.

## 플러그인 만드는 사람을 위한 메모

```
manifest.json               플러그인 정보 · 노드 정의(layout single, 출력 포트 multiple) · 모델(requires.models)
nodes/main.py               NODES = {"yolo": 함수} — 레터박스 전처리, end2end 출력 읽기, OpenPose 색 뼈대, 분할 마스크
nodes/requirements*.txt     받을 패키지 · 운영체제별 해시 잠금 파일
cards/                      예제 사진(참조 이미지 카드)
workflows/                  워크플로우(앱의 워크플로우 저장 모양 그대로)
```

- YOLO26 ONNX 는 NMS 가 모델 안에 들어 있어(end2end) 출력이 `(1, 300, 6 + 추가)` 입니다: `[x1, y1, x2, y2, 점수, 종류, …]`(640 레터박스 좌표). 포즈는 추가가 관절 17개 × (x, y, 점수), 분할은 마스크 계수 32개 + 마스크 바탕 `(1, 32, 160, 160)`.
- 종류가 다른 같은 물체를 두 번 찾는 일이 있어 겹침(IoU 0.7 이상)을 한 번 더 거릅니다.
- 잠금 파일 다시 만들기: `uv pip compile nodes/requirements.in --generate-hashes --python-version 3.12 --python-platform x86_64-pc-windows-msvc --index-url https://pypi.org/simple -o nodes/requirements-win.txt` (맥은 `MACOSX_DEPLOYMENT_TARGET=14.0` + `aarch64-apple-darwin`).

---

## English

A PromBoard community plugin example. It adds one **YOLO pose · objects** node (`YOLO 포즈 · 물체`) to the canvas. Pick the mode with the node's pill:

| Mode | Output |
|---|---|
| Pose skeleton (포즈 뼈대) | One OpenPose-colored skeleton image on black for all people in the photo — connect it to an image generation node as a reference to draw "the same pose" |
| Crop objects (물체 잘라내기) | People and objects cropped one by one as rectangles (most confident first, up to 64) |
| Cut out objects (물체 오려내기) | The same objects as transparent PNGs with the background removed |

- Plugin id: `promboard.yolo`
- Models: official Ultralytics **YOLO26n** pose and segmentation ONNX files (about 12 MB each), downloaded from the original source ([ultralytics/assets v8.4.0](https://github.com/ultralytics/assets/releases/tag/v8.4.0)) with pinned sha256.
- Runs on the CPU with onnxruntime (no torch, no graphics card). About one second per photo.
- Includes five example photos (reference image cards, made with GPT Image 2.5) and the workflow "photo → pose skeleton → new person in the same pose".
- **Install**: PromBoard → Plugin Manager → **Community plugins** → Browse → install. Turn off **Restricted mode** and enable the plugin. The first run prepares the Python packages and the two models (about 23 MB, once). Right-click an empty spot on the canvas: the node is in the plugin group at the bottom of the menu.

The plugin texts in the app are Korean only for now — the plugin manifest takes a single language.

## License

AGPL-3.0 — see [LICENSE](LICENSE). The YOLO26 weights are © Ultralytics, licensed under [AGPL-3.0](https://ultralytics.com/license), and are downloaded from Ultralytics' release, not bundled. onnxruntime (MIT), numpy (BSD-3-Clause) and Pillow (MIT-CMU) are installed from PyPI at prepare time.
