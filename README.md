# flawmap — 촬영 영상으로 물건의 하자를 확인하는 검수 플랫폼

물건 둘레를 찍은 영상을 올리면, 물건의 3D 모델에 촬영을 맞춰 하자를 찾고 웹 뷰어에서 빨갛게 표시합니다. 하자를 누르면 그 하자가 보이는 원본 촬영 프레임과 영상 시각을 보여 주고, 촬영에서 빠진 부분은 빗금으로 보여 줍니다. 지금은 휴대폰을 지원합니다(iPhone 11·12는 애플 도면 치수로 만든 모델, 그 밖은 촬영본에서 잰 판 모양 모델).

웹 서비스로 동작합니다. 검수를 요청한 사람이 촬영본을 올리면 서버가 차례로 처리하고, 요청자는 결과를 확인한 뒤 링크로 공유하거나 공개 목록에 올립니다. 결과를 받은 사람은 같은 결과를 읽기 전용으로 봅니다. 중고 거래, 대여와 반납, 수리 접수, 보험 청구처럼 물건의 상태를 남기고 나눠야 하는 곳에 쓰는 것을 생각하고 있습니다.

2026-2 산학캡스톤디자인, 팀 강화실패.

처음 받은 팀원은 `docs/project_guide.md`(프로젝트 안내서)를 먼저 읽어 주세요. 환경, 데이터 목록과 받는 법, 시스템 구조, 지금까지의 실험 결과와 알게 된 점, 촬영 안내, 할 일이 한 문서에 있습니다.

---

## 1. 동작 흐름과 현재 상태

원본 모델은 애플이 공개한 치수 도면의 숫자로 직접 만들고, 촬영의 모든 프레임을 이 모델의 좌표(mm)에 맞춥니다. 유리와 금속은 사진만으로 형상을 복원하기 어렵기 때문에, 형상은 원본 모델에서 가져오고 촬영본은 외관을 읽는 데 씁니다.

| 단계 | 하는 일 | 상태 | 코드 |
|---|---|---|---|
| 원본 모델 | 도면 치수(YAML)로 몸체, 카메라 단차, 렌즈, 버튼을 만들고 부위와 펼침 좌표를 붙임 | 있음 (iPhone 11, 12) | `flawmap/anchor/parametric.py` |
| 촬영 보드 | A3 ChArUco 보드, 가운데 회색 영역에 기종 외곽선과 화살표 | 있음 | `flawmap/poses/board.py` |
| 촬영 입력 | Stray Scanner 영상, 깊이, ARKit 위치. 선명한 프레임 고르기 | 있음 | `flawmap/io/stray.py`, `flawmap/frames/select.py` |
| 카메라 위치 (보드) | 마커 모서리로 풀고 체스판 모서리로 다듬음, 보드가 안 보이는 장면은 ARKit | 있음 | `flawmap/poses/board.py`, `align.py` |
| 카메라 위치 (보드 없이) | ARKit 위치의 흐름을 기기 외곽으로 바로잡음: 바닥(LiDAR) 위 기기 찾기, 렌즈로 앞뒤 구분, 기기 위치와 흐름을 함께 맞춤 | 있음 (가상 촬영본으로 확인) | `flawmap/poses/boardless.py` |
| 원본 모델 정합 | 원본 모델의 실루엣을 영상의 경계에 맞춤 (6자유도) | 있음 | `flawmap/anchor/register.py` |
| 펼친 외관 지도 | 모든 장면의 색을 액정·후면·프레임 띠 좌표에 모으고, 촬영된 정도를 계산 | 있음 | `flawmap/surface/maps.py` |
| 하자 찾기 | 프레임 까짐(색), 프레임 찍힘(윤곽), 유리 깨짐·흠집(가는 선) | 있음 (가상 하자로만 확인) | `flawmap/detect/` |
| 근거 프레임 | 하자마다 잘 보이는 장면 3장과 영상 시각 | 있음 | `flawmap/evidence/select.py` |
| 웹 뷰어 | 3D 검사 보기, 실제 영상 360 보기, 하자를 누르면 근거 장면과 영상 | 있음 | `web/`, `flawmap/server/app.py` |
| 도면 없는 기기 | 촬영본에서 길이·폭·두께·모서리를 잰 판 모양 모델. 잰 크기로 기종을 추정하고, 좌우 대칭인 구조물(수화부, 홈 버튼)은 하자에서 뺌 | 실험 단계 | `flawmap/anchor/template.py` |
| 검수 서비스 | 촬영본 업로드(또는 공개 영상, 가상 촬영본), 받기 검사, 대기열과 진행률, 요청자 화면, 링크 공유와 공개 목록, 읽기 전용 결과 | 있음 | `flawmap/service/`, `web/service/` |
| 공개 영상 처리 | uCO3D 휴대폰 영상을 도면 없이 처리 (크기는 폭을 가정) | 실험 단계 | `flawmap/io/uco3d.py`, `flawmap uco3d` |
| 실제 촬영본 확인 | 팀이 찍은 아이폰으로 전 과정 확인 | **아직 없음** | `flawmap check`, `flawmap run` |

## 2. 실행 환경

| 항목 | 내용 |
|---|---|
| 확인한 장비 | Ubuntu 24.04, NVIDIA RTX 3080 Ti (12 GB), CPU 24코어, RAM 61 GB |
| 파이썬 | conda 환경 **`env_isaaclab`** (Python 3.11). 새 가상환경을 만들지 말고 이 환경만 씁니다 |

ROS 2가 설치된 셸은 Python 3.10용 `PYTHONPATH`를 내보내므로 먼저 비웁니다.

```bash
conda activate env_isaaclab
export PYTHONPATH=
```

## 3. 설치

```bash
conda activate env_isaaclab && export PYTHONPATH=
bash scripts/install_extra_deps.sh --dry-run   # 기존 패키지가 바뀌지 않는지 먼저 확인
bash scripts/install_extra_deps.sh             # scikit-image, plyfile, gdown, remotezip
pip install -e . --no-deps --no-build-isolation
```

`scripts/constraints-isaacsim.txt`는 이 환경에 고정된 버전(`numpy==1.26.0`, `psutil==5.9.8`, `torch==2.7.0+cu128`)이 추가 패키지 때문에 바뀌지 않게 합니다.

## 4. 데이터

데이터는 git에 없습니다. 애플 도면은 재배포가 금지돼 있고, 공개 데이터셋은 크고 각자의 라이선스가 있기 때문입니다. 팀 서버의 저장소 폴더 `data/`에는 아래 것이 이미 있습니다. 다른 컴퓨터에서는 아래 명령으로 받거나 서버에서 복사합니다. 데이터마다 받은 이유, 형식, 알게 된 점은 `docs/project_guide.md`(3–4절)에 자세히 있습니다.

| 데이터 | 무엇 | 왜, 어디에 쓰나 | 위치 | 크기 | 라이선스 | 상태 |
|---|---|---|---|---|---|---|
| 애플 Dimensional Drawings (`iphone-12`, `-mini`, `-pro`, `-pro-max`, 아이패드 7종) | 기종별 치수 도면 PDF | 원본 모델의 치수. 읽은 숫자만 `configs/devices/*.yaml`에 있고 `flawmap/anchor/parametric.py`가 모델을 만듦 | `data/drawings/apple/` | 각 0.4–0.9 MB | 애플 약관, 재배포 금지 | 받음 |
| 애플 Accessory Design Guidelines 2020-03-10 (웨이백) | 이전 판 가이드라인 | 아이폰 11 치수(현재 도면 목록에 없음) | `data/drawings/apple/archive/` | 30 MB | 애플 약관 | 받음 |
| uCO3D `cellular_telephone` | 크라우드 휴대폰 360° 영상 1,324개, SfM 카메라, 마스크, 분할 점군, 3DGS | 보드·LiDAR·도면 없는 일반 영상. 판 모양 모델 경로와 서비스의 공개 영상 데모. 액정 깨진 휴대폰이 약 200개 | `data/datasets/uco3d/` | 66 GB (메타데이터 13 GB 포함) | CC BY 4.0 | 받음 |
| OmniObject3D 전자기기 7개 범주 | iPhone 12 Pro 360° 영상 + 스캐너 메시 | 정답 형상이 있는 실제 영상으로 정합 정확도 확인 (아직 안 씀) | `data/datasets/omniobject3d/` | 계획 약 20 GB | CC BY 4.0 | Google Drive 할당량으로 14개 중 1개 |
| MobileBrick | LEGO, iPhone LiDAR + ARKit + CAD 정답 | 실제 ARKit 흐름·LiDAR로 위치와 정합 확인 (아직 안 씀) | `data/datasets/mobilebrick/` | 13.1 GB | MIT | 받는 중 (이어받기) |
| Objaverse 1.0 메타데이터 + 아이폰 모델 4개 | Sketchfab 객체 목록 | 남이 만든 아이폰 모델을 앵커로 쓸 수 있는지 비교. 비율이 1–2% 어긋나 쓰지 않음 | `data/datasets/objaverse/` | 577 MB | 객체별 CC | 받음 |
| 가상 촬영본 6개 | 원본 모델을 렌더한 Stray 형식 촬영본 + 정답 `gt.json` (보드 4개, 나무·종이 바닥 2개) | 실제 촬영 전 개발과 정량 평가, 서비스 가상 샘플 | `data/synthetic/` | 7–34 MB씩 | 직접 만듦 | `scripts/fetch_demo_data.sh`로 다시 만듦 |

받지 않은 것과 이유: CO3Dv2(비상업 라이선스), WildRGB-D(전면 TrueDepth, 데이터 라이선스 미표기), MVImgNet(신청서), SAM 3(승인), Roboflow·Kaggle·GrabCAD·Sketchfab(로그인), MSD·SSGD 화면 결함 사진(학습 경로가 확장 목표라 보류).

```bash
bash scripts/fetch_demo_data.sh                    # 데모에 필요한 것만: uCO3D 메타데이터와 데모 영상 5개, 가상 촬영본, 업로드 zip, three.js
bash scripts/download_datasets.sh                  # 전부: omniobject3d, mobilebrick, uco3d (남은 디스크 하한 50 GB를 지킴)
bash scripts/retry_omniobject3d.sh                 # Google Drive 할당량에 막히면 1시간마다 재시도
python -m flawmap.io.download uco3d --sequences <이름> ...   # uCO3D 시퀀스 몇 개만
python scripts/search_objaverse.py                 # Objaverse 메타데이터 검색
```

### 데모용 공개 영상 (uCO3D)

"crack"과 "screen" 캡션, 1080p 이상, 카메라 150대 이상인 86개 가운데 12개를 눈으로 보고 처리해 골랐습니다. 대조용으로 하자 설명이 없는 휴대폰도 처리했습니다. 이 목록은 `configs/demo_public.yaml`에 있고, `/new`의 "공개 영상으로 해 보기"에 나옵니다.

| 순서 | 시퀀스 | 장면 | 처리 결과 (판 모양 모델, 폭 71.5 mm 가정) |
|---|---|---|---|
| 1 | `2712-37354-8763` | 청록 테두리 휴대폰, 나무 판자, 액정 금 | 하자 8 (깨짐·높음 3). 금이 3D 텍스처에도 보임. **가장 추천** |
| 2 | `59015-27660-21826` | 흰 아이폰, 검은 바닥, 액정 금 | 하자 8 (깨짐·높음 5). 수화부·홈 버튼·테두리는 대칭 규칙으로 빠짐 |
| 3 | `24925-17239-690` | 흰 휴대폰, 초록 천, 액정 금 | 하자 8 (깨짐·높음 2) |
| 4 | `29776-61880-12178` | 흰·청록 휴대폰, 나무 판자, 액정 금 | 하자 7 (깨짐·높음 1) |
| 대조군 | `9557-8788-41196` | 흰 아이폰, 하자 설명 없음 | 하자 6 (깨짐·높음 1). 오른쪽 가장자리에 잘못 잡힘 |

판 맞춤은 데이터셋 마스크와 잘 맞습니다(IoU 0.83–0.98). 다만 깨진 것과 깨끗한 것의 차이가 아직 약하고 정답 위치가 없어, 정량 평가가 아닌 정성 시연으로 씁니다. 크기는 가정값이고, 테두리 하자는 판 모델에서 보고하지 않습니다.

## 5. 촬영 방법 (요약)

1. 케이스와 필름을 벗기고 화면을 끄고 닦은 기기를, **기기와 색이 다른 바닥** 위 고른 조명 아래에 놓습니다.
2. 다른 휴대폰(LiDAR가 있는 iPhone Pro 또는 iPad Pro)의 Stray Scanner로, 기기가 화면의 30–60%를 차지하게 천천히 세 높이(약 20°, 45°, 70°)로 한 바퀴씩 돌고 위에서 한 번 훑습니다.
3. 화면이 위, 뒷면이 위로 두 번 찍습니다. 서비스에서는 Stray Scanner 폴더를 파일 앱에서 압축해 올리고, 서버에서 직접 돌릴 때는 `data/captures/<기기>_<면>_<회차>/`에 둡니다.
4. 정밀하게 재려면 인쇄 보드를 씁니다: `flawmap board --device iphone-12`로 만든 PDF를 A3에 **실제 크기(100%)**로 인쇄하고 칸이 25 mm인지 잰 뒤, 회색 영역의 외곽선에 기기를 맞춰 놓습니다. 보드가 찍히면 자동으로 보드로 위치를 잽니다.

## 6. 실행

### 웹 서비스

```bash
bash scripts/fetch_demo_data.sh                          # 처음 한 번: 데모 데이터, 업로드용 zip, three.js (서버에서는 이미 있음)
flawmap serve --port 8000                                # http://localhost:8000 (다른 기기에서 열려면 --host 0.0.0.0)
bash scripts/serve_detached.sh --port 8000               # 로그아웃해도 계속 돌게 띄우기 (끄기: ... stop)
```

| 주소 | 화면 |
|---|---|
| `/new` | 검수 요청: 물건 종류와 기종, 촬영본(zip)을 올리거나 서버의 공개 영상·가상 촬영본으로 시작 |
| `/s/<요청자 토큰>` | 요청자: 대기 순서와 단계별 진행률, 결과, 더 찍을 곳, 링크 공유와 공개 목록, 다시 찍어 올리기 |
| `/gallery` | 공개 목록에 올린 결과 (검수 표) |
| `/r/<공유 토큰>` | 링크를 받은 사람: 읽기 전용 결과 (검수 번호, 처리 시각, 영상 SHA-256, 처리 방식) |
| `/ops` | 운영자 (이 서버에서만): 대기열, 처리 시간, 실패, 남은 디스크 |

공개 영상 목록은 `configs/demo_public.yaml`에 있습니다(uCO3D, CC BY 4.0, 서버에 받아 둔 것만 보입니다). 저장은 `data/service/`(SQLite, 올린 촬영본, 결과)에 합니다. 처리는 한 번에 하나씩 별도 프로세스에서 메모리 감시와 함께 돌고, 결과 파일은 토큰이 맞을 때만 내보냅니다.

### 명령으로 처리

```bash
flawmap board --device iphone-12                         # 인쇄할 보드 PDF
flawmap check data/captures/iphone_back_1 --device iphone-12 --face-up back   # 첫 촬영 점검: 디코딩, 높이별 보드 검출, 모델 맞춤
flawmap run --device iphone-12 \
    --pass back=data/captures/iphone_back_1 --pass screen=data/captures/iphone_screen_1 \
    --out runs/iphone_demo                               # 하자 찾기와 뷰어 묶음 (runs/iphone_demo/bundle)
bash scripts/fetch_web_vendor.sh                         # 뷰어용 three.js를 한 번 받아 둠 (인터넷 없이 시연)
flawmap serve --port 8000                                # http://localhost:8000/?run=iphone_demo
flawmap run --device auto --pass back=... --pass screen=...   # 기종을 모를 때: 바닥·배경 제약 없이 SAM 2로 기기를 찾고, 잰 크기로 추정 (맞는 도면이 없으면 판 모양 모델)
flawmap run --device iphone-12 --poses boardless ...     # 보드가 찍혀 있어도 보드 없이 위치를 잼 (비교용)
flawmap uco3d 59015-27660-21826 --face-up screen         # 공개 uCO3D 영상을 도면 없이 처리
flawmap run --device auto --poses boardless --quality max --pass back=... --pass screen=...   # SAM 2 기기 마스크 + COLMAP 자세 보정 + 선명한 장면 우선 (느림)
flawmap batch data/captures                              # 촬영 규격 v2 폴더(manifest.yaml) 일괄 처리, 요약 runs/batch/summary.md
flawmap board --board configs/board_a2.yaml --size 281.6 215.5 14   # 태블릿용 A2 매트 (도면 없으면 크기로 안내선)
```

일괄 처리용 촬영 규격(매트, 녹화 종류, `manifest.yaml` 형식)은 `docs/project_guide.md` 7.0절, 품질 기준은 `configs/batch.yaml`에 있습니다.

실제 촬영본 없이 시험하려면 가상 촬영본을 만듭니다. Stray Scanner와 같은 형식에 정답(`gt.json`)이 함께 생깁니다.

```bash
python -m flawmap.synth.render_capture data/synthetic/demo_back --face-up back --seed 11          # 166장, 하자 7개 + 찍힘
python -m flawmap.synth.render_capture data/synthetic/dents --face-up back --preset dents --seed 21   # 찍힘 2개만, 50장
python -m flawmap.synth.render_capture data/synthetic/wood_back --face-up back --preset small --background wood --seed 11   # 보드 없이 나무 바닥
python -m flawmap.synth.render_capture data/synthetic/g_dark --face-up back --preset small --background dark --design --shadow --seed 32   # 어두운 바닥, 로고·인쇄, 그림자 (바닥: wood|paper|cloth|dark|pattern)
python -m flawmap.synth.evaluate runs/<이름> --pass back=data/synthetic/<촬영본>   # 정답과 비교: 카메라 위치, 찾은 하자
```

로그는 `runs/logs/`에 남습니다. 오래 걸리는 작업은 `scripts/with_memory_guard.sh`로 감싸 남은 메모리가 8 GB 밑으로 내려가면 멈추게 합니다 (명령은 스크립트 파일로 넘깁니다).

## 7. 저장소 구조

```
flawmap/
  anchor/     parametric.py  도면 치수 -> 원본 모델, 부위 라벨, 펼침 좌표, 보드 위 자세
              register.py    실루엣과 영상 경계(+ 기기 마스크)를 맞추는 정합
              locate.py      보드 없이 기기 찾기: 바닥 평면, SAM 2 마스크, 발자국으로 판 크기
              template.py    도면이 없을 때 촬영본에 맞춘 판 모양 모델, 크기로 기종 추정
  poses/      board.py       인쇄 보드 생성과 보드로 카메라 위치, align.py ARKit 맞추기
              boardless.py   보드 없이: ARKit 흐름을 기기 외곽으로 바로잡음
  segment/    device_masks.py  SAM 2 기기 마스크 (점·상자·마스크 프롬프트로 추적)
  surface/    raycast.py     원본 모델 레이캐스팅 (GPU: raycast_warp.py), maps.py 펼친 외관 지도와 커버리지
  texture/    decompose.py   모든 프레임에서 표면색과 반사 나누기, allframes.py 텍셀별 유효 해상도
  shape/      relief.py      높이 지도 시도 (재지 못하면 판 모델 유지), visual_hull.py 실루엣 형상
  detect/     appearance.py  프레임 까짐과 유리 선, outline.py 윤곽으로 찍힘
              multiview.py   면의 다시점 검출: 표면선·점, 고정 번쩍임, 인쇄·부품·얼룩 분리
              rarity.py      DINOv2 패치의 면 안 희소성 (보조 신호)
  evidence/   select.py      근거 장면 고르기와 빨간 윤곽 그림
  export/     bundle.py      뷰어 묶음 (glTF, 텍스처 아틀라스, defects.json)
  synth/      render_capture.py  정답이 있는 가상 촬영본, evaluate.py 정답과 비교
  io/         stray.py       Stray Scanner 읽기, uco3d.py 공개 uCO3D 읽기, download.py 공개 데이터
  frames/     select.py      선명한 프레임 고르기
  geometry/   좌표 변환, 카메라 (전체 K), 라벨 메시
  service/    store.py 검수 기록(SQLite), intake.py 업로드 받기와 검사, worker.py 대기열, job.py 처리
  server/     app.py         서비스와 뷰어 서버
  pipeline.py, check.py, errors.py, cli.py
web/          뷰어 (index.html, viewer.js, style.css; vendor/는 내려받음), service/ 서비스 화면
configs/      devices/<기종>.yaml (도면 치수), board_a3.yaml, demo_public.yaml (데모용 공개 영상)
scripts/      설치, 내려받기, 데모 데이터, 서버 띄우기, 메모리·디스크 감시
docs/         project_guide.md (프로젝트 안내서: 데이터, 구조, 실험 결과, 촬영 안내, 할 일)
tests/unit/   단위 테스트
data/, runs/  생성물과 내려받은 데이터 (git 제외)
```

`data/`는 `FLAWMAP_DATA_ROOT`로 옮길 수 있습니다.

## 8. 테스트

```bash
pytest          # 단위 테스트 65개, 약 1분 (작은 가상 촬영본을 끝까지 돌리는 시험, 서비스 흐름 시험 포함)
```

## 9. 라이선스

코드는 Apache-2.0입니다. 도면과 데이터셋은 각자의 약관과 라이선스를 따르며, 이 저장소는 그 파일을 배포하지 않습니다.
