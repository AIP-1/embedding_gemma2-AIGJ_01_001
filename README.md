# Jetson EmbeddingGemma 2 실습

Jetson Orin Nano Super 8GB에서 EmbeddingGemma 2(`google/embeddinggemma-2`)로 텍스트·이미지·동영상을 하나의 벡터 공간에서 검색하고, 구성별 자원 사용량을 측정한 실습이다.

## 요약

- **문제**: 2명이 같이 쓰는 8GB 보드에서 멀티모달 임베딩 검색이 되는지, 자원은 얼마나 드는지
- **접근**: JetPack 7용 cu132 torch 위에 sentence-transformers 6.2를 버전 고정으로 설치, USB 웹캠으로 직접 찍은 영상·프레임으로 검색·측정
- **결과** (MAXN_SUPER, bfloat16, 워밍업 뒤 3회 중앙값)

| 구성 | 입력 | 지연 | 로드 RAM | GPU 최대 |
|---|---|---|---|---|
| 텍스트 전용 | 질의 2개 | 160 ms | +0.8 GB | 11% |
| +비전 | 이미지 5장 (640x480) | 2.1 s | +0.8 GB | 99% |
| +비전 | 동영상 10초 (11프레임) | 2.0 s | +1.0 GB | 99% |

- 텍스트 검색: 768/256/128차원 모두 정답 1위
- 이미지 검색: 장면에 맞춘 질의 5개 중 4개 정답 (오답은 렌즈가 가려진 프레임)
- 웹캠 물체 검색(8단계): 라벨 목록 밖 물체는 가장 비슷한 이름으로 억지로 붙는다. YOLOE 박스 + 임베딩 재라벨(`--mode yoloe --relabel`)이 캔·컵·휴대폰을 모두 맞혔다. 약 12 fps, 크롭당 0.75 s, RAM 3.5 GB
- 자세한 결과, 실패 → 원인 → 해결: [reports/results.md](reports/results.md) (1~6단계), [reports/step8.md](reports/step8.md) (8단계)

## 역할

| 단계 | 담당 |
|---|---|
| 1~4, 6단계 (환경, 텍스트·이미지·동영상 검색, 자원 측정) | 민서 (iamminseo) |
| 8단계 (웹캠 물체 검색: yolo·relabel·yoloe 비교, `--relabel` 확장) | 성찬 |
| 7단계 (도메인 미세 조정) | 성찬 (진행 중) |

## 환경

- Jetson Orin Nano Engineering Reference Developer Kit Super, RAM 8GB
- JetPack 7.2.1 (L4T R39.2.1), Ubuntu 24.04, Python 3.12, CUDA 13.2
- torch 2.14.1+cu132, torchvision 0.29.1+cu132, transformers 5.20.1, sentence-transformers 6.2.0
- USB 웹캠 (`/dev/video0`)

## 설치

```bash
python3 -m venv ~/eg2 && source ~/eg2/bin/activate
pip install -U pip wheel
python -m pip install --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/cu132
pip freeze | grep -E '^(torch|torchvision)==' > ~/eg2/torch-pin.txt
pip install -c ~/eg2/torch-pin.txt -U \
  "sentence-transformers[image,audio,video]>=6.1.0" transformers accelerate \
  soundfile librosa av pillow huggingface_hub psutil
pip uninstall -y torchaudio     # PyPI torchaudio(CUDA 13.0)가 torch(13.2)와 충돌
hf download google/embeddinggemma-2
```

## 실행 (`lab/` 폴더에서)

```bash
cd lab && mkdir -p media
python check.py                        # torch·CUDA·버전 확인
python test_text.py                    # 3단계: 한국어 텍스트 검색, 차원 768/256/128 비교
python record.py                       # 웹캠 10초 녹화 → media/clip.mp4, 2초마다 frame_XX.jpg
python test_vision2.py                 # 4단계: 이미지·동영상 임베딩과 검색
python test_vision3.py                 # 4단계 개선: 장면 맞춤 질의 정답률, prompt_name 유무 비교
for m in text image video; do python bench.py $m 2>/dev/null | grep RESULT; done   # 6단계 측정
python app.py --mode yoloe --relabel   # 8단계: 웹캠 물체 검색, http://<Jetson IP>:8200
```

`test_vision3.py` 의 질의와 정답 번호는 이 저장소의 [reports/frames/](reports/frames/) 장면 기준이다. 새로 녹화하면 장면에 맞게 바꿔야 한다.

공용 보드라서 모델을 올리기 전에 `free -h` 로 available이 2.5GB 이상인지 확인한다. 비전 모델을 돌리는 동안 보드 전체 RAM이 5.8GB까지 올라간다.

## 알아둘 점

- `{"video": "clip.mp4"}` 처럼 파일 경로로 넣으면 torchcodec이 필요한데, 이 보드에서는 NVIDIA판 ffmpeg 8 라이브러리를 못 찾아 실패한다. PyAV로 1초에 1장씩 프레임을 뽑아 `{"video": numpy 배열}` 로 넣으면 동작한다.
- `record.py` 로 쓴 mp4는 fps 메타데이터가 정확하지 않아, 읽을 때 15프레임 간격(=1 fps)으로 고정한다.
- 영상·모델 파일은 git에 올리지 않는다 (`.gitignore`).

## AI 활용

Claude가 명령·스크립트를 단계별로 제안하고, 민서가 Jetson에서 직접 실행해 출력으로 검증했다. 결과 문서의 숫자는 모두 실제 실행 출력에서 옮겼다.
