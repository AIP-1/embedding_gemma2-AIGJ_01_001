import time
import soundfile as sf
from eg2 import load

model = load(audio=True)
files = ["media/speech.wav", "media/beep.wav", "media/noise.wav"]
for f in files:
    info = sf.info(f)
    assert info.samplerate == 16000 and info.channels == 1, f"{f}: 16kHz 모노가 아님"

# 파일 경로를 넣으면 torchcodec 으로 디코딩하다 실패한다 → soundfile 로 읽은 배열을 넣는다
inp = [{"audio": sf.read(f, dtype="float32")[0]} for f in files]
model.encode(inp)  # 워밍업
t = time.perf_counter()
aud = model.encode(inp, normalize_embeddings=True)
print(f"오디오 {len(files)}개: {(time.perf_counter()-t)*1000:.0f} ms")

queries = ["덥다고 냉방을 요청하는 목소리", "날카로운 경보음", "웅웅거리는 배경 소음"]
for name, kw in [("SearchQuery", {"prompt_name": "SearchQuery"}), ("프롬프트 없음", {})]:
    q = model.encode(queries, normalize_embeddings=True, **kw)
    sim = q @ aud.T
    print(f"--- {name}")
    for qi, row in zip(queries, sim):
        order = row.argsort()[::-1]
        print(f"{qi} -> {files[order[0]].split('/')[-1]} ({row[order[0]]:.3f}, 2위와 차 {row[order[0]]-row[order[1]]:.3f})")
