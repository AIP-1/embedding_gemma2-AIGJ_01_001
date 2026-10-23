# 5단계 개선: 목소리 질의가 틀린 원인 분리 (녹음 크기 vs 말의 내용)
import soundfile as sf, numpy as np
from eg2 import load

files = ["media/speech.wav", "media/beep.wav", "media/noise.wav"]
for f in files:
    x, _ = sf.read(f); print(f"{f}: 최대 {np.abs(x).max():.3f}, 평균 크기(RMS) {np.sqrt((x**2).mean()):.4f}")
m = load(audio=True)
aud = m.encode([{"audio": sf.read(f, dtype="float32")[0]} for f in files], normalize_embeddings=True)
for q in ["사람이 말하는 목소리", "한국어로 말하는 사람", "a person speaking", "덥다고 냉방을 요청하는 목소리"]:
    s = m.encode([q], prompt_name="SearchQuery", normalize_embeddings=True)[0] @ aud.T
    print(q, "->", " / ".join(f"{f.split('/')[-1]} {v:.3f}" for f, v in zip(files, s)))
