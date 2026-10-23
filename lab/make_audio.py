# 5단계 샘플: 경보음(1 kHz)과 핑크 노이즈를 16 kHz 모노로 만든다.
# 이 보드의 NVIDIA판 ffmpeg 에는 lavfi(sine, anoisesrc)가 없어서 파이썬으로 생성한다.
# 목소리는 웹캠 마이크로 녹음: arecord -D plughw:0,0 -f S16_LE -r 16000 -c 1 -d 5 media/speech.wav
import numpy as np, soundfile as sf
sr = 16000
t = np.arange(3 * sr) / sr
sf.write("media/beep.wav", (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32), sr)
n = 5 * sr
f = np.fft.rfft(np.random.randn(n)); f[1:] /= np.sqrt(np.arange(1, len(f)))   # 핑크 노이즈 (1/f)
x = np.fft.irfft(f, n); x = 0.5 * x / np.abs(x).max()
sf.write("media/noise.wav", x.astype(np.float32), sr)
print("beep, noise 생성 완료")
