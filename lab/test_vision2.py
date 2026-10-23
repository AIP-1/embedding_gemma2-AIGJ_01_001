import glob, time, av, numpy as np, torch, psutil
from eg2 import load
avail = lambda: psutil.virtual_memory().available / 2**30
a0 = avail(); model = load(vision=True)
print(f"로드 후 available {a0:.2f} -> {avail():.2f} GiB")

images = sorted(glob.glob("media/*.jpg"))
model.encode([{"image": images[0]}])  # 워밍업
t = time.perf_counter()
img = model.encode([{"image": p} for p in images], normalize_embeddings=True)
print(f"이미지 {len(images)}장: {(time.perf_counter()-t)*1000:.0f} ms")

# record.py 로 쓴 mp4 는 average_rate 메타데이터가 틀려서 15fps 기준으로 고정 (1초에 1장)
c = av.open("media/clip.mp4"); s = c.streams.video[0]; step = 15
frames = np.stack([f.to_ndarray(format="rgb24") for i, f in enumerate(c.decode(s)) if i % step == 0]); c.close()
t = time.perf_counter()
try:
    vid = model.encode({"video": frames}, normalize_embeddings=True); how = "video 입력"
except Exception as e:
    print("video 입력 실패:", type(e).__name__, str(e)[:100])
    v = model.encode([{"image": f} for f in frames], normalize_embeddings=True).mean(0)
    vid = v / np.linalg.norm(v); how = "프레임 평균"
print(f"동영상 {len(frames)}프레임({how}): {(time.perf_counter()-t)*1000:.0f} ms")

queries = ["사람이 없는 빈 자리", "사람 얼굴이 보이는 장면", "화이트보드나 모니터 화면"]
q = model.encode(queries, prompt_name="SearchQuery", normalize_embeddings=True)
for qi, row in zip(queries, q @ img.T):
    o = row.argsort()[::-1][:3]
    print(qi, "->", [(images[i].split('/')[-1], round(float(row[i]), 3)) for i in o])
print("질의-동영상 유사도:", (q @ vid).round(3))
print(f"GPU 최대 {torch.cuda.max_memory_allocated()/2**30:.2f} GiB, 끝 available {avail():.2f} GiB")
