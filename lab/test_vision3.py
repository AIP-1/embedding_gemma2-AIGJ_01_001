import glob, av, numpy as np
from eg2 import load
model = load(vision=True)
images = sorted(glob.glob("media/frame_*.jpg"))
img = model.encode([{"image": p} for p in images], normalize_embeddings=True)
tests = [("손가락으로 V 사인을 하는 사람", 4), ("천장의 형광등 조명", 5),
         ("책상 위의 흰 키보드와 노트북", 3), ("흔들려서 흐릿한 노트북 사진", 2),
         ("렌즈가 가려진 어둡고 흐린 화면", 1)]
for prompt in ["SearchQuery", None]:
    q = model.encode([t for t, _ in tests], prompt_name=prompt, normalize_embeddings=True)
    hit = 0
    print(f"\n[prompt={prompt}]")
    for (t, ans), row in zip(tests, q @ img.T):
        o = row.argsort()[::-1]; top = int(images[o[0]][-6:-4]); hit += top == ans
        print(f"  {'O' if top==ans else 'X'} {t} -> {top:02d} ({row[o[0]]:.3f}, 2등과 차 {row[o[0]]-row[o[1]]:.3f})")
    print(f"  정답률 {hit}/{len(tests)}")
c = av.open("media/clip.mp4"); fr = np.stack([f.to_ndarray(format="rgb24") for i, f in enumerate(c.decode(video=0)) if i % 15 == 0])
v = model.encode({"video": fr}, normalize_embeddings=True)
q = model.encode(["사람이 V 사인을 한다", "천장 조명", "책상과 키보드"], prompt_name="SearchQuery", normalize_embeddings=True)
print(f"\n동영상 {len(fr)}프레임 질의 유사도:", (q @ v).round(3))
