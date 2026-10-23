import sys, re, glob, time, threading, subprocess, av, numpy as np, torch
mode = sys.argv[1]  # text | image | video
st = {"ram": 0, "gpu": 0, "on": False}
def watch():
    p = subprocess.Popen(["tegrastats", "--interval", "200"], stdout=subprocess.PIPE, text=True)
    for line in p.stdout:
        ram = re.search(r"RAM (\d+)/", line); g = re.search(r"GR3D_FREQ (\d+)%", line)
        st["now"] = int(ram.group(1)) if ram else 0
        if st["on"]:
            st["ram"] = max(st["ram"], st["now"]); st["gpu"] = max(st["gpu"], int(g.group(1)) if g else 0)
threading.Thread(target=watch, daemon=True).start(); time.sleep(1)
ram0 = st["now"]
from eg2 import load
model = load(vision=(mode != "text")); torch.cuda.synchronize(); time.sleep(1)
ram1 = st["now"]
if mode == "text":
    inp, desc = ["수업 중에 졸리고 머리가 맑지 않아요", "여름에 냉방 몇 도로 맞춰야 하나요?"], "질의 2개"
    run = lambda: model.encode(inp, prompt_name="SearchQuery")
elif mode == "image":
    inp = [{"image": p} for p in sorted(glob.glob("media/frame_*.jpg"))]; desc = f"이미지 {len(inp)}장"
    run = lambda: model.encode(inp)
else:
    c = av.open("media/clip.mp4"); fr = np.stack([f.to_ndarray(format="rgb24") for i, f in enumerate(c.decode(video=0)) if i % 15 == 0]); c.close()
    desc = f"동영상 10초({len(fr)}프레임)"; run = lambda: model.encode({"video": fr})
run(); torch.cuda.synchronize()          # 1회차(워밍업)
st["on"] = True; ts = []
for _ in range(3):
    t = time.perf_counter(); run(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1000)
time.sleep(0.5); st["on"] = False
print(f"RESULT|{mode}|{desc}|시스템RAM {ram0}->{ram1}MB (+{ram1-ram0})|실행중 최대 {st['ram']}MB|"
      f"지연 중앙값 {sorted(ts)[1]:.0f}ms (3회 {', '.join(f'{x:.0f}' for x in ts)})|GPU 최대 {st['gpu']}%|"
      f"torch 최대 {torch.cuda.max_memory_allocated()/2**20:.0f}MB")
