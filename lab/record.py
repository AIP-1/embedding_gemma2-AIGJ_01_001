import av, time
SEC, FPS = 10, 15
inp = av.open("/dev/video0", format="v4l2",
              options={"video_size": "640x480", "framerate": str(FPS)})
out = av.open("media/clip.mp4", "w")
codec = "libx264" if "libx264" in av.codecs_available else "mpeg4"
st = out.add_stream(codec, rate=FPS)
st.width, st.height, st.pix_fmt = 640, 480, "yuv420p"
t0, n, saved = time.time(), 0, 0
print("녹화 시작! 카메라를 천천히 돌려줘")
for frame in inp.decode(video=0):
    el = time.time() - t0
    if el > SEC: break
    if el >= saved * 2:
        saved += 1
        frame.to_image().save(f"media/frame_{saved:02d}.jpg")
    f = frame.reformat(width=640, height=480, format="yuv420p")
    f.pts = n; n += 1
    for p in st.encode(f): out.mux(p)
for p in st.encode(): out.mux(p)
out.close(); inp.close()
print(f"끝: {n}프레임, 코덱 {codec}, 사진 {saved}장")
