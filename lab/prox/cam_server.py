"""10단계 확장: 카메라를 다른 보드에 꽂았을 때, 그 보드에서 웹캠 영상을 MJPEG로 내보낸다
proximity_app.py / calib_extrinsics.py 쪽에서 --cams 0 http://<이 보드 IP>:8210/stream 으로 받는다
필요한 것: python3 + opencv (pip install opencv-python-headless). torch·모델 없음, RAM 100MB 안팎
사용법: python cam_server.py --cam 0          →  http://<이 보드 IP>:8210/stream, /snap.jpg
"""
import argparse, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--cam", type=int, default=0)
ap.add_argument("--res", default="640x480")
ap.add_argument("--port", type=int, default=8210)
ap.add_argument("--quality", type=int, default=85)
args = ap.parse_args()
W, H = map(int, args.res.split("x"))

cap = cv2.VideoCapture(args.cam, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
assert cap.isOpened(), f"카메라 {args.cam}을 열 수 없음"

cond, jpg, n = threading.Condition(), None, 0

def grab():
    global jpg, n
    t0 = time.time()
    while True:
        ok, f = cap.read()
        if not ok:
            continue
        assert f.shape[1] == W, f"해상도가 {W}x{H}가 아님: {f.shape[1]}x{f.shape[0]}"
        b = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, args.quality])[1].tobytes()
        with cond:
            jpg, n = b, n + 1
            cond.notify_all()
        if n % 300 == 0:
            print(f"{n} 프레임, {300 / (time.time() - t0):.1f} fps", flush=True); t0 = time.time()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def do_GET(self):
        if self.path.startswith("/snap.jpg"):
            with cond:
                cond.wait_for(lambda: jpg is not None, timeout=5); b = jpg
            self.send_response(200); self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
            return
        if not self.path.startswith("/stream"):
            self.send_error(404); return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache"); self.end_headers()
        last = -1
        try:
            while True:
                with cond:                     # 새 프레임만 보낸다 (밀린 프레임은 버림)
                    cond.wait_for(lambda: n != last, timeout=5); b, last = jpg, n
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(b))
                self.wfile.write(b); self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass

if __name__ == "__main__":
    threading.Thread(target=grab, daemon=True).start()
    print(f"카메라 {args.cam} {W}x{H} → http://0.0.0.0:{args.port}/stream", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()
