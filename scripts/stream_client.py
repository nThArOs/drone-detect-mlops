"""Replay a video (or a folder of images) against the API frame by frame, like a camera feed.

With --serve, annotated frames are streamed as MJPEG on http://localhost:<port>.
"""
import argparse
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import requests

PAGE = b"""<!doctype html><title>drone-detect live</title>
<body style="margin:0;background:#111;display:flex;justify-content:center;align-items:center;height:100vh">
<img src="/stream" style="max-width:100%;max-height:100vh"></body>"""


class Latest:
    def __init__(self):
        self.jpg = None
        self.cond = threading.Condition()

    def put(self, jpg):
        with self.cond:
            self.jpg = jpg
            self.cond.notify_all()

    def wait(self):
        with self.cond:
            self.cond.wait(timeout=5)
            return self.jpg


def serve(latest, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path != "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(PAGE)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    jpg = latest.wait()
                    if jpg is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"live view on http://localhost:{port}")


def draw(frame, result, fps):
    for d in result["detections"]:
        x1, y1, x2, y2 = map(int, d["box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 255), 2)
        cv2.putText(frame, f"{d['label']} {d['conf']:.2f}", (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 255), 2)
    info = (f"{fps:.1f} fps  {result['inference_ms']:.0f} ms  "
            f"v{result.get('model_version', '?')}  {result.get('host', '')}")
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(frame, info, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return frame


def frames(source, shuffle):
    path = Path(source)
    if path.is_dir():
        images = sorted(p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if shuffle:
            random.shuffle(images)
        for p in images:
            frame = cv2.imread(str(p))
            if frame is not None:
                yield frame
        return
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {source}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
    cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="video file or folder of images")
    parser.add_argument("--url", default="http://host.docker.internal:8080")
    parser.add_argument("--fps", type=float, default=5)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--shuffle", action="store_true", help="random order for image folders")
    parser.add_argument("--serve", type=int, metavar="PORT", help="MJPEG live view on this port")
    args = parser.parse_args()

    latest = Latest()
    if args.serve:
        serve(latest, args.serve)

    session = requests.Session()
    period = 1 / args.fps
    sent = errors = detections = 0
    t_start = time.time()

    while True:
        t0 = time.time()
        for frame in frames(args.source, args.shuffle):
            _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            try:
                r = session.post(f"{args.url}/detect", timeout=10,
                                 files={"file": ("frame.jpg", jpg.tobytes(), "image/jpeg")})
                r.raise_for_status()
                result = r.json()
                detections += len(result["detections"])
                if args.serve:
                    fps = 1 / max(1e-3, time.time() - t0)
                    _, out = cv2.imencode(".jpg", draw(frame, result, min(fps, args.fps)))
                    latest.put(out.tobytes())
            except requests.RequestException as e:
                errors += 1
                print(f"error: {e}")
            sent += 1
            if sent % 50 == 0:
                rate = sent / (time.time() - t_start)
                print(f"{sent} frames  {rate:.1f} fps  {detections} detections  {errors} errors")
            time.sleep(max(0.0, period - (time.time() - t0)))
            t0 = time.time()
        if not args.loop:
            break


if __name__ == "__main__":
    main()
