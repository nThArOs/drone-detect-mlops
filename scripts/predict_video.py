"""Run detection (or tracking) on a video and save the annotated result."""
import argparse
import shutil
import subprocess
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from common import ROOT, load_config, resolve_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="video file, or webcam index (0)")
    parser.add_argument("--weights", default=str(ROOT / "models" / "best.pt"))
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--track", action="store_true", help="ByteTrack ids across frames")
    parser.add_argument("--show", action="store_true", help="live window (not available in Docker)")
    args = parser.parse_args()

    cfg = load_config()
    device = resolve_device(cfg["train"]["device"])
    source = int(args.source) if args.source.isdigit() else args.source

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    name = "webcam" if isinstance(source, int) else Path(source).stem
    out_path = ROOT / "results" / f"{name}_detect.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frames, t0 = 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.track:
            r = model.track(frame, conf=args.conf, device=device, persist=True,
                            tracker="bytetrack.yaml", verbose=False)[0]
        else:
            r = model.predict(frame, conf=args.conf, device=device, verbose=False)[0]

        annotated = r.plot()
        frames += 1
        cv2.putText(annotated, f"{frames / (time.time() - t0):.1f} fps", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        writer.write(annotated)

        if args.show:
            cv2.imshow("detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    # OpenCV writes MPEG-4 part 2, which many players don't handle well
    if shutil.which("ffmpeg"):
        tmp = out_path.with_suffix(".tmp.mp4")
        out_path.rename(tmp)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)], check=True)
        tmp.unlink()
    print(f"{frames} frames, {frames / (time.time() - t0):.1f} fps -> {out_path}")


if __name__ == "__main__":
    main()
