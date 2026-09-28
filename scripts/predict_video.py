"""Run detection (or tracking) on videos and save annotated copies to results/videos/."""
import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from common import ROOT, load_config, resolve_device

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}


def to_h264(path):
    # OpenCV writes MPEG-4 part 2, which many players don't handle well
    if not shutil.which("ffmpeg"):
        return
    tmp = path.with_suffix(".tmp.mp4")
    path.rename(tmp)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], check=True)
    tmp.unlink()


def run(model, source, out_path, device, conf, track, show):
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if track:
        model.predictor = None  # reset tracker state between videos

    frames = frames_with_det = 0
    ids, confs = set(), []
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if track:
            r = model.track(frame, conf=conf, device=device, persist=True,
                            tracker="bytetrack.yaml", verbose=False)[0]
        else:
            r = model.predict(frame, conf=conf, device=device, verbose=False)[0]

        frames += 1
        if len(r.boxes):
            frames_with_det += 1
            confs += r.boxes.conf.tolist()
            if r.boxes.id is not None:
                ids.update(int(i) for i in r.boxes.id.tolist())

        annotated = r.plot()
        cv2.putText(annotated, f"{frames / (time.time() - t0):.1f} fps", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        writer.write(annotated)
        if show:
            cv2.imshow("detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()
    to_h264(out_path)

    return {
        "frames": frames,
        "detected_pct": round(100 * frames_with_det / max(frames, 1), 1),
        "tracks": len(ids) if track else None,
        "mean_conf": round(sum(confs) / len(confs), 2) if confs else None,
        "fps": round(frames / (time.time() - t0), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="video file, folder of videos, or webcam index (0)")
    parser.add_argument("--weights", default=str(ROOT / "models" / "best.pt"))
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--track", action="store_true", help="ByteTrack ids across frames")
    parser.add_argument("--show", action="store_true", help="live window (not available in Docker)")
    args = parser.parse_args()

    device = resolve_device(load_config()["train"]["device"])
    model = YOLO(args.weights)
    out_dir = ROOT / "results" / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.source.isdigit():
        sources = [int(args.source)]
    else:
        p = Path(args.source)
        sources = sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXT) if p.is_dir() else [p]

    summary = {}
    for src in sources:
        name = "webcam" if isinstance(src, int) else src.stem
        stats = run(model, src if isinstance(src, int) else str(src), out_dir / f"{name}.mp4",
                    device, args.conf, args.track, args.show)
        summary[name] = stats
        print(f"{name:20s} detections on {stats['detected_pct']:5.1f}% of frames, "
              f"tracks={stats['tracks']}, conf={stats['mean_conf']}, {stats['fps']} fps")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
