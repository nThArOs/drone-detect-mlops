"""Accuracy on the test split + single-image latency."""
import argparse
import statistics
import time
from pathlib import Path

from ultralytics import YOLO

from common import ROOT, hardware_info, load_config, resolve_device, save_json


def latency(model, images, device, imgsz, warmup, runs):
    for img in images[:warmup]:
        model.predict(img, imgsz=imgsz, device=device, verbose=False)
    times = []
    for i in range(runs):
        t = time.perf_counter()
        model.predict(images[i % len(images)], imgsz=imgsz, device=device, verbose=False)
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    mean = statistics.mean(times)
    return {"mean_ms": round(mean, 1), "p95_ms": round(times[int(0.95 * len(times)) - 1], 1),
            "fps": round(1000 / mean, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(ROOT / "models" / "best.pt"))
    parser.add_argument("--tag", default="pytorch_fp32")
    parser.add_argument("--device")
    args = parser.parse_args()

    cfg = load_config()
    device = resolve_device(args.device or cfg["train"]["device"])
    imgsz = cfg["train"]["imgsz"]
    data_dir = ROOT / cfg["dataset"]["out_dir"]
    weights = Path(args.weights)

    model = YOLO(str(weights))
    m = model.val(data=str(data_dir / "data.yaml"), split="test", imgsz=imgsz,
                  device=device, plots=False, verbose=False)

    images = sorted((data_dir / "images" / "test").glob("*.jpg"))
    lat = latency(model, images, device, imgsz, cfg["benchmark"]["warmup"], cfg["benchmark"]["runs"])

    res = {
        "tag": args.tag,
        "size_mb": round(weights.stat().st_size / 1e6, 1),
        "precision": round(float(m.box.mp), 3),
        "recall": round(float(m.box.mr), 3),
        "map50": round(float(m.box.map50), 3),
        "map50_95": round(float(m.box.map), 3),
        "latency": lat,
        "hardware": hardware_info(device),
    }
    save_json(res, ROOT / "results" / f"metrics_{args.tag}.json")
    print(f"mAP50 {res['map50']}  mAP50-95 {res['map50_95']}  "
          f"{lat['mean_ms']} ms/img ({lat['fps']} fps)")


if __name__ == "__main__":
    main()
