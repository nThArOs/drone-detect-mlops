"""Accuracy on the test split + single-image latency."""
import argparse
import statistics
import time
from pathlib import Path

from ultralytics import YOLO

from common import (ROOT, f1, hardware_info, load_config, log_eval, mlflow_enabled,
                    resolve_device, save_json)


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
    parser.add_argument("--list", default="test.txt", help="image list in data/ (e.g. seraphim_test.txt)")
    parser.add_argument("--max-images", type=int, help="evaluate on the first N images only")
    args = parser.parse_args()

    cfg = load_config()
    device = resolve_device(args.device or cfg["train"]["device"])
    imgsz = cfg["train"]["imgsz"]
    weights = Path(args.weights)

    data_dir = ROOT / "data"
    lines = (data_dir / args.list).read_text().split()
    data_yaml = data_dir / "drone.yaml"
    if args.max_images or args.list != "test.txt":
        lines = lines[:args.max_images]
        (data_dir / "test_subset.txt").write_text("\n".join(lines) + "\n")
        data_yaml = data_dir / "drone_subset.yaml"
        data_yaml.write_text("train: train.txt\nval: val.txt\ntest: test_subset.txt\nnames:\n  0: drone\n")

    model = YOLO(str(weights), task="detect")
    m = model.val(data=str(data_yaml), split="test", imgsz=imgsz,
                  device=device, plots=False, verbose=False)

    images = [data_dir / l[2:] for l in lines]
    lat = latency(model, images, device, imgsz, cfg["benchmark"]["warmup"], cfg["benchmark"]["runs"])

    res = {
        "tag": args.tag,
        "test_set": args.list,
        "test_images": len(lines),
        "size_mb": round(sum(f.stat().st_size for f in weights.rglob("*")) / 1e6 if weights.is_dir()
                         else weights.stat().st_size / 1e6, 1),
        "precision": round(float(m.box.mp), 3),
        "recall": round(float(m.box.mr), 3),
        "f1": f1(float(m.box.mp), float(m.box.mr)),
        "map50": round(float(m.box.map50), 3),
        "map50_95": round(float(m.box.map), 3),
        "latency": lat,
        "hardware": hardware_info(device),
    }
    save_json(res, ROOT / "results" / f"metrics_{args.tag}.json")
    if mlflow_enabled():
        log_eval(res, weights.name)
    print(f"mAP50 {res['map50']}  mAP50-95 {res['map50_95']}  "
          f"{lat['mean_ms']} ms/img ({lat['fps']} fps)")


if __name__ == "__main__":
    main()
