import argparse
import shutil
import time

from ultralytics import YOLO

from common import ROOT, hardware_info, load_config, resolve_device, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()

    cfg = load_config()
    tr = cfg["train"]
    tr["epochs"] = args.epochs or tr["epochs"]
    tr["batch"] = args.batch or tr["batch"]
    device = resolve_device(args.device or tr["device"])

    data = ROOT / "data" / "drone.yaml"
    if not data.exists():
        raise SystemExit("no dataset found, run prepare_data.py first")

    model = YOLO(tr["model"])
    start = time.time()
    model.train(
        data=str(data),
        epochs=tr["epochs"],
        imgsz=tr["imgsz"],
        batch=tr["batch"],
        patience=tr["patience"],
        workers=tr["workers"],
        cache=tr.get("cache", False),
        device=device,
        project=str(ROOT / tr["project"]),
        name=tr["name"],
        exist_ok=True,
        seed=cfg["dataset"]["seed"],
    )

    run_dir = ROOT / tr["project"] / tr["name"]
    (ROOT / "models").mkdir(exist_ok=True)
    shutil.copy(run_dir / "weights" / "best.pt", ROOT / "models" / "best.pt")
    save_json({"config": cfg, "hardware": hardware_info(device),
               "minutes": round((time.time() - start) / 60, 1)},
              ROOT / "results" / "train_info.json")


if __name__ == "__main__":
    main()
