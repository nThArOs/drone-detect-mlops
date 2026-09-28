import argparse
import random

from ultralytics import YOLO

from common import ROOT, load_config, resolve_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(ROOT / "models" / "best.pt"))
    parser.add_argument("-n", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    cfg = load_config()
    device = resolve_device(cfg["train"]["device"])
    images = [ROOT / "data" / l[2:] for l in (ROOT / "data" / "test.txt").read_text().split()]
    random.Random(cfg["dataset"]["seed"]).shuffle(images)

    model = YOLO(args.weights)
    out = ROOT / "results" / "demo"
    out.mkdir(parents=True, exist_ok=True)
    for img in images[:args.n]:
        r = model.predict(img, conf=args.conf, device=device, verbose=False)[0]
        r.save(filename=str(out / img.name))


if __name__ == "__main__":
    main()
