"""Download the HF drone dataset and convert it to YOLO format."""
import argparse
import shutil

import yaml
from datasets import load_dataset
from tqdm import tqdm

from common import ROOT, load_config


def coco_to_yolo(bbox, w, h):
    x, y, bw, bh = map(float, bbox)
    # some boxes go slightly outside the image
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(w, x + bw), min(h, y + bh)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 1 or bh <= 1:
        return None
    return (x1 + bw / 2) / w, (y1 + bh / 2) / h, bw / w, bh / h


def export_sample(sample, split, out):
    img = sample["image"].convert("RGB")
    w, h = img.size
    name = f"{split}_{sample['image_id']}"
    img.save(out / "images" / split / f"{name}.jpg", quality=95)

    objs = sample.get("objects") or {}
    lines = []
    for bbox, cls in zip(objs.get("bbox", []), objs.get("category", [])):
        box = coco_to_yolo(bbox, w, h)
        if box:
            lines.append(f"{int(cls)} " + " ".join(f"{v:.6f}" for v in box))
    # empty label file = negative sample
    (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
    return len(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true", help="200/50/50 images, for quick tests")
    parser.add_argument("--force", action="store_true", help="overwrite existing dataset")
    args = parser.parse_args()

    cfg = load_config()["dataset"]
    sizes = (200, 50, 50) if args.small else (cfg["train_size"], cfg["val_size"], cfg["test_size"])
    out = ROOT / cfg["out_dir"]

    if out.exists():
        if not args.force:
            print(f"{out} already exists, use --force to rebuild")
            return
        shutil.rmtree(out)
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)

    ds = load_dataset(cfg["hf_name"])
    n_train, n_val, n_test = sizes
    pool = ds["train"].shuffle(seed=cfg["seed"])
    test = ds["test"].shuffle(seed=cfg["seed"])
    splits = {
        "train": pool.select(range(n_train)),
        "val": pool.select(range(n_train, n_train + n_val)),
        "test": test.select(range(min(n_test, len(test)))),
    }

    for split, subset in splits.items():
        boxes = sum(export_sample(s, split, out) for s in tqdm(subset, desc=split))
        print(f"{split}: {len(subset)} images, {boxes} boxes")

    data = {"path": str(out.resolve()), "train": "images/train", "val": "images/val",
            "test": "images/test", "names": {0: "drone"}}
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))


if __name__ == "__main__":
    main()
