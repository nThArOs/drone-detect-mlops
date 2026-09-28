"""Build train/val/test lists from data/raw (+ optional negatives) without copying images."""
import argparse
import random

import yaml

from common import ROOT, load_config

DATA = ROOT / "data"


def rel(p):
    # "./" paths are resolved by Ultralytics relative to the list file's folder (data/)
    return "./" + p.relative_to(DATA).as_posix()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true", help="200/50/50 images, for quick tests")
    args = parser.parse_args()

    cfg = load_config()["dataset"]
    raw = ROOT / cfg["raw_dir"]
    if not (raw / "train" / "images").exists():
        raise SystemExit("data/raw not found, run export_dataset.py first")

    rng = random.Random(cfg["seed"])
    pool = sorted((raw / "train" / "images").glob("*.jpg"))
    test = sorted((raw / "test" / "images").glob("*.jpg"))
    rng.shuffle(pool)
    rng.shuffle(test)

    n_train, n_val, n_test = (200, 50, 50) if args.small else (cfg["train_size"], cfg["val_size"], cfg["test_size"])
    splits = {
        "train": pool[:n_train],
        "val": pool[n_train:n_train + n_val],
        "test": test[:n_test] if n_test else test,
    }

    # negatives: any image in data/negatives/images gets an empty label and goes to train
    neg_dir = DATA / "negatives"
    negs = sorted((neg_dir / "images").glob("*.jpg")) if (neg_dir / "images").exists() else []
    if negs:
        (neg_dir / "labels").mkdir(exist_ok=True)
        for p in negs:
            (neg_dir / "labels" / f"{p.stem}.txt").touch()
        splits["train"] = splits["train"] + negs

    for name, files in splits.items():
        (DATA / f"{name}.txt").write_text("\n".join(rel(p) for p in files) + "\n")
        print(f"{name}: {len(files)} images")
    if negs:
        print(f"(train includes {len(negs)} negatives)")

    data = {"train": "train.txt", "val": "val.txt", "test": "test.txt", "names": {0: "drone"}}
    (DATA / "drone.yaml").write_text(yaml.safe_dump(data, sort_keys=False))


if __name__ == "__main__":
    main()
