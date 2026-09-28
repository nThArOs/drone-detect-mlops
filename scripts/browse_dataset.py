"""Draw ground-truth boxes on random images of a split: mosaics, or a short slideshow video."""
import argparse
import random

import cv2
import numpy as np

from common import ROOT

DATA = ROOT / "data"


def load(split):
    lines = (DATA / f"{split}.txt").read_text().split()
    return [DATA / l[2:] if l.startswith("./") else DATA / l for l in lines]


def draw(img_path, size=(320, 240)):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    label = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
    n = 0
    for line in label.read_text().split("\n") if label.exists() else []:
        if not line.strip():
            continue
        _, x, y, bw, bh = map(float, line.split())
        p1 = int((x - bw / 2) * w), int((y - bh / 2) * h)
        p2 = int((x + bw / 2) * w), int((y + bh / 2) * h)
        cv2.rectangle(img, p1, p2, (0, 0, 255), max(2, w // 300))
        n += 1
    img = cv2.resize(img, size)
    cv2.putText(img, f"{img_path.stem} ({n})", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("-n", type=int, default=64, help="number of images")
    parser.add_argument("--video", action="store_true", help="slideshow mp4 instead of mosaics")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    files = load(args.split)
    random.Random(args.seed).shuffle(files)
    files = files[:args.n]
    out = ROOT / "results" / "preview"
    out.mkdir(parents=True, exist_ok=True)

    if args.video:
        path = out / f"{args.split}.mp4"
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4, (640, 480))
        for f in files:
            vw.write(draw(f, (640, 480)))
        vw.release()
        print(path)
        return

    for i in range(0, len(files), 16):
        tiles = [draw(f) for f in files[i:i + 16]]
        tiles += [np.zeros_like(tiles[0])] * (16 - len(tiles))
        grid = np.vstack([np.hstack(tiles[r:r + 4]) for r in range(0, 16, 4)])
        cv2.imwrite(str(out / f"{args.split}_{i // 16:02d}.jpg"), grid)
    print(f"{(len(files) + 15) // 16} mosaics in {out}")


if __name__ == "__main__":
    main()
