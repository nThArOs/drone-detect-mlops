"""Export the full HF drone dataset once to data/raw/ as JPG + YOLO labels."""
import argparse
import shutil

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from tqdm import tqdm

from common import ROOT, load_config


def coco_to_yolo(bbox, w, h):
    x, y, bw, bh = map(float, bbox)
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(w, x + bw), min(h, y + bh)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 1 or bh <= 1:
        return None
    return (x1 + bw / 2) / w, (y1 + bh / 2) / h, bw / w, bh / h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-cache", action="store_true", help="delete the HF download afterwards")
    args = parser.parse_args()

    cfg = load_config()["dataset"]
    raw = ROOT / cfg["raw_dir"]
    cache = ROOT / "data" / ".hf_cache"
    try:
        repo = snapshot_download(cfg["hf_name"], repo_type="dataset", cache_dir=cache / "hub")
    except Exception:
        # offline: reuse what is already in the cache
        repo = snapshot_download(cfg["hf_name"], repo_type="dataset", cache_dir=cache / "hub",
                                 local_files_only=True)
    files = sorted((ROOT / repo).glob("data/*.parquet"))

    total = 0
    for f in files:
        split = "test" if f.name.startswith("test") else "train"
        (raw / split / "images").mkdir(parents=True, exist_ok=True)
        (raw / split / "labels").mkdir(parents=True, exist_ok=True)
        done = {p.stem for p in (raw / split / "labels").iterdir()}  # resumable
        pf = pq.ParquetFile(f)
        for rg in tqdm(range(pf.num_row_groups), desc=f.name):
            ids = pf.read_row_group(rg, columns=["image_id"]).column(0).to_pylist()
            total += len(ids)
            if all(f"{i:06d}" in done for i in ids):
                continue
            for r in pf.read_row_group(rg).to_pylist():
                name = f"{r['image_id']:06d}"
                (raw / split / "images" / f"{name}.jpg").write_bytes(r["image"]["bytes"])
                w, h = r["width"], r["height"]
                lines = []
                for bbox, cls in zip(r["objects"]["bbox"], r["objects"]["category"]):
                    box = coco_to_yolo(bbox, w, h)
                    if box:
                        lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in box))
                (raw / split / "labels" / f"{name}.txt").write_text("\n".join(lines))
    print(f"{total} images in {raw}")

    if args.clean_cache:
        shutil.rmtree(cache, ignore_errors=True)


if __name__ == "__main__":
    main()
