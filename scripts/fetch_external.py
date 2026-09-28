"""Download external test sets into data/external/ and write data/<name>_test.txt."""
import argparse
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from common import ROOT

SETS = {
    # CC BY 4.0, 23 public drone datasets merged, 640x640, YOLO labels
    "seraphim": {
        "repo": "lgrzybowski/seraphim-drone-detection-dataset",
        "files": {"images": "test/images/batch_001.zip", "labels": "test/labels/batch_001.zip"},
    },
}


def extract_flat(zip_path, dest, exts):
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            name = Path(info.filename)
            if info.is_dir() or name.suffix.lower() not in exts or name.name.startswith("."):
                continue
            with z.open(info) as src, open(dest / name.name, "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=SETS)
    parser.add_argument("--keep-zip", action="store_true")
    args = parser.parse_args()

    spec = SETS[args.name]
    out = ROOT / "data" / "external" / args.name
    cache = ROOT / "data" / ".hf_cache" / "external"
    exts = {"images": {".jpg", ".jpeg", ".png"}, "labels": {".txt"}}

    for kind, remote in spec["files"].items():
        path = hf_hub_download(spec["repo"], remote, repo_type="dataset", cache_dir=cache)
        n = extract_flat(path, out / kind, exts[kind])
        print(f"{kind}: {n} files")
    if not args.keep_zip:
        shutil.rmtree(cache, ignore_errors=True)

    images = sorted(p for p in (out / "images").iterdir() if p.suffix.lower() in exts["images"])
    lines = [f"./{p.relative_to(ROOT / 'data').as_posix()}" for p in images]
    list_file = ROOT / "data" / f"{args.name}_test.txt"
    list_file.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} images -> {list_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
