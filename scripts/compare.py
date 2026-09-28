"""Build the results table from results/metrics_*.json (and optionally write it into the README)."""
import argparse
import json

from common import ROOT

# tag -> label, in display order
ROWS = {
    "pytorch": "YOLO11n (ours), PyTorch",
    "onnx": "YOLO11n (ours), ONNX Runtime",
    "openvino": "YOLO11n (ours), OpenVINO FP32",
    "openvino_int8": "YOLO11n (ours), OpenVINO INT8",
    "ext_yolov8s_iris": "YOLOv8s, [IRIS](https://huggingface.co/IRIS-Computer-Vision/YOLOv8s_EO_Drone_Detection)",
    "ext_yolo11x_doguilmak": "YOLO11x, [doguilmak](https://huggingface.co/doguilmak/Drone-Detection-YOLOv11x)",
}
START, END = "<!-- results:start -->", "<!-- results:end -->"


def table():
    found = {}
    for f in (ROOT / "results").glob("metrics_*.json"):
        m = json.loads(f.read_text())
        found[m["tag"]] = m

    lines = ["| Model | mAP50 | mAP50-95 | Latency (ms) | Size (MB) | Test images |",
             "| --- | --- | --- | --- | --- | --- |"]
    for tag in list(ROWS) + sorted(set(found) - set(ROWS)):
        label = ROWS.get(tag, tag)
        m = found.get(tag)
        if m is None:
            lines.append(f"| {label} | – | – | – | – | – |")
            continue
        lines.append(f"| {label} | {m['map50']:.3f} | {m['map50_95']:.3f} | {m['latency']['mean_ms']} | "
                     f"{m['size_mb']} | {m.get('test_images', '')} |")

    cpu = next((m["hardware"].get("cpu") or m["hardware"].get("platform") for m in found.values()), None)
    note = f"\nLatency: batch 1, 640 px, CPU only{f' ({cpu})' if cpu else ''}, inside Docker."
    return "\n".join(lines) + "\n" + note


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--readme", action="store_true", help="write the table into README.md")
    args = parser.parse_args()

    t = table()
    print(t)
    if args.readme:
        path = ROOT / "README.md"
        if not path.exists():
            print("README.md not mounted in the container, table not written")
            return
        s = path.read_text(encoding="utf-8")
        i, j = s.index(START) + len(START), s.index(END)
        path.write_text(s[:i] + "\n" + t + "\n" + s[j:], encoding="utf-8")


if __name__ == "__main__":
    main()
