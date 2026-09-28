"""Collect results/metrics_*.json into a markdown table."""
import json

from common import ROOT


def main():
    rows = []
    for f in sorted((ROOT / "results").glob("metrics_*.json")):
        m = json.loads(f.read_text())
        rows.append(f"| {m['tag']} | {m['map50']} | {m['map50_95']} | {m['latency']['mean_ms']} | "
                    f"{m['latency']['fps']} | {m['size_mb']} |")
    print("| Model | mAP50 | mAP50-95 | Latency (ms) | FPS | Size (MB) |")
    print("| --- | --- | --- | --- | --- | --- |")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
