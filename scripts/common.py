import json
import os
import platform
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=ROOT / "configs" / "train.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(device):
    if device not in (None, "auto"):
        return str(device)
    import torch
    return "0" if torch.cuda.is_available() else "cpu"


def hardware_info(device):
    import torch
    info = {"device": device, "platform": platform.platform(), "torch": torch.__version__}
    if device != "cpu":
        info["gpu"] = torch.cuda.get_device_name(int(device))
    else:
        info["cpu"] = platform.processor()
    return info


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def f1(precision, recall):
    return round(2 * precision * recall / (precision + recall), 3) if precision + recall else 0.0


def mlflow_enabled():
    return bool(os.getenv("MLFLOW_TRACKING_URI"))


def log_eval(res, weights=None):
    import mlflow
    mlflow.set_experiment("drone-detect-eval")
    lat = res.get("latency", {})
    with mlflow.start_run(run_name=res["tag"]):
        mlflow.log_params({"tag": res["tag"], "test_set": res.get("test_set", "test.txt"),
                           "test_images": res["test_images"], "weights": weights or "",
                           "size_mb": res.get("size_mb")})
        mlflow.log_metrics({k: v for k, v in {
            "precision": res["precision"], "recall": res["recall"],
            "f1": res.get("f1", f1(res["precision"], res["recall"])),
            "map50": res["map50"], "map50_95": res["map50_95"],
            "latency_ms": lat.get("mean_ms"), "latency_p95_ms": lat.get("p95_ms"),
        }.items() if v is not None})
        mlflow.set_tags({f"hw.{k}": v for k, v in res.get("hardware", {}).items()})
