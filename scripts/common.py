import json
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
