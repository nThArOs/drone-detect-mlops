"""Export the trained model to ONNX, OpenVINO FP32 and OpenVINO INT8."""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from common import ROOT, load_config

# name -> (export args, output path under models/); OpenVINO dirs must end in _openvino_model
FORMATS = {
    "onnx": (dict(format="onnx", simplify=True), "best.onnx"),
    "openvino": (dict(format="openvino"), "best_openvino_model"),
    "openvino_int8": (dict(format="openvino", int8=True), "best_int8_openvino_model"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(ROOT / "models" / "best.pt"))
    parser.add_argument("--formats", nargs="+", default=list(FORMATS), choices=list(FORMATS))
    args = parser.parse_args()

    imgsz = load_config()["train"]["imgsz"]
    data = str(ROOT / "data" / "drone.yaml")  # INT8 calibration uses the val split

    for name in args.formats:
        model = YOLO(args.weights)
        kwargs, target = FORMATS[name]
        out = model.export(imgsz=imgsz, data=data, **kwargs)
        dst = ROOT / "models" / target
        if Path(out).resolve() == dst.resolve():
            print(f"{name}: {dst.relative_to(ROOT)}")
            continue
        if dst.exists():
            shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
        shutil.move(out, dst)
        print(f"{name}: {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
