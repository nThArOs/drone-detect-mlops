#!/usr/bin/env bash
# Train, export and benchmark in one go (meant to run inside the container).
set -euo pipefail
cd "$(dirname "$0")"

python train.py
python export_models.py
python evaluate.py --weights ../models/best.pt --tag pytorch
python evaluate.py --weights ../models/best.onnx --tag onnx
python evaluate.py --weights ../models/best_openvino_model --tag openvino
python evaluate.py --weights ../models/best_int8_openvino_model --tag openvino_int8
python fetch_pretrained.py
python evaluate.py --weights ../models/external/yolov8s_iris.pt --tag ext_yolov8s_iris
python evaluate.py --weights ../models/external/yolo11x_doguilmak.pt --tag ext_yolo11x_doguilmak
python compare.py --readme | tee ../results/compare.md
python predict_video.py ../data/videos --track --weights ../models/best_openvino_model
