# drone-detect-mlops

Drone detection with YOLO11n, from training to a containerized inference service on Kubernetes. CPU only.

## Roadmap

- [ ] Training
- [ ] CPU optimization (ONNX, OpenVINO INT8)
- [ ] Inference API (Docker image)
- [ ] Kubernetes deployment (k3d)
- [ ] Monitoring (Prometheus, Grafana)

## Training

Dataset: [pathikg/drone-detection-dataset](https://huggingface.co/datasets/pathikg/drone-detection-dataset) (MIT), subset of 1500/300/300 images.

```bash
docker compose build
docker compose run --rm train python scripts/prepare_data.py
docker compose run --rm train python scripts/train.py
docker compose run --rm train python scripts/evaluate.py --tag pytorch_cpu
```

## Video

```bash
docker compose run --rm train python scripts/predict_video.py data/videos/clip.mp4 --track
```

Output: `results/clip_detect.mp4`.

## Results

| Model | mAP50 | mAP50-95 | Latency (ms) | Hardware |
| --- | --- | --- | --- | --- |
| YOLO11n PyTorch | | | | CPU |
