"""Register the ONNX model in the MLflow model registry and link it to its Docker image."""
import argparse
import json

import mlflow
import onnx
from mlflow import MlflowClient

from common import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best.onnx")
    parser.add_argument("--name", default="drone-detector")
    parser.add_argument("--image", required=True, help="e.g. leaa1324/drone-detect-api:0.2.1")
    parser.add_argument("--metrics", default="results/metrics_onnx.json")
    parser.add_argument("--alias", default="production")
    args = parser.parse_args()

    metrics = json.loads((ROOT / args.metrics).read_text())
    mlflow.set_experiment("drone-detect-registry")
    with mlflow.start_run(run_name=args.image):
        mlflow.log_metrics({k: metrics[k] for k in ("precision", "recall", "map50", "map50_95")})
        mlflow.log_metric("latency_ms", metrics["latency"]["mean_ms"])
        info = mlflow.onnx.log_model(onnx.load(str(ROOT / args.model)), name="model",
                                     registered_model_name=args.name)

    client = MlflowClient()
    version = info.registered_model_version
    client.set_model_version_tag(args.name, version, "docker_image", args.image)
    client.set_model_version_tag(args.name, version, "eval_set", metrics.get("test_set", "test.txt"))
    client.set_registered_model_alias(args.name, args.alias, version)
    print(f"{args.name} v{version} -> {args.image} (@{args.alias})")


if __name__ == "__main__":
    main()
