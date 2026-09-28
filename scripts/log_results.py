"""Import past results into MLflow: training runs from runs/train/, evaluations from results/."""
import csv
import json

import mlflow
import yaml

from common import ROOT, log_eval


def log_training(run_dir):
    mlflow.set_experiment("drone-detect-train")
    args = yaml.safe_load((run_dir / "args.yaml").read_text())
    with mlflow.start_run(run_name=run_dir.name):
        mlflow.log_params({k: v for k, v in args.items() if v is not None and len(str(v)) < 250})
        with open(run_dir / "results.csv") as f:
            for row in csv.DictReader(f):
                step = int(float(row.pop("epoch")))
                mlflow.log_metrics({k.strip().replace("(B)", ""): float(v) for k, v in row.items()},
                                   step=step)
        for p in sorted(run_dir.glob("*.png")):
            mlflow.log_artifact(str(p), "plots")
        best = run_dir / "weights" / "best.pt"
        if best.exists():
            mlflow.log_artifact(str(best), "weights")
    print(f"train run: {run_dir.name}")


def main():
    for run_dir in sorted((ROOT / "runs" / "train").iterdir()):
        if (run_dir / "results.csv").exists():
            log_training(run_dir)
    for path in sorted((ROOT / "results").glob("metrics_*.json")):
        res = json.loads(path.read_text())
        log_eval(res)
        print(f"eval: {res['tag']}")


if __name__ == "__main__":
    main()
