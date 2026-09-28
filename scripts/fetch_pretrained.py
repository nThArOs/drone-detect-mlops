"""Download public drone detectors to models/external/ for comparison."""
import shutil

from huggingface_hub import hf_hub_download

from common import ROOT

# name -> (repo, file, license)
MODELS = {
    "yolo11x_doguilmak": ("doguilmak/Drone-Detection-YOLOv11x", "weight/best.pt", "MIT"),
    "yolov8s_iris": ("IRIS-Computer-Vision/YOLOv8s_EO_Drone_Detection", "weights-exp-0cb41bb1.pt", "CC BY-NC 4.0"),
}


def main():
    out = ROOT / "models" / "external"
    out.mkdir(parents=True, exist_ok=True)
    for name, (repo, filename, lic) in MODELS.items():
        path = hf_hub_download(repo, filename, cache_dir=ROOT / "data" / ".hf_cache" / "hub")
        shutil.copy(path, out / f"{name}.pt")
        print(f"{name}: {repo} ({lic}) -> models/external/{name}.pt")


if __name__ == "__main__":
    main()
