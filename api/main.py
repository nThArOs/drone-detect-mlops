import io
import os
import time

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageDraw, UnidentifiedImageError
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from detector import Detector

MODEL_PATH = os.getenv("MODEL_PATH", "/models/best.onnx")
MODEL_VERSION = os.getenv("MODEL_VERSION", "dev")
CLASSES = ["drone"]

REQUESTS = Counter("detect_requests_total", "Detection requests", ["status"])
DETECTIONS = Counter("detections_total", "Objects detected")
LATENCY = Histogram("inference_seconds", "Model inference time",
                    buckets=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2))

app = FastAPI(title="drone-detect")
detector = Detector(MODEL_PATH, conf=float(os.getenv("CONF", "0.3")),
                    threads=int(os.getenv("THREADS", "0")))


@app.get("/health")
def health():
    return {"status": "ok", "model": os.path.basename(MODEL_PATH), "version": MODEL_VERSION}


async def run_detection(file, conf):
    try:
        img = Image.open(io.BytesIO(await file.read()))
        img.load()
    except (UnidentifiedImageError, OSError):
        REQUESTS.labels("bad_request").inc()
        raise HTTPException(400, "not a valid image")

    t0 = time.perf_counter()
    dets = detector(img, conf)
    dt = time.perf_counter() - t0
    LATENCY.observe(dt)
    REQUESTS.labels("ok").inc()
    DETECTIONS.inc(len(dets))
    for d in dets:
        d["label"] = CLASSES[d["class"]]
    return img, dets, dt


@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: float | None = Query(None, ge=0, le=1)):
    img, dets, dt = await run_detection(file, conf)
    return {"detections": dets, "inference_ms": round(dt * 1000, 1),
            "image_size": list(img.size), "model_version": MODEL_VERSION}


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...), conf: float | None = Query(None, ge=0, le=1)):
    """Same as /detect but returns the image with boxes drawn on it."""
    img, dets, _ = await run_detection(file, conf)
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    width = max(2, img.width // 300)
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        draw.rectangle((x1, y1, x2, y2), outline=(255, 40, 40), width=width)
        text = f"{d['label']} {d['conf']:.2f}"
        tw = draw.textlength(text)
        draw.rectangle((x1, y1 - 14, x1 + tw + 6, y1), fill=(255, 40, 40))
        draw.text((x1 + 3, y1 - 13), text, fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return Response(buf.getvalue(), media_type="image/jpeg",
                    headers={"X-Detections": str(len(dets))})


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
