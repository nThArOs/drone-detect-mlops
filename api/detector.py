"""YOLO ONNX inference with plain numpy (no torch in the serving image)."""
import numpy as np
import onnxruntime as ort
from PIL import Image


class Detector:
    def __init__(self, model_path, conf=0.3, iou=0.5, threads=0):
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = inp.shape[2]  # square input, e.g. 640
        self.conf = conf
        self.iou = iou

    def preprocess(self, img):
        # letterbox: keep aspect ratio, pad to a square
        w, h = img.size
        scale = self.size / max(w, h)
        nw, nh = round(w * scale), round(h * scale)
        canvas = Image.new("RGB", (self.size, self.size), (114, 114, 114))
        pad_x, pad_y = (self.size - nw) // 2, (self.size - nh) // 2
        canvas.paste(img.convert("RGB").resize((nw, nh), Image.BILINEAR), (pad_x, pad_y))
        x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        return x, scale, pad_x, pad_y

    def __call__(self, img, conf_thr=None):
        x, scale, pad_x, pad_y = self.preprocess(img)
        out = self.session.run(None, {self.input_name: x})[0][0]  # (4 + n_classes, n_anchors)
        boxes, scores = out[:4].T, out[4:].T
        cls = scores.argmax(1)
        conf = scores.max(1)
        keep = conf >= (self.conf if conf_thr is None else conf_thr)
        boxes, conf, cls = boxes[keep], conf[keep], cls[keep]

        # cx, cy, w, h (letterboxed) -> x1, y1, x2, y2 (original image)
        xy = np.empty_like(boxes)
        xy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2 - pad_x) / scale
        xy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2 - pad_y) / scale
        xy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2 - pad_x) / scale
        xy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2 - pad_y) / scale
        w, h = img.size
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, w)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, h)

        idx = nms(xy, conf, self.iou)
        return [
            {"box": [round(float(v), 1) for v in xy[i]], "conf": round(float(conf[i]), 3), "class": int(cls[i])}
            for i in idx
        ]


def nms(boxes, scores, iou_thr):
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou < iou_thr]
    return keep
