"""PlantDoc (YOLOv8n fine-tuné, 29 classes, 416 px) : quantization + comparaison sur le jeu de test officiel.

Variantes : FP32, FP16, INT8 dynamique, INT8 statique (calibration sur 100 images de train).
Métriques : mAP50 et mAP50-95 contre les vraies annotations du split test, taille, latence CPU.
Usage (racine du projet) : python scripts/plantdoc_pipeline.py
Sortie : models/plantdoc_{fp16,int8_dynamic,int8_static}.onnx, results/benchmark_plantdoc.csv
"""
import re
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from onnxconverter_common import float16
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                      quantize_dynamic, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

SIZE = 416
M = Path("models")
SRC = M / "plantdoc_fp32.onnx"
DATA = Path("data/plantdoc_yolo")
TEST = sorted((DATA / "images" / "test").glob("*.jpg"))
CALIB = sorted((DATA / "images" / "train").glob("*.jpg"))[:100]
NC = len([l for l in (DATA / "data.yaml").read_text().splitlines() if re.match(r"^\s+\d+:", l)])


def preprocess(path):
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    r = SIZE / max(h, w)
    nh, nw = round(h * r), round(w * r)
    top, left = (SIZE - nh) // 2, (SIZE - nw) // 2
    canvas = np.full((SIZE, SIZE, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
    x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return x, (h, w), r, left, top


def decode(out, shape, r, left, top, conf=0.001, iou=0.7):
    p = out[0].T
    cls_s = p[:, 4:]
    scores, cls = cls_s.max(1), cls_s.argmax(1)
    m = scores > conf
    p, scores, cls = p[m], scores[m], cls[m]
    h, w = shape
    xyxy = np.stack([p[:, 0] - p[:, 2] / 2, p[:, 1] - p[:, 3] / 2, p[:, 0] + p[:, 2] / 2, p[:, 1] + p[:, 3] / 2], 1)
    xyxy[:, [0, 2]] = ((xyxy[:, [0, 2]] - left) / r).clip(0, w)
    xyxy[:, [1, 3]] = ((xyxy[:, [1, 3]] - top) / r).clip(0, h)
    dets = []
    for c in np.unique(cls):
        k = np.where(cls == c)[0]
        b = xyxy[k]
        xywh = np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)
        keep = np.array(cv2.dnn.NMSBoxes(xywh.tolist(), scores[k].tolist(), conf, iou)).flatten().astype(int)
        dets += [(b[i], float(scores[k][i]), int(c)) for i in keep]
    return dets


def load_gt(img_path, shape):
    h, w = shape
    lab = Path(str(img_path).replace("images", "labels")).with_suffix(".txt")
    gt = []
    for line in lab.read_text().split("\n"):
        if line.strip():
            c, cx, cy, bw, bh = line.split()
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            gt.append((np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]), int(c)))
    return gt


def iou_one(b, boxes):
    x1 = np.maximum(b[0], boxes[:, 0]); y1 = np.maximum(b[1], boxes[:, 1])
    x2 = np.minimum(b[2], boxes[:, 2]); y2 = np.minimum(b[3], boxes[:, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    a = (b[2] - b[0]) * (b[3] - b[1])
    a2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (a + a2 - inter + 1e-9)


def ap(tp, conf, n_gt):
    if len(tp) == 0 or n_gt == 0:
        return 0.0
    tp = tp[np.argsort(-conf)]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / n_gt, ctp / (ctp + cfp)
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.flip(np.maximum.accumulate(np.flip(np.concatenate([[1], prec, [0]]))))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def evaluate(path):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    thr = np.linspace(0.5, 0.95, 10)
    store = {c: {"tp": [], "conf": [], "n": 0} for c in range(NC)}
    times = []
    for f in TEST:
        x, shape, r, left, top = preprocess(f)
        t = time.perf_counter()
        out = sess.run(None, {name: x})[0].astype(np.float32)
        times.append((time.perf_counter() - t) * 1000)
        dets = decode(out, shape, r, left, top)
        gt = load_gt(f, shape)
        for c in range(NC):
            gb = np.array([g[0] for g in gt if g[1] == c]).reshape(-1, 4)
            store[c]["n"] += len(gb)
            pd_ = sorted([d for d in dets if d[2] == c], key=lambda d: -d[1])
            tp = np.zeros((len(pd_), len(thr)))
            used = np.zeros((len(gb), len(thr)), bool)
            for k, (b, s, _) in enumerate(pd_):
                if len(gb) == 0:
                    break
                ious = iou_one(b, gb)
                for j, t_ in enumerate(thr):
                    cand = [(ious[q], q) for q in range(len(gb)) if not used[q, j] and ious[q] >= t_]
                    if cand:
                        q = max(cand)[1]
                        used[q, j] = True
                        tp[k, j] = 1
            store[c]["tp"].append(tp)
            store[c]["conf"].append(np.array([d[1] for d in pd_]))
    aps = []
    for c in range(NC):
        if store[c]["n"] == 0:
            continue
        tp = np.concatenate(store[c]["tp"])
        conf = np.concatenate(store[c]["conf"])
        aps.append([ap(tp[:, j], conf, store[c]["n"]) for j in range(len(thr))])
    aps = np.array(aps)
    return dict(mAP50=float(aps[:, 0].mean()), mAP50_95=float(aps.mean()),
                size_mb=path.stat().st_size / 1e6, latency_ms_mean=float(np.mean(times[3:])),
                latency_ms_median=float(np.median(times[3:])))


def build_variants():
    m = float16.convert_float_to_float16(onnx.load(SRC), keep_io_types=True, op_block_list=["Resize"])
    del m.graph.value_info[:]
    onnx.save(m, M / "plantdoc_fp16.onnx")

    pre = M / "_pre_pd.onnx"
    quant_pre_process(str(SRC), str(pre))
    quantize_dynamic(str(pre), str(M / "plantdoc_int8_dynamic.onnx"), weight_type=QuantType.QUInt8)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{"images": preprocess(f)[0]} for f in CALIB])

        def get_next(self):
            return next(self.it, None)

    # Tête de détection gardée en FP32 (boîtes en pixels et scores 0-1 ne partagent pas la même échelle).
    exclude = [n.name for n in onnx.load(pre).graph.node
               if n.name.startswith("/model.22/") and (n.op_type != "Conv" or re.search(r"cv[23]\.\d\.2/Conv|dfl", n.name))]
    quantize_static(str(pre), str(M / "plantdoc_int8_static.onnx"), Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True,
                    nodes_to_exclude=exclude)
    pre.unlink()


if __name__ == "__main__":
    print(len(TEST), "images de test,", len(CALIB), "images de calibration,", NC, "classes")
    build_variants()
    rows = []
    for v in ["fp32", "fp16", "int8_dynamic", "int8_static"]:
        r = evaluate(M / f"plantdoc_{v}.onnx")
        r["model"] = v
        rows.append(r)
        print(r, flush=True)
    Path("results").mkdir(exist_ok=True)
    df = pd.DataFrame(rows).set_index("model").round(3)
    df.to_csv("results/benchmark_plantdoc.csv")
    print(df.to_string())
