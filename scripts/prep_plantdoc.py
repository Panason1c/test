"""Convertit PlantDoc (CSV, boîtes en pixels) en dataset YOLO.
train officiel -> train (90 %) + val (10 %, choix du meilleur epoch) ; test officiel -> test (évaluation finale).
Les images sont réduites à 640 px max (accélère l'entraînement CPU) et renommées (noms d'origine trop longs/invalides).
Usage: python scripts/prep_plantdoc.py data/plantdoc_raw/PlantDoc-Object-Detection-Dataset
"""
import random
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

src = Path(sys.argv[1])
out = Path("data/plantdoc_yolo")
dfs = {s: pd.read_csv(src / f"{s}_labels.csv") for s in ("train", "test")}
classes = sorted(set(dfs["train"]["class"]) | set(dfs["test"]["class"]))
cid = {c: i for i, c in enumerate(classes)}

random.seed(0)
train_files = sorted(dfs["train"].filename.unique())
random.shuffle(train_files)
n_val = len(train_files) // 10
splits = {"val": [("train", f) for f in train_files[:n_val]],
          "train": [("train", f) for f in train_files[n_val:]],
          "test": [("test", f) for f in sorted(dfs["test"].filename.unique())]}

for name, items in splits.items():
    (out / "images" / name).mkdir(parents=True, exist_ok=True)
    (out / "labels" / name).mkdir(parents=True, exist_ok=True)
    kept = 0
    for k, (orig, f) in enumerate(items):
        p = src / orig.upper() / f
        if not p.exists():
            continue  # noms trop longs pour Windows : non extraits par git
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        w, h = img.size
        rows = dfs[orig][dfs[orig].filename == f]
        lines = []
        for _, r in rows.iterrows():
            x1, y1, x2, y2 = (max(0, r.xmin), max(0, r.ymin), min(w, r.xmax), min(h, r.ymax))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            lines.append(f"{cid[r['class']]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        if not lines:
            continue
        s = 640 / max(w, h)
        if s < 1:
            img = img.resize((round(w * s), round(h * s)))
        stem = f"{name}_{k:05d}"
        img.save(out / "images" / name / f"{stem}.jpg", quality=92)
        (out / "labels" / name / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        kept += 1
    print(name, kept, "images")

names = "\n".join(f"  {i}: {c}" for i, c in enumerate(classes))
(out / "data.yaml").write_text(f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n")
print(len(classes), "classes")
