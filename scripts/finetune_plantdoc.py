"""Fine-tuning de YOLOv8n sur PlantDoc (29 classes, 416 px, CPU) puis export ONNX FP32 -> models/plantdoc_fp32.onnx."""
import shutil
import sys

from ultralytics import YOLO

# Usage : python scripts/finetune_plantdoc.py [epochs]   (25 par défaut ; ex. 60)
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 25
IMGSZ = 416
model = YOLO("yolov8n.pt")
model.train(data="data/plantdoc_yolo/data.yaml", epochs=EPOCHS, patience=20, imgsz=IMGSZ, batch=16, workers=0,
            device="cpu", project="runs/plantdoc", name="train" if EPOCHS == 25 else f"train{EPOCHS}", exist_ok=True,
            plots=False, verbose=False)
best = YOLO(str(model.trainer.best))
p = best.export(format="onnx", imgsz=IMGSZ, opset=17, simplify=True, dynamic=False)
shutil.copy(p, "models/plantdoc_fp32.onnx")
print("saved models/plantdoc_fp32.onnx", model.trainer.best)
