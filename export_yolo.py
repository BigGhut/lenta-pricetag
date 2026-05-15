from ultralytics import YOLO
from utils.config import get

if __name__ == "__main__":
    model = YOLO(get("model.trained"))
    model.export(
        format="onnx",
        simplify=get("export.simplify"),
        imgsz=get("data.imgsz"),
        dynamic=get("export.dynamic"),
    )
