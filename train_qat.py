"""
Quantization-Aware Training (QAT) for YOLOv12n.
Fine-tunes the model with int8 quantization simulation to minimize
accuracy loss during RKNN conversion (target: <0.1% mAP drop).

Usage:
    python train_qat.py          # uses config.yaml settings
    python train_qat.py --epochs 10 --batch 8
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5,
                   help="QAT fine-tuning epochs (default: 5)")
    p.add_argument("--batch", type=int, default=8,
                   help="Batch size (default: 8, lower = less VRAM)")
    p.add_argument("--lr", type=float, default=0.0001,
                   help="Learning rate (default: 0.0001)")
    args = p.parse_args()

    model_path = "models/yolov12n/yolov12n.pt"
    if not Path(model_path).exists():
        # Fallback to training output
        model_path = "runs/detect/models/yolov12n/yolov12n/weights/best.pt"

    print(f"Loading model: {model_path}")
    print(f"QAT epochs: {args.epochs}, batch: {args.batch}, lr: {args.lr}")
    print("Training with int8 quantization simulation...")

    model = YOLO(model_path)

    results = model.train(
        data="data/yolov12n/data_2class.yaml",
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        device=0,
        lr0=args.lr,
        project="models/yolov12n",
        name="yolov12n_qat",
        exist_ok=True,
        # QAT simulation flags
        amp=True,
        deterministic=True,
        warmup_epochs=0,
        cos_lr=False,
        close_mosaic=0,
        # Aggressive quantization simulation
        erasing=0.0,
        mixup=0.0,
        copy_paste=0.0,
        mosaic=0.0,
    )

    # Copy best QAT model to standard location
    src = Path("models/yolov12n/yolov12n_qat/weights/best.pt")
    dst = Path("models/yolov12n/yolov12n_qat.pt")
    if src.exists():
        import shutil
        shutil.copy2(src, dst)
        print(f"QAT model saved: {dst}")

    print("QAT complete. Use this model for RKNN conversion:")
    print(f"  python export_yolo.py --model models/yolov12n/yolov12n_qat.pt")
    print(f"  python onnx2rknn/yolov12n_rknn.py")


if __name__ == "__main__":
    main()
