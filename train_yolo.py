from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer
from utils.config import get


class EarlyStoppingCallback:
    def __init__(self):
        self.patience = get("training.patience")
        self.delta = get("training.delta")
        self.monitor = get("training.monitor")
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def on_train_epoch_end(self, trainer: BaseTrainer):
        if self.monitor not in trainer.metrics:
            return
        score = float(trainer.metrics[self.monitor])
        if self.best_score is None:
            self.best_score = score
            self.counter = 0
        elif score > self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"Early stopping: {self.monitor} did not improve for {self.patience} epochs.")
                trainer.stop = True
        else:
            self.best_score = score
            self.counter = 0


if __name__ == "__main__":
    model = YOLO(get("model.pretrained"))
    model.add_callback("on_train_epoch_end", EarlyStoppingCallback().on_train_epoch_end)
    results = model.train(
        data=get("data.yaml"),
        epochs=get("training.epochs"),
        imgsz=get("data.imgsz"),
        batch=get("training.batch"),
        device=get("training.device"),
        workers=get("training.workers"),
        project=get("paths.models"),
        name="yolov12n",
        exist_ok=True,
    )
