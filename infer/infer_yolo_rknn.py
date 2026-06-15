import numpy as np
import cv2
from utils.config import get


class YOLOv12nRKNN:
    def __init__(self, model_path=None):
        # Импорт rknn и чтение конфига только при создании инстанса,
        # а не на уровне модуля — иначе import падает на любой машине без RKNN.
        from rknn.api import RKNN

        self.model_path = model_path or get("model.rknn")
        self.input_size = get("data.imgsz")
        self._conf_threshold = get("cascade.conf_threshold")
        self._iou_threshold = get("cascade.iou_threshold")
        self._cascade_conf_threshold = get("cascade.cascade_conf")

        self.rknn = RKNN(verbose=False)
        self.rknn.load_rknn(self.model_path)
        self.rknn.init_runtime()

    def preprocess(self, frame, target_size=None):
        if target_size is None:
            target_size = self.input_size
        h, w = frame.shape[:2]
        img = cv2.resize(frame, (target_size, target_size))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return img, h, w

    def postprocess(self, outputs, orig_h, orig_w, conf_thres=None,
                    iou_thres=None, target_size=None):
        if conf_thres is None:
            conf_thres = self._conf_threshold
        if iou_thres is None:
            iou_thres = self._iou_threshold
        if target_size is None:
            target_size = self.input_size
        predictions = np.squeeze(outputs[0])
        if predictions.ndim == 1:
            predictions = predictions.reshape(1, -1)
        bboxes = []

        for pred in predictions:
            obj_conf = pred[4]
            if obj_conf < conf_thres:
                continue
            class_scores = pred[5:]
            class_id = np.argmax(class_scores)
            total_conf = obj_conf * class_scores[class_id]
            if total_conf < conf_thres:
                continue
            x_center, y_center, bw, bh = pred[:4]
            x1 = int((x_center - bw / 2) * orig_w / target_size)
            y1 = int((y_center - bh / 2) * orig_h / target_size)
            x2 = int((x_center + bw / 2) * orig_w / target_size)
            y2 = int((y_center + bh / 2) * orig_h / target_size)
            bboxes.append([max(0, x1), max(0, y1),
                           min(orig_w, x2), min(orig_h, y2),
                           total_conf, int(class_id)])

        return self._nms(bboxes, iou_thres)

    def infer(self, frame, conf_thres=None):
        if conf_thres is None:
            conf_thres = self._conf_threshold
        input_tensor, h, w = self.preprocess(frame)
        outputs = self.rknn.inference(inputs=[input_tensor])
        return self.postprocess(outputs, h, w, conf_thres=conf_thres)

    def infer_on_crop(self, crop, conf_thres=None):
        return self.infer(crop, conf_thres=conf_thres) if crop.size else []

    def refine_pricetag(self, crop):
        bboxes = self.infer_on_crop(crop, conf_thres=self._cascade_conf_threshold)
        pricetags = [b for b in bboxes if b[5] == 0]
        return max(pricetags, key=lambda b: b[4]) if pricetags else None

    def _nms(self, bboxes, iou_thres):
        if not bboxes:
            return []
        bboxes = sorted(bboxes, key=lambda x: x[4], reverse=True)
        keep = []
        while bboxes:
            best = bboxes.pop(0)
            keep.append(best)
            bboxes = [b for b in bboxes if self._iou(best, b) < iou_thres]
        return keep

    def _iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (area1 + area2 - inter + 1e-6)

    def release(self):
        self.rknn.release()
