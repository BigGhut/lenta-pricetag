import sys
import cv2
import json

sys.path.insert(0, ".")
from infer.infer_yolo_rknn import YOLOv12nRKNN
try:
    from infer.infer_paddleocr import recognize_all
except ImportError:
    from infer.infer_easyocr import recognize_all


def process_image(image_path):
    yolo = YOLOv12nRKNN()
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Cannot read {image_path}")
        return

    bboxes = yolo.infer(frame)

    results = []
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox[:4])
        conf = bbox[4]
        class_id = int(bbox[5])

        if class_id != 0:
            continue

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        ocr_results = recognize_all(crop)

        entry = {
            "bbox": [x1, y1, x2, y2],
            "confidence": round(float(conf), 3),
            "ocr": ocr_results,
        }
        results.append(entry)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if ocr_results.get("price"):
            cv2.putText(frame, f"{ocr_results['price']}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if ocr_results.get("product_name"):
            cv2.putText(frame, ocr_results["product_name"][:30], (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    print(json.dumps(results, ensure_ascii=False, indent=2))
    cv2.imshow("Result", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    yolo.release()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python utils/infer_on_image.py <image_path>")
    else:
        process_image(sys.argv[1])
