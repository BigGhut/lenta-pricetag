import cv2
import argparse
from utils.cascade_pipeline import CascadePipeline
from utils.config import get


def run_camera():
    pipe = CascadePipeline()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        pipe.release()
        return

    print("Cascade pipeline started. Press 'q' to quit.")
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        results = pipe.process_frame(frame, "camera", frame_count)

        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            color = (0, 0, 255) if r.get("type") == "red" else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{r['confidence']:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if r.get("ocr", {}).get("price_default"):
                cv2.putText(frame, str(r["ocr"]["price_default"]), (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.putText(frame, f"Frame {frame_count} | Tags: {len(results)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("LENTA Cascade Pipeline", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pipe.release()


def run_video(video_path, skip_frames=None):
    skip_frames = skip_frames or get("video.skip_frames")
    pipe = CascadePipeline()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    filename = video_path.split("/")[-1].split("\\")[-1]

    print(f"Processing: {filename} ({total} frames @ {fps:.1f} fps, step={skip_frames})")
    frame_count = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % skip_frames != 0:
            continue
        processed += 1
        results = pipe.process_frame(frame, filename, frame_count)
        if processed % 10 == 0:
            new = sum(1 for r in results if not r.get('tracked'))
            print(f"  [{frame_count/total*100:5.1f}%] Frame {frame_count} | {len(results)} tags (new={new})")

    cap.release()
    s = pipe.stats
    print(f"\nDone. {processed} frames | OCR: {s['ocr_calls']} | Cache: {s['tracked_hits']}")
    pipe.release()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="LENTA Pricetag Pipeline")
    p.add_argument("--video", "-v", help="Path to video file")
    p.add_argument("--skip", "-s", type=int, default=None, help="Process every Nth frame")
    a = p.parse_args()
    run_video(a.video, a.skip) if a.video else run_camera()
