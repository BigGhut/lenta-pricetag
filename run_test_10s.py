r"""
Test run: first 10 seconds of video through the cascade pipeline.
Outputs: annotated video + CSV results.

Run from Windows PowerShell/CMD:
    cd Z:\Hakaton_project
    .venv_torch\Scripts\python.exe run_test_10s.py --input 26_12-20.mp4 --duration 10 --skip 3
"""

import cv2
import argparse
import time
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.cascade_pipeline import CascadePipeline
from utils.config import get


def main():
    parser = argparse.ArgumentParser(description="LENTA Pipeline — 10s test run")
    parser.add_argument("--input", "-i", default="26_12-20.mp4", help="Input video path")
    parser.add_argument("--output-dir", "-o", default="results", help="Output directory")
    parser.add_argument("--duration", "-d", type=float, default=10.0, help="Duration in seconds")
    parser.add_argument("--skip", "-s", type=int, default=None, help="Process every Nth frame")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    video_path = args.input
    if not os.path.isabs(video_path):
        video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_total = total_frames / fps if fps > 0 else 0

    max_frames = int(args.duration * fps) if fps > 0 else total_frames
    skip = args.skip or get("video.skip_frames", 1)

    print(f"Video: {os.path.basename(video_path)}")
    print(f"  Resolution: {width}x{height}, FPS: {fps:.1f}, Total: {total_frames} frames ({duration_total:.1f}s)")
    print(f"  Processing: first {args.duration}s = {max_frames} frames, skip={skip}")
    print()

    # Output video writer (half resolution for speed)
    out_w, out_h = width // 2, height // 2
    out_video = os.path.join(args.output_dir, "test_10s_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, fps / skip, (out_w, out_h))

    # CSV output
    out_csv = os.path.join(args.output_dir, "test_10s_results.csv")

    pipe = CascadePipeline(csv_path=out_csv)

    frame_count = 0
    processed = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count > max_frames:
            break

        if frame_count % skip != 0:
            continue

        processed += 1
        results = pipe.process_frame(frame, os.path.basename(video_path), frame_count)

        # Draw annotations
        frame_annotated = frame.copy()
        for r in results:
            x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
            color = (0, 0, 255) if r.get("type") == "red" else (0, 255, 0)
            cv2.rectangle(frame_annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_annotated, f"{r['confidence']:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            ocr = r.get("ocr", {})
            if ocr.get("price_default"):
                cv2.putText(frame_annotated, str(ocr["price_default"]), (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if ocr.get("product_name"):
                cv2.putText(frame_annotated, str(ocr["product_name"][:30]), (x1, y2 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        cv2.putText(frame_annotated, f"Frame {frame_count} | Tags: {len(results)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Resize and write
        frame_resized = cv2.resize(frame_annotated, (out_w, out_h))
        writer.write(frame_resized)

        if processed % 10 == 0:
            elapsed = time.time() - t_start
            fps_proc = processed / elapsed if elapsed > 0 else 0
            new = sum(1 for r in results if not r.get("tracked"))
            print(f"  [{frame_count}/{max_frames}] {len(results)} tags (new={new}) | {fps_proc:.1f} fps")

    cap.release()
    writer.release()
    pipe.release()

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s ({processed} frames, {processed / elapsed:.1f} fps)")
    print(f"  Annotated video: {out_video}")
    print(f"  CSV results: {out_csv}")
    s = pipe.stats
    print(f"  Stats: OCR calls={s['ocr_calls']}, tracked hits={s['tracked_hits']}, active tags={s['active_tags']}")


if __name__ == "__main__":
    main()
