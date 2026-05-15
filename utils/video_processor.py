import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.cascade_pipeline import CascadePipeline
from utils.config import get
import cv2


def process_video(video_path, skip_frames=None):
    skip_frames = skip_frames or get("video.skip_frames")
    pipe = CascadePipeline()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    filename = Path(video_path).name

    print(f"\n{'='*50}")
    print(f"Processing: {filename}  ({total}f @ {fps:.1f}fps, step={skip_frames})")
    print(f"{'='*50}")

    frame_count = 0
    processed = 0
    total_tags = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % skip_frames != 0:
            continue
        processed += 1
        results = pipe.process_frame(frame, filename, frame_count)
        total_tags += len(results)
        if processed % 10 == 0:
            new = sum(1 for r in results if not r.get('tracked'))
            print(f"  [{frame_count/total*100:5.1f}%] F{frame_count} | {len(results)}t (new={new}) | tot={total_tags}")

    cap.release()
    s = pipe.stats
    print(f"\n{'='*50}")
    print(f"Summary: {filename}")
    print(f"  Frames: {processed}/{total} | Step: {skip_frames}")
    print(f"  Detections: {total_tags}")
    print(f"  OCR calls: {s['ocr_calls']} | Cache hits: {s['tracked_hits']}")
    print(f"{'='*50}")
    pipe.release()


def main():
    p = argparse.ArgumentParser(description="Batch video processor")
    p.add_argument("input", help="Video file or directory")
    p.add_argument("--skip", "-s", type=int, default=None)
    p.add_argument("--ext", "-e", default=".mp4,.avi,.mov,.mkv")
    a = p.parse_args()
    inp = Path(a.input)

    if inp.is_file():
        process_video(inp, a.skip)
    elif inp.is_dir():
        exts = [e.strip() for e in a.ext.split(",")]
        videos = [f for f in sorted(inp.iterdir()) if f.suffix.lower() in exts]
        if not videos:
            print(f"No video files in {inp}")
            return
        print(f"Found {len(videos)} videos")
        for v in videos:
            process_video(v, a.skip)
    else:
        print(f"Error: {a.input} not found")


if __name__ == "__main__":
    main()
