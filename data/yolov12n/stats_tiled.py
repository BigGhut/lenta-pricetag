"""Quick stats for video_yolo_tiled_v2 labels."""
import glob

areas = []
count = 0
for f in glob.glob(r"Z:\Hakaton_project\data\video_yolo_tiled_v2\labels\train\*.txt"):
    with open(f) as fh:
        for line in fh:
            p = line.strip().split()
            if len(p) == 5:
                try:
                    w, h = float(p[3]), float(p[4])
                    if w > 1 or h > 1:
                        w, h = w / 480, h / 480
                    areas.append(w * h)
                    count += 1
                except:
                    pass

areas.sort()
n = len(areas)
print(f"Total boxes: {count}")
print(f"Unique boxes (after parse): {n}")
print(f"min={areas[0]:.6f}")
print(f"median={areas[n // 2]:.6f}")
print(f"p90={areas[int(n * 0.9)]:.6f}")
print(f"max={areas[-1]:.6f}")

tiny = sum(1 for a in areas if a < 0.001)
small = sum(1 for a in areas if 0.001 <= a < 0.01)
medium = sum(1 for a in areas if 0.01 <= a < 0.1)
large = sum(1 for a in areas if a >= 0.1)
print(f"\nSize distribution:")
print(f"  tiny (<0.001):     {tiny}")
print(f"  small (0.001-0.01): {small}")
print(f"  medium (0.01-0.1):  {medium}")
print(f"  large (>0.1):       {large}")

# Aspect ratios
ar = []
for f in glob.glob(r"Z:\Hakaton_project\data\video_yolo_tiled_v2\labels\train\*.txt"):
    with open(f) as fh:
        for line in fh:
            p = line.strip().split()
            if len(p) == 5:
                try:
                    w, h = float(p[3]), float(p[4])
                    if w > 1 or h > 1:
                        w, h = w / 480, h / 480
                    if h > 0:
                        ar.append(w / h)
                except:
                    pass

ar.sort()
n = len(ar)
print(f"\nAspect ratio (w/h):")
print(f"  min={ar[0]:.2f}")
print(f"  median={ar[n // 2]:.2f}")
print(f"  p90={ar[int(n * 0.9)]:.2f}")
print(f"  max={ar[-1]:.2f}")
