"""Check price_block box sizes in cleaned dataset."""
import glob

areas = []
for f in glob.glob(r"Z:\Hakaton_project\data\yolov12n\labels\train\*.txt"):
    with open(f) as fh:
        for line in fh:
            p = line.strip().split()
            if len(p) == 5 and p[0] == "1":
                w, h = float(p[3]), float(p[4])
                areas.append(w * h)

areas.sort()
n = len(areas)
if n == 0:
    print("No price_block boxes found!")
else:
    print(f"price_block boxes: {n}")
    print(f"min={areas[0]:.6f}")
    print(f"median={areas[n//2]:.6f}")
    print(f"p90={areas[int(n*0.9)]:.6f}")
    print(f"max={areas[-1]:.6f}")

    tiny = sum(1 for a in areas if a < 0.001)
    small = sum(1 for a in areas if 0.001 <= a < 0.01)
    medium = sum(1 for a in areas if 0.01 <= a < 0.1)
    large = sum(1 for a in areas if a >= 0.1)
    print(f"\nSize distribution:")
    print(f"  tiny (<0.001):      {tiny}")
    print(f"  small (0.001-0.01): {small}")
    print(f"  medium (0.01-0.1):  {medium}")
    print(f"  large (>0.1):       {large}")

    # What % of 640x640 is median?
    px = (areas[n//2] ** 0.5) * 640
    print(f"\nMedian box: ~{px:.0f}px at 640x640")
