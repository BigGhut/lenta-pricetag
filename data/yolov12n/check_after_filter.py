"""Count remaining price_block boxes after filter."""
import glob
from collections import Counter

counts = Counter()
areas = []

for f in glob.glob(r"Z:\Hakaton_project\data\yolov12n\labels\train\*.txt"):
    with open(f) as fh:
        for line in fh:
            p = line.strip().split()
            if len(p) == 5:
                counts[int(p[0])] += 1
                if p[0] == "1":
                    w, h = float(p[3]), float(p[4])
                    areas.append(w * h)

print("Class distribution (train):")
for cls in sorted(counts):
    print(f"  class {cls}: {counts[cls]}")

if areas:
    areas.sort()
    n = len(areas)
    print(f"\nprice_block remaining: {n}")
    print(f"min={areas[0]:.6f} (~{int((areas[0]**0.5)*640)}px)")
    print(f"median={areas[n//2]:.6f} (~{int((areas[n//2]**0.5)*640)}px)")
    print(f"p90={areas[int(n*0.9)]:.6f} (~{int((areas[int(n*0.9)]**0.5)*640)}px)")
    
    above_40px = sum(1 for a in areas if (a**0.5)*640 >= 40)
    print(f"\nBoxes >= 40px: {above_40px} ({above_40px/n*100:.1f}%)")
