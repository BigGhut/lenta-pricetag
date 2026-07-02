
## $(date +%Y-%m-%d) - [Optimize IoU Calculation]
**Learning:** Bounding Box intersection code (`iou` in object detection pipelines) is a massive hot path that does heavy array manipulation and float calculation. Implementing an early return for non-intersecting boxes eliminates area calculation and division entirely.
**Action:** When working in ML/CV repositories, look at primitive intersection/distance checks, and implement early zero-checks to avoid division or heavy multiplications on non-overlapping objects.
