## 2024-05-18 - Avoid NumPy in Inner Tracking Loops
**Learning:** Operations on small Python objects (tuples or small arrays) inside tight nested loops (like Hungarian matching NxM) are significantly slower when using NumPy functions (`np.sqrt`) and redundant array instantiations than when using native Python `math` module and `float` casting.
**Action:** Default to Python native modules for scalars or small 4-element structures inside high-frequency loops instead of deferring to NumPy.
