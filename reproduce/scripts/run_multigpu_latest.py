"""Execute the latest dataset-mounted public MultiGPU benchmark notebook."""

import json
from pathlib import Path

matches = list(Path("/kaggle/input").rglob("02_multigpu_boundary_1xt4.ipynb"))
if len(matches) != 1:
    raise RuntimeError(f"expected one mounted MultiGPU notebook, got {matches}")
notebook = json.loads(matches[0].read_text(encoding="utf-8"))
for cell in notebook["cells"]:
    if cell.get("cell_type") == "code":
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        get_ipython().run_cell(source)
