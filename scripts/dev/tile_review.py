"""Create repeated-tile previews from mcp_scripts.py's JSON report (Pillow/numpy required)."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

report_path = Path(sys.argv[1])
report = json.loads(report_path.read_text(encoding="utf-8"))
sources = [("source", report["source"])]
for i, case in enumerate(report["cases"]):
    if not case["ok"]:
        raise RuntimeError(case)
    for j, image in enumerate(case["result"]["images"]):
        label = f'{i}-{case["values"]["direction"]}-restore{case["values"].get("restore", "true")}-{j}'
        sources.append((label, image["path"]))

sheet = Image.new("RGB", (400 * len(sources), 430), "#dddddd")
metrics = {}
for i, (label, path) in enumerate(sources):
    with Image.open(path) as src:
        im = src.convert("RGB")
    a = np.asarray(im, dtype=np.float32)
    metrics[label] = {"x_edge_mae": float(np.abs(a[:, 0] - a[:, -1]).mean()),
                      "y_edge_mae": float(np.abs(a[0] - a[-1]).mean())}
    tile = Image.new("RGB", (im.width * 2, im.height * 2))
    for x in range(2):
        for y in range(2):
            tile.paste(im, (x * im.width, y * im.height))
    tile.save(report_path.with_name(report_path.stem + "-" + label + ".png"))
    sheet.paste(tile.resize((400, 400)), (i * 400, 0))
    ImageDraw.Draw(sheet).text((i * 400, 402), label, fill="black")
sheet.save(report_path.with_suffix(".jpg"))
report_path.with_name(report_path.stem + "-metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(json.dumps(metrics, indent=2))
