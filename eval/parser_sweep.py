"""Score the parser against the synthetic ground truth (no model or network needed).

Usage:
    python eval/parser_sweep.py /path/to/visual-document-qa

Writes eval/parser_results.json and prints LaTeX table rows.
"""

import json
import os
import shutil
import sys
import tempfile
import time

repo = sys.argv[1] if len(sys.argv) > 1 else "."
sys.path.insert(0, os.path.abspath(repo))

import fitz  # noqa: E402
from parser import parse  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus")
truth = json.load(open(os.path.join(CORPUS, "truth.json")))

THRESHOLDS = [10, 20, 40, 80, 0]  # 0 disables the vector rule
results = []

# Per-page drawing counts (descriptive statistic, independent of the threshold).
drawing_counts = {}
for name in truth:
    with fitz.open(os.path.join(CORPUS, name)) as d:
        drawing_counts[name] = {i + 1: len(p.get_drawings()) for i, p in enumerate(d)}

for thr in THRESHOLDS:
    row = dict(threshold=thr, pages=0, vision_calls=0, rendered_vector=0,
               chart_pages=0, chart_detected=0, false_vector_renders=0,
               scanned_pages=0, scanned_detected=0, raster_charts=0,
               raster_extracted=0, embedded_total=0, parse_seconds=0.0)
    for name, pages in truth.items():
        tmp = tempfile.mkdtemp()
        t0 = time.perf_counter()
        parsed = parse(os.path.join(CORPUS, name), image_root=tmp,
                       min_image_size=100, max_images=30, render_dpi=150,
                       vector_chart_min_drawings=thr)
        row["parse_seconds"] += time.perf_counter() - t0
        by_page = {}
        for img in parsed.images:
            by_page.setdefault(img.page, []).append(img.kind)
        shutil.rmtree(tmp, ignore_errors=True)
        row["pages"] += parsed.page_count
        row["vision_calls"] += len(parsed.images)
        for pno, info in pages.items():
            kinds = by_page.get(int(pno), [])
            c = info["content"]
            if "vector_graphics" in kinds:
                row["rendered_vector"] += 1
            if c == "vector_chart":
                row["chart_pages"] += 1
                row["chart_detected"] += int("vector_graphics" in kinds)
            elif "vector_graphics" in kinds:
                row["false_vector_renders"] += 1
            if c == "scanned":
                row["scanned_pages"] += 1
                row["scanned_detected"] += int("scanned_page" in kinds)
            if c == "raster_chart":
                row["raster_charts"] += 1
                row["raster_extracted"] += int("embedded_image" in kinds)
        row["embedded_total"] += sum(k == "embedded_image" for ks in by_page.values() for k in ks)
    results.append(row)

json.dump({"drawing_counts": drawing_counts, "sweep": results},
          open(os.path.join(HERE, "parser_results.json"), "w"), indent=2)

print("drawing counts per page:")
for n, d in drawing_counts.items():
    print(" ", n, d)
print()
for r in results:
    label = "Disabled (0)" if r["threshold"] == 0 else str(r["threshold"]) + (" (default)" if r["threshold"] == 40 else "")
    print(f"{label} & {r['rendered_vector']} & {r['vision_calls']} & "
          f"{r['chart_detected']}/{r['chart_pages']} & {r['false_vector_renders']} \\\\")
print()
r0 = [r for r in results if r["threshold"] == 40][0]
print("scanned detected", r0["scanned_detected"], "/", r0["scanned_pages"])
print("raster charts extracted", r0["raster_extracted"], "/", r0["raster_charts"],
      "| embedded images total", r0["embedded_total"])
print("parse seconds (default thr, all docs):", round(r0["parse_seconds"], 3))
