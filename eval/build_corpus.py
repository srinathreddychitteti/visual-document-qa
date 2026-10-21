"""Build a small synthetic PDF corpus with known ground truth.

Every page records what it contains, so parser detection can be scored without
any language model. Output: eval/corpus/*.pdf and eval/corpus/truth.json.

Run:  python eval/build_corpus.py
"""

import io
import json
import os

import fitz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")
os.makedirs(OUT, exist_ok=True)

LOREM = (
    "This section describes the quarterly results of the organisation. Revenue, "
    "operating cost and headcount are summarised for each region, and the main "
    "drivers of change are discussed in the paragraphs that follow. "
) * 6


def text_block(page, y=72, title="Report"):
    page.insert_text((72, y), title, fontsize=16)
    page.insert_textbox(fitz.Rect(72, y + 20, 520, y + 200), LOREM, fontsize=10)


def vector_chart_pdf(kind: str, n: int) -> bytes:
    """Return a one-page matplotlib PDF (vector paths, no raster)."""
    fig, ax = plt.subplots(figsize=(5, 3))
    xs = list(range(n))
    ys = [3 + (i * 7) % 11 for i in xs]
    if kind == "bar":
        ax.bar(xs, ys)
    else:
        ax.plot(xs, ys, marker="o")
    ax.set_title(f"{kind} chart with {n} points")
    ax.set_xlabel("period")
    ax.set_ylabel("value")
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    plt.close(fig)
    return buf.getvalue()


def raster_chart_png(n: int) -> bytes:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(range(n), [3 + (i * 5) % 9 for i in range(n)])
    ax.set_title(f"raster chart {n}")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return buf.getvalue()


def logo_png() -> bytes:
    fig = plt.figure(figsize=(2, 1))
    fig.text(0.5, 0.5, "LOGO", ha="center", va="center", fontsize=20)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return buf.getvalue()


def add_vector_chart(page, kind, n):
    src = fitz.open("pdf", vector_chart_pdf(kind, n))
    page.show_pdf_page(fitz.Rect(72, 300, 472, 540), src, 0)


def add_ruled_table(page, rows=8, cols=4, top=300):
    """A table drawn with ruling lines: a likely false positive for the chart rule."""
    left, w, h = 72, 100, 18
    for r in range(rows + 1):
        page.draw_line((left, top + r * h), (left + cols * w, top + r * h))
    for c in range(cols + 1):
        page.draw_line((left + c * w, top), (left + c * w, top + rows * h))
    for r in range(rows):
        for c in range(cols):
            page.insert_text((left + c * w + 4, top + r * h + 13), f"{r*cols+c}", fontsize=9)


truth = {}


def record(name, page_no, **kw):
    truth.setdefault(name, {})[page_no] = kw


# --- document 1: vector charts of increasing complexity -------------------
name = "vector_charts.pdf"
doc = fitz.open()
p = doc.new_page(); text_block(p, title="Plain page"); record(name, 1, content="text")
for i, (kind, n) in enumerate([("bar", 4), ("bar", 12), ("line", 8), ("line", 30)], start=2):
    p = doc.new_page(); text_block(p, title=f"{kind} chart")
    add_vector_chart(p, kind, n)
    record(name, i, content="vector_chart", detail=f"{kind}-{n}")
doc.save(os.path.join(OUT, name)); doc.close()

# --- document 2: ruled tables (should not need a chart description) --------
name = "ruled_tables.pdf"
doc = fitz.open()
for i, (r, c) in enumerate([(4, 3), (8, 4), (14, 5)], start=1):
    p = doc.new_page(); text_block(p, title="Table page")
    add_ruled_table(p, rows=r, cols=c)
    record(name, i, content="table_vector", detail=f"{r}x{c}")
doc.save(os.path.join(OUT, name)); doc.close()

# --- document 3: raster charts, repeated logo, tiny icon -------------------
name = "raster_images.pdf"
doc = fitz.open()
logo = logo_png()
icon = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 16, 16), False)
xref_logo = None
for i in range(1, 5):
    p = doc.new_page(); text_block(p, title="Raster page")
    if xref_logo is None:
        xref_logo = p.insert_image(fitz.Rect(400, 20, 520, 80), stream=logo)
    else:
        p.insert_image(fitz.Rect(400, 20, 520, 80), xref=xref_logo)
    p.insert_image(fitz.Rect(300, 30, 316, 46), pixmap=icon)
    if i in (2, 4):
        p.insert_image(fitz.Rect(72, 300, 392, 540), stream=raster_chart_png(6 + i))
        record(name, i, content="raster_chart")
    else:
        record(name, i, content="text")
doc.save(os.path.join(OUT, name)); doc.close()

# --- document 4: scanned pages (image only, no text layer) -----------------
name = "scanned.pdf"
src = fitz.open()
for i in range(3):
    p = src.new_page(); text_block(p, title=f"Scanned page {i+1}")
doc = fitz.open()
for i in range(3):
    pix = src[i].get_pixmap(dpi=100)
    p = doc.new_page()
    p.insert_image(p.rect, stream=pix.tobytes("png"))
    record(name, i + 1, content="scanned")
doc.save(os.path.join(OUT, name)); doc.close()

with open(os.path.join(OUT, "truth.json"), "w") as f:
    json.dump(truth, f, indent=2)
print("Built", len(truth), "documents in", OUT)
