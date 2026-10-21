"""PDF parsing: extract page text and the visual content worth describing.

Visual content comes from three sources:

1. **Embedded images** (photos, screenshots, raster charts). Duplicates (e.g. a
   logo repeated on every page) and tiny images (icons, bullets) are skipped.
2. **Vector graphics**: many charts are drawn with PDF paths rather than stored
   as images. Pages with many drawing operations are rendered to a PNG.
3. **Scanned pages**: pages without any extractable text are rendered to a PNG
   so the vision model can read them.

All images are written as PNG, so the MIME type sent to the model is correct.
"""

import base64
import logging
import os
import uuid
from dataclasses import dataclass

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class PageText:
    page: int  # 1-based page number
    text: str


@dataclass
class ExtractedImage:
    page: int  # 1-based page number
    path: str
    kind: str  # "embedded_image" | "vector_graphics" | "scanned_page"


@dataclass
class ParsedDocument:
    pages: list[PageText]
    images: list[ExtractedImage]
    image_dir: str
    page_count: int


def encode_image(path: str) -> str:
    """Return the file at ``path`` as a base64 string."""
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def _save_embedded_image(doc: fitz.Document, xref: int, out_path: str, min_size: int) -> bool:
    """Save image ``xref`` as PNG. Returns False if it is too small to matter."""
    pix = fitz.Pixmap(doc, xref)
    try:
        if pix.width < min_size or pix.height < min_size:
            return False
        # CMYK and other colour spaces cannot be saved as PNG directly.
        if pix.colorspace and pix.colorspace.n > 3:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        pix.save(out_path)
        return True
    finally:
        pix = None


def parse(
    path: str,
    image_root: str = "temp_images",
    min_image_size: int = 100,
    max_images: int = 30,
    render_dpi: int = 150,
    vector_chart_min_drawings: int = 40,
) -> ParsedDocument:
    """Parse the PDF at ``path``.

    Args:
        path: PDF file to read.
        image_root: Folder under which a unique sub-folder is created for this
            call's images (so concurrent calls never collide).
        min_image_size: Embedded images narrower or shorter than this (pixels)
            are ignored.
        max_images: Upper bound on images returned, protecting API quotas.
            Page renders are prioritised over embedded images on the same page.
        render_dpi: Resolution used when rendering whole pages.
        vector_chart_min_drawings: A page with at least this many vector drawing
            operations is rendered as an image. 0 disables this.

    Returns:
        A :class:`ParsedDocument` with per-page text and extracted images.
    """
    logger.info("Parsing %s", path)
    image_dir = os.path.join(image_root, str(uuid.uuid4()))
    os.makedirs(image_dir, exist_ok=True)

    pages: list[PageText] = []
    images: list[ExtractedImage] = []
    seen_xrefs: set[int] = set()

    with fitz.open(path) as doc:
        page_count = doc.page_count
        for page_index, page in enumerate(doc):
            page_num = page_index + 1
            text = page.get_text().strip()
            if text:
                pages.append(PageText(page=page_num, text=text))

            if len(images) >= max_images:
                continue

            # Decide whether the whole page should be rendered.
            render_kind = None
            if not text:
                render_kind = "scanned_page"
            elif vector_chart_min_drawings > 0:
                try:
                    if len(page.get_drawings()) >= vector_chart_min_drawings:
                        render_kind = "vector_graphics"
                except Exception as e:  # malformed drawing commands
                    logger.warning("Could not inspect drawings on page %d: %s", page_num, e)

            if render_kind:
                out_path = os.path.join(image_dir, f"page{page_num}_render.png")
                page.get_pixmap(dpi=render_dpi).save(out_path)
                images.append(ExtractedImage(page=page_num, path=out_path, kind=render_kind))
                # The render already contains every image on this page.
                continue

            try:
                image_list = page.get_images(full=True)
            except Exception as e:
                logger.warning("Could not list images on page %d: %s", page_num, e)
                image_list = []

            for img_index, img in enumerate(image_list, start=1):
                if len(images) >= max_images:
                    break
                xref = img[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                out_path = os.path.join(image_dir, f"page{page_num}_img{img_index}.png")
                try:
                    if _save_embedded_image(doc, xref, out_path, min_image_size):
                        images.append(
                            ExtractedImage(page=page_num, path=out_path, kind="embedded_image")
                        )
                except Exception as e:
                    logger.warning(
                        "Failed to extract image #%d on page %d: %s", img_index, page_num, e
                    )

    if len(images) >= max_images:
        logger.warning("Reached MAX_IMAGES=%d; remaining images were skipped.", max_images)
    logger.info(
        "Found text on %d of %d pages and %d images.", len(pages), page_count, len(images)
    )
    return ParsedDocument(pages=pages, images=images, image_dir=image_dir, page_count=page_count)
