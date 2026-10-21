"""Offline tests for the parser and RAG pipeline.

LLMs and embeddings are replaced by LangChain's fake models, so no API key or
network access is needed:  python -m pytest
"""

import dataclasses
import os

import fitz
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel

import logic
from parser import parse


def make_pdf(path: str) -> None:
    """Page 1: text + raster image. Page 2: text + vector chart. Page 3: blank (scan)."""
    doc = fitz.open()

    page = doc.new_page()
    page.insert_text((72, 72), "Quarterly revenue grew 12 percent in Q3.")
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 200, 150), False)
    pix.set_rect(pix.irect, (200, 30, 30))
    page.insert_image(fitz.Rect(72, 100, 272, 250), pixmap=pix)
    # A tiny icon that should be ignored.
    icon = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 16, 16), False)
    page.insert_image(fitz.Rect(300, 100, 316, 116), pixmap=icon)

    page = doc.new_page()
    page.insert_text((72, 72), "Figure 2 shows the cost breakdown.")
    for i in range(50):
        page.draw_line((72, 100 + i * 5), (300, 100 + i * 5))

    doc.new_page()
    doc.save(path)
    doc.close()


@pytest.fixture
def pdf_path(tmp_path):
    path = str(tmp_path / "sample.pdf")
    make_pdf(path)
    return path


@pytest.fixture
def fake_models(tmp_path, monkeypatch):
    test_settings = dataclasses.replace(
        logic.settings,
        chroma_dir=str(tmp_path / "chroma"),
        image_dir=str(tmp_path / "images"),
        vision_delay_seconds=0,
        vector_chart_min_drawings=40,
    )
    monkeypatch.setattr(logic, "settings", test_settings)

    vision = FakeListChatModel(responses=["A red rectangle chart."])
    chat = FakeListChatModel(responses=["Revenue grew 12 percent (page 1)."])
    embeddings = DeterministicFakeEmbedding(size=64)
    calls = {"vision": 0}

    def get_vision():
        calls["vision"] += 1
        return vision

    monkeypatch.setattr(logic, "get_vision_model", get_vision)
    monkeypatch.setattr(logic, "get_chat_model", lambda: chat)
    monkeypatch.setattr(logic, "get_embeddings", lambda: embeddings)
    return calls


def test_parse_extracts_text_images_and_renders(pdf_path, tmp_path):
    parsed = parse(pdf_path, image_root=str(tmp_path / "img"), min_image_size=50)

    assert parsed.page_count == 3
    assert [p.page for p in parsed.pages] == [1, 2]
    kinds = {(img.page, img.kind) for img in parsed.images}
    # The 16x16 icon is filtered out; the raster image, vector page and blank page remain.
    assert kinds == {(1, "embedded_image"), (2, "vector_graphics"), (3, "scanned_page")}
    assert all(img.path.endswith(".png") and os.path.exists(img.path) for img in parsed.images)


def test_parse_respects_max_images(pdf_path, tmp_path):
    parsed = parse(pdf_path, image_root=str(tmp_path / "img"), min_image_size=50, max_images=1)
    assert len(parsed.images) == 1


def test_index_and_answer(pdf_path, fake_models):
    document = logic.index_pdf(pdf_path)

    assert not document.from_cache
    assert document.page_count == 3
    assert document.text_chunks == 2
    assert document.image_summaries == 3
    assert fake_models["vision"] == 3

    result = logic.answer_question(document, "How much did revenue grow?")
    assert "12 percent" in result.answer
    assert result.sources
    assert {s.source_type for s in result.sources} <= {
        "text", "embedded_image", "vector_graphics", "scanned_page"
    }


def test_second_index_uses_cache(pdf_path, fake_models):
    first = logic.index_pdf(pdf_path)
    calls_after_first = fake_models["vision"]
    second = logic.index_pdf(pdf_path)

    assert second.from_cache
    assert second.collection_name == first.collection_name
    assert fake_models["vision"] == calls_after_first  # no new vision calls


def test_documents_are_isolated(pdf_path, tmp_path, fake_models):
    other_path = str(tmp_path / "other.pdf")
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "A completely different document.")
    doc.save(other_path)
    doc.close()

    first = logic.index_pdf(pdf_path)
    second = logic.index_pdf(other_path)
    assert first.collection_name != second.collection_name

    store = logic._open_store(second.collection_name)
    contents = [d.page_content for d in store.similarity_search("revenue", k=10)]
    assert all("Quarterly revenue" not in c for c in contents)


def test_failed_images_are_not_indexed_or_cached(pdf_path, fake_models, monkeypatch):
    def broken_summary(image):
        raise RuntimeError("429 rate limited")

    monkeypatch.setattr(logic, "_summarize_image", broken_summary)
    document = logic.index_pdf(pdf_path)

    assert document.image_summaries == 0
    assert len(document.warnings) == 3
    assert not os.path.exists(logic._manifest_path(document.collection_name))
