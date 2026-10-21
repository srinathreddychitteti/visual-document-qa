"""Multimodal RAG pipeline: index a PDF once, then answer questions about it.

Typical use::

    doc = index_pdf("report.pdf")          # slow the first time, cached after
    result = answer_question(doc, "What was Q3 revenue?")
    print(result.answer, result.sources)

Each PDF gets its own Chroma collection, named after a hash of the file's
contents and the embedding model. Re-uploading the same file (even under a
different name) reuses the existing index, so images are only sent to the
vision model once. A small manifest file marks an index as complete; an index
interrupted half-way (e.g. by a rate-limit error) is rebuilt on the next run.
"""

import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from parser import ExtractedImage, encode_image, parse
from providers import get_chat_model, get_embeddings, get_vision_model

logger = logging.getLogger(__name__)

# Called with (message, fraction_complete between 0 and 1).
ProgressCallback = Callable[[str, float], None]

IMAGE_PROMPT = """You are an expert at analyzing images extracted from documents.
Describe this image in detail so that questions about it can be answered from your description alone.
- Chart or graph: give the title, axis labels and units, legend, every data point you can read, and the key trends.
- Table: reproduce all headers, rows and columns as text, preserving the structure.
- Diagram: explain what it shows and how the parts relate.
- Page of text (scan): transcribe the text faithfully.
Respond in plain text."""

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the context below. The context contains text
excerpts and descriptions of charts, tables and images from a PDF. Each excerpt
starts with its page number.

- Cite the page(s) you used, e.g. "(page 4)".
- If the context does not contain the answer, say that you could not find it in the document.

Context:
{context}

Question: {question}"""
)


@dataclass
class IndexedDocument:
    """Handle to an indexed PDF, returned by :func:`index_pdf`."""

    collection_name: str
    file_hash: str
    page_count: int
    text_chunks: int
    image_summaries: int
    from_cache: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class Source:
    page: Optional[int]
    source_type: str  # "text" | "embedded_image" | "vector_graphics" | "scanned_page"
    excerpt: str


@dataclass
class AnswerResult:
    answer: str
    sources: list[Source]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _collection_name(file_hash: str) -> str:
    # Vectors from different embedding models are incompatible, so the model
    # is part of the name.
    embed_id = f"{settings.embedding_provider}:{settings.embedding_model}"
    embed_hash = hashlib.sha256(embed_id.encode()).hexdigest()[:8]
    return f"doc_{file_hash[:24]}_{embed_hash}"


def _manifest_path(collection_name: str) -> str:
    return os.path.join(settings.chroma_dir, "manifests", f"{collection_name}.json")


def _open_store(collection_name: str) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_dir,
    )


def _summarize_image(image: ExtractedImage) -> str:
    data_uri = f"data:image/png;base64,{encode_image(image.path)}"
    message = HumanMessage(
        content=[
            {"type": "text", "text": IMAGE_PROMPT},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ]
    )
    return get_vision_model().invoke([message]).text.strip()


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #


def summarize_images(
    images: list[ExtractedImage], progress: Optional[ProgressCallback] = None
) -> tuple[list[Document], list[str]]:
    """Describe each image with the vision model.

    Returns (documents, warnings). Failed images are reported as warnings and
    are *not* indexed, so error messages never end up in search results.
    """
    documents, warnings = [], []
    for i, image in enumerate(images):
        if progress:
            message = f"Describing image {i + 1} of {len(images)} (page {image.page})"
            progress(message, i / len(images))
        if i > 0 and settings.vision_delay_seconds > 0:
            time.sleep(settings.vision_delay_seconds)
        try:
            summary = _summarize_image(image)
        except Exception as e:
            logger.warning("Vision model failed on %s: %s", image.path, e)
            warnings.append(f"Could not describe an image on page {image.page}: {e}")
            continue
        if not summary:
            continue
        documents.append(
            Document(
                page_content=f"[Visual content on page {image.page}]\n{summary}",
                metadata={"page": image.page, "source_type": image.kind},
            )
        )
    return documents, warnings


def index_pdf(pdf_path: str, progress: Optional[ProgressCallback] = None) -> IndexedDocument:
    """Parse, describe and embed a PDF, or reuse its existing index.

    Args:
        pdf_path: Path to the PDF file.
        progress: Optional callback receiving (message, fraction 0..1).
    """

    def report(message: str, fraction: float) -> None:
        logger.info(message)
        if progress:
            progress(message, fraction)

    file_hash = file_sha256(pdf_path)
    name = _collection_name(file_hash)
    manifest = _manifest_path(name)

    if os.path.exists(manifest):
        with open(manifest, encoding="utf-8") as f:
            cached = IndexedDocument(**json.load(f))
        cached.from_cache = True
        cached.warnings = []
        report("Using cached index for this document.", 1.0)
        return cached

    report("Reading PDF...", 0.0)
    parsed = parse(
        pdf_path,
        image_root=settings.image_dir,
        min_image_size=settings.min_image_size,
        max_images=settings.max_images,
        render_dpi=settings.render_dpi,
        vector_chart_min_drawings=settings.vector_chart_min_drawings,
    )

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        )
        text_docs = splitter.create_documents(
            [p.text for p in parsed.pages],
            metadatas=[{"page": p.page, "source_type": "text"} for p in parsed.pages],
        )

        def image_progress(message: str, fraction: float) -> None:
            report(message, 0.1 + 0.8 * fraction)

        image_docs, warnings = summarize_images(parsed.images, image_progress)
    finally:
        shutil.rmtree(parsed.image_dir, ignore_errors=True)

    all_docs = text_docs + image_docs
    if not all_docs:
        raise ValueError("No text or images could be extracted from this PDF.")

    report("Building search index...", 0.9)
    store = _open_store(name)
    store.reset_collection()  # discard any partial index from an earlier failed run
    store.add_documents(all_docs)

    result = IndexedDocument(
        collection_name=name,
        file_hash=file_hash,
        page_count=parsed.page_count,
        text_chunks=len(text_docs),
        image_summaries=len(image_docs),
        warnings=warnings,
    )
    # Only write the manifest if every image was described; otherwise retry
    # the failed images next time instead of caching an incomplete index.
    if not warnings:
        os.makedirs(os.path.dirname(manifest), exist_ok=True)
        with open(manifest, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2)

    report("Document ready.", 1.0)
    return result


# --------------------------------------------------------------------------- #
# Question answering
# --------------------------------------------------------------------------- #


def _format_context(docs: list[Document]) -> str:
    parts = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        kind = doc.metadata.get("source_type", "text")
        label = "text" if kind == "text" else "visual description"
        parts.append(f"[Page {page} - {label}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def answer_question(document: IndexedDocument, question: str) -> AnswerResult:
    """Answer ``question`` using the indexed ``document``."""
    store = _open_store(document.collection_name)
    docs = store.similarity_search(question, k=settings.retrieval_k)
    if not docs:
        return AnswerResult(answer="The document index is empty.", sources=[])

    # ``.text`` flattens multi-part responses (returned by newer Gemini models).
    chain = ANSWER_PROMPT | get_chat_model()
    answer = chain.invoke({"context": _format_context(docs), "question": question}).text

    sources = [
        Source(
            page=d.metadata.get("page"),
            source_type=d.metadata.get("source_type", "text"),
            excerpt=d.page_content[:300],
        )
        for d in docs
    ]
    return AnswerResult(answer=answer.strip(), sources=sources)


def get_answer_from_pdf(pdf_path: str, question: str) -> str:
    """One-shot helper kept for backwards compatibility: index (or reuse) and answer."""
    return answer_question(index_pdf(pdf_path), question).answer
