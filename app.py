"""Streamlit web app: upload a PDF, then chat with it.

Run with:  streamlit run app.py
"""

import hashlib
import logging
import os
import tempfile

import streamlit as st

from config import settings
from logic import answer_question, index_pdf

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title="Visual Document Q&A", page_icon="📄", layout="centered")

with st.sidebar:
    st.header("Settings")
    st.markdown(
        f"**LLM provider:** `{settings.llm_provider}`  \n"
        f"**Vision model:** `{settings.vision_model}`  \n"
        f"**Answer model:** `{settings.chat_model}`  \n"
        f"**Embeddings:** `{settings.embedding_provider}` / `{settings.embedding_model}`"
    )
    st.caption("Change these in your `.env` file and restart the app.")
    if st.button("Clear chat"):
        st.session_state.messages = []

st.title("📄 Visual Document Q&A")
st.markdown("Upload a PDF and ask questions about its text, charts, tables and images.")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is None:
    st.stop()

pdf_bytes = uploaded_file.getvalue()
upload_id = hashlib.sha256(pdf_bytes).hexdigest()

# Index each uploaded file once per session; logic.index_pdf also caches on disk.
if st.session_state.get("upload_id") != upload_id:
    with st.status("Processing document...", expanded=True) as status:
        progress_bar = st.progress(0.0)

        def on_progress(message: str, fraction: float) -> None:
            progress_bar.progress(min(max(fraction, 0.0), 1.0), text=message)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        try:
            document = index_pdf(tmp_path, progress=on_progress)
        except Exception as e:
            status.update(label="Processing failed", state="error")
            st.error(f"Could not process the document: {e}")
            st.stop()
        finally:
            os.remove(tmp_path)

        for warning in document.warnings:
            st.warning(warning)
        label = "Loaded from cache" if document.from_cache else "Document indexed"
        status.update(label=label, state="complete", expanded=False)

    st.session_state.upload_id = upload_id
    st.session_state.document = document
    st.session_state.messages = []

document = st.session_state.document
st.caption(
    f"{document.page_count} pages · {document.text_chunks} text chunks · "
    f"{document.image_summaries} visual elements described"
)


def render_sources(sources) -> None:
    pages = sorted({s.page for s in sources if s.page is not None})
    with st.expander(f"Sources (pages {', '.join(map(str, pages)) or 'n/a'})"):
        for s in sources:
            kind = "Text" if s.source_type == "text" else "Visual"
            st.markdown(f"**Page {s.page} · {kind}**")
            st.text(s.excerpt + ("…" if len(s.excerpt) >= 300 else ""))


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])

if question := st.chat_input("Ask a question about the document"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = answer_question(document, question)
                answer, sources = result.answer, result.sources
            except Exception as e:
                answer, sources = f"Error while answering: {e}", []
        st.markdown(answer)
        if sources:
            render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
