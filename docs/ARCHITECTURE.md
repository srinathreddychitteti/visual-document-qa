# Architecture and Developer Guide

This guide explains how the code is organised, why it is built this way, and how to extend it. For installation and usage, see the [README](../README.md).

## Module overview

```
                ┌──────────┐      ┌──────────┐
   people ────► │  app.py  │      │ main.py  │ ◄──── AI agents (MCP)
                └────┬─────┘      └────┬─────┘
                     └────────┬────────┘
                              ▼
                        ┌──────────┐
                        │ logic.py │  index_pdf() / answer_question()
                        └──┬────┬──┘
                ┌──────────┘    └───────────┐
                ▼                           ▼
          ┌───────────┐              ┌──────────────┐
          │ parser.py │              │ providers.py │ ──► Gemini / Groq / Ollama / OpenAI / FastEmbed
          └───────────┘              └──────┬───────┘
                                            ▼
                                      ┌───────────┐
                                      │ config.py │ ◄── .env
                                      └───────────┘
```

| Module | What it does | Depends on |
|---|---|---|
| `config.py` | Reads all settings from environment variables into a frozen `Settings` dataclass (`config.settings`). Holds the default model for each provider. | `python-dotenv` |
| `providers.py` | `get_vision_model()`, `get_chat_model()` and `get_embeddings()` build the right LangChain client for the configured provider. Each result is cached with `lru_cache`, and provider packages are imported lazily. | `config` |
| `parser.py` | `parse(path, ...)` returns a `ParsedDocument`: per-page text, plus PNG files for embedded images, vector-chart pages and scanned pages. Has no LLM dependencies. | PyMuPDF |
| `logic.py` | The RAG pipeline. `index_pdf()` parses, describes images, embeds and caches. `answer_question()` retrieves context and generates a cited answer. | `parser`, `providers`, `config` |
| `app.py` | Streamlit UI: upload, progress display, chat history, sources. | `logic` |
| `main.py` | FastMCP server with the tools `query_visual_document` and `index_visual_document`. Supports HTTP and stdio transports. | `logic` |

## Data model

Defined as dataclasses so they serialise easily (the MCP server returns them via `dataclasses.asdict`).

| Class | Module | Fields |
|---|---|---|
| `PageText` | parser | `page` (1-based), `text` |
| `ExtractedImage` | parser | `page`, `path` (PNG), `kind`: `embedded_image` \| `vector_graphics` \| `scanned_page` |
| `ParsedDocument` | parser | `pages`, `images`, `image_dir`, `page_count` |
| `IndexedDocument` | logic | `collection_name`, `file_hash`, `page_count`, `text_chunks`, `image_summaries`, `from_cache`, `warnings` |
| `Source` | logic | `page`, `source_type`, `excerpt` (first 300 characters) |
| `AnswerResult` | logic | `answer`, `sources` |

Every chunk stored in Chroma has the metadata `{"page": int, "source_type": str}`. The `source_type` is `text` for page text, and otherwise one of the `ExtractedImage.kind` values.

## Indexing flow (`logic.index_pdf`)

1. **Hash the file:** SHA-256 of the file's bytes.
2. **Name the collection:** `doc_<first 24 hex chars of hash>_<8-char hash of embedding provider:model>`. Including the embedding model matters because vectors from different models have different sizes and cannot share a collection.
3. **Check the cache:** if `chroma_data/manifests/<collection>.json` exists, the index is complete. The function returns an `IndexedDocument` with `from_cache=True` without parsing anything.
4. **Parse** with `parser.parse()`, applying the settings `MIN_IMAGE_SIZE`, `MAX_IMAGES`, `RENDER_DPI` and `VECTOR_CHART_MIN_DRAWINGS`.
5. **Split the text** per page with `RecursiveCharacterTextSplitter`, so each chunk keeps its page number.
6. **Describe the images** (`summarize_images`):
   - Images are processed one at a time, with a pause of `VISION_DELAY_SECONDS` between calls to stay under free-tier per-minute limits.
   - Each client also retries transient and `429` errors up to `MAX_RETRIES` times.
   - If an image still fails, the failure is recorded in `warnings` and the image is **not** indexed. The old code indexed the error message as if it were content.
7. **Clean up** the temporary image folder. This is done in a `finally` block, so it happens even when an error occurs.
8. **Store:** `reset_collection()` discards any partial index left by an earlier crash, then all chunks are added.
9. **Write the manifest**, but only if there were no warnings. An incomplete index is used for the current session, but it is rebuilt the next time the document is loaded, which retries the failed images.

## Answering flow (`logic.answer_question`)

1. `similarity_search(question, k=RETRIEVAL_K)` on the document's own collection.
2. Each retrieved chunk is labelled `[Page N - text]` or `[Page N - visual description]` (`_format_context`).
3. `ANSWER_PROMPT` tells the model to answer only from the context, cite pages, and say when the answer isn't there.
4. The chat model is invoked and the reply is read with `.text`. This flattens the multi-part content that newer Gemini models return.
5. Returns an `AnswerResult` with the answer and the retrieved `Source`s.

## How the parser picks visual content

The parser applies these rules to each page, in order:

| Condition | Action | `kind` |
|---|---|---|
| The page has no extractable text | Render the whole page at `RENDER_DPI` | `scanned_page` |
| The page has ≥ `VECTOR_CHART_MIN_DRAWINGS` vector paths | Render the whole page | `vector_graphics` |
| Otherwise, for each embedded image | Save it as PNG unless it is smaller than `MIN_IMAGE_SIZE` or was already seen (same `xref`) | `embedded_image` |

- **Rendered pages:** when a page is rendered, its embedded images are not also extracted, because the render already contains them.
- **Image format:** all images are converted to RGB PNG (CMYK is converted). This means the `image/png` data URI sent to the model is always correct.
- **Image cap:** the total number of images is capped by `MAX_IMAGES` to protect API quotas.

## Design decisions

- **Gemini plus local embeddings by default.** Gemini's free tier handles both vision and text well. Running embeddings locally with FastEmbed (`BAAI/bge-small-en-v1.5`, 384 dimensions, CPU-only, no PyTorch) keeps the free API quota for the work that needs a large model.
- **Model aliases instead of pinned versions.** `gemini-flash-latest` follows Google's current Flash model, so the app keeps working when an older model is retired. Pin a specific model with `VISION_MODEL` and `CHAT_MODEL` if you need reproducible results.
- **Caching by content hash, not file name.** Streamlit uploads go to random temporary paths, and the same file can be uploaded under different names. Hashing the content handles both cases.
- **One collection per document.** The original code put every PDF into a single shared collection, so answers could draw on unrelated documents.
- **Logging to stderr.** With the MCP stdio transport, stdout carries the protocol. `print()` there would corrupt it.
- **Lazy provider imports.** You only need the package for the provider you use, and import errors only appear when that provider is selected.

## Extending the project

### Add an LLM provider

1. Add an entry to `DEFAULT_MODELS` (and `DEFAULT_VISION_DELAY`) in `config.py`.
2. Add a branch in `providers._build_chat_model` that returns a LangChain `BaseChatModel`. If the provider needs a key, add it to `API_KEY_ENV`.
3. The image message uses LangChain's standard `{"type": "image_url", "image_url": {"url": "data:..."}}` block, which most chat integrations accept.
4. Add the package to `requirements.txt` and a row to the provider table in the README.

### Add an embedding provider

Add an entry to `DEFAULT_EMBEDDING_MODELS` and a branch in `providers.get_embeddings`. Existing indexes are unaffected, because the embedding model is part of the collection name.

### Change the prompts

`IMAGE_PROMPT` (image descriptions) and `ANSWER_PROMPT` (final answer) are at the top of `logic.py`.

- **`IMAGE_PROMPT`:** changing it only affects newly indexed documents. To re-describe documents that are already cached, delete `chroma_data/`.
- **`ANSWER_PROMPT`:** changes apply to the next question.

### Support other file types

`index_pdf` only depends on `parse()` returning a `ParsedDocument`. A parser for another format, such as DOCX, PPTX or images, can return the same structure and reuse the rest of the pipeline.

## Testing

`tests/test_pipeline.py` builds small PDFs with PyMuPDF and swaps in `FakeListChatModel` and `DeterministicFakeEmbedding` with `monkeypatch`. It does this by patching the names `logic` imports (`logic.get_vision_model`, `logic.settings`, and so on).

| Area | What is covered |
|---|---|
| Parser | Text extraction; image, vector-chart and scanned-page detection; icon filtering; the `MAX_IMAGES` cap |
| Pipeline | Indexing counts; answers with sources; the cache skipping vision calls; separate documents staying separate; failed images not being indexed or cached |

```bash
python -m pytest -v
```

## Configuration reference

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini`, `groq`, `ollama` or `openai` |
| `GOOGLE_API_KEY` / `GROQ_API_KEY` / `OPENAI_API_KEY` | none | Key for the selected provider |
| `VISION_MODEL` | per provider | Model that describes images |
| `CHAT_MODEL` | per provider | Model that writes answers |
| `EMBEDDING_PROVIDER` | `fastembed` | `fastembed`, `gemini`, `ollama` or `openai` |
| `EMBEDDING_MODEL` | per provider | Embedding model name |
| `RETRIEVAL_K` | `6` | Chunks retrieved per question |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Text splitting, in characters |
| `MIN_IMAGE_SIZE` | `100` | Ignore embedded images smaller than this many pixels in either dimension |
| `MAX_IMAGES` | `30` | Maximum number of images described per PDF |
| `RENDER_DPI` | `150` | Resolution of rendered pages |
| `VECTOR_CHART_MIN_DRAWINGS` | `40` | Vector paths needed to render a page (`0` turns this off) |
| `VISION_DELAY_SECONDS` | gemini `6`, groq `2`, others `0` | Pause between vision calls |
| `MAX_RETRIES` | `6` | Retries on rate-limit and transient errors |
| `VISION_MAX_TOKENS` / `ANSWER_MAX_TOKENS` | `1500` / `1500` | Maximum length of image descriptions and answers. Lower them (about 400) for local CPU models. |
| `OLLAMA_KEEP_ALIVE` | `30m` | How long Ollama keeps the model loaded between requests |
| `CHROMA_DIR` | `./chroma_data` | Vector store and manifests |
| `IMAGE_DIR` | `./temp_images` | Scratch space for extracted images |
