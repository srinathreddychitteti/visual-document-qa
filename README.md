# Visual Document Q&A

Ask questions about PDF documents, including their **charts, tables, diagrams and scanned pages**, and get answers that cite page numbers.

The app uses a multimodal RAG (Retrieval-Augmented Generation) pipeline. A vision model turns every visual element into a detailed text description. That description is indexed next to the document's text, so questions like *"What was the highest bar in the Q3 chart?"* can be answered.

It runs on **free-tier LLMs** by default (Google Gemini). It also supports Groq (free tier), Ollama (fully local) and OpenAI (paid).

## Features

- **Free by default:** Gemini Flash for vision and answers, plus local FastEmbed embeddings that need no API key and have no rate limits.
- **Pluggable providers:** switch between Gemini, Groq, Ollama and OpenAI with one line in `.env`.
- **Understands visuals:**
  - embedded images (photos, screenshots, raster charts);
  - **vector charts** drawn with PDF shapes, which are found and rendered automatically;
  - **scanned pages** with no text layer, which are rendered and transcribed.
- **Indexes once, asks many times:** each PDF is indexed once and cached on disk by content hash. Uploading the same file again, or asking more questions, costs no extra vision calls.
- **Each document is kept separate:** every PDF has its own vector collection, so answers never mix content from different documents.
- **Cited answers:** every answer lists the pages and excerpts it was based on.
- **Built for free-tier quotas:** tiny and duplicate images (icons, repeated logos) are skipped, the number of images per PDF is capped, calls are spaced out and rate-limit errors are retried.
- **Two interfaces:**
  1. a Streamlit chat app for people;
  2. an MCP server (HTTP or stdio) that AI agents such as Claude Desktop can use as a tool.

## Quick start

### 1. Install

```bash
git clone https://github.com/srinathreddychitteti/visual-document-qa.git
cd visual-document-qa
python -m venv .venv
```

Activate the virtual environment:

- Windows (PowerShell): `.\.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Then install the dependencies:

```bash
pip install -r requirements.txt
```

Requires Python 3.10+.

### 2. Get a free API key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey) and create an API key. No credit card is needed.
2. Copy the example config and paste your key in:

   ```bash
   cp .env.example .env          # Windows: copy .env.example .env
   ```

   ```ini
   LLM_PROVIDER=gemini
   GOOGLE_API_KEY=your-key-here
   ```

### 3. Run

**Web app:**

```bash
streamlit run app.py
```

Open http://localhost:8501, upload a PDF and start asking questions. The first time you use a PDF it is indexed, which can take a minute or two for image-heavy documents on the free tier. After that, answers are fast.

**MCP server for AI agents:**

```bash
python main.py                      # HTTP: http://127.0.0.1:8000/mcp
python main.py --transport stdio    # stdio, for clients that start the server themselves
```

## Choosing an LLM provider

| `LLM_PROVIDER` | Cost | Key variable | Default model | Notes |
|---|---|---|---|---|
| `gemini` (default) | Free tier | `GOOGLE_API_KEY` | `gemini-flash-latest` | Strong at charts and tables. Free tier is about 10 requests/min. |
| `groq` | Free tier | `GROQ_API_KEY` | `meta-llama/llama-4-scout-17b-16e-instruct` | Very fast. Key from [console.groq.com](https://console.groq.com/keys). |
| `ollama` | Free (local) | none | `llama3.2-vision` | Fully offline. Run `ollama pull llama3.2-vision` first. Needs a decent GPU or patience. |
| `openai` | Paid | `OPENAI_API_KEY` | `gpt-4o-mini` | The original backend. |

- **Changing models:** set `VISION_MODEL` and `CHAT_MODEL` to override the defaults, for example when a provider renames or retires a model.
- **Free-tier limits:** they change often, so check the provider's console for current limits.
- **Privacy:** Google may use free-tier Gemini prompts to improve its products. Use `ollama` for confidential documents.

**Embeddings** are configured separately with `EMBEDDING_PROVIDER`. The default is `fastembed`, which runs locally on the CPU. The model (about 70 MB) is downloaded on first use. The alternatives are `gemini`, `ollama` and `openai`. Changing the embedding model starts a fresh index automatically.

All settings are listed with explanations in [`.env.example`](.env.example).

## Using the MCP server

The server exposes two tools.

| Tool | Arguments | Returns |
|---|---|---|
| `query_visual_document` | `file_path`, `question` | `answer`, `sources` (page, type, excerpt), `warnings` |
| `index_visual_document` | `file_path` | page count and number of text chunks and visuals indexed |

`file_path` must be a path to a PDF on the machine running the server. Calling `index_visual_document` first is optional, since `query_visual_document` indexes on demand.

**Example Claude Desktop config** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "visual-doc-qa": {
      "command": "C:/path/to/visual-document-qa/.venv/Scripts/python.exe",
      "args": ["C:/path/to/visual-document-qa/main.py", "--transport", "stdio"],
      "cwd": "C:/path/to/visual-document-qa"
    }
  }
}
```

## How it works

```
PDF ─► parser.py ──► page text ─────────────► split into chunks ─┐
                └──► images / rendered pages ─► vision LLM ─► descriptions ─┤
                                                                  ▼
                                               embeddings ─► Chroma collection (one per PDF, cached)
                                                                  │
question ─► similarity search (top-k) ─► context with page labels ─► chat LLM ─► answer + sources
```

1. **Parse:** PyMuPDF extracts text page by page. It saves embedded images as PNG, skipping duplicates and icons. It also renders pages that are scanned or contain vector charts.
2. **Describe:** each image is sent to the vision model, which writes a detailed description: chart data points, table contents, diagram structure.
3. **Index:** text chunks and descriptions are embedded and stored in a Chroma collection named after the file's SHA-256 hash, with page numbers as metadata.
4. **Answer:** the most relevant chunks are retrieved and passed to the chat model, which is told to answer only from that context and to cite pages.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for a module-by-module guide, design decisions and how to extend the project.

## Project structure

```
visual-document-qa/
├── app.py            # Streamlit chat UI
├── main.py           # MCP server (HTTP / stdio)
├── logic.py          # Indexing, caching and question answering
├── parser.py         # PDF text and image extraction
├── providers.py      # Builds the LLM and embedding clients for the chosen provider
├── config.py         # Settings loaded from environment / .env
├── .env.example      # All configuration options, documented
├── requirements.txt
├── tests/            # Offline tests (fake models, no API key needed)
└── docs/
    └── ARCHITECTURE.md
```

## Running the tests

```bash
python -m pytest
```

The tests build small PDFs on the fly and replace the LLM and embeddings with LangChain fake models, so they run offline in a few seconds.

## Troubleshooting

| Problem | Fix |
|---|---|
| `GOOGLE_API_KEY is not set` | Create `.env` from `.env.example` and add your key. |
| `429` / `RESOURCE_EXHAUSTED` errors | You hit the free-tier rate limit. Raise `VISION_DELAY_SECONDS`, lower `MAX_IMAGES`, or wait. Failed images are listed as warnings and retried the next time you load the document. |
| `404` / model not found | The provider renamed or retired the model. Set `VISION_MODEL` and `CHAT_MODEL` to a current one. |
| A chart isn't being picked up | It may be drawn with only a few shapes. Lower `VECTOR_CHART_MIN_DRAWINGS`. |
| Too many images are being processed | Raise `MIN_IMAGE_SIZE` or lower `MAX_IMAGES`. |
| You want to rebuild everything | Delete the `chroma_data/` folder. |

## Technology stack

- **Orchestration:** LangChain
- **LLMs:** Google Gemini, Groq, Ollama or OpenAI
- **Embeddings:** FastEmbed (`BAAI/bge-small-en-v1.5`) or a provider's embeddings
- **PDF parsing:** PyMuPDF
- **Vector store:** ChromaDB
- **UI:** Streamlit
- **Agent interface:** FastMCP
