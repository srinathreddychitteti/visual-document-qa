# Visual Document Q&A

This repository contains a Q&A application that analyzes text, charts, and tables in PDF documents using a multimodal AI vision model.

It uses a multimodal RAG (Retrieval-Augmented Generation) pipeline to understand both the textual and visual elements of a document, allowing you to ask complex questions about its content.

## Features

* **Multimodal RAG:** Indexes not just the text, but also AI-generated visual summaries of charts, tables, and diagrams.
* **Vision-Powered Analysis:** Uses `gpt-4o` to interpret and describe images extracted from the PDF.
* **Vector Search:** Employs ChromaDB and OpenAI embeddings to find the most relevant text chunks and image summaries to answer a question.
* **Dual Interfaces:**
    1.  A user-friendly web app built with Streamlit.
    2.  A programmatic API server (FastAPI/FastMCP) that an AI agent can use as a tool.

## How it Works

The application processes a PDF in three phases:

1.  **Phase 1: Parse**
    * The `parser.py` script opens the PDF using `PyMuPDF` (`fitz`).
    * It extracts all plain text blocks page by page.
    * It extracts all images and saves them to a temporary folder (`temp_images`).

2.  **Phase 2: Index**
    * **Text:** Text blocks are split and added to a Chroma vector store.
    * **Images:** Each extracted image is sent to the `gpt-4o` vision model, which generates a detailed text description (e.g., "A bar chart showing... with data points...").
    * These AI-generated image summaries are then *also* added to the vector store.

3.  **Phase 3: Query**
    * When you ask a question, the app searches the vector store for the most relevant context, which could be a mix of text snippets and image descriptions.
    * This retrieved context and the original question are passed to `gpt-4o` to generate a final, grounded answer.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
    cd YOUR_REPO_NAME
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```

3.  **Activate the environment:**
    * On Windows (PowerShell): `.\venv\Scripts\Activate.ps1`
    * On macOS/Linux: `source venv/bin/activate`

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Create your environment file:**
    Create a file named `.env` in the root of the project and add your OpenAI API key:
    ```
    OPENAI_API_KEY=sk-YourSecretKeyHere
    ```
    *Note: This project requires a paid OpenAI plan with quota for `gpt-4o`.*

## How to Run

This project includes two different entry points:

### Option 1: Run the Streamlit Web App (For Human Use)

This is the visual interface for you to use.

1.  Run the following command in your terminal:
    ```bash
    streamlit run app.py
    ```
2.  Open the local URL (e.g., `http://localhost:8501`) in your browser.
3.  Upload a PDF, ask a question, and get an answer.

### Option 2: Run the FastAPI/MCP Server (For AI/Tool Use)

This server exposes the Q&A logic as a tool that other programs or AI agents can call.

1.  Run the following command in your terminal:
    ```bash
    python main.py
    ```
2.  The server will start on `http://127.0.0.1:8000`.
3.  The MCP tool, named `query_visual_document`, is now available for an AI agent to use.

## Technology Stack

* **Orchestration:** LangChain
* **LLM & Vision:** OpenAI (`gpt-4o`, `text-embedding-3-small`)
* **PDF Parsing:** PyMuPDF (`fitz`)
* **Vector Store:** ChromaDB
* **Web App:** Streamlit
* **API Server:** FastAPI, FastMCP
* **Image Handling:** Pillow
