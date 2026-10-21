"""MCP server that exposes the document Q&A pipeline as tools for AI agents.

Run over HTTP (default, endpoint http://127.0.0.1:8000/mcp):
    python main.py

Run over stdio (for clients such as Claude Desktop that launch the server):
    python main.py --transport stdio
"""

import argparse
import asyncio
import logging
import os
import sys
from dataclasses import asdict

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from config import settings
from logic import answer_question, index_pdf

# Log to stderr: with the stdio transport, stdout carries the MCP protocol.
logging.basicConfig(
    level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("visual-doc-qa")

mcp = FastMCP(
    name="visualDocQAServer",
    instructions=(
        "Answers questions about local PDF files, including their charts, tables "
        "and images. Pass an absolute path to the PDF."
    ),
)


def _validate_pdf_path(file_path: str) -> str:
    path = os.path.abspath(os.path.expanduser(file_path))
    if not os.path.isfile(path):
        raise ValueError(f"File not found: {path}")
    if not path.lower().endswith(".pdf"):
        raise ValueError(f"Not a PDF file: {path}")
    return path


@mcp.tool(
    name="query_visual_document",
    description=(
        "Answer a question about a PDF document. Reads text and understands charts, "
        "tables and images. The first call for a document indexes it (can take a "
        "minute); later calls are fast. Returns the answer and the pages it used."
    ),
)
async def query_visual_document(file_path: str, question: str) -> dict:
    path = _validate_pdf_path(file_path)
    logger.info("query_visual_document: %s | %s", path, question)
    document = await asyncio.to_thread(index_pdf, path)
    result = await asyncio.to_thread(answer_question, document, question)
    return {
        "answer": result.answer,
        "sources": [asdict(s) for s in result.sources],
        "warnings": document.warnings,
    }


@mcp.tool(
    name="index_visual_document",
    description=(
        "Pre-index a PDF so later questions about it are answered quickly. "
        "Returns page count and how many text chunks and visuals were indexed."
    ),
)
async def index_visual_document(file_path: str) -> dict:
    path = _validate_pdf_path(file_path)
    logger.info("index_visual_document: %s", path)
    document = await asyncio.to_thread(index_pdf, path)
    return asdict(document)


@mcp.custom_route("/", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "message": "Visual Document Q&A MCP server. MCP endpoint: /mcp",
            "llm_provider": settings.llm_provider,
        }
    )


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    arg_parser.add_argument("--transport", choices=["http", "stdio"], default="http")
    arg_parser.add_argument("--host", default="127.0.0.1")
    arg_parser.add_argument("--port", type=int, default=8000)
    args = arg_parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio", show_banner=False)
    else:
        mcp.run(transport="http", host=args.host, port=args.port)
