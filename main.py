from dotenv import load_dotenv
load_dotenv() 

import asyncio
import sys
import uvicorn
from pydantic import BaseModel
from fastmcp import FastMCP
from logic import get_answer_from_pdf
from starlette.requests import Request
from starlette.responses import JSONResponse
from fastapi import FastAPI


mcp_server = FastMCP(name="visualDocQAServer")
mcp_server.title = "Visual Document Q&A MCP Server"
mcp_server.description = "An MCP server that allows an AI to ask questions about PDFs."


class VisualDocQAInputs(BaseModel):
    file_path: str
    question: str


@mcp_server.tool(
    name="query_visual_document",
    description="Answers a user's question about a specific PDF document. "
                "It can read text, analyze charts, and understand tables.",
)
async def visual_doc_qa_tool(inputs: VisualDocQAInputs) -> str:
    print(f"MCP tool 'query_visual_document' called with:")
    print(f"  File: {inputs.file_path}")
    print(f"  Question: {inputs.question}")

    try:
        answer = await asyncio.to_thread(
            get_answer_from_pdf,
            inputs.file_path,
            inputs.question
        )
        print(f"Logic returned answer: {answer}")
        return answer
    except Exception as e:
        print(f"Error during PDF processing: {e}", file=sys.stderr)
        return f"An error occurred while processing the document: {e}"


@mcp_server.custom_route("/", methods=["GET"])
async def read_root(request: Request) -> JSONResponse:
    return JSONResponse(
        {"message": "Visual Document Q&A MCP Server is running. Visit /mcp for protocol info."}
    )
inner_app = None
if hasattr(mcp_server, "fastapi_app"):
    inner_app = mcp_server.fastapi_app
elif hasattr(mcp_server, "app"):
    inner_app = mcp_server.app
app = FastAPI(title="VisualDocQAServer Wrapper")

@app.get("/")
async def root():
    return JSONResponse({"message": "VisualDocQAServer wrapper is running."})

if inner_app is not None:
    app.mount("/mcp", inner_app)
else:
    @app.get("/mcp")
    async def mcp_placeholder():
        return JSONResponse({
            "message": "FastMCP instance is active but not ASGI-callable. "
                       "Use MCP client to connect directly."
        })

if __name__ == "__main__":
    print("Starting server... Access at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
