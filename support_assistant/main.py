"""
Task 5 — FastAPI application wrapping the LangGraph pipeline.

Run locally with:
    uvicorn main:app --host 0.0.0.0 --port 7860

Example calls (see README.md for recorded raw JSON responses, run with
MOCK_LLM left at its default):

    curl -X POST http://localhost:7860/ask \\
         -H "Content-Type: application/json" \\
         -d '{"query": "Is delivery free on my order?"}'

    curl -X POST http://localhost:7860/ask \\
         -H "Content-Type: application/json" \\
         -d '{"query": "What is the capital of France?"}'
"""

from fastapi import FastAPI

from schemas import AskRequest, AskResponse
from graph import run_query

app = FastAPI(
    title="Zepto Support Assistant",
    description="RAG-based support assistant for Zepto policy questions (Module 3).",
    version="1.0.0",
)


@app.get("/")
def health():
    return {"status": "ok", "service": "zepto-support-assistant"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    return run_query(request.query)
