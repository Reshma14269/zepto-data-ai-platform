"""
Task 3 — LangGraph StateGraph with a TypedDict state and 3 nodes:
    classify_intent     -> policy_question | general_question
    retrieve_and_answer -> for policy_question
    direct_answer       -> for general_question

Every node's generation step branches on the MOCK_LLM env var:
    MOCK_LLM unset or "1"  -> mock branch (required, graded baseline,
                              fully offline, deterministic).
    MOCK_LLM == "0"        -> optional real-LLM branch (ungraded extension).

Retrieval itself (embedding the query + ChromaDB cosine-similarity search)
always runs for real in retrieve_and_answer, in BOTH modes, since it needs
no API key and no network call.

Task 4 — the final answer is validated against the AskResponse Pydantic
schema (answer/sources/confidence). In mock mode it's populated
deterministically from code. In the optional MOCK_LLM=0 path, if the LLM's
raw output fails validation, we retry up to 2 additional times with a
corrective instruction before giving up and returning a clearly marked
error response.
"""

import os
import json
from typing import List, TypedDict

from langgraph.graph import StateGraph, END
from pydantic import ValidationError

from ingestion import retrieve_top_k
from prompts import build_rag_prompt, build_direct_prompt
from schemas import AskResponse

POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]

TOP_K = 3
SNIPPET_LEN = 200


def mock_llm_enabled() -> bool:
    """True = required mock baseline (default). False = optional real-LLM path."""
    return os.environ.get("MOCK_LLM", "1") != "0"


class GraphState(TypedDict):
    query: str
    intent: str
    retrieved_chunks: List[dict]
    answer: str
    sources: List[str]
    confidence: float


# ---------------------------------------------------------------------------
# Node 1: classify_intent
# ---------------------------------------------------------------------------
def classify_intent(state: GraphState) -> GraphState:
    query = state["query"]

    if mock_llm_enabled():
        # Mock mode (graded baseline): keyword heuristic, no LLM call.
        lowered = query.lower()
        intent = "policy_question" if any(kw in lowered for kw in POLICY_KEYWORDS) else "general_question"
    else:
        # Optional MOCK_LLM=0 extension: ask the real LLM to classify.
        from llm_client import call_llm

        classify_prompt = (
            "Classify the following customer question as exactly one word: "
            "'policy_question' if it relates to Zepto's delivery, returns, "
            "membership, tracking, cancellation, gift cards, or support hours "
            "policies, otherwise 'general_question'.\n\n"
            f"Question: {query}\n\nAnswer with exactly one word."
        )
        raw = call_llm(classify_prompt).strip().lower()
        intent = "policy_question" if "policy_question" in raw else "general_question"

    return {**state, "intent": intent}


def _route_after_classify(state: GraphState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


# ---------------------------------------------------------------------------
# Node 2: retrieve_and_answer (policy_question path)
# ---------------------------------------------------------------------------
def _validate_or_retry(raw_json_text: str, retrieved_chunks: List[dict], max_retries: int = 2) -> AskResponse:
    """Try to parse+validate LLM JSON output; retry with corrective prompt on failure."""
    from llm_client import call_llm

    attempt_text = raw_json_text
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            parsed = json.loads(attempt_text)
            return AskResponse(**parsed)
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            if attempt < max_retries:
                corrective_prompt = (
                    "Your previous response was not valid JSON matching this "
                    'schema: {"answer": string, "sources": array of strings, '
                    '"confidence": float 0-1}. Return ONLY the corrected JSON '
                    f"object, nothing else.\n\nPrevious response:\n{attempt_text}\n\n"
                    f"Error: {e}"
                )
                attempt_text = call_llm(corrective_prompt)

    # Exhausted retries — clearly marked error response.
    return AskResponse(
        answer="Error: the model did not return a valid structured response after retries.",
        sources=[c["id"] for c in retrieved_chunks],
        confidence=0.0,
    )


def retrieve_and_answer(state: GraphState) -> GraphState:
    query = state["query"]

    # Retrieval always runs for real, in both modes.
    retrieved_chunks = retrieve_top_k(query, k=TOP_K)

    if mock_llm_enabled():
        # Mock mode (graded baseline): canned templated answer, no LLM call.
        top_chunk_snippet = retrieved_chunks[0]["document"][:SNIPPET_LEN] if retrieved_chunks else ""
        answer_text = f"Based on the retrieved context: {top_chunk_snippet}"
        response = AskResponse(
            answer=answer_text,
            sources=[c["id"] for c in retrieved_chunks],
            confidence=1.0,
        )
    else:
        # Optional MOCK_LLM=0 extension: real LLM grounded in retrieved chunks.
        from llm_client import call_llm

        context = "\n\n".join(f"[{c['id']}] {c['document']}" for c in retrieved_chunks)
        prompt = build_rag_prompt(question=query, context=context)
        raw = call_llm(prompt)
        response = _validate_or_retry(raw, retrieved_chunks)

    return {
        **state,
        "retrieved_chunks": retrieved_chunks,
        "answer": response.answer,
        "sources": response.sources,
        "confidence": response.confidence,
    }


# ---------------------------------------------------------------------------
# Node 3: direct_answer (general_question path)
# ---------------------------------------------------------------------------
def direct_answer(state: GraphState) -> GraphState:
    query = state["query"]

    if mock_llm_enabled():
        # Mock mode (graded baseline): fixed canned string, no LLM call.
        response = AskResponse(
            answer="I can only answer questions about Zepto policies right now.",
            sources=[],
            confidence=1.0,
        )
    else:
        # Optional MOCK_LLM=0 extension: real LLM, no retrieval.
        from llm_client import call_llm

        prompt = build_direct_prompt(question=query)
        raw = call_llm(prompt)
        response = _validate_or_retry(raw, retrieved_chunks=[])

    return {
        **state,
        "retrieved_chunks": [],
        "answer": response.answer,
        "sources": response.sources,
        "confidence": response.confidence,
    }


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("retrieve_and_answer", retrieve_and_answer)
    workflow.add_node("direct_answer", direct_answer)

    workflow.set_entry_point("classify_intent")

    workflow.add_conditional_edges(
        "classify_intent",
        _route_after_classify,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer",
        },
    )

    workflow.add_edge("retrieve_and_answer", END)
    workflow.add_edge("direct_answer", END)

    return workflow.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_query(query: str) -> AskResponse:
    graph = get_graph()
    initial_state: GraphState = {
        "query": query,
        "intent": "",
        "retrieved_chunks": [],
        "answer": "",
        "sources": [],
        "confidence": 0.0,
    }
    final_state = graph.invoke(initial_state)
    return AskResponse(
        answer=final_state["answer"],
        sources=final_state["sources"],
        confidence=final_state["confidence"],
    )
