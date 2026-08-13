# Module 3 — Support Assistant (`/support_assistant`)

A small but complete GenAI service for Zepto: a document corpus that's embedded
and indexed, a LangGraph-orchestrated flow that routes each query and
retrieves grounded context, a structured-output guarantee, and a FastAPI
wrapper you run locally.

Graded with `MOCK_LLM` left at its default (unset / `1`) — fully offline,
deterministic, no signup, no API key, no network call to any LLM provider.

## Architecture

The pipeline has four stages: **ingestion → embedding → retrieval → generation.**

1. **Ingestion** — `ingestion.py: _load_documents()` reads all 8 files from
   `docs/doc_01.txt` … `docs/doc_08.txt`. Each document is treated as a
   single chunk (a simple per-document chunk scheme, appropriate given how
   short each policy doc is), with the chunk id set to the filename stem
   (e.g. `doc_01`).

2. **Embedding** — `ingestion.py: build_collection()` embeds every chunk
   with `all-MiniLM-L6-v2` (via `chromadb`'s
   `SentenceTransformerEmbeddingFunction`) and stores the vectors in a
   persistent **ChromaDB** collection named `zepto_policies`
   (`hnsw:space = cosine`), on disk under `support_assistant/chroma_db/`.

3. **Retrieval** — `graph.py: retrieve_and_answer()` calls
   `ingestion.retrieve_top_k()`, which embeds the incoming query with the
   same model and runs a cosine-similarity search against the
   `zepto_policies` ChromaDB collection, returning the top-3 chunks. This
   retrieval step **always runs for real, in both `MOCK_LLM` modes** —
   embedding and ChromaDB need no API key and no network call.

4. **Generation** — handled by the LangGraph `StateGraph` in `graph.py`,
   built from a `TypedDict` state (`GraphState`) and 3 nodes:
   - **`classify_intent`** — routes the query to `policy_question` or
     `general_question`.
   - **`retrieve_and_answer`** — for `policy_question`, retrieves context
     (step 3 above) then generates the final answer.
   - **`direct_answer`** — for `general_question`, generates an answer with
     no retrieval.

   A conditional edge from `classify_intent` (`_route_after_classify`)
   routes to whichever of the two answer nodes matches the classification.
   The final answer from either node is validated against the `AskResponse`
   Pydantic schema (`schemas.py`: `answer` / `sources` / `confidence`)
   before being returned by the FastAPI endpoint.

### What branches on `MOCK_LLM`

Only the **generation step inside each node** branches on `MOCK_LLM`
(`graph.py: mock_llm_enabled()`) — the routing logic and the retrieval step
itself do not depend on it.

| Node | `MOCK_LLM` default (graded baseline) | `MOCK_LLM=0` (optional extension) |
|---|---|---|
| `classify_intent` | Keyword heuristic over `delivery`, `return`, `refund`, `membership`, `tracking`, `cancel`, `gift card`, `support hours`. No LLM call. | Real LLM call (`llm_client.call_llm`) asked to classify. |
| `retrieve_and_answer` | Canned template: `f"Based on the retrieved context: {top_chunk_snippet}"` (first ~200 chars of the top retrieved chunk). `sources` = ids of the retrieved chunks. `confidence` = `1.0`. | Real LLM prompted with the structured template from `prompts.py` (`build_rag_prompt`), grounded only in the retrieved chunks. Retries up to 2x on schema-validation failure before returning a clearly marked error response. |
| `direct_answer` | Fixed string: `"I can only answer questions about Zepto policies right now."` `sources` = `[]`. `confidence` = `1.0`. | Real LLM prompted directly (`build_direct_prompt`), no retrieval. Same retry-then-error behavior on validation failure. |

The optional real-LLM path uses Groq's free tier (`console.groq.com`, no
card required) via `llm_client.py`, reading the key from `GROQ_API_KEY`.
This path was **not** used for the graded submission — every example call
below was made with `MOCK_LLM` at its default.

## Files

```
support_assistant/
├── docs/doc_01.txt … doc_08.txt   # Zepto policy corpus (Task: exact text)
├── ingestion.py                    # Task 1: load, chunk, embed, ChromaDB
├── prompts.py                      # Task 2: structured prompt template
├── schemas.py                      # Task 4: AskRequest / AskResponse
├── graph.py                        # Task 3 & 4: LangGraph StateGraph
├── llm_client.py                   # optional MOCK_LLM=0 extension only
├── main.py                         # Task 5: FastAPI app (POST /ask)
├── Dockerfile                      # Task 6
└── README.md                       # this file (Task 7)
```

## Running locally

```bash
cd support_assistant
pip install -r requirements.txt
python ingestion.py          # builds the ChromaDB collection (one-time)
uvicorn main:app --host 0.0.0.0 --port 7860
```

## Example calls (recorded with `MOCK_LLM` at its default)

**Call 1 — policy question (triggers retrieval):**

```
POST /ask
{"query": "Is delivery free on my order?"}

200 OK
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del",
  "sources": ["doc_01", "doc_05", "doc_03"],
  "confidence": 1.0
}
```

**Call 2 — general question (no retrieval):**

```
POST /ask
{"query": "What is the capital of France?"}

200 OK
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

## Docker

```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
# then: curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" -d '{"query": "Is delivery free?"}'
```

The image builds and runs the `POST /ask` endpoint locally using only the
required `MOCK_LLM` default baseline — no push to Hugging Face Spaces was
attempted for this submission.
