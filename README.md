# Zepto Data & AI Platform

An end-to-end AI/ML capstone project — a connected data platform with three modules:
1. **`/data_pipeline`** — raw data → cleaned, typed data → relational (SQLite) store
2. **`/analytics`** — full ML pipeline: EDA → feature engineering → model training → evaluation
3. **`/support_assistant`** — a RAG-based support assistant grounded in Zepto's policy documents

These three modules are meant to read as one story: the pipeline feeds clean data to
analysts, the analytics module shows how Zepto would model customer/passenger-style
outcomes end to end, and the support assistant shows how Zepto would put a grounded
GenAI service in front of its own policies.

---

## Repository structure

```
zepto-data-ai-platform/
├── README.md                  ← you are here
├── data_pipeline/
│   ├── README.md
│   ├── requirements.txt
│   └── ...
├── analytics/
│   ├── README.md
│   ├── requirements.txt
│   └── ...
└── support_assistant/
    ├── README.md
    ├── requirements.txt
    ├── docs/              # Zepto policy corpus (doc_01.txt … doc_08.txt)
    ├── ingestion.py        # load, chunk, embed, ChromaDB
    ├── prompts.py          # structured prompt template
    ├── graph.py            # LangGraph StateGraph (classify → retrieve/direct)
    ├── schemas.py           # Pydantic request/response models
    ├── llm_client.py        # optional real-LLM (Groq free tier) extension
    ├── main.py               # FastAPI app (POST /ask)
    └── Dockerfile
```

## Setup

Each module has its own `requirements.txt` (kept separate since the modules use
different dependencies — e.g., `support_assistant` needs an LLM client that the
other two don't). See each module's own README for its exact setup steps.

## How to run each module

- **Data Pipeline:** see [`data_pipeline/README.md`](data_pipeline/README.md)
- **Analytics:** see [`analytics/README.md`](analytics/README.md)
- **Support Assistant:** see [`support_assistant/README.md`](support_assistant/README.md)

## Design decisions

*(Fill this in as each module is built — a short paragraph per module explaining
key choices: why this dataset, why this model, why this chunking strategy, etc.)*

### Data Pipeline
*TBD*

### Analytics
*TBD*

### Support Assistant
A LangGraph-orchestrated RAG pipeline grounded in Zepto's own policy
documents. Each policy doc is treated as a single chunk (short enough that
finer-grained splitting isn't needed) and embedded with `all-MiniLM-L6-v2`
into a persistent ChromaDB collection. A 3-node `StateGraph` classifies each
query (keyword heuristic), retrieves the top-3 most similar chunks for
policy questions, and generates a grounded, schema-validated answer. Every
LLM call is gated behind a `MOCK_LLM` toggle so the graded baseline is
fully offline and deterministic; a real LLM call (Groq's free tier) is an
optional, ungraded extension. See
[`support_assistant/README.md`](support_assistant/README.md) for the full
architecture write-up.
