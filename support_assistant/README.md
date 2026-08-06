# Support Assistant

A RAG-based assistant that answers policy questions grounded in Zepto's own
documents. Architecture reused/adapted from an earlier RAG project — see design
decisions below once filled in.

## What this module does *(fill in as you build)*
- Ingests a set of policy-style documents (chunking strategy: *TBD*)
- Embeds and stores chunks in a vector store
- Retrieves relevant chunks for a user question
- Generates a grounded answer using an LLM, citing/using only retrieved context

## Files
- `rag_engine.py` — chunking, embedding, retrieval, generation logic *(TBD)*
- `app.py` — interface (CLI or Streamlit) *(TBD)*
- `data/` — Zepto policy documents used for grounding *(TBD)*

## Setup

```bash
cd support_assistant
pip install -r requirements.txt
cp .env.example .env   # add your API key
```

## How to run

```bash
python app.py
# or, if Streamlit:
streamlit run app.py
```

## Design decisions

*TBD — explain chunking strategy, vector store choice, and LLM choice, once built.*
