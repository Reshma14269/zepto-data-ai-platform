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
    └── ...
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
*TBD*
