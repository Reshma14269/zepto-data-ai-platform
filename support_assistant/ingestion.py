"""
Task 1 — Ingestion / embedding.

Loads all 8 documents from docs/, chunks them (one chunk per document —
a simple per-document chunk scheme is fine given how short each doc is),
embeds each chunk with all-MiniLM-L6-v2, and stores the embeddings in a
persistent ChromaDB collection called "zepto_policies".

This module is imported by graph.py, which calls get_collection() to run
real retrieval (embedding + ChromaDB cosine-similarity search always runs
for real, in both MOCK_LLM modes — no API key or network call is needed
for embedding/ChromaDB).
"""

import os
import glob

import chromadb
from chromadb.utils import embedding_functions

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL_NAME
)


def _load_documents():
    """Load each docs/doc_NN.txt file as a single chunk (id = filename stem)."""
    chunks = []
    for path in sorted(glob.glob(os.path.join(DOCS_DIR, "doc_*.txt"))):
        doc_id = os.path.splitext(os.path.basename(path))[0]  # e.g. "doc_01"
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        chunks.append({"id": doc_id, "text": text})
    return chunks


def build_collection(persist_directory: str = CHROMA_DIR):
    """
    Build (or rebuild) the ChromaDB collection from the docs/ corpus.
    Safe to call repeatedly — deletes and recreates the collection so
    re-running ingestion doesn't create duplicate entries.
    """
    client = chromadb.PersistentClient(path=persist_directory)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # collection didn't exist yet — fine

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    chunks = _load_documents()
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
    )
    return collection


def get_collection(persist_directory: str = CHROMA_DIR):
    """
    Get the ChromaDB collection, building it first if it doesn't exist yet.
    """
    client = chromadb.PersistentClient(path=persist_directory)
    try:
        return client.get_collection(name=COLLECTION_NAME, embedding_function=_embedding_fn)
    except Exception:
        return build_collection(persist_directory)


def retrieve_top_k(query: str, k: int = 3, persist_directory: str = CHROMA_DIR):
    """
    Embed `query` and return the top-k most similar chunks from ChromaDB,
    via cosine similarity. Returns a list of dicts: {id, document, distance}.
    """
    collection = get_collection(persist_directory)
    results = collection.query(query_texts=[query], n_results=k)

    hits = []
    ids = results["ids"][0]
    docs = results["documents"][0]
    distances = results["distances"][0]
    for doc_id, doc_text, distance in zip(ids, docs, distances):
        hits.append({"id": doc_id, "document": doc_text, "distance": distance})
    return hits


if __name__ == "__main__":
    # Run `python ingestion.py` once to (re)build the ChromaDB collection.
    col = build_collection()
    print(f"Ingested {col.count()} chunks into collection '{COLLECTION_NAME}'.")
