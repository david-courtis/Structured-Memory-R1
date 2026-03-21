"""
Memory retrieval server for Memory-R1.

Serves as the retrieval backend for the Answer Agent, providing
embedding-based similarity search over the memory bank.

Following the Memory-R1 paper, for each question we retrieve the
top-30 most relevant memories per participant (60 total).

The server exposes the same /retrieve API as Search-R1's retrieval server
so it can be used as a drop-in replacement.
"""
import json
import argparse
from typing import List, Dict, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from sklearn.feature_extraction.text import TfidfVectorizer


class MemoryStore:
    """
    In-memory store with TF-IDF retrieval.

    For production use, this could be replaced with a dense embedding
    model (e.g., sentence-transformers) and FAISS index.
    """

    def __init__(self):
        self.memories: List[Dict] = []  # [{id, text, speaker, ...}]
        self.vectorizer = None
        self.tfidf_matrix = None

    def load_memories(self, memories: List[Dict]):
        """Load memories and build TF-IDF index."""
        self.memories = memories
        if not memories:
            return

        texts = [m.get("text", "") for m in memories]
        self.vectorizer = TfidfVectorizer(
            max_features=10_000,
            stop_words="english",
            dtype=np.float32,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def retrieve(
        self,
        query: str,
        topk: int = 30,
        speaker: Optional[str] = None,
    ) -> List[Dict]:
        """
        Retrieve top-k memories matching a query.

        Args:
            query: Search query
            topk: Number of results to return
            speaker: Optional speaker filter

        Returns:
            List of {document, score} dicts
        """
        if not self.memories or self.vectorizer is None:
            return []

        # Filter by speaker if specified
        if speaker:
            indices = [i for i, m in enumerate(self.memories)
                       if m.get("speaker", "").lower() == speaker.lower()]
        else:
            indices = list(range(len(self.memories)))

        if not indices:
            return []

        query_vec = self.vectorizer.transform([query])

        # Compute similarities only for filtered indices
        filtered_matrix = self.tfidf_matrix[indices]
        scores = (query_vec @ filtered_matrix.T).toarray()[0]

        # Get top-k
        top_local = np.argsort(scores)[::-1][:topk]
        results = []
        for local_idx in top_local:
            global_idx = indices[local_idx]
            mem = self.memories[global_idx]
            results.append({
                "document": mem.get("text", ""),
                "score": float(scores[local_idx]),
            })
        return results

    def batch_retrieve(
        self,
        queries: List[str],
        topk: int = 30,
    ) -> List[List[Dict]]:
        """Batch retrieval for multiple queries."""
        if not self.memories or self.vectorizer is None:
            return [[] for _ in queries]

        query_vecs = self.vectorizer.transform(queries)
        all_scores = (query_vecs @ self.tfidf_matrix.T).toarray()

        results = []
        for i, scores in enumerate(all_scores):
            top_indices = np.argsort(scores)[::-1][:topk]
            query_results = []
            for idx in top_indices:
                mem = self.memories[idx]
                query_results.append({
                    "document": mem.get("text", ""),
                    "score": float(scores[idx]),
                })
            results.append(query_results)
        return results


# Global store
memory_store = MemoryStore()

# FastAPI app
app = FastAPI()


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


class UpdateRequest(BaseModel):
    memories: List[Dict]


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    Retrieve memories matching queries.
    Compatible with Search-R1's /retrieve API format.
    """
    topk = request.topk or 30
    results = memory_store.batch_retrieve(request.queries, topk=topk)

    resp = []
    for query_results in results:
        if request.return_scores:
            resp.append(query_results)
        else:
            resp.append([r["document"] for r in query_results])
    return {"result": resp}


@app.post("/update_memories")
def update_memories(request: UpdateRequest):
    """Update the memory store with new memories."""
    memory_store.load_memories(request.memories)
    return {"status": "ok", "num_memories": len(request.memories)}


@app.get("/status")
def status():
    return {"num_memories": len(memory_store.memories)}


def load_memories_from_file(filepath: str) -> List[Dict]:
    """Load memories from a JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "memories" in data:
        return data["memories"]
    return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory-R1 retrieval server")
    parser.add_argument("--memory_file", type=str, default=None,
                        help="JSON file with initial memories")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--topk", type=int, default=30)

    args = parser.parse_args()

    if args.memory_file:
        memories = load_memories_from_file(args.memory_file)
        memory_store.load_memories(memories)
        print(f"Loaded {len(memories)} memories from {args.memory_file}")

    print(f"Starting memory server on port {args.port}...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
