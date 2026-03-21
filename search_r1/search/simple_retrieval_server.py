"""
Lightweight retrieval server that uses TF-IDF over a subset of the wiki corpus.
Designed to run on low-RAM systems to get the Search-R1 pipeline working.
"""
import argparse
import json
import os
from typing import List, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False

app = FastAPI()
vectorizer = None
tfidf_matrix = None
corpus_docs = []
default_topk = 3


def load_corpus_subset(corpus_path: str, max_docs: int = 500_000):
    """Load first max_docs from corpus to fit in RAM."""
    docs = []
    print(f"Loading up to {max_docs} docs from {corpus_path}...")
    with open(corpus_path, "r") as f:
        for i, line in enumerate(f):
            if i >= max_docs:
                break
            doc = json.loads(line.strip())
            docs.append(doc)
            if (i + 1) % 100_000 == 0:
                print(f"  Loaded {i+1} docs...")
    print(f"Loaded {len(docs)} documents.")
    return docs


def build_index(docs):
    """Build TF-IDF index over corpus."""
    global vectorizer, tfidf_matrix
    print("Building TF-IDF index...")
    texts = [doc.get("contents", "") for doc in docs]
    vectorizer = TfidfVectorizer(
        max_features=50_000,
        stop_words="english",
        dtype=np.float32,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"TF-IDF index built: {tfidf_matrix.shape}")


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    topk = request.topk or default_topk
    query_vec = vectorizer.transform(request.queries)
    scores = (query_vec @ tfidf_matrix.T).toarray()

    resp = []
    for i, query_scores in enumerate(scores):
        top_indices = np.argsort(query_scores)[::-1][:topk]
        results = []
        for idx in top_indices:
            doc = corpus_docs[idx]
            contents = doc.get("contents", "")
            if request.return_scores:
                results.append({"document": contents, "score": float(query_scores[idx])})
            else:
                results.append(contents)
        resp.append(results)
    return {"result": resp}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple TF-IDF retrieval server")
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max_docs", type=int, default=500_000,
                        help="Max documents to load (reduce if low on RAM)")
    parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()
    default_topk = args.topk

    corpus_docs = load_corpus_subset(args.corpus_path, args.max_docs)
    build_index(corpus_docs)

    print(f"Starting server on port {args.port}...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
