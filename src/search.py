import json

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


def load_index() -> tuple[np.ndarray, list[dict]]:
    embeddings = np.load("data/embeddings.npy")
    with open("data/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return embeddings, chunks


def search(query: str, model: SentenceTransformer, embeddings: np.ndarray,
           chunks: list[dict], top_k: int = 3) -> list[dict]:
    query_vec = model.encode(query, normalize_embeddings=True)
    scores = embeddings @ query_vec
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "score": float(scores[idx]),
            "chunk_id": chunks[idx]["chunk_id"],
            "page": chunks[idx]["page"],
            "text": chunks[idx]["text"],
        })
    return results


if __name__ == "__main__":
    embeddings, chunks = load_index()
    model = SentenceTransformer(MODEL_NAME)
    print(f"Index ready: {len(chunks)} chunks")

    while True:
        query = input("\nAsk a question (or type 'quit'): ").strip()
        if query.lower() == "quit":
            break
        if not query:
            continue

        for rank, r in enumerate(search(query, model, embeddings, chunks), start=1):
            print(f"\n#{rank}  score={r['score']:.3f}  page={r['page']}")
            print(r["text"][:400])