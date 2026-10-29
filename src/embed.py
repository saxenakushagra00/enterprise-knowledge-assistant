import json

import numpy as np
from sentence_transformers import SentenceTransformer

from ingest import load_pdf, chunk_pages

PDF_PATH = "data/ibm-annual-report-2025.pdf"
MODEL_NAME = "all-MiniLM-L6-v2"


def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> np.ndarray:
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.array(embeddings)


if __name__ == "__main__":
    pages = load_pdf(PDF_PATH)
    chunks = chunk_pages(pages)

    model = SentenceTransformer(MODEL_NAME)
    embeddings = embed_chunks(chunks, model)

    print("Embeddings shape:", embeddings.shape)
    print("First 5 numbers of chunk 0:", embeddings[0][:5])

    np.save("data/embeddings.npy", embeddings)
    with open("data/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    print("Saved embeddings and chunks to data/")