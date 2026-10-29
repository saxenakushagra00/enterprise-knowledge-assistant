import re
from pathlib import Path

from pypdf import PdfReader


def load_pdf(pdf_path: str) -> list[dict]:
    reader = PdfReader(pdf_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()

        if not text:
            continue

        pages.append({
            "source": Path(pdf_path).name,
            "page": page_number,
            "text": text,
        })

    return pages

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_pages(pages: list[dict], chunk_size: int = 1000, overlap: int = 200) -> list[dict]:
    chunks = []
    step = chunk_size - overlap
    seen=set()
    for page in pages:
        text = clean_text(page["text"])
        for start in range(0, len(text), step):
            chunk_text = text[start:start + chunk_size]

            if len(chunk_text) < 100:
                continue
            if chunk_text in seen:
                continue
            seen.add(chunk_text)
            chunks.append({
                "chunk_id": len(chunks),
                "source": page["source"],
                "page": page["page"],
                "text": chunk_text,
            })

    return chunks

if __name__ == "__main__":
    pages = load_pdf("data/ibm-annual-report-2025.pdf")
    chunks = chunk_pages(pages)

    print(f"Pages: {len(pages)}")
    print(f"Chunks: {len(chunks)}")
    print("--- Sample chunk ---")
    print(chunks[50])