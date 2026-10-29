from sentence_transformers import SentenceTransformer

from llm import ask_gemini
from search import load_index, search

EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 6
MIN_SCORE = 0.3


def build_prompt(question: str, results: list[dict]) -> str:
    context = "\n\n".join(f"[Page {r['page']}] {r['text']}" for r in results)
    return f"""You are an assistant answering questions about IBM's 2025 Annual Report.
Use ONLY the excerpts below. If they do not contain the answer, say you could not find it in the report.
Cite the page number like (Page 5) after each fact you use.

EXCERPTS:
{context}

QUESTION: {question}

ANSWER:"""


def answer_question(question, embed_model, embeddings, chunks):
    results = search(question, embed_model, embeddings, chunks, top_k=TOP_K)
    results = [r for r in results if r["score"] >= MIN_SCORE]

    if not results:
        return "I couldn't find anything relevant in the report.", []

    answer = ask_gemini(build_prompt(question, results))
    return answer, results


if __name__ == "__main__":
    embeddings, chunks = load_index()
    embed_model = SentenceTransformer(EMBED_MODEL)
    print(f"Ready: {len(chunks)} chunks")

    while True:
        question = input("\nAsk about the IBM report (or 'quit'): ").strip()
        if question.lower() == "quit":
            break
        if not question:
            continue

        answer, results = answer_question(question, embed_model, embeddings, chunks)
        print("\n" + answer)
        if results:
            pages = sorted({r["page"] for r in results})
            print(f"\nSources: pages {pages}")