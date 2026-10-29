from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from answer import EMBED_MODEL, answer_question
from search import load_index

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["embeddings"], state["chunks"] = load_index()
    state["embed_model"] = SentenceTransformer(EMBED_MODEL)
    yield
    state.clear()


app = FastAPI(title="IBM Annual Report Assistant", lifespan=lifespan)


class Question(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(state["chunks"])}


@app.post("/ask")
def ask(body: Question):
    answer, results = answer_question(
        body.question, state["embed_model"], state["embeddings"], state["chunks"]
    )
    sources = [{"page": r["page"], "score": round(r["score"], 3)} for r in results]
    return {"answer": answer, "sources": sources}