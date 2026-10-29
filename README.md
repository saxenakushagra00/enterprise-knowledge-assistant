# Enterprise Knowledge Assistant

A Retrieval-Augmented Generation (RAG) assistant that answers questions about IBM's 2025 Annual Report, with page citations. Built from scratch in Python: PDF parsing, chunking, embeddings, vector search with NumPy, and answer generation with Gemini.

## How it works

```
PDF -> pages -> overlapping chunks -> embeddings (saved to disk)

Question -> embed -> cosine similarity search -> top chunks
         -> prompt with page labels -> Gemini -> cited answer
```

1. **Ingest** (`ingest.py`): reads the PDF page by page with `pypdf`, cleans whitespace, and splits each page into 1,000-character chunks with 200 characters of overlap. Duplicate chunks (the PDF repeats text across two-page spreads) are dropped.
2. **Embed** (`embed.py`): converts each chunk into a 384-dimension vector with `all-MiniLM-L6-v2` (runs locally, no API key) and saves the vectors to disk.
3. **Search** (`search.py`): embeds the question and scores it against every chunk with a dot product (vectors are normalized, so this equals cosine similarity).
4. **Answer** (`answer.py`): keeps chunks above a similarity cutoff, builds a prompt that tells the model to use only those excerpts and cite pages, and calls Gemini. Questions with no relevant chunks get a "not found" reply without calling the model.
5. **LLM client** (`llm.py`): calls Gemini with retries, exponential backoff, and fallback across several models.
6. **API** (`api.py`): a FastAPI service with `GET /health` and `POST /ask`.

## Tech stack
Python, pypdf, sentence-transformers, NumPy, Google Gen AI SDK (Gemini), FastAPI, Uvicorn.

## Setup (Windows PowerShell)

```powershell
git clone https://github.com/saxenakushagra00/enterprise-knowledge-assistant.git
cd enterprise-knowledge-assistant
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

1. Download IBM's 2025 Annual Report PDF and save it as `data/ibm-annual-report-2025.pdf`.
2. Get a free API key from Google AI Studio and set it for your session:
   ```powershell
   $env:GEMINI_API_KEY="your-key"
   ```
3. Build the index (one time):
   ```powershell
   python src/embed.py
   ```

## Run

Terminal chat:
```powershell
python src/answer.py
```

Web API (then open http://127.0.0.1:8000/docs):
```powershell
python -m uvicorn api:app --app-dir src
```

Example request body for `POST /ask`:
```json
{"question": "Who is the CEO?"}
```

## Project structure

```
src/
  ingest.py   load PDF, clean, chunk, dedupe
  embed.py    build and save embeddings
  search.py   vector search
  llm.py      Gemini client with retries and model fallback
  answer.py   retrieval + generation pipeline
  api.py      FastAPI service
  find_model.py  lists which Gemini models respond for your key
```

## Known limitations
- Financial tables are flattened by PDF extraction, so number-heavy questions retrieve less reliably than narrative ones.
- Chunks are cut at fixed character positions, so a chunk can start mid-word.
- The sources line lists every retrieved page, not only the pages the model cited.
- Gemini model names change over time. If a model returns 404, run `python src/find_model.py` and update `MODELS` in `src/llm.py`.

## Next steps
Hybrid keyword and vector search, sentence-aware chunking, a simple chat UI, and cloud deployment.