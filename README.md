# RAG FastAPI example

This repository contains a minimal FastAPI app with a simple Retrieval-Augmented Generation (RAG) flow.

What I added
- `main.py` - main FastAPI app. `generate_ai_response` now implements a local RAG flow using OpenAI embeddings + ChatCompletion when `OPENAI_API_KEY` is present.
- `knowledge/` - folder to store `.txt` documents used as the knowledge base (one sample file is provided).
- `.rag_index.json` (created at runtime) caches document embeddings.
- `requirements.txt` lists the python dependencies.

Quickstart (macOS / zsh)

1. Create a virtualenv and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2. Create a `.env` file (or copy `.env.example`) and set `OPENAI_API_KEY` plus ERPNext vars:

```
cp .env.example .env
# edit .env and set OPENAI_API_KEY and ERPNext values
```

3. Add knowledge files (plain `.txt`) into the `knowledge/` folder. A sample is included.

4. Run the app:

```bash
uvicorn main:app --reload
```

Notes
- If `OPENAI_API_KEY` isn't set or `openai` isn't installed, `generate_ai_response` falls back to a simulated response.
- For production, use a proper vector DB (Qdrant/Pinecone) and secure secrets management instead of `.env`.
