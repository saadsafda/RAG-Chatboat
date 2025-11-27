import os
import time
import json
import math
import re
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional, List

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DestroAI")

# --- Load Env ---
load_dotenv()

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# --- Global Vector Store (The RAM Cache) ---
VECTOR_DB = []  # Stores [{"text": "...", "embedding": [...], "path": "..."}]

# --- Helpers ---
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not OpenAI:
        logger.error("OpenAI API Key missing or library not installed.")
        return None
    return OpenAI(api_key=api_key)

def embed_text(text: str, client):
    """Generates embedding for a single string."""
    try:
        # Using small model for speed (approx 200ms)
        resp = client.embeddings.create(input=text, model="text-embedding-3-small")
        return resp.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return []

def rebuild_index():
    """Reads all files in /knowledge, creates embeddings, and saves to RAM."""
    global VECTOR_DB
    logger.info("♻️ Building Vector Index in RAM...")
    
    client = get_openai_client()
    if not client: return

    KNOWLEDGE_DIR = Path("knowledge")
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    
    new_db = []
    files = list(KNOWLEDGE_DIR.glob("**/*.txt"))
    
    if not files:
        logger.warning("No knowledge files found.")
        return

    for p in files:
        try:
            text = p.read_text(encoding="utf-8")
            # Only embed if text is long enough
            if len(text) > 10:
                # We truncate to 8000 chars to save costs/errors
                emb = embed_text(text[:8000], client)
                if emb:
                    new_db.append({"path": str(p), "text": text, "embedding": emb})
        except Exception as e:
            logger.error(f"Failed to process {p}: {e}")

    VECTOR_DB = new_db
    logger.info(f"✅ Index built! Loaded {len(VECTOR_DB)} documents into RAM.")

# --- Scraper ---
def fetch_and_save_url(url: str):
    try:
        logger.info(f"🕷️ Scraping: {url}")
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.text, "lxml")
        
        # Simple extraction
        text_parts = [t.get_text(" ", strip=True) for t in soup.find_all(['h1', 'h2', 'h3', 'p', 'li'])]
        body_text = "\n\n".join(text_parts)
        
        if len(body_text) < 50:
            logger.warning("Page content too short.")
            return

        # Save File
        KNOW_DIR = Path("knowledge")
        KNOW_DIR.mkdir(parents=True, exist_ok=True)
        parsed = urlparse(url)
        safe_name = re.sub(r"[^a-zA-Z0-9]", "_", parsed.netloc + parsed.path)
        if len(safe_name) > 50: safe_name = safe_name[:50]
        filename = f"{safe_name}.txt"
        
        (KNOW_DIR / filename).write_text(f"URL: {url}\n\n{body_text}", encoding="utf-8")
        
        # IMPORTANT: Update the RAM index immediately after scraping
        rebuild_index()
        
    except Exception as e:
        logger.error(f"Scraping failed: {e}")

# --- Lifespan (Startup Event) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load index when server starts
    rebuild_index()
    yield
    # Clean up (optional)

app = FastAPI(lifespan=lifespan)

# --- Data Models ---
class ChatMetadata(BaseModel):
    name: Optional[str] = "Guest"
    email: Optional[str] = ""

class ChatRequest(BaseModel):
    message: str
    metadata: Optional[ChatMetadata] = None

# --- RAG Logic (Optimized) ---
def generate_fast_response(message: str, user_name: str):
    global VECTOR_DB
    client = get_openai_client()
    if not client: return "AI Service Unavailable."
    
    # 1. Embed Query (The ONLY OpenAI Call for search)
    q_emb = embed_text(message, client)
    if not q_emb: return "I didn't understand that."

    # 2. Vector Search (Math in RAM - Microseconds)
    def cosine_sim(a, b):
        return sum(x*y for x,y in zip(a,b)) # Simplified for speed

    scored = []
    for doc in VECTOR_DB:
        score = cosine_sim(q_emb, doc["embedding"])
        scored.append((score, doc))
    
    # Sort and pick Top 3
    scored.sort(key=lambda x: x[0], reverse=True)
    top_docs = [d for s, d in scored[:3] if s > 0.3] # Filter low relevance

    if not top_docs:
        context_text = "No specific context found."
    else:
        context_text = "\n---\n".join([d["text"] for d in top_docs])

    # 3. Generate Answer (Streaming capable logic, but returning block for n8n)
    system_prompt = (
        f"You are Destro's AI assistant for destrotechnologies.com, helpful assistant for {user_name}. "
        "Answer strictly based on the Context below. Keep it short (under 3 sentences)."
        "Destro helps businesses with AI chatbots, AI calling agents, sales CRM automation, ecommerce solutions, and digital acceleration services."
        """Your goals:
- Greet visitors warmly and professionally
- Ask clarifying questions about their business, current challenges, and goals.
- Explain Destro's services clearly, using simple language and concrete examples.
- Suggest next steps (e.g., schedule a call, share more details, or request a proposal).
- Never invent services Destro does not provide.
- If you are not sure about something (pricing, very specific technical detail, internal policy), say you
will pass this to a human expert.
Always keep answers concise and helpful. When appropriate, summarize what you understood
about their needs and what Destro can do for them."""
    )
    
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {message}"}
            ],
            max_tokens=200,
            temperature=0.3
        )
        return completion.choices[0].message.content
    except Exception as e:
        return "I'm having trouble thinking right now."

# --- API Endpoints ---
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    start_time = time.time()
    
    user_name = request.metadata.name if request.metadata else "User"
    reply = generate_fast_response(request.message, user_name)
    
    process_time = time.time() - start_time
    logger.info(f"⚡ Chat processed in {process_time:.2f}s")
    
    return {"reply": reply}

@app.post("/ingest")
async def ingest_endpoint(payload: dict, background_tasks: BackgroundTasks):
    url = payload.get("url")
    if url:
        background_tasks.add_task(fetch_and_save_url, url)
        return {"status": "Ingest started. Index will update automatically."}
    return {"error": "No URL provided"}


@app.get("/health")
async def health(deep: bool = False):
    """Basic health check.

    Parameters:
      - deep: if true, perform a lightweight OpenAI capability check (instantiate client).
        Avoids expensive calls (no embedding/chat). Returns degraded status if fails.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_state = "missing"
    if openai_key and OpenAI is not None:
        try:
            # Instantiate client (cheap). If deep, optionally hit models list (still light).
            client = OpenAI(api_key=openai_key)
            if deep:
                # Light call: list one model (wrapped in try to avoid raising to caller)
                try:
                    _ = client.models.list().data[:1]
                except Exception as e:
                    logger.warning(f"Deep OpenAI check failed: {e}")
                    openai_state = "error"
                    return {"status": "degraded", "openai": openai_state, "knowledge_files": 0}
            openai_state = "ready"
        except Exception as e:
            logger.warning(f"OpenAI client init failed: {e}")
            openai_state = "error"
    # Count knowledge files
    knowledge_dir = Path("knowledge")
    if knowledge_dir.exists():
        k_files = sum(1 for _ in knowledge_dir.glob("**/*.txt"))
    else:
        k_files = 0
    status = "ok" if openai_state in {"ready", "missing"} else "degraded"
    return {
        "status": status,
        "openai": openai_state,
        "knowledge_files": k_files
    }