# import os
# from dotenv import load_dotenv
# from fastapi import FastAPI, HTTPException, BackgroundTasks
# from pydantic import BaseModel
# from typing import Optional
# import requests
# import uuid
# import json
# import math
# import glob
# from pathlib import Path
# import re
# from urllib.parse import urlparse
# from bs4 import BeautifulSoup

# try:
#     # OpenAI v1+: use the OpenAI client
#     from openai import OpenAI
# except Exception:
#     OpenAI = None

# # --- Configuration ---
# app = FastAPI()
# # Load local `.env` file into environment (if present). Install via: pip install python-dotenv
# load_dotenv()
# ERPNEXT_URL = os.getenv("ERPNEXT_URL")  # e.g., https://erp.destrotechnologies.website
# ERPNEXT_API_KEY = os.getenv("ERPNEXT_API_KEY")
# ERPNEXT_API_SECRET = os.getenv("ERPNEXT_API_SECRET")

# # --- Data Models ---
# class ChatMetadata(BaseModel):
#     name: str
#     email: str
#     message_number: str

# class ChatRequest(BaseModel):
#     session_id: Optional[str] = None
#     message: str
#     metadata: ChatMetadata

# # --- Helper: ERPNext Integration ---
# def create_or_update_erpnext_lead(metadata: ChatMetadata, session_id: str, summary: str):
#     """
#     Creates a lead if email doesn't exist, or updates it.
#     """
#     headers = {
#         "Authorization": f"token {ERPNEXT_API_KEY}:{ERPNEXT_API_SECRET}",
#         "Content-Type": "application/json"
#     }

#     print("Checking \n\n\n\n\n\n")
    
#     # 1. Check if Lead exists
#     search_url = f"{ERPNEXT_URL}/api/resource/Lead?filters=[[\"email_id\",\"=\",\"{metadata.email}\"]]"
#     response = requests.get(search_url, headers=headers)
#     data = response.json()
    
#     lead_payload = {
#         "lead_name": metadata.name,
#         "email_id": metadata.email,
#         "mobile_no": metadata.message_number,
#         "source": "Website AI Chatbot",
#         "custom_session_id": session_id,
#         # "notes": summary  # You might want to append to notes rather than overwrite
#     }

#     if data.get('data'):
#         # Update existing Lead
#         lead_name = data['data'][0]['name']
#         requests.put(f"{ERPNEXT_URL}/api/resource/Lead/{lead_name}", json=lead_payload, headers=headers)
#         return lead_name
#     else:
#         # Create New Lead
#         lead_payload["status"] = "Lead"
#         create_resp = requests.post(f"{ERPNEXT_URL}/api/resource/Lead", json=lead_payload, headers=headers)
#         if create_resp.status_code == 200:
#             return create_resp.json()['data']['name']
#     return None


# def fetch_and_save_url(url: str) -> str:
#     """
#     Fetch a URL, extract visible text, and save it to the `knowledge/` folder as a .txt file.
#     Returns the path to the saved file.

#     Notes:
#     - This is a simple scraper: it collects <h1>-<h3>, <p>, and meta description.
#     - Respect robots.txt and rate limits for production use. This helper does not check robots.txt.
#     """
#     try:
#         resp = requests.get(url, timeout=15)
#         resp.raise_for_status()
#     except Exception as e:
#         raise RuntimeError(f"Failed to fetch {url}: {e}")

#     soup = BeautifulSoup(resp.text, "lxml")
#     parts = []

#     # Title
#     title = soup.title.string.strip() if soup.title and soup.title.string else ""
#     if title:
#         parts.append(title)

#     # Meta description
#     desc = ""
#     md = soup.find("meta", attrs={"name": "description"})
#     if md and md.get("content"):
#         desc = md.get("content").strip()
#         parts.append(desc)

#     # Headings and paragraphs
#     for tag in soup.find_all(["h1", "h2", "h3", "p"]):
#         text = (tag.get_text(separator=" ") or "").strip()
#         if text:
#             parts.append(text)

#     # Join and sanitize whitespace
#     body_text = "\n\n".join(parts)
#     body_text = re.sub(r"\s+", " ", body_text).strip()

#     # Create knowledge directory if missing
#     KNOW_DIR = Path("knowledge")
#     KNOW_DIR.mkdir(parents=True, exist_ok=True)

#     # Create a safe filename from the URL
#     parsed = urlparse(url)
#     safe_path = parsed.netloc + parsed.path
#     safe_path = re.sub(r"[^0-9A-Za-z_-]", "_", safe_path)
#     if not safe_path:
#         safe_path = "page"
#     filename = f"{safe_path}.txt"
#     file_path = KNOW_DIR / filename

#     # Write content to file
#     try:
#         file_path.write_text(f"URL: {url}\n\n{body_text}", encoding="utf-8")
#     except Exception as e:
#         raise RuntimeError(f"Failed to save scraped content: {e}")

#     return str(file_path)


# # --- Helper: RAG / OpenAI Logic (Placeholder) ---
# def generate_ai_response(message: str, history: list):
#     """
#     RAG flow (simple local implementation):
#     - If OPENAI_API_KEY is present and `openai` is installed, compute embeddings for the query
#       and for documents in `knowledge/` (cached in `.rag_index.json`), retrieve top-k docs,
#       then call OpenAI ChatCompletion with the retrieved context and return the assistant reply.
#     - If OpenAI isn't available, return a simulated response.

#     Inputs:
#       - message: user query
#       - history: list of past messages (not used in this minimal example)

#     Outputs:
#       - string reply
#     """
#     # TODO: Connect to Qdrant/Pinecone here
#     # TODO: Connect to OpenAI API here
#     # Configuration
#     KNOWLEDGE_DIR = Path("knowledge")
#     INDEX_PATH = Path(".rag_index.json")
#     TOP_K = 3

#     OPENAI_KEY = os.getenv("OPENAI_API_KEY")
#     if not OPENAI_KEY or OpenAI is None:
#         # fallback if OpenAI not configured or package missing
#         return "This is a simulated AI response based on Destro services. (OpenAI not configured)"

#     # Initialize OpenAI client (v1+)
#     try:
#         client = OpenAI(api_key=OPENAI_KEY)
#     except Exception as e:
#         return f"Failed to initialize OpenAI client: {e}"

#     def embed_text(text: str):
#         # Use OpenAI embeddings API
#         try:
#             resp = client.embeddings.create(input=text, model="text-embedding-3-small")
#             # resp.data[0].embedding is a list of floats
#             return resp.data[0].embedding
#         except Exception as e:
#             raise RuntimeError(f"Embedding request failed: {e}")

#     def cosine_sim(a, b):
#         # avoid numpy to keep deps small
#         dot = sum(x * y for x, y in zip(a, b))
#         na = math.sqrt(sum(x * x for x in a))
#         nb = math.sqrt(sum(y * y for y in b))
#         if na == 0 or nb == 0:
#             return 0.0
#         return dot / (na * nb)

#     def build_or_load_index():
#         # Load cached index if available and knowledge dir hasn't changed timestamp-wise.
#         docs = []
#         for p in sorted(KNOWLEDGE_DIR.glob("**/*.txt")):
#             text = p.read_text(encoding="utf-8")
#             docs.append({"path": str(p), "text": text})

#         if not docs:
#             return []

#         # If we have cache, try to reuse embeddings for unchanged files
#         cache = {}
#         if INDEX_PATH.exists():
#             try:
#                 cache = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
#             except Exception:
#                 cache = {}

#         index = []
#         for d in docs:
#             path = d["path"]
#             mtime = Path(path).stat().st_mtime
#             cache_entry = cache.get(path)
#             if cache_entry and cache_entry.get("mtime") == mtime and cache_entry.get("embedding"):
#                 emb = cache_entry["embedding"]
#             else:
#                 emb = embed_text(d["text"])
#                 cache[path] = {"mtime": mtime, "embedding": emb, "text": d["text"]}
#             index.append({"path": path, "text": d["text"], "embedding": emb})

#         # persist cache
#         try:
#             INDEX_PATH.write_text(json.dumps(cache), encoding="utf-8")
#         except Exception:
#             pass

#         return index

#     def retrieve_docs(query: str, k=TOP_K):
#         idx = build_or_load_index()
#         if not idx:
#             return []
#         q_emb = embed_text(query)
#         scored = []
#         for item in idx:
#             score = cosine_sim(q_emb, item["embedding"])
#             scored.append((score, item))
#         scored.sort(key=lambda x: x[0], reverse=True)
#         return [it for _, it in scored[:k]]

#     # Retrieval
#     try:
#         docs = retrieve_docs(message, TOP_K)
#     except Exception as e:
#         # If embeddings fail for some reason, fallback
#         return f"Failed to retrieve docs: {e}"

#     # Build prompt/context for ChatCompletion
#     context_text = "\n\n---\n\n".join([f"Source: {Path(d['path']).name}\n{d['text']}" for d in docs])

#     system_prompt = (
#         "You are an assistant that helps answer user questions using the provided source documents. "
#         "When the answer cannot be found in the sources, be honest and say you don't know."
#     )

#     user_prompt = (
#         f"User question:\n{message}\n\nRelevant sources:\n{context_text}\n\n"
#         "Provide a concise answer and cite the source filenames you used."
#     )

#     try:
#         completion = client.chat.completions.create(
#             model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt}
#             ],
#             max_tokens=512,
#             temperature=0.2,
#         )
#         # For the v1 client, the text is in choices[0].message.content
#         return completion.choices[0].message.content.strip()
#     except Exception as e:
#         return f"OpenAI ChatCompletion failed: {e}"

# # --- API Endpoints ---
# @app.post("/chat")
# async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
#     # 1. Manage Session
#     session_id = request.session_id or str(uuid.uuid4())
    
#     # 2. RAG Logic
#     # Retrieve history from DB based on session_id (omitted for brevity)
#     ai_reply = generate_ai_response(request.message, [])
    
#     # 3. ERPNext Sync (Run in background to keep chat fast)
#     # Only sync if we have valid contact info
#     if request.metadata.email:
#         print("Checking \n\n\n\n\n\n")
#         background_tasks.add_task(
#             create_or_update_erpnext_lead, 
#             request.metadata, 
#             session_id, 
#             f"User: {request.message}\nAI: {ai_reply}"
#         )

#     return {
#         "reply": ai_reply,
#         "session_id": session_id,
#         "lead_created": True # Simplified logic
#     }


# @app.post("/ingest")
# async def ingest_url_endpoint(payload: dict, background_tasks: BackgroundTasks):
#     """Accept JSON {"url":"https://..."} and fetch the page into the local knowledge folder."""
#     url = payload.get("url") if isinstance(payload, dict) else None
#     if not url:
#         raise HTTPException(status_code=400, detail="Missing 'url' in request body")

#     # Run scrape in background to keep API responsive
#     try:
#         background_tasks.add_task(fetch_and_save_url, url)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

#     return {"status": "ingest_started", "url": url}


import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import uuid
import json
import math
import glob
from pathlib import Path
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import requests
from typing import Optional, List
import logging
import time

try:
    from openai import OpenAI
    # Try importing common exception classes (available in openai>=1.x)
    try:
        from openai import APIError, AuthenticationError, RateLimitError, OpenAIError
    except Exception:
        APIError = AuthenticationError = RateLimitError = OpenAIError = Exception
except Exception:
    OpenAI = None
    APIError = AuthenticationError = RateLimitError = OpenAIError = Exception

# --- Configuration ---
app = FastAPI()
load_dotenv()

# Basic logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rag-app")

# Domain-specific internal error to signal 500-worthy failures
class InternalAIError(Exception):
    pass

# --- Data Models ---
class ChatMetadata(BaseModel):
    name: Optional[str] = ""
    email: Optional[str] = ""
    message_number: Optional[str] = ""

# NOTE: ERPNext logic removed. This is now a pure RAG Microservice.

# --- Data Models ---
class ChatRequest(BaseModel):
    message: str
    metadata: Optional[ChatMetadata] = None
    history: Optional[List[dict]] = None # n8n will pass history here if needed

# --- Helper: Web Scraping (Kept exactly as is) ---
def fetch_and_save_url(url: str) -> str:
    # ... [Keep your existing fetch_and_save_url code here] ...
    # (For brevity, assume your existing scraper code is here)
    # Basic URL validation
    parsed = urlparse(url or "")
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError("Invalid URL. Only http/https are allowed.")

    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch {url}: {e}")

    soup = BeautifulSoup(resp.text, "lxml")
    parts = []
    
    # Title & Meta
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    if title: parts.append(title)
    md = soup.find("meta", attrs={"name": "description"})
    if md and md.get("content"): parts.append(md.get("content").strip())

    # Content
    for tag in soup.find_all(["h1", "h2", "h3", "p"]):
        text = (tag.get_text(separator=" ") or "").strip()
        if text: parts.append(text)

    body_text = "\n\n".join(parts)
    body_text = re.sub(r"\s+", " ", body_text).strip()

    KNOW_DIR = Path("knowledge")
    KNOW_DIR.mkdir(parents=True, exist_ok=True)
    
    safe_path = re.sub(r"[^0-9A-Za-z_-]", "_", parsed.netloc + parsed.path) or "page"
    filename = f"{safe_path}.txt"
    (KNOW_DIR / filename).write_text(f"URL: {url}\n\n{body_text}", encoding="utf-8")
    return filename

# --- Helper: RAG Logic (Kept similar but cleaner) ---
def generate_ai_response(message: str, user_name: str):
    KNOWLEDGE_DIR = Path("knowledge")
    INDEX_PATH = Path(".rag_index.json")
    TOP_K = 3
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    
    if not message or not message.strip():
        # Bad request: no question provided
        raise HTTPException(status_code=400, detail="Please provide a question.")

    if not OPENAI_KEY or OpenAI is None:
        logger.warning("OpenAI not configured or missing key; raising InternalAIError.")
        raise InternalAIError("AI service is unavailable. Please try again later.")

    client = OpenAI(api_key=OPENAI_KEY)


    # --- PERSONALIZED PROMPT ---
    system_prompt = (
        f"You are a helpful assistant for Destro Technologies talking to {user_name}. " 
        "Answer the question using the provided context. "
        "Be polite, professional, and concise."
    )

    def _retry_sleep(attempt: int):
        # Exponential backoff with jitter
        delay = min(2 ** attempt, 8) + (0.05 * attempt)
        time.sleep(delay)

    def embed_text(text: str):
        last_err = None
        for attempt in range(3):
            try:
                return client.embeddings.create(input=text, model="text-embedding-3-small").data[0].embedding
            except (RateLimitError, APIError) as e:
                last_err = e
                logger.warning(f"Embedding attempt {attempt+1} failed: {e}")
                _retry_sleep(attempt)
            except (AuthenticationError,) as e:
                logger.error("Authentication with OpenAI failed (check OPENAI_API_KEY).")
                # send proper error
                raise InternalAIError("I'm having trouble authenticating with the AI service. Please try again later.")
            except Exception as e:
                last_err = e
                break
        raise RuntimeError(f"Embedding request failed: {last_err}")

    def cosine_sim(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    # Build Index
    docs = []
    for p in KNOWLEDGE_DIR.glob("**/*.txt"):
        docs.append({"path": str(p), "text": p.read_text(encoding="utf-8")})
    
    if not docs:
        return "I have no knowledge base yet. Please ingest some URLs."

    # Simplified Indexing (In-Memory for now)
    # In production, implement the caching logic you had before
    scored = []
    q_emb = embed_text(message)
    
    for d in docs:
        # Compute embedding on the fly (slow) or load from cache (fast)
        # For this demo, we assume you implement the caching logic you had
        d_emb = embed_text(d['text'][:2000]) # Truncate for embedding cost
        score = cosine_sim(q_emb, d_emb)
        scored.append((score, d))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    top_docs = [item for score, item in scored[:TOP_K]]

    context_text = "\n---\n".join([d['text'] for d in top_docs])
    
    # Chat completion with small retry and graceful fallback
    last_err = None
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {message}"}
                ],
                max_tokens=512,
                temperature=0.2,
            )
            return completion.choices[0].message.content
        except (RateLimitError, APIError) as e:
            last_err = e
            logger.warning(f"Chat attempt {attempt+1} failed: {e}")
            _retry_sleep(attempt)
        except (AuthenticationError,) as e:
            logger.error("Authentication with OpenAI failed (check OPENAI_API_KEY).")
            return "I'm having trouble authenticating with the AI service. Please try again later."
        except Exception as e:
            last_err = e
            break
    logger.error(f"Chat failed after retries: {last_err}")
    raise InternalAIError("I'm sorry, I'm having trouble generating a response right now. Please try again shortly.")

# --- API Endpoints ---
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # Pure RAG response. No side effects.
    user_name = (request.metadata.name if request.metadata and request.metadata.name else "User")
    try:
        ai_reply = generate_ai_response(request.message, user_name)

        return {"reply": ai_reply}
    except InternalAIError as e:
        # Convert internal AI failures to HTTP 500 for clients that expect error signaling
        logger.warning(f"InternalAIError: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except HTTPException:
        # Re-raise explicit HTTPExceptions (e.g., 400 for bad input)
        raise
    except Exception as e:
        # Unexpected errors -> 500
        logger.exception("Unhandled error in /chat", e)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/ingest")
async def ingest_url_endpoint(payload: dict, background_tasks: BackgroundTasks):
    if not isinstance(payload, dict):
        raise HTTPException(400, "Invalid JSON body")
    url = payload.get("url")
    if not url:
        raise HTTPException(400, "Missing url")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(400, "Invalid URL. Only http/https are allowed.")
    background_tasks.add_task(fetch_and_save_url, url)
    return {"status": "ingest_started"}


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