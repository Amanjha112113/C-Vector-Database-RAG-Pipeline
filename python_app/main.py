from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import os

from database import VectorDB, DocumentDB
from llm_client import OllamaClient, chunk_text

app = FastAPI(title="VectorDB API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DIMS = 16
db = VectorDB(DIMS)
docDB = DocumentDB()
ollama = OllamaClient()

# Load Demo Data
def load_demo():
    demo_data = [
        ("Linked List: nodes connected by pointers", "cs", [0.90,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06]),
        ("Binary Search Tree: O(log n) search and insert", "cs", [0.88,0.82,0.78,0.74,0.15,0.10,0.08,0.12,0.06,0.07,0.08,0.05,0.09,0.06,0.07,0.10]),
        ("Dynamic Programming: memoization overlapping subproblems", "cs", [0.82,0.76,0.88,0.80,0.20,0.18,0.12,0.09,0.07,0.06,0.08,0.07,0.08,0.09,0.06,0.07]),
        ("Graph BFS and DFS: breadth and depth first traversal", "cs", [0.85,0.80,0.75,0.82,0.18,0.14,0.10,0.08,0.06,0.09,0.07,0.06,0.10,0.08,0.09,0.07]),
        ("Hash Table: O(1) lookup with collision chaining", "cs", [0.87,0.78,0.70,0.76,0.13,0.11,0.09,0.14,0.08,0.07,0.06,0.08,0.07,0.10,0.08,0.09]),
        ("Calculus: derivatives integrals and limits", "math", [0.12,0.15,0.18,0.10,0.91,0.86,0.78,0.72,0.08,0.06,0.07,0.09,0.07,0.08,0.06,0.10]),
        ("Linear Algebra: matrices eigenvalues eigenvectors", "math", [0.20,0.18,0.15,0.12,0.88,0.90,0.82,0.76,0.09,0.07,0.08,0.06,0.10,0.07,0.08,0.09]),
        ("Probability: distributions random variables Bayes theorem", "math", [0.15,0.12,0.20,0.18,0.84,0.80,0.88,0.82,0.07,0.08,0.06,0.10,0.09,0.06,0.09,0.08]),
        ("Number Theory: primes modular arithmetic RSA cryptography", "math", [0.22,0.16,0.14,0.20,0.80,0.85,0.76,0.90,0.08,0.09,0.07,0.06,0.08,0.10,0.07,0.06]),
        ("Combinatorics: permutations combinations generating functions", "math", [0.18,0.20,0.16,0.14,0.86,0.78,0.84,0.80,0.06,0.07,0.09,0.08,0.06,0.09,0.10,0.07]),
        ("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food", [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.90,0.86,0.78,0.72,0.08,0.06,0.09,0.07]),
        ("Sushi: vinegared rice raw fish and nori rolls", "food", [0.06,0.08,0.07,0.09,0.09,0.06,0.08,0.07,0.86,0.90,0.82,0.76,0.07,0.09,0.06,0.08]),
        ("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food", [0.09,0.07,0.06,0.08,0.08,0.09,0.07,0.06,0.82,0.78,0.90,0.84,0.09,0.07,0.08,0.06]),
        ("Tacos: corn tortillas with carnitas salsa and cilantro", "food", [0.07,0.09,0.08,0.06,0.06,0.07,0.09,0.08,0.78,0.82,0.86,0.90,0.06,0.08,0.07,0.09]),
        ("Croissant: laminated pastry with buttery flaky layers", "food", [0.06,0.07,0.10,0.09,0.10,0.06,0.07,0.10,0.85,0.80,0.76,0.82,0.09,0.07,0.10,0.06]),
        ("Basketball: fast-paced shooting dribbling slam dunks", "sports", [0.09,0.07,0.08,0.10,0.08,0.09,0.07,0.06,0.08,0.07,0.09,0.06,0.91,0.85,0.78,0.72]),
        ("Football: tackles touchdowns field goals and strategy", "sports", [0.07,0.09,0.06,0.08,0.09,0.07,0.10,0.08,0.07,0.09,0.08,0.07,0.87,0.89,0.82,0.76]),
        ("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports", [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.09,0.06,0.07,0.08,0.83,0.80,0.88,0.82]),
        ("Chess: openings endgames tactics strategic board game", "sports", [0.25,0.20,0.22,0.18,0.22,0.18,0.20,0.15,0.06,0.08,0.07,0.09,0.80,0.84,0.78,0.90]),
        ("Swimming: butterfly freestyle backstroke Olympic competition", "sports", [0.06,0.08,0.07,0.09,0.08,0.06,0.09,0.07,0.10,0.08,0.06,0.07,0.85,0.82,0.86,0.80])
    ]
    for meta, cat, emb in demo_data:
        db.insert(meta, cat, emb)

load_demo()

# Models
class InsertDemoReq(BaseModel):
    metadata: str
    category: str
    embedding: List[float]

class InsertDocReq(BaseModel):
    title: str
    text: str

class AskReq(BaseModel):
    question: str
    k: int = 3

# Static Files
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))

# Endpoints
@app.get("/search")
def search(v: str, k: int = 5, metric: str = "cosine", algo: str = "hnsw"):
    try:
        q = [float(x) for x in v.split(',')]
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid vector format")
    if len(q) != DIMS:
        raise HTTPException(status_code=400, detail=f"Need {DIMS}D vector")
    
    out = db.search(q, k, metric, algo)
    return {
        "results": out["hits"],
        "latencyUs": out["latency_us"],
        "algo": out["algo"],
        "metric": out["metric"]
    }

@app.post("/insert")
def insert(req: InsertDemoReq):
    if len(req.embedding) != DIMS:
        raise HTTPException(status_code=400, detail=f"Need {DIMS}D vector")
    vid = db.insert(req.metadata, req.category, req.embedding)
    return {"id": vid, "message": "Inserted successfully"}

@app.delete("/delete/{vid}")
def delete_item(vid: int):
    if db.remove(vid):
        return {"message": "Deleted successfully"}
    raise HTTPException(status_code=404, detail="Not found")

@app.get("/items")
def get_items():
    return db.all()

@app.get("/benchmark")
def benchmark(v: str, k: int = 5, metric: str = "cosine"):
    try:
        q = [float(x) for x in v.split(',')]
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid vector format")
    if len(q) != DIMS:
        raise HTTPException(status_code=400, detail=f"Need {DIMS}D vector")
    
    res = db.benchmark(q, k, metric)
    return {
        "bruteforceUs": res["bf_us"],
        "kdtreeUs": res["kd_us"],
        "hnswUs": res["hnsw_us"],
        "n": res["n"]
    }

@app.get("/hnsw-info")
def hnsw_info():
    return db.hnsw_info()

@app.get("/status")
def status():
    return {
        "ollamaAvailable": ollama.is_available(),
        "embedModel": ollama.embed_model,
        "genModel": ollama.gen_model,
        "docCount": docDB.size(),
        "docDims": docDB.dims
    }

@app.post("/doc/insert")
def doc_insert(req: InsertDocReq):
    chunks = chunk_text(req.text)
    if not chunks:
        return {"error": "Empty text"}
    
    for chunk in chunks:
        emb = ollama.embed(chunk)
        if not emb:
            return {"error": "Ollama failed to generate embedding"}
        docDB.insert(req.title, chunk, emb)
        
    return {"message": "Inserted", "chunks": len(chunks), "dims": docDB.dims}

@app.get("/doc/list")
def doc_list():
    docs = docDB.all()
    out = []
    for d in docs:
        out.append({
            "id": d["id"],
            "title": d["title"],
            "preview": d["text"][:100] + "..." if len(d["text"]) > 100 else d["text"],
            "words": len(d["text"].split())
        })
    return out

@app.delete("/doc/delete/{vid}")
def doc_delete(vid: int):
    if docDB.remove(vid):
        return {"message": "Deleted"}
    raise HTTPException(status_code=404, detail="Not found")

@app.post("/doc/ask")
def doc_ask(req: AskReq):
    q_emb = ollama.embed(req.question)
    if not q_emb:
        return {"answer": "Error: Ollama embedding failed.", "contexts": []}
    
    hits = docDB.search(q_emb, req.k)
    
    context_str = ""
    contexts_out = []
    for hit in hits:
        doc = hit["doc"]
        context_str += f"Title: {doc['title']}\nExcerpt: {doc['text']}\n---\n"
        contexts_out.append({"title": doc["title"], "text": doc["text"], "distance": hit.get("distance", 0.0)})
        
    prompt = (
        "Use the following excerpts to answer the question. If you don't know the answer based on the excerpts, say so.\n\n"
        f"{context_str}\n"
        f"Question: {req.question}\n"
        "Answer:"
    )
    
    answer = ollama.generate(prompt)
    return {"answer": answer, "contexts": contexts_out}
