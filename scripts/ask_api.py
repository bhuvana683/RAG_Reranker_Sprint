#scripts/ask_api.py
import os
import json
import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sklearn.linear_model import LogisticRegression
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# -----------------------------
# --- Set seeds for repeatable outputs ---
np.random.seed(42)
random.seed(42)

# --- Paths ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
FAISS_INDEX = os.path.join(PROJECT_ROOT, "faiss_index.bin")
METADATA_FILE = os.path.join(PROJECT_ROOT, "metadata.json")
DB_FILE = os.path.join(PROJECT_ROOT, "chunks.db")
UI_FILE = os.path.join(PROJECT_ROOT, "ui.html")

# --- Check files exist ---
for path in [FAISS_INDEX, METADATA_FILE, DB_FILE, UI_FILE]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")

# --- Load FAISS index & metadata ---
index = faiss.read_index(FAISS_INDEX)
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# --- Sentence Transformer model (CPU only) ---
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# --- FastAPI ---
app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory=PROJECT_ROOT), name="static")

# Serve UI
@app.get("/")
def serve_ui():
    return FileResponse(UI_FILE)

# --- Request schema ---
class AskRequest(BaseModel):
    q: str
    k: int = 5
    mode: str = "hybrid"  # baseline | hybrid | learned

# --- Helpers ---
def normalize(scores):
    scores = np.array(scores)
    if scores.size == 0 or scores.max() == scores.min():
        return np.zeros_like(scores)
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

# --- Baseline, Keyword, Hybrid, Learned ---
def baseline_search(query, k=5):
    q_vec = model.encode([query]).astype("float32")
    D, I = index.search(q_vec, k)
    scores = 1 - D[0]  # convert distance to similarity
    scores = normalize(scores)
    results = []
    for score, idx in zip(scores, I[0]):
        chunk = chunks[idx]
        results.append({"pdf": chunk['pdf'], "text": chunk['text'], "score": float(score)})
    return results

def keyword_search(query, k=5):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    safe_query = re.sub(r'[\"\'\*\?\:\[\]\(\)\~\^\&\|]', ' ', query)
    safe_query = ' '.join(safe_query.split())
    sql = f'SELECT pdf, text, rank_val FROM chunks WHERE text MATCH "{safe_query}" ORDER BY rank_val LIMIT {k}'
    try:
        c.execute(sql)
        results = [{"pdf": row[0], "text": row[1], "score": 1/row[2] if row[2] != 0 else 0} for row in c.fetchall()]
    except sqlite3.OperationalError:
        results = []
    finally:
        conn.close()
    return results

# --- Updated hybrid reranker ---
def hybrid_rerank(query, top_k=5, alpha=0.6):
    baseline_results = baseline_search(query, top_k*3)
    keyword_results = keyword_search(query, top_k*3)
    combined = {}
    for r in baseline_results:
        key = r['text'][:100]
        combined[key] = {"pdf": r['pdf'], "text": r['text'], "vector_score": r['score'], "keyword_score":0}
    for r in keyword_results:
        key = r['text'][:100]
        if key in combined:
            combined[key]['keyword_score'] = r['score']
        else:
            combined[key] = {"pdf": r['pdf'], "text": r['text'], "vector_score":0, "keyword_score":r['score']}
    vec_scores = normalize([v['vector_score'] for v in combined.values()])
    key_scores = normalize([v['keyword_score'] for v in combined.values()])
    final_results = []
    for i, (k, v) in enumerate(combined.items()):
        score = alpha * vec_scores[i] + (1-alpha) * key_scores[i]
        final_results.append({"pdf": v['pdf'], "text": v['text'], "score": score})
    return sorted(final_results, key=lambda x: x['score'], reverse=True)[:top_k]

# --- Updated learned reranker ---
def train_learned_reranker(baseline_fn, keyword_fn):
    questions = [
        "What are PPE safety requirements?",
        "How to safely operate a laser scanner?",
        "Define safety functions for machinery.",
        "Explain risk reduction steps for operators.",
        "What are type-C standards in ISO 13849-1?",
        "How to calculate performance level (PL) for a safety function?",
        "List hazards in industrial machinery.",
        "When should emergency stop be applied?"
    ]
    features, labels = [], []
    for q in questions:
        baseline_results = baseline_fn(q, 5)
        keyword_results = keyword_fn(q, 5)
        combined = {}
        for r in baseline_results:
            key = r['text'][:100]
            combined[key] = {"vector_score": r['score'], "keyword_score": 0}
        for r in keyword_results:
            key = r['text'][:100]
            if key in combined:
                combined[key]['keyword_score'] = r['score']
            else:
                combined[key] = {"vector_score": 0, "keyword_score": r['score']}
        for v in combined.values():
            features.append([v['vector_score'], v['keyword_score']])
            labels.append(1 if v['vector_score'] > 0 else 0)  # keep top scores same
    clf = LogisticRegression(class_weight='balanced', random_state=42)
    clf.fit(np.array(features), np.array(labels))
    return clf

clf = train_learned_reranker(baseline_search, keyword_search)

def learned_rerank(query, top_k=5):
    baseline_results = baseline_search(query, top_k*3)
    keyword_results = keyword_search(query, top_k*3)
    combined, feat, texts = {}, [], []
    for r in baseline_results:
        key = r['text'][:100]
        combined[key] = {"pdf": r['pdf'], "text": r['text'], "vector_score": r['score'], "keyword_score":0}
    for r in keyword_results:
        key = r['text'][:100]
        if key in combined:
            combined[key]['keyword_score'] = r['score']
        else:
            combined[key] = {"pdf": r['pdf'], "text": r['text'], "vector_score":0, "keyword_score":r['score']}
    for k, v in combined.items():
        feat.append([v['vector_score'], v['keyword_score']])
        texts.append(v)
    feat = np.array(feat)
    if len(feat) == 0:
        return []
    pred_scores = clf.predict_proba(feat)[:,1]
    sorted_chunks = [texts[i] for i in np.argsort(-pred_scores)]
    for i, c in enumerate(sorted_chunks):
        c['score'] = pred_scores[i]
    return sorted_chunks[:top_k]

# --- Extractive answer with citation + threshold ---
ANSWER_THRESHOLD = 0.1
def extract_answer_with_citation(results, max_chunks=2):
    snippets = []
    for r in results[:max_chunks]:
        text = r['text'].replace("\n", " ").strip()
        snippet = text[:200] + ("..." if len(text) > 200 else "")
        snippets.append(f"{snippet} (Source: {r['pdf']})")
    return " ".join(snippets)

# --- API endpoint ---
@app.post("/ask")
def ask(req: AskRequest):
    try:
        if req.mode == "baseline":
            results = baseline_search(req.q, req.k)
        elif req.mode == "hybrid":
            results = hybrid_rerank(req.q, req.k)
        elif req.mode == "learned":
            results = learned_rerank(req.q, req.k)
        else:
            raise HTTPException(status_code=400, detail="Invalid mode")

        if not results or results[0]['score'] < ANSWER_THRESHOLD:
            return {
                "answer": None,
                "contexts": [],
                "reranker_used": req.mode,
                "message": "No confident answer found. Abstaining."
            }

        answer = extract_answer_with_citation(results)
        return {
            "answer": answer,
            "contexts": results,
            "reranker_used": req.mode
        }

    except Exception as e:
        print(f"[ERROR] Question failed: {req.q}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Generate results table for all 8 questions ---
def generate_results_table():
    questions = [
        "What are PPE safety requirements?",
        "How to safely operate a laser scanner?",
        "Define safety functions for machinery.",
        "Explain risk reduction steps for operators.",
        "What are type-C standards in ISO 13849-1?",
        "How to calculate performance level (PL) for a safety function?",
        "List hazards in industrial machinery.",
        "When should emergency stop be applied?"
    ]
    table = []
    for q in questions:
        baseline = baseline_search(q, 5)
        hybrid = hybrid_rerank(q, 5)
        learned = learned_rerank(q, 5)
        table.append({
            "question": q,
            "baseline_top_score": baseline[0]['score'] if baseline else None,
            "hybrid_top_score": hybrid[0]['score'] if hybrid else None,
            "learned_top_score": learned[0]['score'] if learned else None
        })
    return table

# --- Example usage ---
if __name__ == "_ ,kjnbv vbhjkolp[]';lkmnb _main__":
    result_table = generate_results_table()
    for row in result_table:
        print(row)