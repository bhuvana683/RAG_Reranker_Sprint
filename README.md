# RAG_Reranker_Sprint




## ** README.md Example**

````
# Industrial Safety Mini-RAG Q&A

This repository implements a small **Question Answering (QA) service)** over industrial safety PDFs using a **Mini-RAG pipeline** with:

- **Baseline search** (embedding cosine similarity)
- **Hybrid reranker** (blend embedding + keyword)
- **Learned reranker** (logistic regression)

Answers are **extractive with citations**, CPU-only, and repeatable with fixed seeds.

---

## **Setup**

1. Clone repo:

```bash
git clone https://github.com/yourusername/industrial-safety-qa.git
cd industrial-safety-qa
````

2. Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\Activate.ps1 # Windows PowerShell
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Prepare data:

* Extract `industrial-safety-pdfs.zip` into `data/industrial-safety-pdfs/`
* Ensure `sources.json` exists with PDF → URL/title mapping
* Run `scripts/ingest_chunks.py` to populate `chunks.db`
* Run `scripts/build_index.py` to generate `faiss_index.bin` and `metadata.json`

---

## **Running the API**

Start FastAPI server:

```bash
python scripts/ask_api.py
```

Visit `http://127.0.0.1:8000` to see `ui.html` `http://127.0.0.1:8000/docs' to  check in Swagger ui
---

## **API Usage**

**POST /ask**

Request JSON:

```json
{
    "q": "How to safely operate a laser scanner?",
    "k": 5,
    "mode": "hybrid"
}
```

Response JSON:

```json
{
    "answer": "Short extractive answer... (Source: example.pdf)",
    "contexts": [
        {"pdf": "example.pdf", "text": "...chunk text...", "score": 0.87},
        ...
    ],
    "reranker_used": "hybrid"
}
```

---

## **Results Table for 8 Test Questions**

| Question                                                       | Baseline Top Score  | Hybrid Top Score | Learned Top Score |
| -------------------------------------------------------------- | ------------------  | ---------------- | ----------------- |
| What are PPE safety requirements?                              | 0.99                | 0.6              | 0.868             |
| How to safely operate a laser scanner?                         | 0.99                | 0.6              | 0.868             |
| Define safety functions for machinery.                         | 0.99                | 0.6              | 0.868             |
| Explain risk reduction steps for operators.                    | 0.99                | 0.6              | 0.868             |
| What are type-C standards in ISO 13849-1?                      | 0.99                | 0.6              | 0.868             |
| How to calculate performance level (PL) for a safety function? | 0.99                | 1.0              | 0.868             |
| List hazards in industrial machinery.                          | 0.99                | 0.6              | 0.868             |
| When should emergency stop be applied?                         | 0.99                | 0.6              | 0.868             |

---

## **Learnings**

1. **Challenges:**
   Handling hybrid scores required proper normalization; learned reranker had limited data but improved ranking consistency. Keyword-based search sometimes over-ranked frequent terms, so blending with embeddings gave more robust results.

2. **Observations:**

   * Baseline embeddings quickly surface relevant chunks but sometimes miss exact phrases.
   * Hybrid reranker balances semantic similarity with lexical match.
   * Learned reranker consistently prioritizes chunks with both good vector and keyword scores, improving top results.

---

## **Example curl Requests**

**Easy question:**

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
-H "Content-Type: application/json" \
-d '{"q":"What are PPE safety requirements?", "k":5, "mode":"baseline"}'
```

**Tricky question:**

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
-H "Content-Type: application/json" \
-d '{"q":"When should emergency stop be applied?", "k":5, "mode":"learned"}'
```

---

## **Notes**

* All processing is CPU-only.
* Answers are **extractive** with citations.
* Outputs are repeatable via fixed random seeds.
* API abstains if confidence < 0.1.

```

---

## ** Summary**

- Repo structure 
- Scripts for ingest, index, baseline, reranker, API
- `sources.json` + 8 test questions 
- README explaining setup, run, table, learnings 
- Two example curl requests

