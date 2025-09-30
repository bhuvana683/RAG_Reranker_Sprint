import json
import requests
import time

# --- Config ---
API_URL = "http://127.0.0.1:8000/ask"
QUESTIONS_FILE = "questions.json"
RESULTS_FILE = "results.json"
K = 5                   # number of top chunks
MODES = ["baseline", "hybrid", "learned"]  # all modes
RETRY_DELAY = 2          # seconds to wait before retrying
MAX_RETRIES = 3

# --- Load questions ---
with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
    questions = json.load(f)

results = []

for q in questions:
    question_text = q.get("q")
    question_result = {"question": question_text, "results": {}}

    for mode in MODES:
        attempt = 0
        success = False
        while attempt < MAX_RETRIES and not success:
            try:
                resp = requests.post(API_URL, json={"q": question_text, "k": K, "mode": mode}, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                question_result["results"][mode] = data
                print(f"Done: {question_text} ({mode})")
                success = True
            except requests.exceptions.RequestException as e:
                attempt += 1
                print(f"Error with question '{question_text}' ({mode}): {e}")
                if attempt < MAX_RETRIES:
                    print(f"Retrying in {RETRY_DELAY} seconds... (Attempt {attempt}/{MAX_RETRIES})")
                    time.sleep(RETRY_DELAY)
                else:
                    question_result["results"][mode] = {"answer": None, "contexts": [], "reranker_used": mode, "error": str(e)}
                    print(f"Skipped after {MAX_RETRIES} attempts: {question_text} ({mode})")

    results.append(question_result)

# --- Save results ---
with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"All results saved to {RESULTS_FILE}")
