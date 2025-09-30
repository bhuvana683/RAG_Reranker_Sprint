import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Paths
INDEX_FILE = "./faiss_index.bin"       # FAISS index path
METADATA_FILE = "./metadata.json"      # Metadata mapping chunks to PDFs

# Load FAISS index
index = faiss.read_index(INDEX_FILE)
print("FAISS index loaded.")

# Load metadata
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)
print(f"Loaded {len(metadata)} chunks metadata.")

# Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def search_faiss(query, top_k=5):
    # Convert query to embedding
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    # Search FAISS
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        chunk = metadata[idx]
        results.append({
            "pdf": chunk["pdf"],
            "text": chunk["text"],
            "distance": dist
        })
    return results

def display_results(results):
    print("\nTop results:\n")
    for i, r in enumerate(results, 1):
        snippet = r["text"][:500].replace("\n", " ")  # first 500 chars
        print(f"{i}. PDF: {r['pdf']}\n   Distance: {r['distance']:.4f}\n   Snippet: {snippet}\n")

if __name__ == "__main__":
    print("Enter your question (or 'exit' to quit):")
    while True:
        query = input(">> ").strip()
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        if not query:
            continue
        
        results = search_faiss(query, top_k=5)
        display_results(results)
        
        # Optional: combine top chunks for a consolidated answer
        combined_text = " ".join([r["text"] for r in results])
        print("Combined context snippet (first 1000 chars):")
        print(combined_text[:1000].replace("\n", " "))
        print("\n---\n")
