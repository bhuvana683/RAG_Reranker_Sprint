import json
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Absolute paths
BASE_DIR = r"D:\industrial-safety-qa"
CHUNKS_JSON = os.path.join(BASE_DIR, "chunks.json")       # Input from pdfread.py
INDEX_FILE = os.path.join(BASE_DIR, "faiss_index.bin")    # FAISS index output
METADATA_FILE = os.path.join(BASE_DIR, "metadata.json")   # Metadata output

# Load chunks from JSON
with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks from {CHUNKS_JSON}.")

# Initialize SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")  # lightweight & fast

# Create embeddings
texts = [chunk["text"] for chunk in chunks]
print("Generating embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")  # FAISS requires float32

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print(f"FAISS index created with {index.ntotal} vectors.")

# Save FAISS index
faiss.write_index(index, INDEX_FILE)
print(f"Saved FAISS index to {INDEX_FILE}")

# Save metadata for reference (mapping back to PDFs/chunks)
with open(METADATA_FILE, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)
print(f"Saved metadata to {METADATA_FILE}")
