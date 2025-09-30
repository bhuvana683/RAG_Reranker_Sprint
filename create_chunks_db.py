import os
import sqlite3
import json

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "chunks.db")
METADATA_FILE = os.path.join(BASE_DIR, "metadata.json")

# --- Load metadata ---
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# --- Connect to SQLite ---
conn = sqlite3.connect(DB_FILE)
c = conn.cursor()

# --- Create FTS5 table ---
# Avoid 'rank' since it's reserved; use 'rank_val' instead
c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING FTS5(pdf, text, rank_val UNINDEXED)")

# --- Insert chunks with unique numbered PDF-like names ---
pdf_chunk_counters = {}  # track chunk numbers per PDF

for i, chunk in enumerate(chunks):
    rank_val = i + 1
    pdf_name = chunk['pdf']

    # Initialize counter if first chunk for this PDF
    if pdf_name not in pdf_chunk_counters:
        pdf_chunk_counters[pdf_name] = 1
    else:
        pdf_chunk_counters[pdf_name] += 1

    # Create unique chunk name with .pdf
    numbered_pdf_name = f"{os.path.splitext(pdf_name)[0]}_chunk{pdf_chunk_counters[pdf_name]}.pdf"

    c.execute(
        "INSERT INTO chunks (pdf, text, rank_val) VALUES (?, ?, ?)",
        (numbered_pdf_name, chunk['text'], rank_val)
    )

conn.commit()
conn.close()
print(f"Database created with {len(chunks)} chunks.")
