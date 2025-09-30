# scripts/create_sqlite.py
import sqlite3
import json
import os

CHUNKS_JSON = r"D:\industrial-safety-qa\chunks.json"

DB_FILE = "../chunks.db"

# Load chunks
with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
    chunks = json.load(f)

conn = sqlite3.connect(DB_FILE)
c = conn.cursor()

# Create table with FTS for keyword matching
c.execute("DROP TABLE IF EXISTS chunks")
c.execute("""
CREATE VIRTUAL TABLE chunks USING fts5(
    pdf, text, content='',
    tokenize='porter'
)
""")

# Insert chunks
for i, chunk in enumerate(chunks):
    c.execute("INSERT INTO chunks (pdf, text) VALUES (?, ?)", (chunk['pdf'], chunk['text']))

conn.commit()
conn.close()
print(f"SQLite FTS DB created with {len(chunks)} chunks.")
