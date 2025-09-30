import os
from PyPDF2 import PdfReader
import json
from urllib.parse import unquote, urlparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER = os.path.join(SCRIPT_DIR, "../ispdfs")
SOURCE_JSON = os.path.join(SCRIPT_DIR, "../sources.json")
METADATA_JSON = os.path.join(SCRIPT_DIR, "../metadata.json")

# Load PDF titles from sources.json
with open(SOURCE_JSON, "r", encoding="utf-8") as f:
    sources_list = json.load(f)

sources = {}
for item in sources_list:
    pdf_name = os.path.basename(urlparse(item['url']).path)
    pdf_name = unquote(pdf_name)
    sources[pdf_name] = item['title']

metadata = []
pdf_counter = {}

for pdf_file in os.listdir(PDF_FOLDER):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        reader = PdfReader(pdf_path)
        text = ""
        page_texts = []
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                page_texts.append(page_text)
                text += page_text + "\n"

        words = text.split()
        chunk_size = 300
        base_name = pdf_file.rsplit(".pdf", 1)[0]

        # Keep track of words covered per chunk to estimate pages
        word_index = 0
        total_chunks = (len(words) + chunk_size - 1) // chunk_size

        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i+chunk_size]
            chunk_text = " ".join(chunk_words)
            pdf_counter[base_name] = pdf_counter.get(base_name, 0) + 1
            numbered_pdf_name = f"{base_name}_chunk{pdf_counter[base_name]}.pdf"

            # Estimate pages for chunk (roughly)
            approx_words_per_page = max(1, len(words)//len(page_texts))
            page_start = (i // approx_words_per_page) + 1
            page_end = min(len(page_texts), ((i + chunk_size - 1) // approx_words_per_page) + 1)

            metadata.append({
                "chunk_id": len(metadata)+1,
                "pdf": numbered_pdf_name,
                "source_pdf": pdf_file,
                "title": sources.get(pdf_file, pdf_file),
                "text": chunk_text,
                "chunk_number": pdf_counter[base_name],
                "total_chunks": total_chunks,
                "chunk_len": len(chunk_words),
                "page_start": page_start,
                "page_end": page_end,
                "is_first_paragraph": 1 if i == 0 else 0
            })

# Save metadata
with open(METADATA_JSON, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"Created metadata.json with {len(metadata)} chunks.")
