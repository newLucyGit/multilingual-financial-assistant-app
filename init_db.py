# init_db.py
# ------------------------------------------
# Loads all PDFs from docs/, extracts text and OCR from images,
# chunks them, embeds using SentenceTransformer,
# and stores in FAISS with metadata mapping.
# ------------------------------------------

import os
import io
import faiss
import pickle
import numpy as np
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

DOCS_FOLDER = "docs"

def extract_text_and_ocr(pdf_path):
    pages = []

    # Extract text from PDF
    reader = PdfReader(pdf_path)
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append((i + 1, text, "text"))

    # Extract images for OCR
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        for img in doc[page_num].get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            ocr_text = pytesseract.image_to_string(image)
            if ocr_text.strip():
                pages.append((page_num + 1, ocr_text, "ocr"))

    return pages

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

if __name__ == "__main__":
    all_chunks = []
    metadata = []

    for file in os.listdir(DOCS_FOLDER):
        if file.endswith(".pdf"):
            file_path = os.path.join(DOCS_FOLDER, file)
            print(f"Processing {file}...")
            pages = extract_text_and_ocr(file_path)

            for page_num, page_text, source_type in pages:
                chunks = chunk_text(page_text)
                for chunk in chunks:
                    all_chunks.append(chunk)
                    metadata.append({
                        "doc": file,
                        "page": page_num,
                        "source": source_type
                    })

    print(f"✅ Created {len(all_chunks)} chunks from {len(os.listdir(DOCS_FOLDER))} PDFs")

    # Embeddings
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(all_chunks, convert_to_numpy=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # FAISS
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, "vector_index.faiss")
    with open("doc_mapping.pkl", "wb") as f:
        pickle.dump({"chunks": all_chunks, "metadata": metadata}, f)

    print("✅ FAISS index + metadata saved successfully!")
