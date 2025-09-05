# qa_minimal.py
# ------------------------------------------
# Handles translation, FAISS search, and calling Mistral.
# ------------------------------------------

import os
import faiss
import pickle
import numpy as np
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator

load_dotenv()

# Load FAISS + metadata
embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("vector_index.faiss")
with open("doc_mapping.pkl", "rb") as f:
    mapping = pickle.load(f)

# ------------------------
# Translation
# ------------------------
def translate_to_english(text: str) -> str:
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return text

def translate_back(text: str, target_lang: str) -> str:
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except Exception:
        return text

# ------------------------
# Vector search
# ------------------------
def search_docs(query: str, k: int = 3):
    q_vec = embedder.encode([query], convert_to_numpy=True)
    q_vec = q_vec / np.linalg.norm(q_vec, axis=1, keepdims=True)

    D, I = index.search(q_vec, k)
    results = []
    for idx in I[0]:
        if idx < len(mapping["chunks"]):
            results.append({
                "text": mapping["chunks"][idx],
                "meta": mapping["metadata"][idx]
            })
    return results

# ------------------------
# Call Mistral
# ------------------------
def call_mistral(prompt: str) -> str:
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('MISTRAL_API_KEY')}"
    }
    payload = {
        "model": "mistral-small-latest",
        "messages": [{"role": "user", "content": prompt}],
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=30)

    if resp.status_code >= 400:
        try:
            data = resp.json()
        except Exception:
            raise ValueError(f"❌ API error {resp.status_code}: {resp.text}")
        raise ValueError(f"❌ API error {resp.status_code}: {data}")

    data = resp.json()
    if "choices" not in data or not data["choices"]:
        raise ValueError(f"❌ Unexpected response: {data}")

    choice = data["choices"][0]
    if "message" in choice and "content" in choice["message"]:
        return choice["message"]["content"]
    if "text" in choice:
        return choice["text"]
    return str(choice)
