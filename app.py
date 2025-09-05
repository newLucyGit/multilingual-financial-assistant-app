# app.py
# ------------------------------------------
# Streamlit app: input → translation → retrieval → Mistral → translation back.
# ------------------------------------------

import streamlit as st
from dotenv import load_dotenv
from qa_minimal import translate_to_english, translate_back, call_mistral, search_docs

load_dotenv()

st.set_page_config(page_title="Financial Assistant", layout="centered")
st.title("📊 Financial Document Assistant")
st.markdown("Ask questions across **financial reports, earnings transcripts, and investor documents**.")

query = st.text_input("Ask your question (English or Hindi):")

if query:
    # Step 1: Translate to English if needed
    query_en = translate_to_english(query)

    # Step 2: Retrieve docs
    docs = search_docs(query_en, k=3)
    context = "\n\n".join([f"From {d['meta']['doc']} (p.{d['meta']['page']}, {d['meta']['source']}): {d['text']}" for d in docs])

    # Step 3: Prompt
    final_prompt = f"""You are a specialized **Financial & Business Assistant** with expertise in analyzing corporate financial reports, investor communications, earnings transcripts, and audit statements.

IMPORTANT RULES:
1. Use ONLY the information given in the context. Do not rely on external knowledge. 
2. If the context does not provide enough information, clearly say: 
   "I cannot provide a definitive answer based on the available financial documents."
3. If multiple companies or documents are mentioned, make it clear which document you are referencing.
4. Be precise, professional, and structured in your response.
5. If numerical data is available (e.g., revenue, EBITDA, margins), present it clearly. 
6. Do not speculate beyond what is explicitly stated in the context.

Context (from financial documents):
{context}

User Question: {query_en}

Answer:
"""

    with st.spinner("Analyzing documents..."):
        answer_en = call_mistral(final_prompt)

    # Step 4: Translate back if needed
    if query != query_en:
        answer_final = translate_back(answer_en, "hi")
    else:
        answer_final = answer_en

    st.subheader("💬 Answer:")
    st.write(answer_final)

    with st.expander("📚 Sources used"):
        for d in docs:
            st.markdown(f"- {d['meta']['doc']} (p.{d['meta']['page']}, {d['meta']['source']})")
