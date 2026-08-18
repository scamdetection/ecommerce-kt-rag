import os
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import fitz

load_dotenv()

PDF_PATH = Path("Knowledge Transfer Document – E-Commerce Order Management System.pdf")
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

st.set_page_config(page_title="E-Commerce KT RAG", page_icon="RAG", layout="wide")
st.title("E-Commerce Order Management System - RAG")
st.caption("Ask questions using the uploaded Knowledge Transfer document as the knowledge base.")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBED_MODEL)

@st.cache_resource
def build_index(pdf_path):
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    chunks = []

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if not text:
            continue

        # Small overlapping chunks for better retrieval.
        words = text.split()
        chunk_size = 180
        overlap = 35

        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end]).strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "page": page_number
                })
            if end == len(words):
                break
            start = end - overlap

    model = load_embedding_model()
    embeddings = model.encode(
        [c["text"] for c in chunks],
        normalize_embeddings=True,
        show_progress_bar=False
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks

def retrieve(question, index, chunks, k=TOP_K):
    model = load_embedding_model()
    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False
    )
    query_embedding = np.asarray(query_embedding, dtype="float32")
    scores, ids = index.search(query_embedding, min(k, len(chunks)))

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx >= 0:
            item = dict(chunks[idx])
            item["score"] = float(score)
            results.append(item)
    return results

def generate_answer(question, results):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to the .env file.")

    model_name = os.getenv("OPENAI_MODEL", "gpt-5.6")
    client = OpenAI(api_key=api_key)

    context = "\n\n".join(
        f"[Page {r['page']}]\n{r['text']}" for r in results
    )

    prompt = f"""You are a RAG assistant for the E-Commerce Order Management System.

Answer ONLY from the supplied KT document context.
If the answer is not supported by the context, say:
"I could not find that information in the KT document."

Do not invent APIs, database fields, technologies, business rules, or workflows.
Keep the answer clear and suitable for a new developer.
When useful, mention the relevant KT page number.

KT CONTEXT:
{context}

QUESTION:
{question}
"""

    response = client.responses.create(
        model=model_name,
        input=prompt
    )
    return response.output_text.strip()

try:
    index, chunks = build_index(PDF_PATH)
    st.success(f"Knowledge base loaded: {len(chunks)} chunks from the KT PDF.")
except Exception as e:
    st.error(str(e))
    st.stop()

question = st.text_input(
    "Ask a question",
    placeholder="Example: What is the order lifecycle?"
)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Retrieving relevant KT content and generating answer..."):
        try:
            results = retrieve(question, index, chunks)
            answer = generate_answer(question, results)

            st.subheader("Answer")
            st.write(answer)

            with st.expander("Retrieved KT sources"):
                for r in results:
                    st.markdown(
                        f"**Page {r['page']} | similarity {r['score']:.3f}**"
                    )
                    st.write(r["text"])

        except Exception as e:
            st.error(f"RAG error: {e}")

st.sidebar.header("RAG Pipeline")
st.sidebar.write("1. Load KT PDF")
st.sidebar.write("2. Extract page text")
st.sidebar.write("3. Split text into chunks")
st.sidebar.write("4. Create embeddings")
st.sidebar.write("5. Store vectors in FAISS")
st.sidebar.write("6. Retrieve top relevant chunks")
st.sidebar.write("7. Send context to LLM")
st.sidebar.write("8. Generate grounded answer")
