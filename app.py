import os
import tempfile
from pathlib import Path

import streamlit as st
from groq import Groq
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.set_page_config(page_title="RAG Document Assistant", page_icon="📚", layout="wide")

# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 5


# -----------------------------
# Load embedding model
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


embedder = load_embedding_model()


# -----------------------------
# Extract text
# -----------------------------
def extract_pdf(file):
    reader = PdfReader(file)
    pages = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)

    return "\n".join(pages)


def extract_docx(file):
    document = Document(file)
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def extract_txt(file):
    return file.read().decode("utf-8", errors="ignore")


def extract_text(file):
    suffix = Path(file.name).suffix.lower()

    if suffix == ".pdf":
        return extract_pdf(file)
    elif suffix == ".docx":
        return extract_docx(file)
    elif suffix == ".txt":
        return extract_txt(file)

    return ""


# -----------------------------
# Split text into chunks
# -----------------------------
def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# -----------------------------
# Create FAISS vector database
# -----------------------------
def create_vector_database(chunks):
    embeddings = embedder.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


# -----------------------------
# Retrieve relevant chunks
# -----------------------------
def retrieve_chunks(question, chunks, index, top_k=TOP_K):
    question_embedding = embedder.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    k = min(top_k, len(chunks))
    scores, indices = index.search(question_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append((chunks[idx], float(score)))

    return results


# -----------------------------
# Groq LLM
# -----------------------------
def generate_answer(question, retrieved_chunks, api_key):
    client = Groq(api_key=api_key)

    context = "\n\n".join(
        [
            f"Source {i + 1}:\n{chunk}"
            for i, (chunk, score) in enumerate(retrieved_chunks)
        ]
    )

    prompt = f"""
You are a helpful RAG assistant.

Answer the user's question using the provided context.

Rules:
1. Use the context as the primary source of truth.
2. Do not invent information that is not supported by the context.
3. If the answer is not available in the context, clearly say:
   "I could not find this information in the uploaded documents."
4. Give a concise but useful answer.
5. When possible, mention which source/chunk supports the answer.

CONTEXT:
{context}

USER QUESTION:
{question}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are a precise document question-answering assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_completion_tokens=1500
    )

    return response.choices[0].message.content


# -----------------------------
# UI
# -----------------------------
st.title("📚 RAG Document Assistant")
st.write(
    "Upload PDF, Word, or text documents and ask questions using "
    "FAISS retrieval and Groq's open-source LLM."
)

with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Your key is used only for the current Streamlit session."
    )

    top_k = st.slider(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=10,
        value=TOP_K
    )

    st.info(
        f"LLM: {MODEL_NAME}\n\n"
        f"Embeddings: {EMBEDDING_MODEL}\n\n"
        "Vector DB: FAISS"
    )


uploaded_files = st.file_uploader(
    "Upload your source documents",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)


if uploaded_files:
    all_chunks = []

    for uploaded_file in uploaded_files:
        try:
            text = extract_text(uploaded_file)
            chunks = split_text(text)

            for chunk in chunks:
                all_chunks.append(
                    {
                        "text": chunk,
                        "source": uploaded_file.name
                    }
                )

        except Exception as e:
            st.error(f"Could not process {uploaded_file.name}: {e}")

    if all_chunks:
        texts = [item["text"] for item in all_chunks]
        index = create_vector_database(texts)

        st.success(
            f"Processed {len(uploaded_files)} document(s) "
            f"into {len(texts)} text chunks."
        )

        question = st.text_input(
            "Ask a question about your documents"
        )

        if question:
            if not api_key:
                st.warning("Please enter your Groq API key in the sidebar.")
            else:
                with st.spinner("Searching documents and generating answer..."):
                    retrieved = retrieve_chunks(
                        question,
                        texts,
                        index,
                        top_k=top_k
                    )

                    try:
                        answer = generate_answer(
                            question,
                            retrieved,
                            api_key
                        )

                        st.subheader("Answer")
                        st.write(answer)

                        with st.expander("🔎 Retrieved Context"):
                            for i, (chunk, score) in enumerate(retrieved):
                                source = all_chunks[
                                    texts.index(chunk)
                                ]["source"]

                                st.markdown(
                                    f"**Chunk {i + 1} — {source}** "
                                    f"(similarity: {score:.3f})"
                                )
                                st.write(chunk)

                    except Exception as e:
                        st.error(f"Groq API error: {e}")

    else:
        st.warning("No readable text was found in the uploaded files.")

else:
    st.info("Upload one or more PDF, DOCX, or TXT files to begin.")
