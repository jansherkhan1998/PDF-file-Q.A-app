import os
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
# Using LLaMA 3.3 70B default (or swap with 'openai/gpt-oss-120b' if enabled on your Groq key)
MODEL_NAME = "openai/gpt-oss-120b" 
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 5


# -----------------------------
# Load embedding model (Cached)
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

embedder = load_embedding_model()


# -----------------------------
# Extract text functions
# -----------------------------
def extract_pdf(file):
    reader = PdfReader(file)
    pages = [page.extract_text() for page in reader.pages if page.extract_text()]
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
def create_vector_database(chunk_objects):
    texts = [c["text"] for c in chunk_objects]
    embeddings = embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


# -----------------------------
# Retrieve relevant chunks (Fixed Metadata Tracking)
# -----------------------------
def retrieve_chunks(question, chunk_objects, index, top_k=TOP_K):
    question_embedding = embedder.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    k = min(top_k, len(chunk_objects))
    scores, indices = index.search(question_embedding, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            # Attach score directly to the chunk object dictionary
            item = chunk_objects[idx].copy()
            item["score"] = float(score)
            results.append(item)

    return results


# -----------------------------
# Groq LLM Completion
# -----------------------------
def generate_answer(question, retrieved_chunks, api_key):
    client = Groq(api_key=api_key)

    context = "\n\n".join(
        [
            f"Source {i + 1} ({chunk['source']}):\n{chunk['text']}"
            for i, chunk in enumerate(retrieved_chunks)
        ]
    )

    prompt = f"""You are a precise document question-answering assistant.

Answer the user's question strictly using the provided context.

Rules:
1. Use the context as the primary source of truth.
2. Do not invent information that is not supported by the context.
3. If the answer is not available in the context, clearly say:
   "I could not find this information in the uploaded documents."
4. Give a concise but useful answer.
5. Mention which source file supports your answer.

CONTEXT:
{context}

USER QUESTION:
{question}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a precise document question-answering assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_completion_tokens=1500
    )

    return response.choices[0].message.content


# -----------------------------
# Streamlit Interface
# -----------------------------
st.title("📚 RAG Document Assistant")
st.write("Upload PDF, Word, or TXT documents and ask questions using FAISS & Groq.")

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Your key is used only for the current Streamlit session."
    ) or os.environ.get("GROQ_API_KEY")

    top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=TOP_K)
    st.info(f"LLM: {MODEL_NAME}\n\nEmbeddings: {EMBEDDING_MODEL}\n\nVector DB: FAISS")


uploaded_files = st.file_uploader(
    "Upload your source documents",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    # Use session_state to prevent re-indexing on every widget re-render
    if "index" not in st.session_state or st.sidebar.button("Re-process Documents"):
        all_chunks = []
        for uploaded_file in uploaded_files:
            try:
                text = extract_text(uploaded_file)
                chunks = split_text(text)
                for chunk in chunks:
                    all_chunks.append({"text": chunk, "source": uploaded_file.name})
            except Exception as e:
                st.error(f"Could not process {uploaded_file.name}: {e}")

        if all_chunks:
            with st.spinner("Embedding documents and building FAISS index..."):
                st.session_state.all_chunks = all_chunks
                st.session_state.index = create_vector_database(all_chunks)
                st.success(f"Processed {len(uploaded_files)} document(s) into {len(all_chunks)} chunks.")

    # Execute Search and QA if index exists
    if "index" in st.session_state:
        question = st.text_input("Ask a question about your documents:")

        if question:
            if not api_key:
                st.warning("Please enter your Groq API key in the sidebar.")
            else:
                with st.spinner("Searching context & generating response..."):
                    retrieved = retrieve_chunks(
                        question,
                        st.session_state.all_chunks,
                        st.session_state.index,
                        top_k=top_k
                    )

                    try:
                        answer = generate_answer(question, retrieved, api_key)
                        st.subheader("Answer")
                        st.write(answer)

                        # Render context breakdown
                        with st.expander("🔎 Retrieved Context"):
                            for i, item in enumerate(retrieved):
                                st.markdown(
                                    f"**Chunk {i + 1} — {item['source']}** "
                                    f"(similarity score: {item['score']:.3f})"
                                )
                                st.write(item["text"])

                    except Exception as e:
                        st.error(f"Groq API Error: {e}")
else:
    st.info("Upload one or more PDF, DOCX, or TXT files to begin.")
