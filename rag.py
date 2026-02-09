import pickle
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from docx import Document


# =========================================================
# Load base RAG artifacts (pre-built index)
# =========================================================

def load_rag_artifacts():
    """
    Loads prebuilt FAISS index, text chunks, and embedding model
    """
    # Load chunks
    with open("text_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    # Load FAISS index
    index = faiss.read_index("faiss_index.index")

    # Load embedding model
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    return chunks, index, embedder


# =========================================================
# Retrieve relevant context
# =========================================================

def retrieve_context(query, chunks, index, embedder, top_k=5):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]


# =========================================================
# Document ingestion (PDF / TXT / DOCX)
# =========================================================

def read_document(file):
    filename = file.name.lower()

    # ---- PDF ----
    if filename.endswith(".pdf"):
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
        return text

    # ---- TXT ----
    elif filename.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")

    # ---- DOCX ----
    elif filename.endswith(".docx"):
        doc = Document(file)
        return "\n".join(para.text for para in doc.paragraphs)

    else:
        return ""


# =========================================================
# Chunking + FAISS build (runtime documents)
# =========================================================

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def build_faiss_index(chunks, embedder):
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, chunks
