
import streamlit as st
from PIL import Image

from vision import load_vision_model, generate_caption
from rag import (
    load_rag_artifacts,
    retrieve_context,
    read_document,
    chunk_text,
    build_faiss_index,
)
from llm import load_llm, generate_answer

# ------------------------------------------------------
# Page config
# ------------------------------------------------------
st.set_page_config(page_title="Multimodal RAG Assistant", layout="wide")

st.title("🧠 Multimodal AI Knowledge Assistant")
st.write("Image understanding + Document RAG + LLM reasoning")

# --------- Limits (important for free hosting) ----------
MAX_FILE_SIZE_MB = 10

# ------------------------------------------------------
# Load Models (cached)
# ------------------------------------------------------
@st.cache_resource
def load_models():
    # Vision (pipeline-based, CPU/GPU auto)
    caption_pipeline = load_vision_model()

    # Base RAG artifacts
    chunks, index, embedder = load_rag_artifacts()

    # LLM
    tokenizer, llm = load_llm()

    return (
        caption_pipeline,
        chunks,
        index,
        embedder,
        tokenizer,
        llm,
    )

(
    caption_pipeline,
    chunks,
    index,
    embedder,
    tokenizer,
    llm,
) = load_models()

# ------------------------------------------------------
# UI Tabs
# ------------------------------------------------------
tab1, tab2 = st.tabs(["🖼 Image", "📄 Document Q&A"])

# ===================== IMAGE TAB ======================
with tab1:
    st.subheader("Image Understanding")

    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "png", "jpeg"],
    )

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Generate Caption"):
            caption = generate_caption(image, caption_pipeline)
            st.success(caption)

# =================== DOCUMENT TAB =====================
with tab2:
    st.subheader("Ask from Uploaded Document")

    st.info(
        "Upload a document to ask questions. "
    "For best results, use files under 10 MB."
    )

    uploaded_doc = st.file_uploader(
        "Upload a document (PDF, TXT, or DOCX)",
        type=["pdf", "txt", "docx"],
    )

    if uploaded_doc:
        # ---- File size check ----
        file_size_mb = uploaded_doc.size / (1024 * 1024)

        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(
                f"❌ File too large ({file_size_mb:.1f} MB). "
                f"Please upload a file smaller than {MAX_FILE_SIZE_MB} MB."
            )
            st.stop()

        with st.spinner("Processing document..."):
            doc_text = read_document(uploaded_doc)
            doc_chunks = chunk_text(doc_text)
            doc_index, doc_chunks = build_faiss_index(
                doc_chunks, embedder
            )

        st.success("Document processed successfully!")

        question = st.text_input("Ask a question from the document")

        if st.button("Get Answer") and question:
            context_chunks = retrieve_context(
                question, doc_chunks, doc_index, embedder
            )
            context = " ".join(context_chunks)

            answer = generate_answer(
                question, context, tokenizer, llm
            )

            st.markdown("### ✅ Answer")
            st.write(answer)
