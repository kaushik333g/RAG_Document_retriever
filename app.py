import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the pretrained model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

st.title("📄 Simple RAG Document Retriever")

st.markdown("""
This application allows you to upload documents (PDF or TXT) and retrieve the most relevant text chunks based on your query.
""")

# --- Ingestion Phase ---
st.header("1. Ingestion: Upload Your Documents")

uploaded_files = st.file_uploader(
    "Choose your documents (PDF or TXT)",
    type=['pdf', 'txt'],
    accept_multiple_files=True
)

documents = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.pdf'):
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                text = "".join(page.get_text() for page in doc)
                documents.append({"name": uploaded_file.name, "text": text})
        elif uploaded_file.name.endswith('.txt'):
            text = uploaded_file.read().decode('utf-8')
            documents.append({"name": uploaded_file.name, "text": text})
    st.success(f"✅ Successfully loaded {len(documents)} documents.")

# --- Chunking and Embedding ---
chunks = []
if documents:
    st.header("2. Processing: Chunking and Embedding")
    with st.spinner("Splitting documents into chunks..."):
        for doc in documents:
            paragraphs = doc['text'].split('\\n\\n')
            for para in paragraphs:
                clean_para = para.strip().replace('\\n', ' ')
                if len(clean_para) > 30:
                    chunks.append({
                        "text": clean_para,
                        "source": doc['name']
                    })
    st.info(f"Split documents into {len(chunks)} chunks.")

    with st.spinner("Generating embeddings for each chunk... This may take a moment."):
        for chunk in chunks:
            chunk['embedding'] = model.encode(chunk['text'], convert_to_numpy=True)
    st.success("✅ Embeddings generated and stored.")

# --- Inference (Retrieval) Phase ---
st.header("3. Inference: Ask a Question")

if chunks:
    query = st.text_input("Enter your query:")

    if query:
        # Embed the query
        query_embedding = model.encode(query, convert_to_numpy=True)

        # Compute cosine similarity
        similarities = []
        for chunk in chunks:
            sim = cosine_similarity(
                [query_embedding],
                [chunk['embedding']]
            )[0][0]
            similarities.append((chunk, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top 3 chunks
        top_chunks = similarities[:3]

        st.subheader("Top 3 Relevant Chunks:")
        for i, (chunk, score) in enumerate(top_chunks):
            st.markdown(f"**Rank {i+1} | Score: {score:.4f} | Source: {chunk['source']}**")
            st.write(chunk['text'])
            st.markdown("---")
else:
    st.warning("Please upload documents to begin.")
