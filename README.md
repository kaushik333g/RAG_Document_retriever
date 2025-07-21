# Simple RAG-style Document Retriever

This project is a simple Retrieval-Augmented Generation (RAG) style document retriever built with Streamlit. It allows users to upload documents, and the system will retrieve the most relevant chunks of text based on a user's query. This implementation does not use a Large Language Model (LLM) for generation and focuses on the retrieval part of the RAG pipeline.

## How to Run

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

## SentenceTransformer Model Used

*   **Model:** `all-MiniLM-L6-v2`
*   **Reasoning:** This model was chosen because it provides a good balance between performance and size. It is a small, fast model that is well-suited for tasks like semantic search and clustering. Its small size makes it ideal for a lightweight Streamlit application, while still providing high-quality embeddings.

## Chunking Strategy

*   **Strategy:** Paragraph-based chunking.
*   **Reasoning:** The documents are split into chunks based on double newlines (`\\n\\n`), which typically separate paragraphs. This strategy was chosen because paragraphs usually represent a complete thought or idea. By keeping paragraphs intact, we can preserve the semantic context of the text, which leads to more relevant search results. Very short chunks (less than 30 characters) are discarded to avoid noise.

## Similarity Metric

*   **Metric:** Cosine Similarity
*   **Reasoning:** Cosine similarity is used to measure the similarity between the query embedding and the document chunk embeddings. It is a suitable metric for semantic retrieval because it measures the cosine of the angle between two vectors in a multi-dimensional space. This means that it is not affected by the magnitude of the vectors, only their direction. In the context of text embeddings, this allows us to find chunks that are semantically similar to the query, regardless of their length.
