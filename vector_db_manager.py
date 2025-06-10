import os
import logging
import streamlit as st
import ollama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from config import EMBEDDING_MODEL, VECTOR_STORE_NAME
from document_processor import ingest_pdf, split_documents

logger = logging.getLogger(__name__)

@st.cache_resource(hash_funcs={str: lambda x: x})
def load_vector_db_for_doc(doc_temp_path):
    
    if not doc_temp_path:
        return None

    try:
        logger.info(f"Attempting to pull embedding model: {EMBEDDING_MODEL}")
        with st.spinner(f"Ensuring embedding model '{EMBEDDING_MODEL}' is available..."):
            ollama.pull(EMBEDDING_MODEL)
        logger.info(f"Embedding model '{EMBEDDING_MODEL}' ensured.")
    except ollama.ResponseError as e:
        logger.error(f"Ollama connection error or model pull issue: {e}")
        st.error(f"Ollama Error: {e}. Please ensure the Ollama server is running and you have pulled the `{EMBEDDING_MODEL}` model (`ollama pull {EMBEDDING_MODEL}`).")
        return None

    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

    logger.info(f"Building knowledge base from uploaded PDF: {doc_temp_path}")
    with st.spinner("🚀 Building knowledge base from your PDF ..."):
        data = ingest_pdf(doc_temp_path)
        if data is None:
            return None

        chunks = split_documents(data)

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name=VECTOR_STORE_NAME,
        )
    logger.info("Vector database created in memory for the uploaded document.")
    return vector_db