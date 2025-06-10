import os
import logging
import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)
# PDF is ingested based on the provisions made in STREAMLIT 
def ingest_pdf(file_path):
 
    if os.path.exists(file_path):
        try:
            loader = UnstructuredPDFLoader(file_path=file_path)
            data = loader.load()
            logger.info(f"PDF loaded successfully from : {file_path}.")
            return data
        except Exception as e:
            logger.error(f"Error loading PDF from {file_path}: {e}")
            st.error(f"Error loading PDF: {e}")
            return None
    else:
        logger.error(f"PDF file not found at path: {file_path}")
        st.error(f"PDF file not found at: `{file_path}`.")
        return None

#Docx ChunKer
def split_documents(documents):
   
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Documents split into {len(chunks)} chunks.")
    return chunks