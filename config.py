import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_NAME = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "uploaded_doc_rag"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300