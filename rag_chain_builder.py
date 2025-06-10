

import logging
import streamlit as st
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

from config import MODEL_NAME

logger = logging.getLogger(__name__)

def create_retriever(vector_db, llm_model_instance=None):
    
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from
a vector database. By generating multiple perspectives on the user question, your
goal is to help the user overcome some of the limitations of the distance-based
similarity search. Provide these alternative questions separated by newlines.
Original question: {question}""",
    )

    llm = llm_model_instance
    try:
        if not isinstance(llm, ChatOllama):
            llm = ChatOllama(model=MODEL_NAME)
            logger.info("Initialized ChatOllama for retriever (or re-initialized due to type mismatch).")

        retriever = MultiQueryRetriever.from_llm(
            vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
        )
        logger.info("Retriever created.")
        return retriever
    except Exception as e:
        logger.error(f"Failed to create retriever: {e}", exc_info=True)
        st.error(f" Error creating retriever. Ensure LLM model '{MODEL_NAME}' is pulled. Error: {e}")
        return None


def create_chain(retriever, llm_model_instance=None):

    template = """You are a helpful document assistant. Answer the question thoroughly and concisely based ONLY on the following context. If the answer cannot be found in the context, politely state that you don't have enough information from the provided document.

Context:
{context}

Question: {question}
"""


    prompt = ChatPromptTemplate.from_template(template)

    llm = llm_model_instance
    try:
        if not isinstance(llm, ChatOllama):
            llm = ChatOllama(model=MODEL_NAME)
            logger.info("Initialized ChatOllama for chain (or re-initialized due to type mismatch).")

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        logger.info("Chain created with preserved syntax.")
        return chain
    except Exception as e:
        logger.error(f"Failed to create chain: {e}", exc_info=True)
        st.error(f"Error creating chain. Ensure LLM model '{MODEL_NAME}' is pulled. Error: {e}")
        return None