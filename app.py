
import streamlit as st
import os
import logging
import tempfile
import ollama

from config import MODEL_NAME, EMBEDDING_MODEL
from vector_db_manager import load_vector_db_for_doc
from rag_chain_builder import create_retriever, create_chain
from langchain_ollama import ChatOllama 

logger = logging.getLogger(__name__)

def main():
    st.set_page_config(page_title="Document Assistant (Ollama RAG)", layout="wide", page_icon="📄")
    st.title("📄 Document query Parser Using RAG")
    st.markdown("Upload a PDF and ask questions about its content!")

    if "history" not in st.session_state:
        st.session_state.history = []
    if "user_input_key" not in st.session_state:
        st.session_state.user_input_key = ""
    if "uploaded_file_path" not in st.session_state:
        st.session_state.uploaded_file_path = None
    if "file_display_name" not in st.session_state:
        st.session_state.file_display_name = None
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "llm_instance" not in st.session_state:
        st.session_state.llm_instance = None


    with st.sidebar:
        st.header("Upload Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", help="Upload a PDF document to query.")

        if uploaded_file is not None:
            current_file_identifier = getattr(uploaded_file, "file_id", uploaded_file.name)

            if st.session_state.uploaded_file_path != current_file_identifier:
                logger.info(f"New file uploaded: {uploaded_file.name} (Identifier: {current_file_identifier})")
                
                if st.session_state.uploaded_file_path and os.path.exists(st.session_state.uploaded_file_path):
                    try:
                        os.remove(st.session_state.uploaded_file_path)
                        logger.info(f"Removed previous temp file: {st.session_state.uploaded_file_path}")
                    except Exception as e:
                        logger.warning(f"Could not remove old temp file {st.session_state.uploaded_file_path}: {e}")

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_file_path = tmp_file.name

                st.session_state.uploaded_file_path = temp_file_path
                st.session_state.file_display_name = uploaded_file.name

                st.session_state.history = []
                st.session_state.user_input_key = ""
                st.session_state.vector_db = None
                st.session_state.retriever = None
                st.session_state.chain = None
                st.session_state.llm_instance = None
                
                st.success(f"PDF '{uploaded_file.name}' uploaded successfully!")
                st.info("Building knowledge base from your new document. Please wait...")
            
        
        if st.session_state.uploaded_file_path and st.session_state.file_display_name:
            st.write(f"**Current Document:** `{st.session_state.file_display_name}`")
            if st.button("Load New Document", help="Click to upload a different PDF. This will clear current progress and history."):
                if st.session_state.uploaded_file_path and os.path.exists(st.session_state.uploaded_file_path):
                    try:
                        os.remove(st.session_state.uploaded_file_path)
                        logger.info(f"Cleaned up temp file for new upload: {st.session_state.uploaded_file_path}")
                    except Exception as e:
                        logger.warning(f"Error cleaning up temp file: {e}")

                st.session_state.uploaded_file_path = None
                st.session_state.file_display_name = None
                st.session_state.history = []
                st.session_state.user_input_key = ""
                st.session_state.vector_db = None
                st.session_state.retriever = None
                st.session_state.chain = None
                st.session_state.llm_instance = None
                st.experimental_rerun()

            st.markdown("---")


        st.header("⏳ Query History")
        if st.session_state.history:
            for i, (query, response) in enumerate(reversed(st.session_state.history)):
                with st.expander(f"➡️ Q: {query[:70]}..."):
                    st.markdown(f"**Question:**\n```\n{query}\n```")
                    st.markdown(f"**Response:**\n```\n{response}\n```")
    
        else:
            st.info("No queries yet. Your questions and the AI's answers will appear here.")
        
        if st.button("Clear History", help="Clear all past queries and responses from the sidebar."):
            st.session_state.history = []
            st.session_state.user_input_key = ""
            st.experimental_rerun()



    if st.session_state.uploaded_file_path:
        if st.session_state.vector_db is None:
            st.session_state.vector_db = load_vector_db_for_doc(st.session_state.uploaded_file_path)
            if st.session_state.vector_db:
                st.success("Knowledge base built successfully! You can now ask questions.")
            else:
                st.error("Failed to build knowledge base from the document. Please check the console for errors and try again.")
                return

        if st.session_state.llm_instance is None:
            try:
                st.session_state.llm_instance = ChatOllama(model=MODEL_NAME)
                logger.info(f"Initialized ChatOllama instance with model: {MODEL_NAME}")
            except Exception as e:
                logger.error(f"Failed to initialize ChatOllama instance: {e}", exc_info=True)
                st.error(f"Failed to initialize language model. Ensure Ollama is running and '{MODEL_NAME}' is pulled. Error: {e}")
                return

        if st.session_state.retriever is None:
            st.session_state.retriever = create_retriever(st.session_state.vector_db, st.session_state.llm_instance)
            if st.session_state.retriever is None:
                return

        if st.session_state.chain is None:
            st.session_state.chain = create_chain(st.session_state.retriever, st.session_state.llm_instance)
            if st.session_state.chain is None:
                return

  
        with st.form("query_form", clear_on_submit=False): 
            user_input = st.text_area("Enter your question:", value=st.session_state.user_input_key, key="main_user_input_area", height=100)
            
            col1, col2 = st.columns([1, 1]) 
            with col1:
                submitted = st.form_submit_button("Submit")
            with col2:
                reset_button = st.form_submit_button("Reset Query") 

            if submitted and user_input:
                st.session_state.user_input_key = user_input 
                with st.spinner("Loading..."):
                    try:
                        response = st.session_state.chain.invoke(input=user_input)

                        st.subheader("Assistant's Answer:")
                        st.success(f"✨ {response}")

                        st.session_state.history.append((user_input, response))
                        
                        
                    except ollama.ResponseError as e:
                        logger.error(f"Ollama generation error: {e}")
                        st.error(f"Ollama Generation Error: {e}. Please ensure the Ollama server is running and you have pulled the `{MODEL_NAME}` model (`ollama pull {MODEL_NAME}`).")
                    except Exception as e:
                        logger.error(f"An unexpected error occurred during response generation: {e}", exc_info=True)
                        st.error(f"An unexpected error occurred: {str(e)}. Check console for details.")
            elif submitted and not user_input:
                st.warning("Please enter a question before clicking 'Ask the Document!'.")
            
 
            if reset_button:
                st.session_state.user_input_key = "" 
                st.experimental_rerun() 
    else:
        st.info("⬆️ Please upload a PDF document in the sidebar to get started.")
        try:
            with st.spinner(f"Ensuring essential Ollama models are available..."):
                ollama.pull(EMBEDDING_MODEL)
                ollama.pull(MODEL_NAME)
            st.success("Ollama models confirmed! Ready for document upload.")
        except ollama.ResponseError as e:
            st.error(f"Pre-check Ollama Error: {e}. Ensure Ollama server is running and you have internet access.")
        except Exception as e:
            st.error(f"An unexpected error during model pre-check: {e}")


if __name__ == "__main__":
    main()