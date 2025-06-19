# Local RAG Document Assistant with Streamlit and Ollama

A powerful and user-friendly web application that allows you to upload any PDF document and get answers to your questions based *only* on its content. This project leverages Retrieval Augmented Generation (RAG) using local Large Language Models (LLMs) powered by Ollama and built with LangChain and Streamlit.

## Features

* **Document Upload:** Easily upload any PDF document directly through the Streamlit interface.
* **Local LLM Integration:** Utilizes Ollama to run powerful LLMs (e.g., Llama 3.2) entirely on your local machine, ensuring privacy and reducing API costs.
* **Retrieval Augmented Generation (RAG):** Answers your questions by first retrieving relevant information from your uploaded document, then using the LLM to synthesize a concise and accurate response based *only* on the retrieved context.
* **Interactive UI:** A user-friendly Streamlit interface for seamless interaction, including:
    * Real-time processing feedback.
    * Query history display in the sidebar.
    * Automatic input clearing for intuitive next questions.
* **Modular Codebase:** Organized into separate Python modules for clarity, maintainability, and reusability.
## File Structure: 
```
ollama_rag_app/
├── app.py
├── config.py
├── document_processor.py
├── vector_db_manager.py
├── rag_chain_builder.py
├── requirements.txt
└── README.md
```

## Getting Started

Follow these steps to set up and run the application on your local machine.

### Prerequisites

* **Git:** For cloning the repository.
* **Python 3.8+:** The application is built with Python.
* **Ollama:** A local LLM runtime.
    * Download and install Ollama from [ollama.com](https://ollama.com/).
    * Ensure the Ollama server is running (usually starts automatically after installation, or run `ollama serve` in your terminal).
      

### Installation

1.  **Clone the Repository:**
    ```
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Python Dependencies:**
    Create a `requirements.txt` file in your project root with the following content:
    ```
    streamlit
    langchain
    langchain-community
    langchain-text-splitters
    langchain-ollama
    pypdf
    unstructured
    chromadb
    numpy
    ollama # The Python client for Ollama
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Pull Ollama Models:**
    The application uses specific Ollama models. It will attempt to pull them if they are missing, but it's good practice to pull them manually beforehand. Open a terminal and run:
    ```bash
    ollama pull llama3.2:3b
    ollama pull nomic-embed-text
    ```
    You can verify Ollama is running and models are available by trying a `curl` command directly, for example:
    ```bash
    curl http://localhost:11434/api/generate -d '{
      "model": "llama3.2",
      "prompt": "What color is the sky at different times of the day? Respond using JSON",
      "format": "json",
      "stream": false
    }'
    ```

Once all prerequisites and installations are complete, run the Streamlit application from your project root directory:

```bash
streamlit run app.py
