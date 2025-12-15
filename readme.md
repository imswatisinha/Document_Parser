# 📄 AI Document & Audio Processing Assistant

This application is a feature-rich platform built with **Streamlit** that utilizes local and cloud-based Large Language Models (LLMs) to perform advanced processing on various file types, including PDF documents and audio files.

It prioritizes local LLMs via **Ollama** for privacy and performance, while offering robust RAG (Retrieval-Augmented Generation) capabilities and essential data normalization to ensure high-quality, structured output.

## ✨ Key Features

### 🧠 AI Document Parsing
* **Local-First Processing:** Uses a local LLM via **Ollama** (e.g., `llama3.2:3b`, `phi3:mini`) to parse documents, ensuring **data privacy** by default.
* **Robust Fallback Chain:** Automatically falls back to commercial providers (OpenAI, Gemini) if the local Ollama connection fails, ensuring high reliability.
* **Structured JSON Output:** Extracts complex data (like résumés or reports) into a consistent, validated JSON schema.
* **Skill Classification:** Utilizes a custom keyword classifier (`keyword_classifier.py`) to categorize and score skills within the document.

### 🔍 Retrieval-Augmented Generation (RAG)
* **Intelligent Q&A:** Index large PDF documents using advanced chunking, vector embeddings, and RAG to answer complex, context-specific questions. 
* **Vector Store Options:** Supports both **Pinecone** (for persistent, cloud-based RAG) and a **Local Faiss Index** (for in-session, zero-setup RAG).
* **Smart Chunking:** Employs optimized document chunking (`document_chunker.py`) to prepare documents for embedding, improving retrieval accuracy.

### 🎧 Audio & Media Processing
* **Audio Transcription:** Transcribes uploaded audio files using a local **Whisper** model (via `transformers` pipeline).
* **AI Summarization:** Generates concise summaries of the full audio transcript using the LLM.

## ⚙️ Setup and Installation

### Prerequisites

1.  **Python 3.8+**
2.  **Ollama:** Must be installed and running locally.
    * Download and install from [Ollama Website](https://ollama.com/).
    * Run the service: `ollama serve`
    * Pull a model (e.g., `llama3.2:3b`): `ollama pull llama3.2:3b`

### Installation Steps

1.  **Clone the Repository**

    ```bash
    git clone [your_repo_url]
    cd Document_Parser
    ```

2.  **Create and Activate Environment**

    ```bash
    # Create environment (optional but recommended)
    python -m venv venv
    
    # Activate (macOS/Linux)
    source venv/bin/activate
    
    # Activate (Windows)
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**

    Copy the example file and fill in your details:

    ```bash
    cp .env.example .env
    ```

    Edit the new `.env` file with your **Pinecone** and preferred **Ollama** model settings.

## 🚀 Running the Application

Start the Streamlit web application from the root directory:

```bash
streamlit run app.py
