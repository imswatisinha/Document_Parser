# 📄 AI Document Parsing and Information Retrieval

This app parses resumes/documents using local LLMs via Ollama by default. It 
extracts structured info and exports validated JSON, including per-section 
downloads and processes and retrieves information from documents such as 
Resumes in PDF, multi-page PDFs and audio.

It prioritizes local LLMs via **Ollama** for privacy and performance,
while offering robust RAG (Retrieval-Augmented Generation) capabilities
and essential data normalization to ensure high-quality, structured
output.

## ✨ Key Features

### 🧠 AI Document Parsing

-   **Local-First Processing:** Uses a local LLM via **Ollama** (e.g.,
    `llama3.2:3b`, `phi3:mini`) to parse documents, ensuring **data
    privacy** by default.
-   **Robust Fallback Chain:** Automatically falls back to commercial
    providers (OpenAI, Gemini) if the local Ollama connection fails,
    ensuring high reliability.
-   **Structured JSON Output:** Extracts complex data (like résumés or
    reports) into a consistent, validated JSON schema.
-   **Skill Classification:** Utilizes a custom keyword classifier
    (`keyword_classifier.py`) to categorize and score skills within the
    document.

### 🔍 Retrieval-Augmented Generation (RAG)

-   **Intelligent Q&A:** Index large PDF documents using advanced
    chunking, vector embeddings, and RAG to answer complex,
    context-specific questions.
-   **Vector Store Options:** Supports both **Pinecone** (for
    persistent, cloud-based RAG) and a **Local Faiss Index** (for
    in-session, zero-setup RAG).
-   **Smart Chunking:** Employs optimized document chunking
    (`document_chunker.py`) to prepare documents for embedding,
    improving retrieval accuracy.

### 🎧 Audio & Media Processing

-   **Audio Transcription:** Transcribes uploaded audio files using a
    local **Whisper** model (via `transformers` pipeline).
-   **AI Summarization:** Generates concise summaries of the full audio
    transcript using the LLM.

### 🔒 Privacy & Benefits
- **Local AI Processing**: Your documents never leave your machine for AI processing
- **Persistent Storage**: Store documents in Pinecone for long-term access
- **No Repeated Setup**: Configure API key once, use seamlessly
- **Cost Effective**: Only pay for Pinecone storage, Ollama is free

## 🎮 Usage

### Upload Documents
1. Go to the **"📤 Upload Documents"** tab
2. Drag & drop or browse for your files
3. Click **"🔄 Process Documents"**

### Ask Questions  
1. Go to the **"❓ Ask Questions"** tab
2. Type your question about the uploaded documents
3. Get AI-powered answers with source references

## ⚙️ Setup and Installation

### Prerequisites

1.  **Python 3.8+**
2.  **Ollama:** Must be installed and running locally.
    -   Download and install from [Ollama Website](https://ollama.com/).
    -   Run the service: `ollama serve`
    -   Pull a model (e.g., `llama3.2:3b`): `ollama pull llama3.2:3b`

### Installation Steps

1.  **Clone the Repository**

``` bash
git clone [your_repo_url]
cd Document_Parser
```

2.  **Create and Activate Environment**

``` bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
.\venv\Scripts\activate  # Windows
```

3.  **Install Dependencies**

``` bash
pip install -r requirements.txt
```

4.  **Configure Environment Variables**

``` bash
cp .env.example .env
Update `.env` file with your Pinecone API key:
```env
PINECONE_API_KEY=your_actual_api_key_here
Get your free API key at [pinecone.io](https://pinecone.io)
```
```
Edit the `.env` file with your **Pinecone** and preferred **Ollama**
model settings.
```

## 📁 Project Structure
```
├── app.py                      # Main Streamlit UI and orchestration.
├── readme.md   
├── parser/
│   ├── rag_agent.py            # RAG Agent: Assembles prompt, calls LLM, and generates the final answer.
│   ├── semantic_classifier.py  # Zero-shot classification for skills and visual radar chart generation.
│   ├── chunking.py             # Core logic for smart document splitting/chunking (new implementation).
│   └── rag_engine.py           # Core RAG retrieval logic (embedding, indexing, vector search, re-ranking).
├── utils/
│   ├── audio_transcribe.py     # Transcribes audio files using a local Whisper model.
│   ├── document_chunker.py     # Fallback utility for structured document chunking (legacy logic).
│   ├── json_formatter.py       # Coerces and validates raw LLM output against a standard JSON schema.
│   ├── ollama_parser.py        # Handles synchronous/asynchronous API calls to the local Ollama service.
│   ├── ollama_singleton.py     # Manages the single, thread-safe instance of the Ollama client.
│   ├── pdf_rag.py              # High-level RAG integration and index management (Pinecone/Faiss setup).
│   ├── pdf_parser.py           # Utility for extracting raw text from PDF files.
│   ├── normalizers.py          # Low-level utilities for safe string/type coercion.
│   └── pinecone_vector_store.py# Pinecone vector database client and utility functions.
├── recommend_model.py          # Standalone script to test and recommend Ollama models.
├── .env.example                # Template for environment configuration.
└── requirements.txt            # Python dependencies.
├──.............                # Some additional metadata        
```

## Troubleshooting

**"Pinecone API key not configured"**
- Update your `.env` file with a valid Pinecone API key

**"Ollama connection failed"**
- Make sure Ollama is running: `ollama serve`
- Check if the model is available: `ollama list`

**App won't start**
- Check all dependencies are installed: `pip install -r requirements.txt`
- Verify Python version compatibility (3.8+)


## 🚀 Running the Application

``` bash
streamlit run app.py
```
