# Resume Parser (Local LLM First, Fallback to Cloud)

This app parses resumes/documents using local LLMs via Ollama by default, with automatic fallback to OpenAI and then Gemini if needed. It extracts structured info and exports validated JSON, including per-section downloads.

## What's new in this update

- Robust normalization and validation of model outputs to a standard JSON schema
- Automatic provider fallback chain: Ollama → OpenAI → Gemini → offline
- Safer string handling preventing `'dict' object has no attribute lower'` errors
- Chunk summaries and enhanced chunk metadata
- Download buttons for full JSON and per-section files (experience, education, skills, chunks, summary)

## Root cause of `'dict' object has no attribute lower'`

Some parts of the code performed string-only operations (like `.lower()`) on values that sometimes arrived as dicts or lists from varying model responses. This happened especially in provenance checks and post-processing. We fixed it by:

- Adding safe guards that coerce values to strings before calling `.lower()`
- Centralizing output normalization in `utils/normalizers.py` with `safe_extract_text`, `extract_json_block`, and `validate_and_normalize`
- Ensuring every parsing path returns a validated schema with correct types

## Standard JSON schema

```json
{
  "summary": "str or null",
  "experience": [{
    "title": "",
    "company_name": "",
    "location": "str|null",
    "start_time": "str|null",
    "end_time": "str|null",
    "summary": "str|null"
  }],
  "education": [{
    "institution": "",
    "degree": "str|null",
    "year": "str|null",
    "location": "str|null",
    "gpa": "str|null"
  }],
  "skills": ["str", ...],
  "technologies": ["str", ...],
  "tags": ["str", ...],
  "chunks": [{
    "chunk_id": 0,
    "text": "",
    "summary": "str|null",
    "page": 1,
    "start_char": 0,
    "end_char": 100
  }],
  "provider": "ollama|openai|gemini|offline",
  "parsing_method": "model|offline_basic",
  "parsing_time": "YYYY-MM-DDTHH:MM:SSZ"
}
```

## How the fallback works

1. Try Ollama (local). If it errors or returns invalid JSON, we attempt to normalize it.
2. If still invalid, fallback to OpenAI if `OPENAI_API_KEY` is present.
3. If OpenAI fails or no key, fallback to Gemini if `GEMINI_API_KEY` or `GOOGLE_API_KEY` is present.
4. If all fail, return an offline minimal structure with warnings.

## Environment variables

- `OLLAMA_BASE_URL` (default `http://localhost:11434`)
- `OPENAI_API_KEY` (optional fallback)
- `OPENAI_MODEL` (default `gpt-4o-mini`)
- `GEMINI_API_KEY` or `GOOGLE_API_KEY` (optional fallback)
- `GEMINI_MODEL` (default `gemini-1.5-flash`)

## Run locally

1. Ensure Python environment uses only packages from `requirements.txt`.
2. Start Ollama and have a model: `ollama pull llama3.2:3b`
3. Optionally set API keys in `.env` for fallbacks.
4. Run:

```bash
streamlit run app.py
```

Upload a PDF resume and click “Extract & Parse”. You’ll see provider info and download buttons for full JSON and per-section JSONs.

## Testing instructions

- Verify Ollama: `ollama list` (and optionally `ollama run <model>`)
- `streamlit run app.py`
- Upload a sample resume and click "Extract & Parse"
- Confirm UI shows `Parsed using: ollama` (or fallback) and that the six download buttons produce valid JSON
- Remove OpenAI/Gemini keys to test offline fallback behavior

# 📄 AI Document Processing Assistant

## 🎯 Overview
A simplified document processing application that uses:
- **Ollama/Llama** for local AI processing
- **Pinecone** for vector storage and retrieval  
- **Streamlit** for clean, intuitive interface

## ✨ Features
- **Simple Upload**: Just drag & drop documents (PDF, DOCX, TXT)
- **Smart Processing**: Automatic chunking and embedding generation
- **Ask Questions**: Natural language Q&A about your documents
- **Persistent Storage**: Documents stored in Pinecone cloud database
- **No API Keys Required from Users**: Configure once in .env file

## 🚀 Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup Ollama
```bash
# Install and start Ollama
ollama serve

# Pull the required model
ollama pull llama3.2:3b
```

### 3. Configure Environment
Update `.env` file with your Pinecone API key:
```env
PINECONE_API_KEY=your_actual_api_key_here
```
Get your free API key at [pinecone.io](https://pinecone.io)

### 4. Run the App
```bash
streamlit run app.py
```

## 🎮 Usage

### Upload Documents
1. Go to the **"📤 Upload Documents"** tab
2. Drag & drop or browse for your files
3. Click **"🔄 Process Documents"**

### Ask Questions  
1. Go to the **"❓ Ask Questions"** tab
2. Type your question about the uploaded documents
3. Get AI-powered answers with source references

## 🔧 Configuration

The app automatically loads settings from `.env`:
- `PINECONE_API_KEY`: Your Pinecone API key (required)
- `PINECONE_ENVIRONMENT`: Pinecone region (default: us-east-1) 
- `PINECONE_INDEX_NAME`: Index name (default: ai-documents)
- `OLLAMA_MODEL`: AI model (default: llama3.2:3b)

## 🆘 Troubleshooting

**"Pinecone API key not configured"**
- Update your `.env` file with a valid Pinecone API key

**"Ollama connection failed"**
- Make sure Ollama is running: `ollama serve`
- Check if the model is available: `ollama list`

**App won't start**
- Check all dependencies are installed: `pip install -r requirements.txt`
- Verify Python version compatibility (3.8+)

## 📁 Project Structure
```
├── app.py                 # Main Streamlit application
├── utils/
│   ├── pinecone_vector_store.py    # Pinecone integration
│   ├── document_chunker.py         # Document processing  
│   └── llm_parser.py              # Ollama integration
├── .env                   # Environment configuration
└── requirements.txt       # Python dependencies
```

## 🔒 Privacy & Benefits
- **Local AI Processing**: Your documents never leave your machine for AI processing
- **Persistent Storage**: Store documents in Pinecone for long-term access
- **No Repeated Setup**: Configure API key once, use seamlessly
- **Cost Effective**: Only pay for Pinecone storage, Ollama is free

---
🎉 **Ready to process documents with AI!**