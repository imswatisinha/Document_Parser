# utils/pdf_rag.py
import os
import io
import math
import re
from typing import List, Dict, Tuple

import pdfplumber
import numpy as np

# Embeddings
from sentence_transformers import SentenceTransformer

# Transformers pipelines for summarization & generation (small models)
from transformers import pipeline

# Pinecone optional
try:
    import pinecone
    PINECONE_AVAILABLE = True
except Exception:
    PINECONE_AVAILABLE = False

# Faiss optional local fallback
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

# --- Configurable defaults ---
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = 384
SUMMARIZER_MODEL = os.environ.get("SUMMARIZER_MODEL", "sshleifer/distilbart-cnn-12-6")
GEN_MODEL = os.environ.get("GEN_MODEL", "google/flan-t5-small")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX", "pdf-docs")
PINECONE_NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "pdfparser")
BATCH_UPSERT = int(os.environ.get("BATCH_UPSERT", 64))

# Lazy-loaded globals
_embed_model = None
_summarizer = None
_generator = None
_local_faiss_index = None

# --- Model getters ---
def get_embedder():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model

def get_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline("summarization", model=SUMMARIZER_MODEL, device=-1)
    return _summarizer

def get_generator():
    global _generator
    if _generator is None:
        _generator = pipeline("text2text-generation", model=GEN_MODEL, device=-1, truncation=True)
    return _generator

# --- PDF extraction & chunking ---
def extract_text_and_tables_from_pdf_bytes(pdf_bytes: bytes) -> List[Dict]:
    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, p in enumerate(pdf.pages, start=1):
            text = (p.extract_text() or "")
            tables_texts = []
            try:
                tables = p.extract_tables()
                for t in tables:
                    rows = ["\t".join([cell or "" for cell in row]) for row in t]
                    tables_texts.append("\n".join(rows))
            except Exception:
                tables_texts = []
            pages.append({"page": i, "text": text, "tables": tables_texts})
    return pages

def chunk_pages(pages: List[Dict], chunk_size: int = 3500, overlap: int = 1000) -> List[Dict]:
    """
    Chunk pages by character count instead of page numbers.
    Default: 3500 characters per chunk with 1000 character overlap.
    """
    # First, combine all pages into a single text with page markers
    full_text_parts = []
    page_markers = []  # Track where each page starts in the full text
    
    for p in pages:
        current_pos = len("".join(full_text_parts))
        page_markers.append({'page': p['page'], 'start_pos': current_pos})
        
        full_text_parts.append(f"[Page {p['page']}]\n")
        if p['text']:
            full_text_parts.append(p['text'])
            full_text_parts.append("\n")
        for t in p.get("tables", []):
            full_text_parts.append("[Table]\n")
            full_text_parts.append(t)
            full_text_parts.append("\n")
    
    full_text = "".join(full_text_parts)
    
    # Create chunks based on character count
    chunks = []
    chunk_id = 0
    start = 0
    
    while start < len(full_text):
        end = start + chunk_size
        chunk_text = full_text[start:end]
        
        # Find which pages this chunk spans
        start_page = 1
        end_page = len(pages)
        
        for i, marker in enumerate(page_markers):
            if marker['start_pos'] <= start:
                start_page = marker['page']
            if marker['start_pos'] < end:
                end_page = marker['page']
        
        chunk_id += 1
        chunks.append({
            "id": f"chunk_{chunk_id}_p{start_page}-{end_page}",
            "start_page": start_page,
            "end_page": end_page,
            "content": chunk_text.strip(),
            "char_start": start,
            "char_end": end
        })
        
        # Move to next chunk with overlap
        start = start + chunk_size - overlap
    
    return chunks

# --- Embedding & index helpers ---
def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_embedder()
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return emb

def init_pinecone(api_key: str, environment: str = None):
    if not PINECONE_AVAILABLE:
        raise RuntimeError("Pinecone client not installed.")
    pinecone.init(api_key=api_key, environment=environment)
    if PINECONE_INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(PINECONE_INDEX_NAME, dimension=EMBED_DIM, metric="cosine")
    return pinecone.Index(PINECONE_INDEX_NAME)

def upsert_chunks_to_pinecone(index, chunks: List[Dict], namespace: str = PINECONE_NAMESPACE, store_content_in_metadata: bool = False):
    texts = [c["content"] for c in chunks]
    ids = [c["id"] for c in chunks]
    vectors = embed_texts(texts)
    to_upsert = []
    for uid, vec, c in zip(ids, vectors, chunks):
        metadata = {"start_page": c["start_page"], "end_page": c["end_page"]}
        if store_content_in_metadata:
            metadata["content"] = c["content"]
        to_upsert.append((uid, vec.tolist(), metadata))
    for i in range(0, len(to_upsert), BATCH_UPSERT):
        batch = to_upsert[i:i+BATCH_UPSERT]
        index.upsert(vectors=batch, namespace=namespace)

# --- Local FAISS fallback (simple) ---
class LocalFaissIndex:
    def __init__(self, dim):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadatas = []

    def add(self, vectors: np.ndarray, metadatas: List[Dict]):
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.metadatas.extend(metadatas)

    def query(self, qvec: np.ndarray, top_k=5):
        faiss.normalize_L2(qvec)
        distances, indices = self.index.search(qvec, top_k)
        results = []
        for idx_list, dist_list in zip(indices, distances):
            for idx, score in zip(idx_list, dist_list):
                md = self.metadatas[idx] if idx < len(self.metadatas) else {}
                results.append({"metadata": md, "score": float(score), "idx": int(idx)})
        return results

def build_local_faiss(chunks: List[Dict]):
    global _local_faiss_index
    if not FAISS_AVAILABLE:
        raise RuntimeError("faiss-cpu not installed.")
    texts = [c["content"] for c in chunks]
    metadatas = [{"id": c["id"], "start_page": c["start_page"], "end_page": c["end_page"], "content": c["content"]} for c in chunks]
    vectors = embed_texts(texts)
    dim = vectors.shape[1]
    _local_faiss_index = LocalFaissIndex(dim)
    _local_faiss_index.add(vectors, metadatas)
    return _local_faiss_index

def pinecone_query(index, query_text: str, top_k: int = 5, namespace: str = PINECONE_NAMESPACE):
    qvec = embed_texts([query_text])[0].tolist()
    res = index.query(vector=qvec, top_k=top_k, include_metadata=True, namespace=namespace)
    matches = []
    for m in res["matches"]:
        matches.append({"id": m["id"], "score": m["score"], "metadata": m.get("metadata", {})})
    return matches

# --- Summarization (map-reduce) ---
def summarize_chunks(chunks: List[Dict], max_chunk_chars: int = 1800, audio = False) -> str:
    summarizer = get_summarizer()

    # summarization for audio data
    if audio:
        try:
            out = summarizer(chunks[0]["audio_transcript"], max_length = 300, min_length = 50, do_sample=False, truncation = True)
            res = out[0]["summary_text"]
            return res
        except Exception as e:
            return None
    
    chunk_texts = [c["content"] for c in chunks if c.get("content", "").strip()]
    
    if not chunk_texts:
        return "No content available to summarize."
    
    chunk_summaries = []
    for txt in chunk_texts:
        truncated = txt[:max_chunk_chars]
        if not truncated.strip():
            continue
        try:
            out = summarizer(truncated, max_length=150, min_length=30, do_sample=False)
            if out and isinstance(out, list) and len(out) > 0 and "summary_text" in out[0]:
                chunk_summaries.append(out[0]["summary_text"])
        except Exception as e:
            # Skip chunks that fail to summarize
            continue
    
    if not chunk_summaries:
        return "Unable to generate summary from the provided content."
    
    concatenated = "\n\n".join(chunk_summaries)
    try:
        final = summarizer(concatenated, max_length=300, min_length=80, do_sample=False)
        if final and isinstance(final, list) and len(final) > 0 and "summary_text" in final[0]:
            return final[0]["summary_text"]
        else:
            # Return concatenated summaries if final summarization fails
            return concatenated[:500] + "..."
    except Exception as e:
        # Return concatenated summaries if final summarization fails
        return concatenated[:500] + "..."

# --- RAG QA ---
def rag_answer_with_retrieval_pinecone(index, question: str, top_k: int = 5, namespace: str = PINECONE_NAMESPACE, chunks_mapping: List[Dict] = None) -> Dict:
    generator = get_generator()
    matches = pinecone_query(index, question, top_k=top_k, namespace=namespace)
    contexts = []
    sources = []
    # If content stored in metadata, use it; otherwise use chunks_mapping to resolve ids
    for m in matches:
        meta = m.get("metadata", {})
        sources.append({"id": m.get("id"), "score": m.get("score"), "metadata": meta})
        if "content" in meta and meta["content"]:
            contexts.append(f"Source (pages {meta.get('start_page')}-{meta.get('end_page')}):\n{meta['content']}")
        elif chunks_mapping:
            # find by id
            found = next((c for c in chunks_mapping if c["id"] == m.get("id")), None)
            if found:
                contexts.append(f"Source (pages {found.get('start_page')}-{found.get('end_page')}):\n{found.get('content')}")
    prompt_parts = []
    if contexts:
        for i, c in enumerate(contexts, start=1):
            prompt_parts.append(f"Context chunk {i}:\n{c}\n")
    prompt_parts.append(f"Question: {question}\nAnswer concisely and cite page ranges of sources if known.")
    prompt = "\n\n".join(prompt_parts)
    gen_out = generator(prompt, max_length=512, do_sample=False)
    answer = gen_out[0].get("generated_text") or gen_out[0].get("text") or ""
    return {"answer": answer, "sources": sources}

def rag_answer_with_local_faiss(local_index: LocalFaissIndex, question: str, top_k: int = 5) -> Dict:
    generator = get_generator()
    qvec = embed_texts([question])
    matches = local_index.query(qvec, top_k=top_k)
    contexts = []
    sources = []
    for m in matches:
        md = m.get("metadata", {})
        sources.append({"metadata": md, "score": m.get("score")})
        if md.get("content"):
            contexts.append(f"Source (pages {md.get('start_page')}-{md.get('end_page')}):\n{md.get('content')}")
    prompt = "\n\n".join(contexts + [f"Question: {question}\nAnswer concisely and cite page ranges of sources if known."])
    gen_out = generator(prompt, max_length=512, do_sample=False)
    answer = gen_out[0].get("generated_text") or gen_out[0].get("text") or ""
    return {"answer": answer, "sources": sources}

# --- Convenience pipeline to parse & index ---
def process_and_index_pdf(pdf_bytes: bytes, doc_id: str = "doc", use_pinecone: bool = True, pinecone_api_key: str = None, pine_env: str = None, store_content_in_metadata: bool = False):
    """
    Process and index PDF with fixed chunking: 3500 characters per chunk, 1000 character overlap.
    """
    pages = extract_text_and_tables_from_pdf_bytes(pdf_bytes)
    chunks = chunk_pages(pages)  # Uses default 3500 char chunks with 1000 char overlap
    if use_pinecone:
        if not pinecone_api_key:
            raise ValueError("pinecone_api_key required when use_pinecone=True")
        idx = init_pinecone(pinecone_api_key, environment=pine_env)
        upsert_chunks_to_pinecone(idx, chunks, namespace=PINECONE_NAMESPACE, store_content_in_metadata=store_content_in_metadata)
        return {"index_type": "pinecone", "index": idx, "chunks": chunks, "pages_count": len(pages)}
    else:
        idx = build_local_faiss(chunks)
        return {"index_type": "faiss", "index": idx, "chunks": chunks, "pages_count": len(pages)}
