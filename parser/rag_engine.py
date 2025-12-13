from typing import List
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import faiss

# Load models once globally (fast!)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def boosted_score(text: str, base_score: float, heading: str = "") -> float:
    """
    Boost score if the chunk is from a section likely relevant for algorithm-related questions.
    heading: optional heading/title of the chunk if available.
    """
    boost_words = [
        "method", "methods", "algorithm", "approach", "implementation",
        "ocr", "crnn", "text recognition", "pipeline", "architecture"
    ]

    context = (heading + " " + text).lower()

    # If the chunk mentions any boosting keyword, increase the score
    if any(w in context for w in boost_words):
        return base_score * 1.40   # 40% boost — you can tune this
    return base_score


def build_index(chunks: List[str]):
    embeddings = embed_model.encode(
        chunks, 
        batch_size=32, 
        convert_to_numpy=True, 
        normalize_embeddings=True
    )
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity via inner product (normalized)
    index.add(embeddings)

    return index, embeddings


def retrieve_and_rerank(query: str, index, chunks: List[str], embeddings, topk=50, rerank_top=10):
    # Dense retrieval
    q_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q_emb, topk)

    dense_candidates = [chunks[i] for i in I[0]]

    # Rerank top candidates using cross-encoder
    pairs = [[query, c] for c in dense_candidates[:rerank_top]]
    scores = reranker.predict(pairs)

    # Sort by cross-encoder score
    reranked = sorted(
        zip(dense_candidates[:rerank_top], scores),
        key=lambda x: -x[1]
    )

    return reranked
# parser/rag_engine.py
"""
RAG engine: dense indexing (FAISS) + TF-IDF lexical retrieval + cross-encoder reranking.
Drop this file into parser/rag_engine.py and import the functions you need from other parts of your app.

Functions provided:
- build_index(chunks)
- save_index(path, index, embeddings, chunks, tfidf_vectorizer, tfidf_matrix)
- load_index(path) -> (index, embeddings, chunks, vectorizer, tfidf_matrix)
- init_tfidf(chunks)
- tfidf_retrieve(query, topn=20)
- retrieve_dense(query, topk=50)
- retrieve_and_rerank(query, index, chunks, embeddings, topk=50, rerank_top=10)
- hybrid_retrieve(query, index, chunks, embeddings, topk=50, rerank_top=10, tfidf_top=20)
"""

from typing import List, Tuple, Optional, Any
import os
import pickle
import logging

import numpy as np
import faiss

from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- Configuration / Globals ----
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Load heavy models once (module import)
logging.info(f"Loading embedder {EMBED_MODEL_NAME} and reranker {RERANKER_MODEL_NAME}...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
reranker = CrossEncoder(RERANKER_MODEL_NAME)

# TF-IDF globals (initialized when init_tfidf is called)
_tfidf_vectorizer: Optional[TfidfVectorizer] = None
_tfidf_matrix: Optional[Any] = None  # scipy sparse matrix

# Keep last-indexed chunks (useful for saving / returning metadata)
_last_index_chunks: Optional[List[str]] = None


# ---- Indexing / Persistence ----
def build_index(chunks: List[str]) -> Tuple[faiss.Index, np.ndarray]:
    """
    Build FAISS index from text chunks.
    Returns (faiss_index, embeddings_array)
    embeddings_array is shape (n_chunks, dim), normalized.
    """
    global _last_index_chunks
    _last_index_chunks = chunks

    # Compute embeddings (normalized for cosine via inner product)
    embeddings = embed_model.encode(
        chunks,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine similarity
    index.add(embeddings.astype(np.float32))
    logging.info(f"Built FAISS IndexFlatIP with dim={dim}, n={embeddings.shape[0]}")
    return index, embeddings


def save_index(path: str, index: faiss.Index, embeddings: np.ndarray, chunks: List[str],
               tfidf_vectorizer: Optional[TfidfVectorizer] = None, tfidf_matrix: Optional[Any] = None) -> None:
    """
    Save FAISS index, embeddings, chunks, and TF-IDF artifacts to `path` directory.
    """
    os.makedirs(path, exist_ok=True)
    faiss.write_index(index, os.path.join(path, "faiss.index"))
    np.save(os.path.join(path, "embeddings.npy"), embeddings.astype(np.float32))
    with open(os.path.join(path, "chunks.pkl"), "wb") as fh:
        pickle.dump(chunks, fh)
    if tfidf_vectorizer is not None and tfidf_matrix is not None:
        with open(os.path.join(path, "tfidf_vectorizer.pkl"), "wb") as fh:
            pickle.dump(tfidf_vectorizer, fh)
        with open(os.path.join(path, "tfidf_matrix.pkl"), "wb") as fh:
            pickle.dump(tfidf_matrix, fh)
    logging.info(f"Saved index and artifacts to {path}")


def load_index(path: str) -> Tuple[faiss.Index, np.ndarray, List[str], Optional[TfidfVectorizer], Optional[Any]]:
    """
    Load index artifacts from `path` directory.
    Returns (index, embeddings, chunks, tfidf_vectorizer or None, tfidf_matrix or None).
    """
    faiss_index_path = os.path.join(path, "faiss.index")
    embeddings_path = os.path.join(path, "embeddings.npy")
    chunks_path = os.path.join(path, "chunks.pkl")

    if not os.path.exists(faiss_index_path):
        raise FileNotFoundError(f"{faiss_index_path} not found")

    index = faiss.read_index(faiss_index_path)
    embeddings = np.load(embeddings_path)
    with open(chunks_path, "rb") as fh:
        chunks = pickle.load(fh)

    tfidf_vectorizer = None
    tfidf_matrix = None
    vpath = os.path.join(path, "tfidf_vectorizer.pkl")
    mpath = os.path.join(path, "tfidf_matrix.pkl")
    if os.path.exists(vpath) and os.path.exists(mpath):
        with open(vpath, "rb") as fh:
            tfidf_vectorizer = pickle.load(fh)
        with open(mpath, "rb") as fh:
            tfidf_matrix = pickle.load(fh)

    # store last chunks for convenience
    global _last_index_chunks, _tfidf_vectorizer, _tfidf_matrix
    _last_index_chunks = chunks
    _tfidf_vectorizer = tfidf_vectorizer
    _tfidf_matrix = tfidf_matrix

    logging.info(f"Loaded FAISS index from {path} (n_chunks={len(chunks)})")
    return index, embeddings, chunks, tfidf_vectorizer, tfidf_matrix


# ---- TF-IDF lexical retrieval ----
def init_tfidf(chunks: List[str]) -> Any:
    """
    Initialize TF-IDF vectorizer and compute matrix for a list of chunks.
    Must be called once after (re)indexing.
    Stores vectorizer and matrix in module globals.
    Returns the tfidf_matrix.
    """
    global _tfidf_vectorizer, _tfidf_matrix
    _tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    _tfidf_matrix = _tfidf_vectorizer.fit_transform(chunks)
    logging.info("Initialized TF-IDF vectorizer and matrix")
    return _tfidf_matrix


def tfidf_retrieve(query: str, chunks: List[str], topn: int = 20) -> List[Tuple[str, float, int]]:
    """
    Retrieve topn chunks with TF-IDF lexical similarity.
    Returns list of tuples: (chunk_text, score, chunk_index)
    Requires init_tfidf(chunks) to have been called.
    """
    if _tfidf_vectorizer is None or _tfidf_matrix is None:
        raise RuntimeError("TF-IDF not initialized. Call init_tfidf(chunks) after indexing.")

    q_vec = _tfidf_vectorizer.transform([query])
    sims = cosine_similarity(q_vec, _tfidf_matrix)[0]  # dense 1 x n
    top_idx = np.argsort(-sims)[:topn]
    results = [(chunks[i], float(sims[i]), int(i)) for i in top_idx]
    return results


# ---- Dense retrieval + rerank ----
def retrieve_dense(query: str, index: faiss.Index, chunks: List[str],
                   embeddings: np.ndarray, topk: int = 50) -> List[Tuple[str, float, int]]:
    """
    Retrieve topk dense candidates from FAISS.
    Returns list of tuples: (chunk_text, score, chunk_index), where score is cosine (inner product).
    """
    if index.ntotal == 0:
        return []

    q_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    D, I = index.search(q_emb, topk)
    scores = D[0].tolist()
    indices = I[0].tolist()
    candidates = []
    for s, idx in zip(scores, indices):
        if idx < 0 or idx >= len(chunks):
            continue
        candidates.append((chunks[idx], float(s), int(idx)))
    return candidates


def retrieve_and_rerank(query: str, index: faiss.Index, chunks: List[str], embeddings: np.ndarray,
                        topk: int = 50, rerank_top: int = 10) -> List[Tuple[str, float, int]]:
    """
    Dense retrieval followed by cross-encoder rerank.
    Returns reranked list of tuples (chunk, rerank_score, chunk_index), sorted descending by rerank_score.
    """
    dense = retrieve_dense(query, index, chunks, embeddings, topk=topk)
    if not dense:
        return []

    # take top candidates to rerank
    cand_texts = [c[0] for c in dense[:rerank_top]]
    pairs = [[query, txt] for txt in cand_texts]
    rerank_scores = reranker.predict(pairs)  # numpy array
    reranked = sorted(
        [(cand_texts[i], float(rerank_scores[i]), int(dense[i][2])) for i in range(len(cand_texts))],
        key=lambda x: -x[1]
    )
    return reranked


# ---- Hybrid retrieval ----
def hybrid_retrieve(query: str, index: faiss.Index, chunks: List[str], embeddings: np.ndarray,
                    topk: int = 50, rerank_top: int = 10, tfidf_top: int = 20) -> List[Tuple[str, float, int, str]]:
    """
    Hybrid retrieval that unions dense top-k and TF-IDF top-k, then reranks with cross-encoder.
    Returns list of tuples: (chunk_text, rerank_score, chunk_index, source) where source in {"dense","tfidf"}.
    """
    # Dense candidates
    dense = retrieve_dense(query, index, chunks, embeddings, topk=topk)
    dense = dense[:topk]

    # TF-IDF candidates (if initialized)
    tfidf_candidates = []
    try:
        tfidf_candidates = tfidf_retrieve(query, chunks, topn=tfidf_top)
    except RuntimeError:
        # TF-IDF not initialized — ignore
        tfidf_candidates = []

    # Union while preserving provenance & de-duping
    candidate_map = {}  # idx -> (text, provenance_score, source)
    for text, score, idx in dense:
        if idx not in candidate_map:
            candidate_map[idx] = (text, float(score), "dense")
    for text, score, idx in tfidf_candidates:
        if idx not in candidate_map:
            candidate_map[idx] = (text, float(score), "tfidf")

    # Build list and limit to rerank_top * 4 for efficiency (you can tune)
    merged = list(candidate_map.items())  # list of (idx, (text, score, source))
    # sort by the raw retrieval score desc (dense inner product is in second pos, tfidf score also there)
    merged_sorted = sorted(merged, key=lambda x: -x[1][1])
    merged_sorted = merged_sorted[: max(rerank_top * 4, rerank_top + 10)]

    # Prepare rerank pairs (query, chunk_text)
    cand_texts = [entry[1][0] for entry in merged_sorted]
    cand_idxs = [entry[0] for entry in merged_sorted]
    pairs = [[query, t] for t in cand_texts]
    if len(pairs) == 0:
        return []

    rerank_scores = reranker.predict(pairs)
    reranked = sorted(
        [
            (cand_texts[i], float(rerank_scores[i]), int(cand_idxs[i]), merged_sorted[i][1][2])
            for i in range(len(cand_texts))
        ],
        key=lambda x: -x[1]
    )

    # return top rerank_top results
    return reranked[:rerank_top]


# ---- Utility helpers ----
def encode_query(query: str) -> np.ndarray:
    """Return normalized embedding for a query (float32)"""
    q_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    return q_emb


# ---- If run as script, quick smoke test (requires small test corpus) ----
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # quick local smoke test (very small)
    sample_chunks = [
        "This section describes the algorithm. The algorithm uses a CRNN-based model for text recognition.",
        "Author affiliations and metadata: School of Electronic and Computer Engineering, Georgia Institute of Technology.",
        "We apply perspective transforms and Gaussian noise for augmentation.",
        "We trained a DnCNN model for grid removal."
    ]

    # Build index + TF-IDF
    idx, embs = build_index(sample_chunks)
    init_tfidf(sample_chunks)

    q = "what does algorithm use?"
    print("Dense top:")
    print(retrieve_dense(q, idx, sample_chunks, embs, topk=4))
    print("\nTF-IDF top:")
    print(tfidf_retrieve(q, sample_chunks, topn=4))
    print("\nReranked:")
    print(retrieve_and_rerank(q, idx, sample_chunks, embs, topk=4, rerank_top=3))
    print("\nHybrid:")
    print(hybrid_retrieve(q, idx, sample_chunks, embs, topk=4, rerank_top=3, tfidf_top=4))
