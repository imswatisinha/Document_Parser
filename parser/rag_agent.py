# parser/rag_agent.py
from typing import List, Dict, Tuple, Callable, Optional
import textwrap

# Import your retrieval functions
from parser.rag_engine import hybrid_retrieve, retrieve_dense, tfidf_retrieve, init_tfidf

# You should already have these modules: rag_engine handles retrieval + reranking
# This module assembles the system + user prompt and calls an LLM via a pluggable llm_call function.

SYSTEM_PROMPT = """You are a precise assistant specialized in extracting factual information from technical documents.
When asked about implementation details, algorithms, or design, prioritize information that appears in sections
titled 'Methods', 'Methodology', 'Algorithm', 'Approach', 'Implementation', 'Character removal', 'Grid removal', or 'Pipeline'.
Cite the exact chunk ID and section heading used for each claim. If the document text contradicts itself, clearly say so and cite both sources.
Be concise and ONLY report facts found in the provided chunks. If you are not confident, reply with "I couldn't find that in the document"."""

# Short template for user-plus-context prompt:
USER_PROMPT_TEMPLATE = """
User question:
{user_query}

Context:
{context_chunks}

Instructions:
- Answer concisely in 1-4 sentences.
- If the question asks for "what the algorithm uses" or similar, prefer facts appearing in headings containing 'method', 'algorithm', 'approach', 'implementation', or 'pipeline'.
- After the answer, list the citation(s) exactly in the form: [doc: <chunk_id> — <heading>].
- If no relevant info is found in the context, reply: "I couldn't find that in the document."
"""

def make_chunk_display(chunk: Dict, max_chars: int = 1000) -> str:
    """
    Format a chunk dict to show inside the prompt. Truncate chunk text if too long.
    chunk is expected to have keys: 'id' (int or str), 'heading' (str or None), 'text' (str)
    """
    text = chunk.get("text", "")
    if len(text) > max_chars:
        text = text[:max_chars-3] + "..."
    heading = chunk.get("heading") or "NoHeading"
    chunk_id = str(chunk.get("id", "unknown"))
    return f"[chunk:{chunk_id} — {heading}]\n{text}\n"

def build_context_from_retrieved(retrieved: List[Tuple[str, float, int, str]],
                                 chunks_meta: List[Dict],
                                 max_chars_per_chunk: int = 900,
                                 max_context_chars: int = 3500) -> Tuple[str, List[Dict]]:
    """
    Convert retrieved tuples into the concatenated context string for the prompt.
    retrieved items from hybrid_retrieve() are expected as: (text, score, chunk_idx, source)
    chunks_meta is the list of chunk metadata dicts where index matches chunk_idx.
    Returns (context_str, used_chunks_metadata)
    """
    context_parts = []
    used = []
    chars = 0
    for text, score, idx, source in retrieved:
        if idx < 0 or idx >= len(chunks_meta):
            continue
        meta = chunks_meta[idx]
        # ensure meta contains id & heading & text
        chunk_dict = {
            "id": meta.get("chunk_id", idx),
            "heading": meta.get("heading", "NoHeading"),
            "text": meta.get("text", text)
        }
        part = make_chunk_display(chunk_dict, max_chars=max_chars_per_chunk)
        if chars + len(part) > max_context_chars:
            break
        context_parts.append(part)
        used.append(chunk_dict)
        chars += len(part)
    context_str = "\n---\n".join(context_parts)
    return context_str, used

def assemble_prompt(user_query: str, context_str: str) -> str:
    """
    Combine system prompt (sent separately in a chat API) and user prompt template.
    Return a text prompt suitable for an LLM that accepts single string prompts.
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(user_query=user_query.strip(), context_chunks=context_str)
    # Combine both with clear delimiters (if using chat API, pass SYSTEM_PROMPT in system message)
    full_prompt = SYSTEM_PROMPT + "\n\n" + user_prompt
    return full_prompt

def default_llm_call(prompt: str, max_tokens: int = 256) -> Dict:
    """
    Default dummy LLM call. Replace this with your actual LLM invocation (OLLaMA/OpenAI).
    Must return a dict with at least {'text': '...'} containing the assistant response.
    """
    # Example placeholder:
    return {"text": "LLM call not implemented. Replace default_llm_call with your provider function."}

def generate_answer(
    user_query: str,
    index,
    chunks: List[str],
    chunks_meta: List[Dict],
    embeddings,
    llm_call: Callable[[str, int], Dict] = default_llm_call,
    topk: int = 50,
    rerank_top: int = 10,
    tfidf_top: int = 20
) -> Dict:
    """
    Main entrypoint:
    - Run hybrid retrieval (dense + tfidf)
    - Build prompt favoring Method/Algorithm sections (system prompt enforces that)
    - Call LLM (pluggable via llm_call)
    - Return a dict: { 'answer': str, 'used_chunks': [...], 'prompt': str, 'llm_raw': {...} }
    ----------
    chunks_meta: list of metadata dict per chunk (same length as chunks). Expected keys:
        - 'chunk_id' (unique id)
        - 'heading' (str)
        - 'text' (str)  # optional if chunks contains text too
    """
    # 1) retrieve
    retrieved = hybrid_retrieve(user_query, index, chunks, embeddings,
                                topk=topk, rerank_top=rerank_top, tfidf_top=tfidf_top)
    # retrieved: list of tuples (text, rerank_score, chunk_idx, source)

    # 2) build context from retrieved results (truncate to fit token budget)
    context_str, used_chunks = build_context_from_retrieved(retrieved, chunks_meta,
                                                           max_chars_per_chunk=900,
                                                           max_context_chars=3500)

    # 3) assemble prompt
    prompt = assemble_prompt(user_query, context_str)

    # 4) call LLM (pluggable)
    llm_raw = llm_call(prompt, max_tokens=256)
    answer_text = llm_raw.get("text", "").strip()

    # 5) sanitize output: ensure citations exist (if LLM missed it, include used chunk list)
    if "I couldn't find that in the document" in answer_text:
        # return with used chunks for debugging
        return {"answer": answer_text, "used_chunks": used_chunks, "prompt": prompt, "llm_raw": llm_raw}

    return {"answer": answer_text, "used_chunks": used_chunks, "prompt": prompt, "llm_raw": llm_raw}
