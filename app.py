# app.py (patched)
import os
import asyncio
import time
import re
import json
import io
import streamlit as st
from dotenv import load_dotenv

# Local imports (assume these modules exist as in your repo)
from parser.semantic_classifier import generate_radar_chart, zero_shot_skill_scores
from utils.pdf_parser import extract_text_from_pdf
from utils.json_formatter import safe_json_dumps

# NEW: Use the smart chunker and RAG engine / agent we designed
from parser.chunking import smart_chunk_text
from parser.rag_engine import (
    build_index,
    init_tfidf,
    save_index,
    load_index,
    retrieve_dense,
    retrieve_and_rerank,
    hybrid_retrieve
)
from parser.rag_agent import generate_answer

# replaced integrate_ollama_parser imports with singleton-aware loader
from utils.document_chunker import DocumentChunker  # keep for fallback if needed
from utils.pinecone_vector_store import setup_pinecone_interface, display_pinecone_stats

# NEW: pdf_rag helpers for large-doc tab (make sure utils/pdf_rag.py exists)
from utils.pdf_rag import (
    process_and_index_pdf,
    summarize_chunks,
    rag_answer_with_retrieval_pinecone,
    rag_answer_with_local_faiss,
)

# Page config
st.set_page_config(
    page_title="Document Parser LLM",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    load_dotenv()
except Exception:
    pass

# Create uploads dir
os.makedirs("uploads", exist_ok=True)

from parser.semantic_classifier import load_skill_labels_from_csv
CANDIDATE_LABELS = load_skill_labels_from_csv()


@st.cache_data(show_spinner=False)
def load_css():
    """Load and cache CSS to avoid re-reading on every rerun.""" 
    css_path = "assets/style.css"
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            return f"<style>{f.read()}</style>"
    return ""

# -------------------------
# Ollama singleton loader
# -------------------------
def _import_ollama_singleton():
    """
    Try several common symbol names/locations for an Ollama singleton or factory in your utils.
    Returns the singleton instance or None.
    """
    candidates = [
        ("utils.ollama_parser", "Ollama_Singleton"),
        ("utils.ollama_parser", "ollama_Singleton"),
        ("utils.ollama_parser", "OllamaSingleton"),
        ("utils.ollama_parser", "ollama_singleton"),
        ("utils.ollama_parser", "OllamaParser"),
        ("utils.ollama_singleton", "Ollama_Singleton"),
        ("utils.ollama_singleton", "OllamaSingleton"),
        ("utils.ollama_singleton", "ollama_singleton"),
        ("utils", "ollama_Singleton"),
        ("utils", "ollama_singleton"),
    ]

    for module_name, symbol in candidates:
        try:
            module = __import__(module_name, fromlist=[symbol])
            obj = getattr(module, symbol, None)
            if obj is None:
                continue

            # If obj is a callable factory or class, try to get instance
            try:
                # If it's already an instance (has typical attrs), return directly
                if not callable(obj) and hasattr(obj, "is_connected"):
                    return obj

                # If class has an 'instance' or 'get_instance' method, use it
                if hasattr(obj, "instance") and callable(getattr(obj, "instance")):
                    return obj.instance()
                if hasattr(obj, "get_instance") and callable(getattr(obj, "get_instance")):
                    return obj.get_instance()

                # If callable, attempt to call/instantiate
                if callable(obj):
                    try:
                        inst = obj()
                        return inst
                    except Exception:
                        continue
            except Exception:
                continue
        except Exception:
            continue
    return None

def _ensure_ollama_in_session():
    """Load the singleton and set session state keys like older integrate_ollama_parser did."""
    if st.session_state.get("ollama_parser") is None:
        singleton = _import_ollama_singleton()
        if singleton:
            st.session_state["ollama_parser"] = singleton
            # ensure a few common session keys exist
            st.session_state.setdefault("ollama_model", getattr(singleton, "selected_model", None))
            # store available models if present
            available = getattr(singleton, "available_models", None)
            if available is not None:
                st.session_state["ollama_available_models"] = available
            return True
        else:
            return False
    else:
        return True

# -------------------------
# Small helpers
# -------------------------
def _extract_response_from_stream_chunk(raw_chunk: str) -> str:
    """
    Extract only the 'response' field value from a streaming JSON chunk produced by Ollama.
    Returns an empty string if nothing meaningful found.
    """
    if not raw_chunk:
        return ""
    # If chunk is bytes-like, ensure str
    try:
        if isinstance(raw_chunk, (bytes, bytearray)):
            raw_chunk = raw_chunk.decode("utf-8", errors="ignore")
    except Exception:
        raw_chunk = str(raw_chunk)

    # Try direct JSON parse
    try:
        data = json.loads(raw_chunk)
        if isinstance(data, dict) and "response" in data:
            return data.get("response") or ""
    except Exception:
        # partial JSON may arrive; try regex fallback
        m = re.search(r'"response"\s*:\s*"(.*?)"(?:,|\}|$)', raw_chunk, re.DOTALL)
        if m:
            # unescape simple escaped quotes
            return m.group(1).replace('\\"', '"')
    # sometimes chunk contains only plain text (not JSON) - return it if it's not noise
    cleaned = raw_chunk.strip()
    if len(cleaned) > 2 and not re.fullmatch(r'["\':,\[\]\{\}\s]+', cleaned):
        return cleaned
    return ""

def _is_noise_chunk(text: str) -> bool:
    """Return True for tiny / punctuation-only chunks we want to ignore in the preview."""
    if not text:
        return True
    t = text.strip()
    if len(t) <= 2 and re.fullmatch(r'["\':,\[\]\{\}\s]+', t):
        return True
    return False

def _show_streaming_output(generator, progress_bar, output_box, timeout=300):
    """
    Consume async generator returned by OllamaParser.parse_resume_stream
    and update UI with a clean preview. Returns the accumulated (clean) string.

    This version extracts only the 'response' text from streaming JSON to avoid printing raw protocol lines.
    """
    accumulated = ""
    start = time.time()

    try:
        async def _consume():
            nonlocal accumulated
            last_preview = ""
            async for chunk in generator:
                # chunk may be bytes or str
                try:
                    if isinstance(chunk, (bytes, bytearray)):
                        chunk = chunk.decode("utf-8", errors="ignore")
                except Exception:
                    chunk = str(chunk)

                # Extract the model's textual response from the protocol chunk
                resp = _extract_response_from_stream_chunk(chunk)
                if _is_noise_chunk(resp):
                    # update progress indicator but don't flood UI
                    elapsed = time.time() - start
                    progress_bar.progress(min(0.95, elapsed / max(5.0, timeout)))
                    continue

                accumulated += resp
                # Build a cleaned preview: collapse long runs of whitespace
                preview_raw = accumulated[-10000:]
                preview = re.sub(r'\s{2,}', ' ', preview_raw).strip()
                if preview != last_preview:
                    # show a short snippet so UI stays responsive
                    output_box.code(preview[:4000])
                    last_preview = preview

                elapsed = time.time() - start
                progress_bar.progress(min(0.95, elapsed / max(5.0, timeout)))
            return accumulated

        return asyncio.run(_consume())
    except Exception as e:
        # user-facing friendly message
        st.warning("Live streaming unavailable — using stable blocking parse instead.")
        return accumulated

# -------------------------
# Main (with persistent radio for tabs)
# -------------------------
def main():
    # Load cached CSS
    css = load_css()
    if css:
        st.markdown(css, unsafe_allow_html=True)

    st.title("🤖 AI Document Parser")
    st.markdown("Upload documents, get insights, summaries, and answers instantly")

    # persistent tab selector using session_state, so it survives reruns (e.g., file upload)
    if "main_tab" not in st.session_state:
        st.session_state["main_tab"] = "Analyze Resume"
    
    st.subheader("What would you like to do?")
    
    # Create tab-like buttons using columns
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 Analyze Resume", use_container_width=True, type="primary" if st.session_state["main_tab"] == "Analyze Resume" else "secondary"):
            st.session_state["main_tab"] = "Analyze Resume"
            st.rerun()
    
    with col2:
        if st.button("📊 Analyze Large PDF", use_container_width=True, type="primary" if st.session_state["main_tab"] == "Analyze Large PDF" else "secondary"):
            st.session_state["main_tab"] = "Analyze Large PDF"
            st.rerun()
    
    current_tab = st.session_state["main_tab"]

    # Sidebar: Ollama setup + vector DB selection + chunking options
    with st.sidebar:
        st.header("🦙 Ollama Local AI")
        ollama_ready = _ensure_ollama_in_session()
        if ollama_ready:
            st.success("✅ Ollama loaded into session")
            parser_obj = st.session_state.get("ollama_parser")

            # show available model selection stored by singleton
            available_models = st.session_state.get("ollama_available_models") or getattr(parser_obj, "available_models", []) or []
            if available_models:
                st.write("Available models:")
                for m in available_models:
                    st.write(f"• {m}")
            
            # Initialize OllamaRAGRetriever if not already in session
            if "ollama_rag" not in st.session_state:
                from utils.rag_retriever import OllamaRAGRetriever
                st.session_state.ollama_rag = OllamaRAGRetriever()
        else:
            print("Ollama singleton not found in utils. Please ensure OllamaParser or similar is defined.")
            

        st.header("🗄️ Vector Database")
        vector_db_option = "🧠 In-Memory (Local)"
        st.info("Using in-memory vector storage (FAISS)")
        pinecone_store = None

        st.header("📄 Document Processing")
        chunking_strategy = st.selectbox("Chunking strategy:", ["sections", "sliding_window", "pages"])
        if chunking_strategy == "sliding_window":
            # Use sentence-based UI but DocumentChunker may accept character sizes; adapt accordingly
            chunk_size_sentences = st.slider("Chunk size (sentences):", 3, 12, 6)
            overlap_sentences = st.slider("Overlap (sentences):", 0, 4, 1)
        else:
            chunk_size_sentences = None
            overlap_sentences = None

       

    # -------------------------
    #   Resume Parser view (keeps your existing flow)   
    # -------------------------
    if current_tab == "Analyze Resume":
        col1, col2 = st.columns([1, 2])

        with col1:
            st.header("📁 Upload Document")
            uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
            if uploaded_file is not None:
                st.success(f"✅ Uploaded: {uploaded_file.name}")
                file_size_kb = uploaded_file.size / 1024
                st.write(f"**Size:** {file_size_kb:.1f} KB")

                if st.button("🚀 Extract & Parse Document"):
                    # Check Ollama readiness
                    parser_obj = st.session_state.get("ollama_parser")
                    if not parser_obj or (not getattr(parser_obj, "is_connected", False) and not getattr(parser_obj, "is_ready", False)):
                        print("Ollama not ready. Please start Ollama and install a model.")
                        return

                    # Extract text
                    with st.spinner("Extracting text from PDF..."):
                        combined_text, pages = extract_text_from_pdf(uploaded_file)
                        if not combined_text:
                            print("Failed to extract text from PDF")
                            return
                        st.session_state["raw_text"] = combined_text
                        st.session_state["pages"] = pages
                        st.success(f"Extracted {len(combined_text)} characters from {len(pages)} page(s)")

                    # Chunk document using smart_chunk_text (new)
                    with st.spinner("Chunking document with smart_chunk_text..."):
                        try:
                            # smart_chunk_text returns list[str] — convert to chunk metadata dicts expected elsewhere
                            raw_chunks = smart_chunk_text(combined_text, max_tokens=500, overlap_tokens=100)
                            chunks_meta = []
                            for i, c in enumerate(raw_chunks):
                                chunks_meta.append({
                                    "chunk_id": f"{uploaded_file.name}_chunk_{i}",
                                    "heading": "Unknown",
                                    "text": c
                                })
                            # store both raw text chunks (list of strings) and metadata
                            st.session_state["chunks"] = raw_chunks
                            st.session_state["chunks_meta"] = chunks_meta
                            st.success(f"Created {len(raw_chunks)} chunks (smart_chunk_text)")
                        except Exception as e:
                            # Silently fallback to DocumentChunker if smart_chunk_text fails
                            try:
                                chunker = DocumentChunker()
                                fallback = chunker.chunk_by_sections(combined_text)
                                st.session_state["chunks"] = [c.get("content", "") for c in fallback]
                                st.session_state["chunks_meta"] = [{
                                    "chunk_id": c.get("id", f"chunk_{i}"),
                                    "heading": c.get("title", "Unknown"),
                                    "text": c.get("content", "")
                                } for i, c in enumerate(fallback)]
                                st.success(f"Created {len(fallback)} chunks (DocumentChunker)")
                            except Exception as e2:
                                print(f"Document chunking failed. Please check the document format.")
                                return

                    # If document small, use blocking sync parse (fast)
                    use_streaming = False

                    parsed_data = None
                    # UI elements for streaming feedback
                    progress_bar = st.empty()
                    output_box = st.empty()
                    try:
                        if use_streaming:
                            progress_widget = st.progress(0.0)
                            output_box.info("Processing ")
                            # Kick off async streaming generator from OllamaParser
                            try:
                                generator = parser_obj.parse_resume_stream(combined_text, model=None)
                                accumulated = _show_streaming_output(generator, progress_widget, output_box, timeout=300)
                                # Attempt to extract JSON from accumulated string
                                parsed = None
                                if hasattr(parser_obj, "_extract_json_from_text"):
                                    parsed = parser_obj._extract_json_from_text(accumulated)
                                else:
                                    # naive extraction
                                    m = re.search(r'\{.*\}', accumulated, re.DOTALL)
                                    if m:
                                        try:
                                            parsed = json.loads(m.group())
                                        except Exception:
                                            parsed = None
                                if parsed:
                                    parsed_data = parsed
                                    st.success("✅ Parsing complete (stream).")
                                else:
                                    st.info("Could not find structured JSON in streamed output — trying stable parse.")
                            except Exception as e:
                                st.info("Live streaming failed — using stable blocking parse.")
                                st.warning(str(e))

                        if parsed_data is None:
                            # Blocking sync parse fallback (safe)
                            with st.spinner("Running blocking parse (this may take a moment)..."):
                                parsed_result = parser_obj.parse_resume_sync(combined_text, model=None)
                                parsed_data = parsed_result if isinstance(parsed_result, dict) else {}
                                st.success("✅ Parsing complete (sync).")

                        # Normalize structure if the parsed_data is raw dict coming from OllamaParser or service
                        st.session_state["parsed_data"] = parsed_data

                        # After parsing, set up vector store or in-memory RAG
                        if vector_db_option.startswith("🌲") and st.session_state.get("pinecone_store"):
                            pinecone_store = st.session_state.get("pinecone_store")
                            with st.spinner("Storing chunks in Pinecone..."):
                                success = pinecone_store.add_documents(st.session_state["chunks_meta"], f"doc_{uploaded_file.name}")
                                if success:
                                    qa_ok = pinecone_store.setup_qa_chain(parser_obj.selected_model or "llama3.2:3b")
                                    if qa_ok:
                                        st.session_state["pinecone_qa_ready"] = True
                                        st.success("✅ Pinecone storage & QA ready")
                                    else:
                                        st.warning("⚠️ Pinecone QA chain setup failed")
                                else:
                                    print("❌ Failed to upsert to Pinecone")
                        else:
                            # in-memory RAG: build FAISS index using rag_engine
                            try:
                                with st.spinner("Building in-memory FAISS index..."):
                                    chunks = st.session_state.get("chunks", [])
                                    # build index and TF-IDF (store in session_state so it persists)
                                    index, embeddings = build_index(chunks)
                                    st.session_state["in_memory_index"] = index
                                    st.session_state["in_memory_embeddings"] = embeddings
                                    init_tfidf(chunks)  # populates module-level tfidf in rag_engine
                                    # keep chunks_meta for agent prompt assembly
                                    st.session_state["chunks_meta"] = st.session_state.get("chunks_meta", [{
                                        "chunk_id": f"chunk_{i}", "heading": "Unknown", "text": chunks[i]
                                    } for i in range(len(chunks))])
                                    st.session_state["in_memory_rag_ready"] = True
                                    st.success("✅ In-memory RAG ready (FAISS index built)")
                            except Exception as e:
                                st.warning(f"Failed to set up in-memory RAG: {e}")
                    except Exception as e:
                        print("Parsing failed — see details below.")
                        st.exception(e)

        # Right column: results and QA
        with col2:
            st.header("📊 Document Analysis")
            if "parsed_data" in st.session_state:
                parsed_struct = st.session_state["parsed_data"]

                # Try friendly formatted display first (shared helper)
                displayed = False
                try:
                    from utils.llm_parser import format_document_display
                    format_document_display(parsed_struct)
                    displayed = True
                except Exception:
                    displayed = False

                # Friendly header
                try:
                    provider = parsed_struct.get("ai_provider") or parsed_struct.get("_source") or parsed_struct.get("provider", "AI")
                    model = parsed_struct.get("model") or parsed_struct.get("_model", st.session_state.get("ollama_model", "unknown"))
                    st.info(f"Parsed by: {provider} • Model: {model}")
                except Exception:
                    pass

                # Chunk stats
                chunks = st.session_state.get("chunks", [])
                st.subheader("📄 Document Chunks")
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Total Chunks", len(chunks))
                avg_len = sum(len(c) for c in chunks) // len(chunks) if chunks else 0
                col_b.metric("Avg Chunk Length", f"{avg_len} chars")
                col_c.metric("Strategy", chunking_strategy.title())

                # with st.expander("🔍 View Document Chunks"):
                #     for i, chunk_text in enumerate(chunks):
                #         meta = st.session_state.get("chunks_meta", [{}])[i] if st.session_state.get("chunks_meta") else {}
                #         st.write(f"**Chunk {i+1}** ({meta.get('heading','text')})")
                #         if meta.get('page'):
                #             st.caption(f"Page: {meta['page']}")
                #         st.text_area(f"chunk_{i+1}", value=chunk_text[:1500], height=140, disabled=True)

                # Raw text view
                with st.expander("📝 Raw Extracted Text"):
                    pages = st.session_state.get("pages", [])
                    if pages:
                        idx = st.selectbox("Select page", list(range(1, len(pages) + 1)))
                        st.text_area(f"Page {idx}", value=pages[idx - 1], height=300, disabled=True)
                    else:
                        st.text_area("Combined text", value=st.session_state.get("raw_text", ""), height=300, disabled=True)

                # Raw JSON accordion
                with st.expander("🔧 View Raw JSON Data", expanded=False):
                    # Download button at the top
                    try:
                        full_json = safe_json_dumps(parsed_struct)
                        st.download_button("💾 Download JSON", data=full_json, file_name="parsed_resume.json", mime="application/json", use_container_width=True)
                    except Exception:
                        pass
                    
                    st.markdown("---")
                    
                    # Display JSON
                    try:
                        st.json(parsed_struct)
                    except Exception:
                        try:
                            full_json = safe_json_dumps(parsed_struct)
                            st.code(full_json, language="json")
                        except Exception:
                            st.text(str(parsed_struct))

                # If formatted display didn't run, show a quick glance
                if not displayed:
                    personal = parsed_struct.get("personal_info", {}) or parsed_struct.get("personal", {})
                    if personal:
                        name = personal.get("name") or personal.get("full_name")
                        email = personal.get("email")
                        if name:
                            st.subheader(f"👤 {name}")
                        if email:
                            st.write(f"📧 {email}")

                # Semantic classification
                # Semantic classification (robust)
                st.subheader("📊 Semantic Skill Classification")
                if st.session_state.get("raw_text"):
                    with st.spinner("Running zero-shot classification..."):
                        try:
                            # get labels from session or CSV
                            try:
                                from parser.semantic_classifier import load_skill_labels_from_csv
                                labels_to_use = st.session_state.get("candidate_labels") or load_skill_labels_from_csv()
                            except Exception:
                                labels_to_use = st.session_state.get("candidate_labels", [])

                            if not labels_to_use:
                                st.warning("No candidate labels available. Please add `assets/skills.csv` or upload labels in the sidebar.")
                            else:
                                scores = zero_shot_skill_scores(st.session_state["raw_text"], labels_to_use)
                                if not scores or not scores.get("labels"):
                                    st.warning("Classifier returned no labels — check model availability or loader.")
                                else:
                                    skill_pairs = list(zip(scores.get("labels", []), scores.get("scores", [])))
                                    skill_pairs.sort(key=lambda x: x[1], reverse=True)
                                    st.table([{"Skill": s, "Confidence": f"{p:.2f}"} for s, p in skill_pairs[:50]])
                                    if skill_pairs:
                                        labels, vals = zip(*skill_pairs)
                                        chart_path = generate_radar_chart(list(labels)[:12], list(vals)[:12])
                                        st.image(chart_path, caption="Skill confidence radar")
                        except Exception as e:
                            pass
                            # st.warning(f"Skill classification failed: {e}")
                else:
                    st.info("Run parsing to enable skill classification.")


                # Q&A
                qa_ready = st.session_state.get("pinecone_qa_ready", False) or st.session_state.get("in_memory_rag_ready", False)
                if qa_ready:
                    st.divider()
                    st.header("💬 Document Q&A")
                    if st.session_state.get("pinecone_qa_ready", False):
                        st.info("Using Pinecone + LangChain")
                    else:
                        # st.info("Using RAG (FAISS + TF-IDF + Cross-Encoder)")
                        pass

                    question = st.text_area("Ask a question about the document:", height=80)
                    if st.button("Get AI Analysis"):
                        if not question:
                            st.warning("Please enter a question.")
                        else:
                            with st.spinner("Generating answer..."):
                                try:
                                    if st.session_state.get("pinecone_qa_ready", False):
                                        res = st.session_state["pinecone_store"].ask_question(question)
                                        if "error" in res:
                                            print(res["error"])
                                        else:
                                            st.write(res.get("answer", ""))
                                            # show sources if present
                                            if res.get("source_documents"):
                                                with st.expander("Source documents"):
                                                    for sd in res["source_documents"]:
                                                        st.text_area("Source", value=sd.get("content","")[:800], height=120, disabled=True)
                                    else:
                                        # in-memory RAG -> use rag_agent to assemble prompt & call LLM
                                        if st.session_state.get("in_memory_rag_ready"):
                                            index = st.session_state.get("in_memory_index")
                                            embeddings = st.session_state.get("in_memory_embeddings")
                                            chunks = st.session_state.get("chunks", [])
                                            chunks_meta = st.session_state.get("chunks_meta", [{
                                                "chunk_id": f"chunk_{i}", "heading": "Unknown", "text": chunks[i]
                                            } for i in range(len(chunks))])

                                            # call rag_agent.generate_answer (pluggable llm)
                                            # We will try to use the ollama singleton as llm_call if available, otherwise rely on default that needs implementing
                                            ollama = st.session_state.get("ollama_parser")
                                            def _ollama_llm_call(prompt: str, max_tokens: int = 256):
                                                # expect singleton to have a simple method like `generate(prompt, max_tokens)` - adapt as needed
                                                try:
                                                    if hasattr(ollama, "sync_generate"):
                                                        out = ollama.sync_generate(prompt = prompt, model = st.session_state.get("ollama_model"))
                                                        return {"text": out.get("response", "").strip()}
                                                    elif hasattr(ollama, "generate") and callable(ollama.generate):
                                                        out = ollama.generate(prompt, max_tokens=max_tokens)
                                                        # adapt return shape
                                                        if isinstance(out, dict) and out.get("text"):
                                                            return {"text": out.get("text")}
                                                        if isinstance(out, str):
                                                            return {"text": out}
                                                        # try to extract simple field
                                                        return {"text": str(out)}
                                                    elif hasattr(ollama, "chat") and callable(ollama.chat):
                                                        out = ollama.chat(prompt)
                                                        return {"text": out.get("response", str(out)) if isinstance(out, dict) else str(out)}
                                                    else:
                                                        # Adapter not present — return placeholder
                                                        return {"text": "OLLaMA adapter not implemented. Provide a llm_call adapter to rag_agent.generate_answer."}
                                                except Exception as e:
                                                    return {"text": f"LLM call failed: {e}"}

                                            result = generate_answer(
                                                user_query=question,
                                                index=index,
                                                chunks=chunks,
                                                chunks_meta=chunks_meta,
                                                embeddings=embeddings,
                                                llm_call=_ollama_llm_call,
                                                topk=50,
                                                rerank_top=10,
                                                tfidf_top=20
                                            )

                                            # display answer + used chunks
                                            st.subheader("Answer")
                                            st.write(result.get("answer", "No answer returned."))
                                            if result.get("used_chunks"):
                                                with st.expander("Used chunks (for citations & debugging)"):
                                                    for ch in result["used_chunks"]:
                                                        st.markdown(f"**{ch.get('id')}** — {ch.get('heading')}")
                                                        st.text_area(f"chunk_{ch.get('id')}", value=ch.get("text","")[:1200], height=140, disabled=True)
                                        else:
                                            pass
                                            # print("In-memory RAG not initialized.")
                                except Exception as e:
                                    print(f"QA failed: {e}")
            else:
                st.info("Upload a PDF to parse and analyze.")
                with st.expander("💡 Vector DB options"):
                    st.write("🧠 FAISS: in-memory, fast, ephemeral.")
                    # st.write("🌲 Pinecone: persistent, multi-document, requires keys.")

    # -------------------------
    # PDF PARSER view (large-document flow)
    # -------------------------
    else:
        st.header("📚 PDF Parser — Large Documents")
        st.markdown("Perfect for analyzing large documents! Upload your PDF and get an AI-powered summary or ask questions about the content.")

        col_left, col_right = st.columns([1, 2])

        with col_left:

            file_type = st.radio("Select type of file:", ["PDF Document", "Audio File"], key = "file_type")

            if file_type == "PDF Document":
                uploaded_pdf = st.file_uploader("Upload a PDF (for large-doc parsing)", type=["pdf"], key="pdf_rag_uploader")
                if uploaded_pdf:
                    st.success(f"Selected: {uploaded_pdf.name}")
                    bytes_data = uploaded_pdf.read()
                    # quick page count check
                    import pdfplumber
                    try:
                        with pdfplumber.open(io.BytesIO(bytes_data)) as pdf:
                            detected_pages = len(pdf.pages)
                    except Exception:
                        detected_pages = None
                    if detected_pages:
                        st.write(f"Detected **{detected_pages}** pages.")
                    else:
                        st.write("Could not detect page count (file may be corrupted).")

                    # st.subheader("Indexing options")
                    use_pinecone = False
                    # st.info("Using local FAISS in-memory index")
                    store_content_in_metadata = False

                    # Use local FAISS only
                    pine_key = None
                    pine_env = None

                    # Guarded indexing action
                    if st.button("🔎 Click to Parse"):
                        # validate Pinecone key if chosen
                        if use_pinecone and not pine_key:
                            print(
                                "Pinecone selected but no API key provided. Choose one of:\n\n"
                                "• Paste your Pinecone API key in the input above,\n"
                                "• Set environment variable `PINECONE_API_KEY`, or\n"
                                "• Switch to Local FAISS (in-memory) instead."
                            )
                            with st.expander("How to set Pinecone API key"):
                                st.markdown(
                                    "- **Temporary (current session)**: paste it into the \"Pinecone API Key\" field above.\n"
                                    "- **Persistent (Windows)**: run `setx PINECONE_API_KEY \"<YOUR_KEY>\"` in Powershell/Command Prompt and restart your session.\n"
                                    "- **Persistent (.env)**: add `PINECONE_API_KEY=<YOUR_KEY>` to your `.env` file in the project root (you use python-dotenv)."
                                )
                        else:
                            try:
                                with st.spinner("Processing..."):
                                    res = process_and_index_pdf(
                                        pdf_bytes=bytes_data,
                                        doc_id=os.path.splitext(uploaded_pdf.name)[0],
                                        use_pinecone=use_pinecone,
                                        pinecone_api_key=pine_key,
                                        pine_env=pine_env,
                                        store_content_in_metadata=store_content_in_metadata,
                                    )
                                    st.success(f"Indexed {len(res['chunks'])} chunks from {res['pages_count']} pages (index_type={res['index_type']}).")
                                    st.session_state["pdf_index_info"] = res
                                    st.session_state["pdf_chunks"] = res["chunks"]
                                    # ensure we stay on Analyze Large PDF tab and show summary area
                                    st.session_state["main_tab"] = "Analyze Large PDF"
                                    # optionally pre-generate a high-level summary (comment/uncomment to auto generate)
                                    # st.session_state["last_pdf_summary"] = summarize_chunks(res["chunks"])
                            except Exception as e:
                                print(f"Indexing failed: {e}")
                                st.exception(e)

                else:
                    pass
                    # st.info("Upload a PDF file to use the large-doc parser.")

            elif file_type == "Audio File":
                from utils.audio_transcribe import audio_transcribe
                uploaded_audio = st.file_uploader("Upload an audio file", type = ["wav", "mp3", "flac", "ogg"], key = "audio_uploader")
                if uploaded_audio:
                    st.success("Audio File uploaded")

                    if st.button("Transcribe Audio"):
                        with st.spinner("Transcribing.."):
                            try:
                                audio_bytes = uploaded_audio.read()
                                audio_buffer = io.BytesIO(audio_bytes)
                                audio_buffer.seek(0)
                                transcript = audio_transcribe(audio_buffer)
                                if not transcript:
                                    st.error("Transcription Failed")
                                st.session_state["Audio_Transcription"] = transcript
                                st.success("Transcription Complete")
                            except Exception as e:
                                st.error(f"Transcription Failed {e}")
            
            # If there is an indexed doc in session, show quick controls
            if st.session_state.get("pdf_index_info"):
                st.markdown("---")
                st.write("Indexed document is available in session. You can summarize or ask questions below.")

        with col_right:
            if st.session_state.get("pdf_index_info"):
                idx_info = st.session_state["pdf_index_info"]
                st.subheader("Indexed Document Overview")
                st.write(f"Index type: **{idx_info['index_type']}**")
                st.write(f"Pages: **{idx_info['pages_count']}** — Chunks: **{len(idx_info['chunks'])}**")

                # with st.expander("Preview first 6 chunks"):
                #     for c in idx_info["chunks"][:6]:
                #         st.markdown(f"**{c['id']}** — pages {c['start_page']}-{c['end_page']}")
                #         st.text_area(c['id'], value=c['content'][:800], height=120, disabled=True)

                # Summarization: store result in session_state to persist across reruns
                if st.button("📝 Generate Document Summary"):
                    try:
                        with st.spinner("Summarizing"):
                            summary = summarize_chunks(idx_info["chunks"])
                            st.session_state["last_pdf_summary"] = summary
                            st.success("Summary generated.")
                    except Exception as e:
                        pass
                        # print(f"{e}")

                # display summary if available
                if st.session_state.get("last_pdf_summary"):
                    st.subheader("Document Summary")
                    st.write(st.session_state.get("last_pdf_summary"))

                # RAG Q&A
                st.subheader("💬Q&A")
                # keep the question persistent as well
                st.session_state.setdefault("pdf_rag_question", "")
                question = st.text_area("Ask a question about the indexed document:", height=120, key="pdf_rag_question")
                top_k = st.slider("Select Top K Value", 1, 10, 5, key="pdf_rag_topk")
                if st.button("Get Answer"):
                    if not question or not question.strip():
                        st.warning("Please enter a question.")
                    else:
                        try:
                            with st.spinner("Retrieving top chunks and generating answer..."):
                                if idx_info["index_type"] == "pinecone":
                                    pine_idx = idx_info["index"]
                                    chunks_map = idx_info["chunks"]
                                    res = rag_answer_with_retrieval_pinecone(pine_idx, question, top_k=top_k, chunks_mapping=chunks_map)
                                    st.subheader("Answer")
                                    st.write(res.get("answer", ""))
                                    if res.get("sources"):
                                        with st.expander("Retrieved sources"):
                                            for s in res["sources"]:
                                                st.write(s)
                                else:
                                    # local FAISS: we expect process_and_index_pdf to have returned a FAISS index + chunks mapping
                                    local_idx = idx_info["index"]
                                    chunks_map = idx_info["chunks"]
                                    # If pdf_rag returned embeddings and TF-IDF we can use rag_agent similar to in-memory flow
                                    # We'll attempt to call rag_answer_with_local_faiss which should implement its own LLM call or return answer+sources
                                    res = rag_answer_with_local_faiss(local_idx, question, top_k=top_k)
                                    st.subheader("Answer")
                                    st.write(res.get("answer", ""))
                                    if res.get("sources"):
                                        with st.expander("Retrieved sources"):
                                            for s in res["sources"]:
                                                st.write(s)
                        except Exception as e:
                            pass
                            # print(f"RAG failed: {e}")
            
            elif st.session_state.get("Audio_Transcription"):

                # displaying the full transcription
                st.subheader("Audio_Transcription")
                st.write(st.session_state.get("Audio_Transcription"))

                # Summarization: store result in session_state to persist across reruns
                if st.button("Generate Audio Summary"):
                    try:
                        with st.spinner("Generating Audio Summary"):
                            
                            summary = summarize_chunks([{"audio_transcript": st.session_state.get("Audio_Transcription")}], audio = True)
                            st.session_state['Audio_summary'] = summary
                            st.success("Summary generated")
                    except Exception as e:
                        st.write(f"Unable to generate summary due to error {e}")

                # display summary if available
                if st.session_state.get("Audio_summary"):
                    st.subheader("Audio Summary")
                    st.write(st.session_state.get("Audio_summary"))

            else:
                pass
                # st.info("No indexed document found in session. Upload & index a PDF in the left column first.")

    # Footer
    st.markdown("---")
    st.caption("Local AI Document Parser • Built with Streamlit & Ollama")

if __name__ == "__main__":
    main()
