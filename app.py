# # app.py (optimized Streamlit entrypoint) — updated to use an Ollama singleton
# import os
# import asyncio
# import time
# import streamlit as st
# from dotenv import load_dotenv

# # Local imports (assume these modules exist as in your repo)
# from parser.semantic_classifier import generate_radar_chart, zero_shot_skill_scores
# from utils.pdf_parser import extract_text_from_pdf
# from utils.json_formatter import safe_json_dumps
# # replaced integrate_ollama_parser imports with singleton-aware loader
# from utils.document_chunker import DocumentChunker
# from utils.pinecone_vector_store import setup_pinecone_interface, display_pinecone_stats

# # Page config
# st.set_page_config(
#     page_title="Document Parser LLM",
#     page_icon="📄",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# try:
#     load_dotenv()
# except Exception:
#     pass

# # Create uploads dir
# os.makedirs("uploads", exist_ok=True)

# CANDIDATE_LABELS = [
#     "Python", "C++", "Machine Learning", "Deep Learning",
#     "Cloud", "Azure", "AWS", "Communication",
#     "Leadership", "Research", "Data Analysis",
#     "Teamwork", "Web Development", "Frontend",
#     "Backend", "DevOps"
# ]

# @st.cache_data(show_spinner=False)
# def load_css():
#     """Load and cache CSS to avoid re-reading on every rerun.""" 
#     css_path = "assets/style.css"
#     if os.path.exists(css_path):
#         with open(css_path, "r", encoding="utf-8") as f:
#             return f"<style>{f.read()}</style>"
#     return ""

# # -------------------------
# # Ollama singleton loader
# # -------------------------
# def _import_ollama_singleton():
#     """
#     Try several common symbol names/locations for an Ollama singleton or factory in your utils.
#     Returns the singleton instance or None.
#     """
#     candidates = [
#         # common module / symbol combos to try
#         ("utils.ollama_parser", "Ollama_Singleton"),
#         ("utils.ollama_parser", "ollama_Singleton"),
#         ("utils.ollama_parser", "OllamaSingleton"),
#         ("utils.ollama_parser", "ollama_singleton"),
#         ("utils.ollama_parser", "OllamaParser"),  # fallback class
#         ("utils.ollama_singleton", "Ollama_Singleton"),
#         ("utils.ollama_singleton", "OllamaSingleton"),
#         ("utils.ollama_singleton", "ollama_singleton"),
#         ("utils", "ollama_Singleton"),
#         ("utils", "ollama_singleton"),
#     ]

#     for module_name, symbol in candidates:
#         try:
#             module = __import__(module_name, fromlist=[symbol])
#             obj = getattr(module, symbol, None)
#             if obj is None:
#                 continue

#             # If obj is a callable factory or class, try to get instance
#             try:
#                 # If it's already an instance (has typical attrs), return directly
#                 if not callable(obj) and hasattr(obj, "is_connected"):
#                     return obj

#                 # If class has an 'instance' or 'get_instance' method, use it
#                 if hasattr(obj, "instance") and callable(getattr(obj, "instance")):
#                     return obj.instance()
#                 if hasattr(obj, "get_instance") and callable(getattr(obj, "get_instance")):
#                     return obj.get_instance()

#                 # If it's a class, instantiate (no args). If it's a function, call it.
#                 if callable(obj):
#                     try:
#                         inst = obj()  # attempt no-arg constructor/factory
#                         return inst
#                     except Exception:
#                         # can't instantiate — skip
#                         continue
#             except Exception:
#                 continue
#         except Exception:
#             continue
#     return None

# def _ensure_ollama_in_session():
#     """Load the singleton and set session state keys like older integrate_ollama_parser did."""
#     if st.session_state.get("ollama_parser") is None:
#         singleton = _import_ollama_singleton()
#         if singleton:
#             st.session_state["ollama_parser"] = singleton
#             # ensure a few common session keys exist
#             if "ollama_model" not in st.session_state:
#                 st.session_state["ollama_model"] = getattr(singleton, "selected_model", None) or None
#             # store available models if present
#             available = getattr(singleton, "available_models", None)
#             if available is not None:
#                 st.session_state["ollama_available_models"] = available
#             return True
#         else:
#             return False
#     else:
#         return True

# # -------------------------
# # Small helpers
# # -------------------------
# def _show_streaming_output(generator, progress_bar, output_box, timeout=300):
#     """
#     Consume async generator returned by OllamaParser.parse_resume_stream
#     and update UI with partial content. Returns the accumulated string.
#     """
#     accumulated = ""
#     start = time.time()
#     try:
#         # If running in Streamlit (synchronous), use asyncio.run to consume the async generator
#         async def _consume():
#             nonlocal accumulated
#             async for chunk in generator:
#                 accumulated += chunk
#                 # update a small trailing preview (keeps UI snappy)
#                 preview = accumulated[-8000:]  # show last N chars
#                 output_box.code(preview)
#                 elapsed = time.time() - start
#                 # update progress bar heuristically (can't know exact progress)
#                 # alternate progress to indicate liveliness
#                 progress_bar.progress(min(0.95, elapsed / max(10.0, timeout)))
#             return accumulated

#         return asyncio.run(_consume())
#     except Exception as e:
#         # If streaming failed (e.g., aiohttp not available), just return partial accumulated text
#         st.warning(f"Streaming unavailable or failed: {e}. Falling back to blocking parse.")
#         return accumulated

# # -------------------------
# # Main
# # -------------------------
# def main():
#     # Load cached CSS
#     css = load_css()
#     if css:
#         st.markdown(css, unsafe_allow_html=True)

#     st.title("📄 Document Parser with Local AI")
#     st.markdown("Upload PDFs and get intelligent analysis with local LLMs (Ollama) + RAG")

#     # Sidebar: Ollama setup + vector DB selection + chunking options
#     with st.sidebar:
#         st.header("🦙 Ollama Local AI (singleton)")
#         ollama_ready = _ensure_ollama_in_session()
#         if ollama_ready:
#             st.success("✅ Ollama singleton loaded into session")
#             parser_obj = st.session_state.get("ollama_parser")
#             # Determine readiness using likely attributes
#             is_connected = getattr(parser_obj, "is_connected", None)
#             is_ready = getattr(parser_obj, "is_ready", None)
#             ready_flag = bool(is_connected or is_ready or getattr(parser_obj, "connected", False))
#             if ready_flag:
#                 st.success("✅ Ollama is ready")
#             else:
#                 st.info("Ollama singleton loaded but not currently connected/ready.")

#             # show available model selection stored by singleton
#             available_models = st.session_state.get("ollama_available_models") or getattr(parser_obj, "available_models", []) or []
#             if available_models:
#                 st.write("Available models:")
#                 for m in available_models:
#                     st.write(f"• {m}")
#                 st.info("Model selection is automatic; you may override in advanced settings.")
#         else:
#             st.error("⚠️ Ollama singleton not found. Make sure your utils module exposes a singleton object (Ollama_Singleton / OllamaSingleton / ollama_singleton / OllamaParser).")

#         st.header("🗄️ Vector Database")
#         vector_db_option = st.radio("Choose vector storage:", ["🧠 In-Memory (Local)", "🌲 Pinecone (Cloud)"])
#         pinecone_store = None
#         if vector_db_option.startswith("🌲"):
#             pinecone_store = setup_pinecone_interface()
#             if pinecone_store and getattr(pinecone_store, "is_initialized", False):
#                 display_pinecone_stats(pinecone_store)
#                 st.session_state["pinecone_store"] = pinecone_store

#         st.header("📄 Document Processing")
#         chunking_strategy = st.selectbox("Chunking strategy:", ["sections", "sliding_window", "pages"])
#         if chunking_strategy == "sliding_window":
#             # Use sentence-based UI but DocumentChunker may accept character sizes; adapt accordingly
#             chunk_size_sentences = st.slider("Chunk size (sentences):", 3, 12, 6)
#             overlap_sentences = st.slider("Overlap (sentences):", 0, 4, 1)
#         else:
#             chunk_size_sentences = None
#             overlap_sentences = None

#     col1, col2 = st.columns([1, 2])

#     with col1:
#         st.header("📁 Upload Document")
#         uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
#         if uploaded_file is not None:
#             st.success(f"✅ Uploaded: {uploaded_file.name}")
#             file_size_kb = uploaded_file.size / 1024
#             st.write(f"**Size:** {file_size_kb:.1f} KB")

#             if st.button("🚀 Extract & Parse Document"):
#                 # Check Ollama readiness
#                 parser_obj = st.session_state.get("ollama_parser")
#                 if not parser_obj or not getattr(parser_obj, "is_connected", False) and not getattr(parser_obj, "is_ready", False):
#                     st.error("Ollama not ready. Please start Ollama and install a model.")
#                     return

#                 # Extract text
#                 with st.spinner("Extracting text from PDF..."):
#                     combined_text, pages = extract_text_from_pdf(uploaded_file)
#                     if not combined_text:
#                         st.error("Failed to extract text from PDF")
#                         return
#                     st.session_state["raw_text"] = combined_text
#                     st.session_state["pages"] = pages
#                     st.success(f"Extracted {len(combined_text)} characters from {len(pages)} page(s)")

#                 # Chunk document (use DocumentChunker implementation)
#                 with st.spinner("Chunking document..."):
#                     chunker = DocumentChunker()
#                     if chunking_strategy == "sections":
#                         chunks = chunker.chunk_by_sections(combined_text)
#                     elif chunking_strategy == "sliding_window":
#                         # If your chunker expects sentence counts, pass those, else adapt
#                         chunks = chunker.chunk_by_sliding_window(combined_text, chunk_size_sentences, overlap_sentences)
#                     else:
#                         chunks = chunker.chunk_by_pages(pages)
#                     st.session_state["chunks"] = chunks
#                     st.success(f"Created {len(chunks)} chunks")

#                 # If document small, use blocking sync parse (fast)
#                 use_streaming = hasattr(parser_obj, "parse_resume_stream") and len(combined_text) >= 2000

#                 parsed_data = None
#                 # UI elements for streaming feedback
#                 progress_bar = st.empty()
#                 output_box = st.empty()
#                 try:
#                     if use_streaming:
#                         progress_widget = st.progress(0.0)
#                         output_box.info("Streaming parsing output (accumulating...)")
#                         # Kick off async streaming generator from OllamaParser
#                         try:
#                             generator = parser_obj.parse_resume_stream(combined_text, model=None)
#                             accumulated = _show_streaming_output(generator, progress_widget, output_box, timeout=300)
#                             # Attempt to extract JSON from accumulated string
#                             # reuse helper from OllamaParser if available
#                             parsed = None
#                             if hasattr(parser_obj, "_extract_json_from_text"):
#                                 parsed = parser_obj._extract_json_from_text(accumulated)
#                             else:
#                                 # naive extraction
#                                 import re, json
#                                 m = re.search(r'\{.*\}', accumulated, re.DOTALL)
#                                 if m:
#                                     parsed = json.loads(m.group())
#                             if parsed:
#                                 parsed_data = parsed
#                                 st.success("✅ Parsing complete (stream).")
#                             else:
#                                 st.warning("Could not extract structured JSON from streamed output — trying blocking parse.")
#                         except Exception as e:
#                             st.warning(f"Streaming path failed: {e}. Falling back to blocking parse.")

#                     if parsed_data is None:
#                         # Blocking sync parse fallback (safe)
#                         with st.spinner("Running blocking parse (this may take a moment)..."):
#                             parsed_result = parser_obj.parse_resume_sync(combined_text, model=None)
#                             parsed_data = parsed_result if isinstance(parsed_result, dict) else {}
#                             st.success("✅ Parsing complete (sync).")

#                     # Normalize structure if the parsed_data is raw dict coming from OllamaParser or service
#                     st.session_state["parsed_data"] = parsed_data

#                     # After parsing, set up vector store or in-memory RAG
#                     if vector_db_option.startswith("🌲") and st.session_state.get("pinecone_store"):
#                         pinecone_store = st.session_state.get("pinecone_store")
#                         with st.spinner("Storing chunks in Pinecone..."):
#                             success = pinecone_store.add_documents(chunks, f"doc_{uploaded_file.name}")
#                             if success:
#                                 qa_ok = pinecone_store.setup_qa_chain(parser_obj.selected_model or "llama3.2:3b")
#                                 if qa_ok:
#                                     st.session_state["pinecone_qa_ready"] = True
#                                     st.success("✅ Pinecone storage & QA ready")
#                                 else:
#                                     st.warning("⚠️ Pinecone QA chain setup failed")
#                             else:
#                                 st.error("❌ Failed to upsert to Pinecone")
#                     else:
#                         # in-memory RAG (your existing class)
#                         if "ollama_rag" in st.session_state:
#                             with st.spinner("Setting up in-memory vector search..."):
#                                 rag_success = st.session_state.ollama_rag.setup_rag_with_chunks(chunks)
#                                 if rag_success:
#                                     st.session_state["in_memory_rag_ready"] = True
#                                     st.success("✅ In-memory RAG ready")
#                                 else:
#                                     st.warning("⚠️ In-memory RAG setup failed")
#                         else:
#                             st.info("In-memory RAG unavailable; skip vector setup.")
#                 except Exception as e:
#                     st.error(f"Parsing failed: {e}")
#                     st.exception(e)

#     # Right column: results and QA
#     with col2:
#         st.header("📊 Document Analysis")
#         if "parsed_data" in st.session_state:
#             parsed_struct = st.session_state["parsed_data"]

#             # Try friendly formatted display first (shared helper)
#             displayed = False
#             try:
#                 from utils.llm_parser import format_document_display
#                 format_document_display(parsed_struct)
#                 displayed = True
#             except Exception:
#                 displayed = False

#             # Friendly header
#             try:
#                 provider = parsed_struct.get("ai_provider") or parsed_struct.get("_source") or parsed_struct.get("provider", "AI")
#                 model = parsed_struct.get("model") or parsed_struct.get("_model", st.session_state.get("ollama_model", "unknown"))
#                 st.info(f"Parsed by: {provider} • Model: {model}")
#             except Exception:
#                 pass

#             # Chunk stats
#             chunks = st.session_state.get("chunks", [])
#             st.subheader("📄 Document Chunks")
#             col_a, col_b, col_c = st.columns(3)
#             col_a.metric("Total Chunks", len(chunks))
#             avg_len = sum(len(c.get("content", "")) for c in chunks) // len(chunks) if chunks else 0
#             col_b.metric("Avg Chunk Length", f"{avg_len} chars")
#             col_c.metric("Strategy", chunking_strategy.title())

#             with st.expander("🔍 View Document Chunks"):
#                 for i, chunk in enumerate(chunks):
#                     st.write(f"**Chunk {i+1}** ({chunk.get('type', 'text')})")
#                     if 'page' in chunk:
#                         st.caption(f"Page: {chunk['page']}")
#                     st.text_area(f"chunk_{i+1}", value=chunk.get('content', '')[:1500], height=140, disabled=True)

#             # Raw text view
#             with st.expander("📝 Raw Extracted Text"):
#                 pages = st.session_state.get("pages", [])
#                 if pages:
#                     idx = st.selectbox("Select page", list(range(1, len(pages) + 1)))
#                     st.text_area(f"Page {idx}", value=pages[idx - 1], height=300, disabled=True)
#                 else:
#                     st.text_area("Combined text", value=st.session_state.get("raw_text", ""), height=300, disabled=True)

#             # Raw JSON accordion
#             with st.expander("🔧 View Raw JSON Data (click to expand)", expanded=False):
#                 try:
#                     st.json(parsed_struct)
#                 except Exception:
#                     try:
#                         full_json = safe_json_dumps(parsed_struct)
#                         st.code(full_json, language="json")
#                     except Exception:
#                         st.text(str(parsed_struct))
#                 try:
#                     full_json = safe_json_dumps(parsed_struct)
#                     st.download_button("💾 Download full JSON", data=full_json, file_name="parsed_full.json", mime="application/json")
#                 except Exception:
#                     pass

#             # If formatted display didn't run, show a quick glance
#             if not displayed:
#                 personal = parsed_struct.get("personal_info", {}) or parsed_struct.get("personal", {})
#                 if personal:
#                     name = personal.get("name") or personal.get("full_name")
#                     email = personal.get("email")
#                     if name:
#                         st.subheader(f"👤 {name}")
#                     if email:
#                         st.write(f"📧 {email}")

#             # Continue with Semantic classification and Q&A (unchanged)
#             # ...
#         # Semantic classification
#                 st.subheader("📊 Semantic Skill Classification")
#                 if st.session_state.get("raw_text"):
#                     with st.spinner("Running zero-shot classification..."):
#                         try:
#                             scores = zero_shot_skill_scores(st.session_state["raw_text"], CANDIDATE_LABELS)
#                             skill_pairs = list(zip(scores.get("labels", []), scores.get("scores", [])))
#                             skill_pairs.sort(key=lambda x: x[1], reverse=True)
#                             st.table([{"Skill": s, "Confidence": f"{p:.2f}"} for s, p in skill_pairs])
#                             if skill_pairs:
#                                 labels, vals = zip(*skill_pairs)
#                                 chart_path = generate_radar_chart(list(labels), list(vals))
#                                 st.image(chart_path, caption="Skill confidence radar")
#                         except Exception as e:
#                             st.warning(f"Skill classification failed: {e}")
#                 else:
#                     st.info("Run parsing to enable skill classification.")

#                 # Q&A
#                 qa_ready = st.session_state.get("pinecone_qa_ready", False) or st.session_state.get("in_memory_rag_ready", False)
#                 if qa_ready:
#                     st.divider()
#                     st.header("💬 Document Q&A")
#                     if st.session_state.get("pinecone_qa_ready", False):
#                         st.info("Using Pinecone + LangChain")
#                     else:
#                         st.info("Using in-memory RAG")

#                     question = st.text_area("Ask a question about the document:", height=80)
#                     if st.button("Get AI Analysis"):
#                         if not question:
#                             st.warning("Please enter a question.")
#                         else:
#                             with st.spinner("Generating answer..."):
#                                 try:
#                                     if st.session_state.get("pinecone_qa_ready", False):
#                                         res = st.session_state["pinecone_store"].ask_question(question)
#                                         if "error" in res:
#                                             st.error(res["error"])
#                                         else:
#                                             st.write(res.get("answer", ""))
#                                             # show sources if present
#                                             if res.get("source_documents"):
#                                                 with st.expander("Source documents"):
#                                                     for sd in res["source_documents"]:
#                                                         st.text_area("Source", value=sd.get("content","")[:800], height=120, disabled=True)
#                                     else:
#                                         # in-memory RAG
#                                         rag = st.session_state.get("ollama_rag")
#                                         if rag:
#                                             res = rag.ask_question(question)
#                                             if "error" in res:
#                                                 st.error(res["error"])
#                                             else:
#                                                 st.write(res.get("answer", ""))
#                                                 if res.get("sources"):
#                                                     with st.expander("Source chunks"):
#                                                         for s in res["sources"]:
#                                                             st.text_area("Chunk", value=s.get("content","")[:800], height=120, disabled=True)
#                                         else:
#                                             st.error("In-memory RAG not initialized.")
#                                 except Exception as e:
#                                     st.error(f"QA failed: {e}")
#         else:
#             st.info("Upload a PDF to parse and analyze.")
#             with st.expander("💡 Vector DB options"):
#                 st.write("🧠 In-memory: fast, ephemeral.")
#             st.write("🌲 Pinecone: persistent, multi-document, requires keys.")


            


#     st.markdown("---")
#     st.caption("Local AI Document Parser • Built with Streamlit & Ollama")

# if __name__ == "__main__":
#     main()

# app.py (optimized Streamlit entrypoint) — uses utils.ollama_singleton.get_ollama
import os
import asyncio
import time
import re
import json
import streamlit as st
from dotenv import load_dotenv

# Local imports (assume these modules exist as in your repo)
from parser.semantic_classifier import generate_radar_chart, zero_shot_skill_scores
from utils.pdf_parser import extract_text_from_pdf
from utils.json_formatter import safe_json_dumps
# replaced integrate_ollama_parser imports with singleton-aware loader
from utils.document_chunker import DocumentChunker
from utils.pinecone_vector_store import setup_pinecone_interface, display_pinecone_stats

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
# Main
# -------------------------
def main():
    # Load cached CSS
    css = load_css()
    if css:
        st.markdown(css, unsafe_allow_html=True)

    st.title("📄 Document Parser with Local AI")
    st.markdown("Upload PDFs and get intelligent analysis with local LLMs (Ollama) + RAG")

    # Sidebar: Ollama setup + vector DB selection + chunking options
    with st.sidebar:
        st.header("🦙 Ollama Local AI (singleton)")
        ollama_ready = _ensure_ollama_in_session()
        if ollama_ready:
            st.success("✅ Ollama singleton loaded into session")
            parser_obj = st.session_state.get("ollama_parser")
            # Determine readiness using likely attributes
            is_connected = getattr(parser_obj, "is_connected", None)
            is_ready = getattr(parser_obj, "is_ready", None)
            ready_flag = bool(is_connected or is_ready or getattr(parser_obj, "connected", False))
            if ready_flag:
                st.success("✅ Ollama is ready")
            else:
                st.info("Ollama singleton loaded but not currently connected/ready. Start Ollama and ensure a model is installed.")

            # show available model selection stored by singleton
            available_models = st.session_state.get("ollama_available_models") or getattr(parser_obj, "available_models", []) or []
            if available_models:
                st.write("Available models:")
                for m in available_models:
                    st.write(f"• {m}")
                st.info("Model selection is automatic; override in advanced settings if needed.")
        else:
            st.error("⚠️ Ollama singleton not found. Make sure your utils module exposes a singleton (e.g., `ollama_singleton` or `OllamaParser`).")

        st.header("🗄️ Vector Database")
        vector_db_option = st.radio("Choose vector storage:", ["🧠 In-Memory (Local)", "🌲 Pinecone (Cloud)"])
        pinecone_store = None
        if vector_db_option.startswith("🌲"):
            pinecone_store = setup_pinecone_interface()
            if pinecone_store and getattr(pinecone_store, "is_initialized", False):
                display_pinecone_stats(pinecone_store)
                st.session_state["pinecone_store"] = pinecone_store

        st.header("📄 Document Processing")
        chunking_strategy = st.selectbox("Chunking strategy:", ["sections", "sliding_window", "pages"])
        if chunking_strategy == "sliding_window":
            # Use sentence-based UI but DocumentChunker may accept character sizes; adapt accordingly
            chunk_size_sentences = st.slider("Chunk size (sentences):", 3, 12, 6)
            overlap_sentences = st.slider("Overlap (sentences):", 0, 4, 1)
        else:
            chunk_size_sentences = None
            overlap_sentences = None

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
                    st.error("Ollama not ready. Please start Ollama and install a model.")
                    return

                # Extract text
                with st.spinner("Extracting text from PDF..."):
                    combined_text, pages = extract_text_from_pdf(uploaded_file)
                    if not combined_text:
                        st.error("Failed to extract text from PDF")
                        return
                    st.session_state["raw_text"] = combined_text
                    st.session_state["pages"] = pages
                    st.success(f"Extracted {len(combined_text)} characters from {len(pages)} page(s)")

                # Chunk document (use DocumentChunker implementation)
                with st.spinner("Chunking document..."):
                    chunker = DocumentChunker()
                    if chunking_strategy == "sections":
                        chunks = chunker.chunk_by_sections(combined_text)
                    elif chunking_strategy == "sliding_window":
                        # If your chunker expects sentence counts, pass those, else adapt
                        chunks = chunker.chunk_by_sliding_window(combined_text, chunk_size_sentences, overlap_sentences)
                    else:
                        chunks = chunker.chunk_by_pages(pages)
                    st.session_state["chunks"] = chunks
                    st.success(f"Created {len(chunks)} chunks")

                # If document small, use blocking sync parse (fast)
                use_streaming = False

                parsed_data = None
                # UI elements for streaming feedback
                progress_bar = st.empty()
                output_box = st.empty()
                try:
                    if use_streaming:
                        progress_widget = st.progress(0.0)
                        output_box.info("Processing (live preview shown)...")
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
                            success = pinecone_store.add_documents(chunks, f"doc_{uploaded_file.name}")
                            if success:
                                qa_ok = pinecone_store.setup_qa_chain(parser_obj.selected_model or "llama3.2:3b")
                                if qa_ok:
                                    st.session_state["pinecone_qa_ready"] = True
                                    st.success("✅ Pinecone storage & QA ready")
                                else:
                                    st.warning("⚠️ Pinecone QA chain setup failed")
                            else:
                                st.error("❌ Failed to upsert to Pinecone")
                    else:
                        # in-memory RAG (your existing class)
                        if "ollama_rag" in st.session_state:
                            with st.spinner("Setting up in-memory vector search..."):
                                rag_success = st.session_state.ollama_rag.setup_rag_with_chunks(chunks)
                                if rag_success:
                                    st.session_state["in_memory_rag_ready"] = True
                                    st.success("✅ In-memory RAG ready")
                                else:
                                    st.warning("⚠️ In-memory RAG setup failed")
                        else:
                            st.info("In-memory RAG unavailable; skip vector setup.")
                except Exception as e:
                    st.error("Parsing failed — see details below.")
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
            avg_len = sum(len(c.get("content", "")) for c in chunks) // len(chunks) if chunks else 0
            col_b.metric("Avg Chunk Length", f"{avg_len} chars")
            col_c.metric("Strategy", chunking_strategy.title())

            with st.expander("🔍 View Document Chunks"):
                for i, chunk in enumerate(chunks):
                    st.write(f"**Chunk {i+1}** ({chunk.get('type', 'text')})")
                    if 'page' in chunk:
                        st.caption(f"Page: {chunk['page']}")
                    st.text_area(f"chunk_{i+1}", value=chunk.get('content', '')[:1500], height=140, disabled=True)

            # Raw text view
            with st.expander("📝 Raw Extracted Text"):
                pages = st.session_state.get("pages", [])
                if pages:
                    idx = st.selectbox("Select page", list(range(1, len(pages) + 1)))
                    st.text_area(f"Page {idx}", value=pages[idx - 1], height=300, disabled=True)
                else:
                    st.text_area("Combined text", value=st.session_state.get("raw_text", ""), height=300, disabled=True)

            # Raw JSON accordion
            with st.expander("🔧 View Raw JSON Data (click to expand)", expanded=False):
                try:
                    st.json(parsed_struct)
                except Exception:
                    try:
                        full_json = safe_json_dumps(parsed_struct)
                        st.code(full_json, language="json")
                    except Exception:
                        st.text(str(parsed_struct))
                try:
                    full_json = safe_json_dumps(parsed_struct)
                    st.download_button("💾 Download full JSON", data=full_json, file_name="parsed_full.json", mime="application/json")
                except Exception:
                    pass

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
                        st.warning(f"Skill classification failed: {e}")
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
                    st.info("Using in-memory RAG")

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
                                        st.error(res["error"])
                                    else:
                                        st.write(res.get("answer", ""))
                                        # show sources if present
                                        if res.get("source_documents"):
                                            with st.expander("Source documents"):
                                                for sd in res["source_documents"]:
                                                    st.text_area("Source", value=sd.get("content","")[:800], height=120, disabled=True)
                                else:
                                    # in-memory RAG
                                    rag = st.session_state.get("ollama_rag")
                                    if rag:
                                        res = rag.ask_question(question)
                                        if "error" in res:
                                            st.error(res["error"])
                                        else:
                                            st.write(res.get("answer", ""))
                                            if res.get("sources"):
                                                with st.expander("Source chunks"):
                                                    for s in res["sources"]:
                                                        st.text_area("Chunk", value=s.get("content","")[:800], height=120, disabled=True)
                                    else:
                                        st.error("In-memory RAG not initialized.")
                            except Exception as e:
                                st.error(f"QA failed: {e}")
        else:
            st.info("Upload a PDF to parse and analyze.")
            with st.expander("💡 Vector DB options"):
                st.write("🧠 In-memory: fast, ephemeral.")
                st.write("🌲 Pinecone: persistent, multi-document, requires keys.")

    st.markdown("---")
    st.caption("Local AI Document Parser • Built with Streamlit & Ollama")

if __name__ == "__main__":
    main()
