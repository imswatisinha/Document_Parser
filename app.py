import streamlit as st
import os
from dotenv import load_dotenv
from parser.semantic_classifier import generate_radar_chart, zero_shot_skill_scores
from utils.pdf_parser import extract_text_from_pdf
from utils.llm_parser import parse_resume_with_ollama, format_document_display, show_ollama_status, show_model_recommendations, get_available_ollama_models
from utils.json_formatter import safe_json_dumps
from utils.ollama_parser import integrate_ollama_parser, OllamaParser
from utils.rag_retriever import create_ollama_rag_interface, display_ollama_resume_insights
from utils.document_chunker import DocumentChunker
from utils.pinecone_vector_store import PineconeVectorStore, setup_pinecone_interface, display_pinecone_stats

# Page configuration
st.set_page_config(
    page_title="Resume Parser LLM",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables for API fallback keys
try:
    load_dotenv()
except Exception:
    pass

# Custom CSS
def load_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

CANDIDATE_LABELS = [
    "Python", "C++", "Machine Learning", "Deep Learning",
    "Cloud", "Azure", "AWS", "Communication",
    "Leadership", "Research", "Data Analysis",
    "Teamwork", "Web Development", "Frontend",
    "Backend", "DevOps"
]

def main():
    # Header
    st.title("📄 Document Parser with Local AI")
    st.markdown("Upload documents (PDF, Resume, etc.) and get intelligent analysis using Ollama/Llama with advanced chunking and vector search")
    
    # Sidebar
    with st.sidebar:
        # Ollama Service Setup
        st.header(" Ollama Local AI")
        
        # Initialize Ollama service
        ollama_ready = integrate_ollama_parser()
        if ollama_ready:
            st.success("✅ Ollama is ready!")
            show_ollama_status()
            
            # Show intelligent model recommendations
            available_models = get_available_ollama_models()
            if available_models:
                show_model_recommendations(available_models)
                st.info("💡 **Smart Model Selection:** The app will automatically choose the best model for your document!")
            
            # Initialize RAG system
            if 'ollama_rag' not in st.session_state:
                from utils.rag_retriever import OllamaRAGRetriever
                st.session_state.ollama_rag = OllamaRAGRetriever()
        else:
            st.error("⚠️ Ollama not ready. Please follow setup instructions.")
            
        # Vector Database Selection
        st.header("🗄️ Vector Database")
        
        vector_db_option = st.radio(
            "Choose vector storage:",
            ["🧠 In-Memory (Local)", "🌲 Pinecone (Cloud)"],
            help="In-memory is faster but temporary, Pinecone is persistent"
        )
        
        # Pinecone setup
        pinecone_store = None
        if vector_db_option.startswith("🌲"):
            pinecone_store = setup_pinecone_interface()
            if pinecone_store and pinecone_store.is_initialized:
                display_pinecone_stats(pinecone_store)
        
        # Document Chunking Options  
        st.header("📄 Document Processing")
        
        chunking_strategy = st.selectbox(
            "Choose chunking strategy:",
            ["sections", "sliding_window", "pages"],
            help="How to split the document for better analysis"
        )
        
        if chunking_strategy == "sliding_window":
            chunk_size = st.slider("Chunk size (sentences):", 3, 10, 5)
            overlap = st.slider("Overlap (sentences):", 0, 3, 1)
        else:
            chunk_size = None
            overlap = None
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📁 Upload Resume")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file", 
            type=["pdf"],
            help="Upload a resume in PDF format"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Display file info
            file_size = uploaded_file.size / 1024  # KB
            st.write(f"**File size:** {file_size:.1f} KB")
            
            # Extract and parse button
            if st.button("🚀 Extract & Parse Document", type="primary"):
                if not ollama_ready:
                    st.error("Ollama is not ready. Please install and start Ollama first.")
                    return
                    
                with st.spinner("Processing document..."):
                    # Step 1: Extract text from PDF (supports multi-page)
                    with st.status("Extracting text from PDF...", expanded=False) as status:
                        combined_text, pages = extract_text_from_pdf(uploaded_file)
                        if combined_text or (pages and len(pages) > 0):
                            st.write(f"✅ Extracted {len(combined_text)} characters from {len(pages)} page(s)")
                            status.update(label="Text extraction complete!", state="complete")
                        else:
                            st.error("Failed to extract text from PDF")
                            return
                    
                    # Step 2: Document chunking
                    with st.status("Chunking document...", expanded=False) as status:
                        chunker = DocumentChunker()
                        
                        if chunking_strategy == "sections":
                            chunks = chunker.chunk_by_sections(combined_text)
                        elif chunking_strategy == "sliding_window":
                            chunks = chunker.chunk_by_sliding_window(
                                combined_text, chunk_size, overlap
                            )
                        else:  # pages
                            chunks = chunker.chunk_by_pages(pages)
                        
                        st.write(f"✅ Created {len(chunks)} chunks")
                        status.update(label="Document chunking complete!", state="complete")
                    
                    # Step 3: Intelligent AI Processing
                    with st.status("🧠 Analyzing document and selecting optimal AI model...", expanded=True) as status:
                        if 'ollama_parser' in st.session_state:
                            # Use intelligent model selection (auto-selects based on document)
                            parsed_data = parse_resume_with_ollama(combined_text, chunks, model_name=None, use_expanders=False)
                            
                            if "error" in parsed_data and parsed_data.get("setup_required"):
                                st.error("Please install an Ollama model first. See sidebar for instructions.")
                                return
                        else:
                            st.error("Ollama parser not initialized")
                            return
                            
                        status.update(label="🎯 AI processing complete with optimal model!", state="complete")
                    
                    # Store results in session state
                    st.session_state.parsed_data = parsed_data
                    st.session_state.raw_text = combined_text
                    st.session_state.pages = pages
                    st.session_state.chunks = chunks
                    st.session_state.chunking_strategy = chunking_strategy
                    
                    # Setup vector storage based on user choice
                    if vector_db_option.startswith("🌲") and pinecone_store and pinecone_store.is_initialized:
                        # Use Pinecone for persistent storage
                        with st.status("Storing in Pinecone...", expanded=False) as status:
                            pinecone_success = pinecone_store.add_documents(chunks, f"doc_{uploaded_file.name}")
                            if pinecone_success:
                                # Setup QA chain
                                qa_setup = pinecone_store.setup_qa_chain(st.session_state.get('ollama_model', 'llama3.2:3b'))
                                if qa_setup:
                                    st.session_state.pinecone_qa_ready = True
                                    st.write("✅ Pinecone vector storage and QA ready")
                                    status.update(label="Pinecone storage complete!", state="complete")
                                else:
                                    st.warning("⚠️ QA chain setup failed")
                                    status.update(label="QA setup failed", state="error")
                            else:
                                st.error("❌ Failed to store in Pinecone")
                                status.update(label="Pinecone storage failed", state="error")
                    
                    else:
                        # Use in-memory RAG system
                        if 'ollama_rag' in st.session_state:
                            with st.status("Setting up in-memory vector search...", expanded=False) as status:
                                rag_success = st.session_state.ollama_rag.setup_rag_with_chunks(chunks)
                                if rag_success:
                                    st.write("✅ In-memory vector search ready for Q&A")
                                    status.update(label="Vector search setup complete!", state="complete")
                                else:
                                    st.warning("⚠️ Vector search setup failed")
                                    status.update(label="Vector search setup failed", state="error")
    
    with col2:
        st.header("📊 Document Analysis")
        
        if 'parsed_data' in st.session_state:
            # Display results
            format_document_display(st.session_state.parsed_data)
            
            # Show chunking statistics
            if 'chunks' in st.session_state:
                st.subheader("📄 Document Chunks")
                chunks = st.session_state.chunks
                strategy = st.session_state.get('chunking_strategy', 'sections')
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Total Chunks", len(chunks))
                with col_b:
                    avg_length = sum(len(chunk['content']) for chunk in chunks) // len(chunks) if chunks else 0
                    st.metric("Avg Chunk Length", f"{avg_length} chars")
                with col_c:
                    st.metric("Strategy", strategy.title())
                
                # Show chunk previews
                with st.expander("🔍 View Document Chunks"):
                    for i, chunk in enumerate(chunks):
                        st.write(f"**Chunk {i+1}** ({chunk.get('type', 'text')})")
                        if 'page' in chunk:
                            st.caption(f"Page: {chunk['page']}")
                        if chunk.get('summary'):
                            st.caption(f"Summary: {chunk['summary'][:200]}" + ("..." if len(chunk['summary']) > 200 else ""))
                        st.text_area(
                            f"chunk_{i+1}",
                            value=chunk['content'][:500] + ("..." if len(chunk['content']) > 500 else ""),
                            height=150,
                            disabled=True,
                            label_visibility="collapsed"
                        )
            
            # Raw text section (expandable)
            with st.expander("📝 View Raw Extracted Text"):
                pages = st.session_state.get('pages', [])
                if pages:
                    page_index = st.selectbox("Select page to view:", list(range(1, len(pages) + 1)))
                    st.text_area(
                        f"Raw text - Page {page_index}",
                        value=pages[page_index - 1] or "",
                        height=300,
                        disabled=True
                    )
                    if st.checkbox("Show combined text", value=False):
                        st.text_area(
                            "Combined text:",
                            value=st.session_state.raw_text,
                            height=300,
                            disabled=True
                        )
                else:
                    st.text_area(
                        "Raw text from PDF:",
                        value=st.session_state.raw_text,
                        height=300,
                        disabled=True
                    )
            
            # Provider status and warnings
            provider = st.session_state.parsed_data.get("provider") or st.session_state.parsed_data.get("ai_provider")
            model_used = st.session_state.parsed_data.get("_model") or st.session_state.parsed_data.get("model")
            if provider:
                st.info(f"Parsed using: {provider} (model: {model_used})")
            warnings = st.session_state.parsed_data.get("_warnings", [])
            for w in warnings:
                st.warning(w)

            # Download structured data (full and per-section)
            parsed_struct = st.session_state.parsed_data
            full_json = safe_json_dumps(parsed_struct)
            st.download_button(
            	label="💾 Download Full JSON",
            	data=full_json,
            	file_name="parsed_full.json",
            	mime="application/json"
            )

            exp = parsed_struct.get("experience", [])
            edu = parsed_struct.get("education", [])
            skills = parsed_struct.get("skills", [])
            chunks_list = parsed_struct.get("chunks", []) or st.session_state.get('chunks', [])
            summary_text = parsed_struct.get("summary")

            cols_dl = st.columns(3)
            with cols_dl[0]:
                if exp:
                    st.download_button("📥 Experience JSON", safe_json_dumps({"experience": exp}), file_name="experience.json", mime="application/json")
                else:
                    st.button("📥 Experience JSON", disabled=True)
                if edu:
                    st.download_button("📥 Education JSON", safe_json_dumps({"education": edu}), file_name="education.json", mime="application/json")
                else:
                    st.button("📥 Education JSON", disabled=True)
            with cols_dl[1]:
                if skills:
                    st.download_button("📥 Skills JSON", safe_json_dumps({"skills": skills}), file_name="skills.json", mime="application/json")
                else:
                    st.button("📥 Skills JSON", disabled=True)
                if chunks_list:
                    from utils.normalizers import validate_and_normalize
                    norm = validate_and_normalize({}, chunks=chunks_list)
                    st.download_button("📥 Chunks JSON", safe_json_dumps({"chunks": norm.get("chunks", [])}), file_name="chunks.json", mime="application/json")
                else:
                    st.button("📥 Chunks JSON", disabled=True)
            with cols_dl[2]:
                if summary_text:
                    st.download_button("📥 Summary JSON", safe_json_dumps({"summary": summary_text}), file_name="summary.json", mime="application/json")
                else:
                    st.button("📥 Summary JSON", disabled=True)

            st.subheader("📊 Semantic Skill Classification (BART MNLI)")
            if st.session_state.get('raw_text'):
                skill_table_col, chart_col = st.columns(2)
                try:
                    with st.spinner("Running zero-shot skill classification..."):
                        scores = zero_shot_skill_scores(st.session_state.raw_text, CANDIDATE_LABELS)
                    skill_pairs = list(zip(scores.get("labels", []), scores.get("scores", [])))
                    skill_pairs.sort(key=lambda item: item[1], reverse=True)

                    with skill_table_col:
                        st.markdown("**Ranked skills**")
                        st.table([{"Skill": label, "Confidence": f"{score:.2f}"} for label, score in skill_pairs])

                    if skill_pairs:
                        chart_labels, chart_scores = zip(*skill_pairs)
                        with chart_col:
                            chart_path = generate_radar_chart(list(chart_labels), list(chart_scores))
                            st.image(chart_path, caption="Skill confidence radar", use_column_width=True)
                except Exception as e:
                    st.error(f"Semantic skill classification unavailable: {e}")
            else:
                st.info("Semantic skill classification will appear after parsing the resume text.")
            
            # Show Q&A section for document analysis
            qa_available = (
                ('ollama_rag' in st.session_state and 'chunks' in st.session_state) or
                st.session_state.get('pinecone_qa_ready', False)
            )
            
            if qa_available:
                st.divider()
                st.header("💬 Document Q&A with Vector Search")
                
                # Show which system is being used
                if st.session_state.get('pinecone_qa_ready', False):
                    st.info("🌲 Using **Pinecone** vector database with **LangChain** RAG pipeline")
                else:
                    st.info("🧠 Using **in-memory** vector search with enhanced chunking")
                
                st.write("Ask detailed questions about the document using intelligent vector similarity search.")
                
                # Question input
                question = st.text_area(
                    "Enter your question:",
                    placeholder="e.g., What are the key skills mentioned? Summarize the experience section. What projects are described? What education background is listed?",
                    height=80
                )
                
                if st.button("🤖 Get AI Analysis", type="secondary"):
                    if question:
                        with st.spinner("Analyzing document and generating response..."):
                            try:
                                # Use Pinecone QA chain if available
                                if st.session_state.get('pinecone_qa_ready', False) and 'pinecone_store' in st.session_state:
                                    result = st.session_state.pinecone_store.ask_question(question)
                                    
                                    if "error" in result:
                                        st.error(f"❌ {result['error']}")
                                    else:
                                        st.success("**AI Analysis (via Pinecone + LangChain):**")
                                        st.write(result["answer"])
                                        
                                        # Show source documents from Pinecone
                                        if result.get("source_documents"):
                                            sources = result["source_documents"]
                                            st.caption(f"🌲 Retrieved from Pinecone | Sources: {len(sources)}")
                                            
                                            with st.expander(f"📄 Source Documents ({len(sources)})"):
                                                for i, source in enumerate(sources):
                                                    st.write(f"**Source {i+1}**")
                                                    
                                                    # Show metadata
                                                    metadata = source.get('metadata', {})
                                                    meta_cols = st.columns(4)
                                                    with meta_cols[0]:
                                                        st.metric("Type", metadata.get('chunk_type', 'N/A'))
                                                    with meta_cols[1]:
                                                        st.metric("Page", metadata.get('page', 1))
                                                    with meta_cols[2]:
                                                        st.metric("Section", metadata.get('section', 'N/A'))
                                                    with meta_cols[3]:
                                                        st.metric("Length", f"{len(source.get('content', ''))} chars")
                                                    
                                                    # Show content
                                                    st.text_area(
                                                        f"Content {i+1}:",
                                                        value=source.get('content', ''),
                                                        height=100,
                                                        disabled=True,
                                                        label_visibility="collapsed"
                                                    )
                                
                                else:
                                    # Use in-memory RAG system
                                    result = st.session_state.ollama_rag.ask_question(question)
                                    
                                    if "error" in result:
                                        st.error(f"❌ {result['error']}")
                                    else:
                                        st.success("**AI Analysis (via In-Memory RAG):**")
                                        st.write(result["answer"])
                                        
                                        # Show retrieval metadata
                                        metadata_info = []
                                        if result.get("model_used"):
                                            metadata_info.append(f"Model: {result['model_used']}")
                                        if result.get("retrieval_method"):
                                            metadata_info.append(f"Method: {result['retrieval_method']}")
                                        if result.get("total_chunks_used"):
                                            metadata_info.append(f"Chunks: {result['total_chunks_used']}")
                                        
                                        if metadata_info:
                                            st.caption(" | ".join(metadata_info))
                                        
                                        # Show relevant chunks with enhanced metadata
                                        if result.get("sources"):
                                            sources = result["sources"]
                                            with st.expander(f"📄 Source Chunks ({len(sources)})"):
                                                for i, source in enumerate(sources):
                                                    st.write(f"**Chunk {i+1}**")
                                                    
                                                    # Show metadata
                                                    meta_cols = st.columns(4)
                                                    with meta_cols[0]:
                                                        st.metric("Type", source.get('type', 'N/A'))
                                                    with meta_cols[1]:
                                                        st.metric("Page", source.get('page', 1))
                                                    with meta_cols[2]:
                                                        score = source.get('similarity_score', 0)
                                                        st.metric("Relevance", f"{score:.2f}")
                                                    with meta_cols[3]:
                                                        st.metric("Length", f"{len(source.get('content', ''))} chars")
                                                    
                                                    # Show content
                                                    st.text_area(
                                                        f"Content {i+1}:",
                                                        value=source.get('content', ''),
                                                        height=100,
                                                        disabled=True,
                                                        label_visibility="collapsed"
                                                    )
                                
                            except Exception as e:
                                st.error(f"Error generating response: {str(e)}")
                    else:
                        st.warning("Please enter a question.")
            # Show Pinecone document history if available
            if st.session_state.get('pinecone_documents'):
                st.subheader("🗄️ Stored Documents")
                docs = st.session_state.pinecone_documents
                
                for doc in docs[-5:]:  # Show last 5 documents
                    with st.expander(f"� {doc['id']} ({doc['chunks']} chunks)"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Chunks:** {doc['chunks']}")
                        with col2:
                            st.write(f"**Created:** {doc['created_at'][:16]}")
                        
                        if st.button(f"🗑️ Delete {doc['id']}", key=f"delete_{doc['id']}"):
                            if 'pinecone_store' in st.session_state:
                                success = st.session_state.pinecone_store.delete_documents(doc['id'])
                                if success:
                                    # Remove from session state
                                    st.session_state.pinecone_documents = [
                                        d for d in st.session_state.pinecone_documents 
                                        if d['id'] != doc['id']
                                    ]
                                    st.experimental_rerun()
        
        else:
            st.info("👆 Upload a document PDF to see parsed information here")
            
            # Show available vector database options
            with st.expander("💡 Vector Database Options"):
                st.write("**🧠 In-Memory Storage:**")
                st.write("- ✅ Fast processing")
                st.write("- ✅ No setup required") 
                st.write("- ❌ Data lost when session ends")
                st.write("- ❌ Limited to single document")
                
                st.write("**🌲 Pinecone Cloud Storage:**") 
                st.write("- ✅ Persistent across sessions")
                st.write("- ✅ Multiple document storage")
                st.write("- ✅ Advanced filtering & search")
                st.write("- ✅ LangChain integration")
                st.write("- ⚙️ Requires API key setup")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🦙 **Local AI Document Parser** • Built with Streamlit & Ollama • "
        "Features: Document Chunking, Vector Search, Local LLM Processing | "
        "[View Source Code](https://github.com/your-repo)"
    )

if __name__ == "__main__":
    # Load custom CSS if available
    if os.path.exists("assets/style.css"):
        load_css()
    
    main()