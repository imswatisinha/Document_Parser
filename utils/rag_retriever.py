import requests
import json
import numpy as np
import streamlit as st
from typing import List, Dict, Any, Optional

# Try to import optional dependencies with fallbacks
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("⚠️ scikit-learn not available. Install with: pip install scikit-learn")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    st.info("ℹ️ sentence-transformers not available. Install with: pip install sentence-transformers")

class OllamaRAGRetriever:
    """Local RAG functionality using Ollama for enhanced resume parsing and Q&A."""
    
    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_base_url
        self.chunks = []
        self.chunk_objects = []  # Store structured chunk objects
        self.embeddings = None
        self.vectorizer = None
        self.embedding_model = None
        self.available_models = []
        self.selected_model = None
        self.chunk_metadata = []  # Store metadata for each chunk
        self.setup_embedding_model()
        self.check_ollama_connection()
    
    def check_ollama_connection(self):
        """Check if Ollama is running and get available models."""
        supported_models = ["phi3:mini", "llama3.2:3b"]
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                all_models = [model['name'] for model in models_data.get('models', [])]
                self.available_models = [m for m in all_models if m in supported_models]
                if self.available_models:
                    self.selected_model = self.available_models[0]  # Use first available model
                return True
            return False
        except Exception as e:
            st.warning(f"Ollama connection failed: {str(e)}")
            return False
    
    def setup_embedding_model(self):
        """Setup local embedding model for semantic search."""
        try:
            # Try sentence-transformers first (best option)
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.embedding_method = "sentence_transformers"
                st.success("✅ Using Sentence Transformers for embeddings")
                return
        except Exception as e:
            st.warning(f"Failed to load sentence-transformers: {str(e)}")
        
        try:
            # Fallback to TF-IDF (requires scikit-learn)
            if SKLEARN_AVAILABLE:
                self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                self.embedding_method = "tfidf"
                st.info("ℹ️ Using TF-IDF embeddings (install sentence-transformers for better results)")
                return
        except Exception as e:
            st.warning(f"Failed to setup TF-IDF: {str(e)}")
        
        # Final fallback to simple keyword matching
        self.embedding_method = "keyword_matching"
        st.warning("⚠️ Using basic keyword matching. Install scikit-learn and sentence-transformers for better results.")
    
    def split_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for text chunks."""
        if self.embedding_method == "sentence_transformers":
            return self.embedding_model.encode(texts)
        elif self.embedding_method == "tfidf" and SKLEARN_AVAILABLE:
            return self.vectorizer.fit_transform(texts).toarray()
        else:
            # Simple word count fallback - no external dependencies
            vocab = set()
            for text in texts:
                vocab.update(text.lower().split())
            vocab = list(vocab)
            
            embeddings = []
            for text in texts:
                words = text.lower().split()
                embedding = [words.count(word) for word in vocab]
                embeddings.append(embedding)
            
            return np.array(embeddings, dtype=float)
    
    def setup_rag(self, resume_text: str) -> bool:
        """
        Set up RAG system with the resume text.
        
        Args:
            resume_text (str): The extracted resume text
            
        Returns:
            bool: True if setup successful, False otherwise
        """
        try:
            # Split text into chunks
            self.chunks = self.split_text(resume_text, chunk_size=300, overlap=50)
            
            if not self.chunks:
                st.error("No text chunks created")
                return False
            
            # Create embeddings
            self.embeddings = self.create_embeddings(self.chunks)
            
            st.success(f"✅ RAG setup complete: {len(self.chunks)} chunks, {self.embedding_method} embeddings")
            return True
            
        except Exception as e:
            st.error(f"Failed to setup RAG: {str(e)}")
            return False
    
    def setup_rag_with_chunks(self, chunk_objects: List[Dict[str, Any]]) -> bool:
        """
        Set up RAG system with pre-processed chunk objects.
        
        Args:
            chunk_objects (List[Dict]): List of chunk objects with 'content', 'type', etc.
            
        Returns:
            bool: True if setup successful, False otherwise
        """
        try:
            self.chunk_objects = chunk_objects
            self.chunks = [chunk['content'] for chunk in chunk_objects]
            self.chunk_metadata = [
                {
                    'type': chunk.get('type', 'text'),
                    'page': chunk.get('page', 1),
                    'section': chunk.get('section', 'unknown'),
                    'length': len(chunk['content'])
                }
                for chunk in chunk_objects
            ]
            
            if not self.chunks:
                st.error("No text chunks provided")
                return False
            
            # Create embeddings for all chunks
            self.embeddings = self.create_embeddings(self.chunks)
            
            # Calculate chunk statistics
            total_chars = sum(len(chunk) for chunk in self.chunks)
            avg_length = total_chars // len(self.chunks) if self.chunks else 0
            
            st.success(f"✅ Enhanced RAG setup complete: {len(self.chunks)} chunks, "
                      f"avg length: {avg_length} chars, method: {self.embedding_method}")
            
            return True
            
        except Exception as e:
            st.error(f"Failed to setup enhanced RAG: {str(e)}")
            return False
    
    def calculate_similarity(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """Calculate similarity between query and document embeddings."""
        if SKLEARN_AVAILABLE:
            return cosine_similarity(query_embedding, doc_embeddings)[0]
        else:
            # Manual cosine similarity calculation
            similarities = []
            query_norm = np.linalg.norm(query_embedding)
            
            for doc_embedding in doc_embeddings:
                doc_norm = np.linalg.norm(doc_embedding)
                if query_norm == 0 or doc_norm == 0:
                    similarities.append(0)
                else:
                    dot_product = np.dot(query_embedding[0], doc_embedding)
                    similarity = dot_product / (query_norm * doc_norm)
                    similarities.append(similarity)
            
            return np.array(similarities)
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve most relevant text chunks for a query with metadata."""
        if not self.chunks or self.embeddings is None:
            return []
        
        try:
            # Create embedding for query
            if self.embedding_method == "sentence_transformers":
                query_embedding = self.embedding_model.encode([query])
                similarities = self.calculate_similarity(query_embedding, self.embeddings)
            
            elif self.embedding_method == "tfidf" and SKLEARN_AVAILABLE:
                query_embedding = self.vectorizer.transform([query]).toarray()
                similarities = self.calculate_similarity(query_embedding, self.embeddings)
            
            else:
                # Enhanced keyword matching with section weighting
                query_words = set(query.lower().split())
                similarities = []
                for i, chunk in enumerate(self.chunks):
                    chunk_words = set(chunk.lower().split())
                    overlap = len(query_words & chunk_words)
                    base_score = overlap / max(len(query_words), 1)
                    
                    # Boost score based on chunk type/section if available
                    if self.chunk_metadata and i < len(self.chunk_metadata):
                        metadata = self.chunk_metadata[i]
                        section_type = metadata.get('type', '').lower()
                        
                        # Boost relevant sections for specific query types
                        if any(skill_word in query.lower() for skill_word in ['skill', 'technology', 'programming']):
                            if 'skill' in section_type:
                                base_score *= 1.5
                        elif any(exp_word in query.lower() for exp_word in ['experience', 'work', 'job']):
                            if 'experience' in section_type or 'work' in section_type:
                                base_score *= 1.5
                        elif any(edu_word in query.lower() for edu_word in ['education', 'degree', 'university']):
                            if 'education' in section_type:
                                base_score *= 1.5
                    
                    similarities.append(base_score)
                similarities = np.array(similarities)
            
            # Get top-k most similar chunks with their metadata
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            relevant_chunks = []
            for idx in top_indices:
                if similarities[idx] > 0.05:  # Lower threshold for better recall
                    chunk_data = {
                        'content': self.chunks[idx],
                        'similarity_score': float(similarities[idx]),
                        'chunk_index': idx
                    }
                    
                    # Add metadata if available
                    if self.chunk_metadata and idx < len(self.chunk_metadata):
                        chunk_data.update(self.chunk_metadata[idx])
                    
                    # Add chunk object data if available
                    if self.chunk_objects and idx < len(self.chunk_objects):
                        original_chunk = self.chunk_objects[idx]
                        chunk_data['original_type'] = original_chunk.get('type', 'text')
                        chunk_data['source_page'] = original_chunk.get('page', 1)
                    
                    relevant_chunks.append(chunk_data)
            
            return relevant_chunks
            
        except Exception as e:
            st.warning(f"Retrieval failed: {str(e)}")
            # Fallback: return first few chunks with basic structure
            return [
                {
                    'content': self.chunks[i],
                    'similarity_score': 0.5,
                    'chunk_index': i,
                    'type': 'fallback'
                }
                for i in range(min(top_k, len(self.chunks)))
            ]
    
    def ask_ollama_question(self, question: str, context: str, model: str = None) -> Dict[str, Any]:
        """Ask question to Ollama with context."""
        if not model:
            model = self.selected_model
        
        if not model:
            return {"error": "No Ollama model available"}
        
        prompt = f"""
        Based on the following resume information, answer the question accurately and concisely.
        
        Resume Context:
        {context}
        
        Question: {question}
        
        Answer: Provide a direct, helpful answer based only on the information in the resume context above.
        """
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,  # Low temperature for factual answers
                        "max_tokens": 300
                    }
                },
                timeout=180
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                return {
                    "answer": answer,
                    "model_used": model,
                    "context_chunks": len(context.split('\n\n'))
                }
            else:
                return {"error": f"Ollama API error: {response.status_code}"}
                
        except Exception as e:
            return {"error": f"Failed to get answer from Ollama: {str(e)}"}
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Ask a question about the document using local RAG.
        
        Args:
            question (str): Question about the document
            
        Returns:
            dict: Answer and source information
        """
        if not self.chunks:
            return {"error": "RAG system not initialized. Please setup RAG first."}
        
        if not self.available_models:
            return {"error": "No Ollama models available. Please install a model first."}
        
        try:
            # Retrieve relevant chunks
            relevant_chunks = self.retrieve_relevant_chunks(question, top_k=3)
            
            if not relevant_chunks:
                return {"error": "No relevant information found in the document."}
            
            # Combine chunks as context with metadata
            context_parts = []
            chunk_sources = []
            
            for i, chunk_data in enumerate(relevant_chunks):
                content = chunk_data['content']
                metadata = []
                
                if chunk_data.get('type'):
                    metadata.append(f"Section: {chunk_data['type']}")
                if chunk_data.get('page'):
                    metadata.append(f"Page: {chunk_data['page']}")
                if chunk_data.get('similarity_score'):
                    metadata.append(f"Relevance: {chunk_data['similarity_score']:.2f}")
                
                chunk_header = f"[Chunk {i+1}" + (f" - {', '.join(metadata)}" if metadata else "") + "]"
                context_parts.append(f"{chunk_header}\n{content}")
                
                # Store source info
                chunk_sources.append({
                    'content': content[:200] + "..." if len(content) > 200 else content,
                    'type': chunk_data.get('type', 'unknown'),
                    'page': chunk_data.get('page', 1),
                    'similarity_score': chunk_data.get('similarity_score', 0),
                    'chunk_index': chunk_data.get('chunk_index', i)
                })
            
            context = "\n\n".join(context_parts)
            
            # Ask Ollama
            result = self.ask_ollama_question(question, context, self.selected_model)
            
            if "error" not in result:
                result["sources"] = chunk_sources
                result["retrieval_method"] = self.embedding_method
                result["total_chunks_used"] = len(relevant_chunks)
                result["context_length"] = len(context)
            
            return result
            
        except Exception as e:
            return {"error": f"Failed to answer question: {str(e)}"}
    
    def get_resume_insights(self, resume_text: str) -> Dict[str, Any]:
        """
        Generate comprehensive insights about the resume using local RAG.
        
        Args:
            resume_text (str): The resume text
            
        Returns:
            dict: Various insights about the resume
        """
        if not self.setup_rag(resume_text):
            return {"error": "Could not setup RAG for insights"}
        
        insights = {}
        
        # Predefined questions for insights
        questions = {
            "experience_summary": "Summarize the candidate's work experience and key achievements in 2-3 sentences.",
            "skill_analysis": "What are the main technical and soft skills mentioned in this resume?",
            "career_progression": "How has the candidate's career progressed over time? What growth patterns do you see?",
            "education_relevance": "How relevant is the candidate's education to their career path?",
            "strengths": "What are the candidate's main strengths based on their resume?",
            "years_of_experience": "How many years of professional experience does this candidate have?",
            "notable_achievements": "What are the most notable achievements or accomplishments mentioned?",
            "industry_focus": "What industry or field does this candidate primarily work in?"
        }
        
        st.info(f"🔍 Generating insights using {self.selected_model}...")
        
        for key, question in questions.items():
            with st.spinner(f"Analyzing: {key.replace('_', ' ').title()}"):
                result = self.ask_question(question)
                insights[key] = result.get("answer", "Could not generate insight")
                
                if "error" in result:
                    insights[key] = f"Error: {result['error']}"
        
        insights["_metadata"] = {
            "model_used": self.selected_model,
            "embedding_method": self.embedding_method,
            "total_chunks": len(self.chunks),
            "ollama_available": len(self.available_models) > 0,
            "dependencies": {
                "sklearn": SKLEARN_AVAILABLE,
                "sentence_transformers": SENTENCE_TRANSFORMERS_AVAILABLE
            }
        }
        
        return insights
    
    def answer_question(self, question: str, context: str = None) -> str:
        """
        Simple interface for answering questions - compatible with app.py
        
        Args:
            question (str): The question to answer
            context (str): Optional context (if not provided, uses RAG retrieval)
            
        Returns:
            str: The answer text
        """
        if context:
            # Use provided context directly
            result = self.ask_ollama_question(question, context, self.selected_model)
            return result.get("answer", "Could not generate answer")
        else:
            # Use RAG system
            result = self.ask_question(question)
            if "error" in result:
                return f"Error: {result['error']}"
            return result.get("answer", "Could not generate answer")
    
    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models."""
        self.check_ollama_connection()
        return self.available_models
    
    def set_model(self, model_name: str) -> bool:
        """Set the Ollama model to use."""
        if model_name in self.available_models:
            self.selected_model = model_name
            return True
        return False

# Helper functions for Streamlit integration
def create_ollama_rag_interface(document_text: str = None, chunk_objects: List[Dict[str, Any]] = None):
    """Create Ollama RAG interface in Streamlit sidebar."""
    st.sidebar.header("🤖 Local AI Q&A System")
    
    # Show dependency status
    with st.sidebar.expander("📊 Dependency Status", expanded=False):
        st.write("**scikit-learn:**", "✅ Available" if SKLEARN_AVAILABLE else "❌ Not installed")
        st.write("**sentence-transformers:**", "✅ Available" if SENTENCE_TRANSFORMERS_AVAILABLE else "❌ Not installed")
        
        if not SKLEARN_AVAILABLE:
            st.code("pip install scikit-learn")
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            st.code("pip install sentence-transformers")
    
    # Initialize RAG system
    if 'ollama_rag' not in st.session_state:
        st.session_state.ollama_rag = OllamaRAGRetriever()
    
    rag = st.session_state.ollama_rag
    
    # Check Ollama status
    if not rag.available_models:
        st.sidebar.error("❌ No Ollama models available")
        st.sidebar.info("Please install Ollama and a model first:")
        st.sidebar.code("ollama pull llama3.2:3b")
        return False
    
    # Model selection
    selected_model = st.sidebar.selectbox(
        "Select Model:",
        rag.available_models,
        help="Choose which Ollama model to use for Q&A"
    )
    
    if selected_model != rag.selected_model:
        rag.set_model(selected_model)
        st.sidebar.success(f"✅ Model set to {selected_model}")
    
    # Setup RAG button
    if st.sidebar.button("🔧 Initialize Local Q&A System"):
        with st.spinner("Setting up local AI Q&A system..."):
            setup_success = False
            
            if chunk_objects:
                # Use structured chunks if available
                setup_success = rag.setup_rag_with_chunks(chunk_objects)
                st.sidebar.info("Using structured document chunks")
            elif document_text:
                # Fall back to text-based setup
                setup_success = rag.setup_rag(document_text)
                st.sidebar.info("Using text-based chunking")
            else:
                st.sidebar.error("No document data provided")
                return False
                
            if setup_success:
                st.session_state.rag_ready = True
                st.sidebar.success("✅ Local Q&A system ready!")
            else:
                st.sidebar.error("❌ Failed to initialize Q&A system")
                return False
    
    # Q&A Interface
    if st.session_state.get('rag_ready', False):
        st.sidebar.subheader("❓ Ask Questions")
        
        # Predefined questions
        predefined_questions = [
            "What is the candidate's main expertise?",
            "How many years of experience do they have?", 
            "What are their key technical skills?",
            "What notable achievements are mentioned?",
            "What is their educational background?",
            "What companies have they worked for?"
        ]
        
        selected_predefined = st.sidebar.selectbox(
            "Quick Questions:",
            ["Choose a question..."] + predefined_questions
        )
        
        # Custom question input
        custom_question = st.sidebar.text_input("Or ask your own question:")
        
        question_to_ask = custom_question if custom_question else (
            selected_predefined if selected_predefined != "Choose a question..." else None
        )
        
        if question_to_ask and st.sidebar.button("🔍 Get Answer"):
            with st.spinner(f"Asking {rag.selected_model}..."):
                result = rag.ask_question(question_to_ask)
                
                if "error" in result:
                    st.sidebar.error(f"❌ {result['error']}")
                else:
                    st.sidebar.success("✅ Answer generated!")
                    st.sidebar.write("**Answer:**")
                    st.sidebar.write(result["answer"])
                    
                    # Show metadata
                    st.sidebar.caption(f"Model: {result.get('model_used', 'unknown')} | "
                                     f"Method: {result.get('retrieval_method', 'unknown')}")
                    
                    # Show sources
                    if result.get("sources"):
                        with st.sidebar.expander("📚 Source Context"):
                            for i, source in enumerate(result["sources"], 1):
                                st.sidebar.write(f"**Chunk {i}:**")
                                st.sidebar.write(source[:150] + "...")
        
        # Generate comprehensive insights
        if st.sidebar.button("🧠 Generate Complete Analysis"):
            with st.spinner("Generating comprehensive insights..."):
                # Use document text or reconstruct from chunks
                text_for_insights = document_text
                if not text_for_insights and chunk_objects:
                    text_for_insights = "\n\n".join([chunk['content'] for chunk in chunk_objects])
                
                if text_for_insights:
                    insights = rag.get_resume_insights(text_for_insights)
                    st.session_state.resume_insights = insights
                    if "error" not in insights:
                        st.sidebar.success("✅ Complete analysis generated!")
                else:
                    st.sidebar.error("No text available for analysis")
    
    return st.session_state.get('rag_ready', False)

def display_ollama_resume_insights():
    """Display comprehensive resume insights generated by local AI."""
    if 'resume_insights' not in st.session_state:
        return
    
    insights = st.session_state.resume_insights
    
    if "error" in insights:
        st.error(f"❌ {insights['error']}")
        return
    
    # Show metadata
    metadata = insights.get("_metadata", {})
    st.info(f"🦙 Analysis by: {metadata.get('model_used', 'Unknown')} | "
            f"Method: {metadata.get('embedding_method', 'Unknown')} | "
            f"Chunks: {metadata.get('total_chunks', 0)}")
    
    st.header("🧠 Comprehensive Resume Analysis")
    
    # Create tabs for different insights
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", "💼 Experience", "🛠️ Skills", "🎓 Education", "🏆 Achievements", "📈 Analysis"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👤 Professional Summary")
            st.write(insights.get("experience_summary", "No summary available"))
            
            st.subheader("💪 Key Strengths")
            st.write(insights.get("strengths", "No strengths analysis available"))
        
        with col2:
            st.subheader("⏳ Experience Level")
            st.write(insights.get("years_of_experience", "Experience level not determined"))
            
            st.subheader("🏭 Industry Focus")
            st.write(insights.get("industry_focus", "Industry focus not determined"))
    
    with tab2:
        st.subheader("💼 Work Experience Analysis")
        st.write(insights.get("experience_summary", "No experience analysis available"))
        
        st.subheader("📈 Career Progression")
        st.write(insights.get("career_progression", "No career progression analysis available"))
    
    with tab3:
        st.subheader("🛠️ Skills Assessment")
        st.write(insights.get("skill_analysis", "No skills analysis available"))
    
    with tab4:
        st.subheader("🎓 Educational Background")
        st.write(insights.get("education_relevance", "No education analysis available"))
    
    with tab5:
        st.subheader("🏆 Notable Achievements")
        st.write(insights.get("notable_achievements", "No achievements analysis available"))
    
    with tab6:
        st.subheader("📊 Complete Analysis Summary")
        
        # Create a comprehensive summary
        summary_sections = [
            ("Experience", insights.get("experience_summary", "N/A")),
            ("Skills", insights.get("skill_analysis", "N/A")),
            ("Career Growth", insights.get("career_progression", "N/A")),
            ("Education", insights.get("education_relevance", "N/A")),
            ("Strengths", insights.get("strengths", "N/A"))
        ]
        
        for section, content in summary_sections:
            with st.expander(f"📋 {section} Analysis"):
                st.write(content)
    
    # Download insights as JSON
    insights_json = json.dumps(insights, indent=2)
    st.download_button(
        label="💾 Download Analysis as JSON",
        data=insights_json,
        file_name=f"resume_analysis_{metadata.get('model_used', 'ollama')}.json",
        mime="application/json"
    )