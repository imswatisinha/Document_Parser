"""
Pinecone Vector Database Integration with LangChain
Enhanced document storage, retrieval, and persistent vector search
"""

import os
import hashlib
import streamlit as st
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime

# Core LangChain imports
LANGCHAIN_AVAILABLE = False
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Pinecone
    from langchain_core.documents import Document
    from langchain_core.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    pass

# Pinecone imports
PINECONE_AVAILABLE = False
try:
    from pinecone import Pinecone as PineconeClient, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    pass

# Fallback to sentence-transformers if available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

class PineconeVectorStore:
    """Advanced vector store using Pinecone and LangChain for persistent document storage."""
    
    def __init__(self, 
                 api_key: str = None, 
                 environment: str = "us-east-1-aws",
                 index_name: str = "resume-parser",
                 dimension: int = 384):
        """
        Initialize Pinecone vector store.
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment 
            index_name: Name of the Pinecone index
            dimension: Embedding dimension (384 for all-MiniLM-L6-v2)
        """
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.dimension = dimension
        self.index = None
        self.vectorstore = None
        self.embeddings = None
        self.text_splitter = None
        self.retrieval_qa = None
        
        # Status flags
        self.is_initialized = False
        self.dependencies_ok = self._check_dependencies()
        
        if self.dependencies_ok and api_key:
            self._initialize_pinecone()
            self._setup_embeddings()
            self._setup_text_splitter()
    
    def _check_dependencies(self) -> bool:
        """Check if all required dependencies are available."""
        missing = []
        if not LANGCHAIN_AVAILABLE:
            missing.append("langchain (pip install langchain langchain-community langchain-text-splitters)")
        if not PINECONE_AVAILABLE:
            missing.append("pinecone-client (pip install pinecone-client)")
        
        if missing:
            st.error("Missing dependencies:")
            for dep in missing:
                st.write(f"• {dep}")
            return False
        return True
    
    def _initialize_pinecone(self):
        """Initialize Pinecone client and index."""
        try:
            # Initialize Pinecone
            pc = PineconeClient(api_key=self.api_key)
            
            # Create index if it doesn't exist
            existing_indexes = pc.list_indexes()
            index_names = [index.name for index in existing_indexes.indexes]
            
            if self.index_name not in index_names:
                st.info(f"Creating new Pinecone index: {self.index_name}")
                pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=self.environment.split('-')[0] + "-" + self.environment.split('-')[1]
                    )
                )
                st.success(f"✅ Created Pinecone index: {self.index_name}")
            
            # Connect to index
            self.index = pc.Index(self.index_name)
            st.success(f"✅ Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            st.error(f"Failed to initialize Pinecone: {str(e)}")
            return False
        
        return True
    
    def _setup_embeddings(self):
        """Setup embedding model for vector generation."""
        try:
            if LANGCHAIN_AVAILABLE:
                # Use LangChain HuggingFace embeddings
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                st.success("✅ Using LangChain HuggingFace embeddings")
            
            elif SENTENCE_TRANSFORMERS_AVAILABLE:
                # Fallback to direct sentence transformers
                self.embeddings = SentenceTransformer('all-MiniLM-L6-v2')
                st.info("Using direct Sentence Transformers (install LangChain for better integration)")
            
            else:
                st.error("No embedding model available. Please install sentence-transformers or transformers.")
                return False
                
        except Exception as e:
            st.error(f"Failed to setup embeddings: {str(e)}")
            return False
        
        return True
    
    def _setup_text_splitter(self):
        """Setup intelligent text splitting using LangChain."""
        if not LANGCHAIN_AVAILABLE:
            return
        
        # Recursive text splitter - best for most documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        st.success("✅ LangChain text splitter initialized")
    
    def setup_vectorstore(self):
        """Setup LangChain Pinecone vectorstore."""
        if not all([self.index, self.embeddings, LANGCHAIN_AVAILABLE]):
            st.error("Cannot setup vectorstore - missing components")
            return False
        
        try:
            self.vectorstore = Pinecone(
                index=self.index,
                embedding=self.embeddings,
                text_key="content"
            )
            self.is_initialized = True
            st.success("✅ Pinecone vectorstore ready!")
            return True
            
        except Exception as e:
            st.error(f"Failed to setup vectorstore: {str(e)}")
            return False
    
    def add_documents(self, documents: List[Dict[str, Any]], document_id: str = None) -> bool:
        """
        Add documents to Pinecone with enhanced metadata.
        
        Args:
            documents: List of document chunks with content and metadata
            document_id: Unique identifier for the document
            
        Returns:
            bool: Success status
        """
        if not self.is_initialized:
            st.error("Vectorstore not initialized")
            return False
        
        try:
            # Generate document ID if not provided
            if not document_id:
                content_hash = hashlib.md5(
                    "".join([doc.get('content', '') for doc in documents]).encode()
                ).hexdigest()[:8]
                document_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{content_hash}"
            
            # Convert to LangChain Documents
            langchain_docs = []
            
            for i, doc in enumerate(documents):
                content = doc.get('content', '')
                if not content.strip():
                    continue
                
                # Enhanced metadata
                metadata = {
                    'document_id': document_id,
                    'chunk_index': i,
                    'chunk_type': doc.get('type', 'text'),
                    'page': doc.get('page', 1),
                    'section': doc.get('section', 'unknown'),
                    'created_at': datetime.now().isoformat(),
                    'length': len(content)
                }
                
                # Add any additional metadata
                for key, value in doc.items():
                    if key not in ['content'] and key not in metadata:
                        metadata[key] = str(value)
                
                langchain_docs.append(Document(
                    page_content=content,
                    metadata=metadata
                ))
            
            if not langchain_docs:
                st.warning("No valid documents to add")
                return False
            
            # Add to Pinecone via LangChain
            self.vectorstore.add_documents(langchain_docs)
            
            st.success(f"✅ Added {len(langchain_docs)} chunks to Pinecone (ID: {document_id})")
            
            # Store document ID for later reference
            if 'pinecone_documents' not in st.session_state:
                st.session_state.pinecone_documents = []
            st.session_state.pinecone_documents.append({
                'id': document_id,
                'chunks': len(langchain_docs),
                'created_at': datetime.now().isoformat()
            })
            
            return True
            
        except Exception as e:
            st.error(f"Failed to add documents to Pinecone: {str(e)}")
            return False
    
    def similarity_search(self, query: str, k: int = 3, filter_dict: Dict = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents using Pinecone.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Metadata filters
            
        Returns:
            List of similar documents with metadata
        """
        if not self.is_initialized:
            st.error("Vectorstore not initialized")
            return []
        
        try:
            # Perform similarity search
            results = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filter_dict
            )
            
            # Convert to our format
            formatted_results = []
            for doc in results:
                result = {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'document_id': doc.metadata.get('document_id'),
                    'chunk_type': doc.metadata.get('chunk_type', 'text'),
                    'page': doc.metadata.get('page', 1),
                    'section': doc.metadata.get('section', 'unknown')
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            st.error(f"Search failed: {str(e)}")
            return []
    
    def setup_qa_chain(self, model_name: str = "llama3.2:3b"):
        """
        Setup LangChain QA chain with Ollama and Pinecone retriever.
        
        Args:
            model_name: Ollama model name to use
        """
        if not self.is_initialized:
            st.error("Vectorstore not initialized")
            return False
        
        try:
            # For now, we'll use a simpler approach without RetrievalQA
            # Store the model name for later use
            self.qa_model = model_name
            st.success(f"✅ QA ready with {model_name} (using direct retrieval)")
            return True
            
        except Exception as e:
            st.error(f"Failed to setup QA: {str(e)}")
            return False
    
    def ask_question(self, question: str, k: int = 3) -> Dict[str, Any]:
        """
        Ask a question using RAG with improved timeout and error handling.
        
        Args:
            question (str): Question to ask
            k (int): Number of relevant chunks to retrieve
            
        Returns:
            dict: Answer with metadata and source information
        """
        if not self.pinecone_available:
            return {"error": "Pinecone not available"}
        
        if not self.ollama_parser:
            return {"error": "Ollama parser not available"}
        
        if not self.index_name:
            return {"error": "No Pinecone index configured"}
        
        try:
            # Step 1: Create embeddings for the question
            with st.spinner("🔍 Searching relevant document sections..."):
                question_embedding = self.embeddings.embed_query(question)
            
            # Step 2: Search for relevant chunks
            with st.spinner("📚 Retrieving context..."):
                search_results = self.index.query(
                    vector=question_embedding,
                    top_k=k,
                    include_metadata=True,
                    include_values=False
                )
            
            if not search_results.matches:
                return {
                    "answer": "I couldn't find relevant information in the uploaded documents to answer your question.",
                    "sources": [],
                    "success": False
                }
            
            # Step 3: Prepare context from retrieved chunks
            context_chunks = []
            sources = []
            
            for match in search_results.matches:
                if match.score > 0.3:  # Only use relevant matches
                    chunk_text = match.metadata.get('text', '')
                    context_chunks.append(chunk_text)
                    sources.append({
                        'text': chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                        'score': float(match.score),
                        'page': match.metadata.get('page', 'Unknown'),
                        'section': match.metadata.get('section', 'Unknown')
                    })
            
            if not context_chunks:
                return {
                    "answer": "The retrieved information doesn't seem relevant to your question. Please try rephrasing or asking a more specific question.",
                    "sources": [],
                    "success": False
                }
            
            # Step 4: Combine context and ask Ollama with progress indicator
            context = "\n\n".join(context_chunks)
            
            with st.spinner("🤖 Generating AI response... This may take a moment for complex questions."):
                # Add a timeout warning
                timeout_warning = st.empty()
                timeout_warning.warning("⏱️ Processing your question... Large documents may take up to 3 minutes.")
                
                ollama_response = self.ollama_parser.ask_question(
                    question=question,
                    context=context,
                    max_tokens=800
                )
                
                timeout_warning.empty()  # Clear the warning
            
            # Step 5: Process the response
            if ollama_response.get("success", False):
                return {
                    "answer": ollama_response["answer"],
                    "sources": sources,
                    "model_used": ollama_response.get("model_used", "unknown"),
                    "chunks_used": len(context_chunks),
                    "success": True
                }
            else:
                # Handle Ollama errors gracefully
                error_msg = ollama_response.get("error", "Unknown error")
                suggestion = ollama_response.get("suggestion", "")
                
                return {
                    "error": error_msg,
                    "suggestion": suggestion,
                    "sources": sources,  # Still provide sources for reference
                    "success": False
                }
                
        except Exception as e:
            return {
                "error": f"Failed to process question: {str(e)}",
                "suggestion": "Try asking a simpler question or check if Ollama is running properly.",
                "success": False
            }
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the Pinecone index."""
        if not self.index:
            return {"error": "Index not initialized"}
        
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": list(stats.namespaces.keys()) if stats.namespaces else []
            }
        except Exception as e:
            return {"error": f"Failed to get stats: {str(e)}"}
    
    def delete_documents(self, document_id: str = None, filter_dict: Dict = None):
        """Delete documents from Pinecone index."""
        if not self.index:
            st.error("Index not initialized")
            return False
        
        try:
            if document_id:
                # Delete by document ID
                filter_dict = {"document_id": document_id}
            
            if filter_dict:
                self.index.delete(filter=filter_dict)
                st.success(f"✅ Deleted documents matching filter: {filter_dict}")
                return True
            else:
                st.warning("No filter provided for deletion")
                return False
                
        except Exception as e:
            st.error(f"Failed to delete documents: {str(e)}")
            return False

# Streamlit integration functions
def setup_pinecone_interface():
    """Setup Pinecone configuration in Streamlit sidebar."""
    st.sidebar.header("🌲 Pinecone Vector Database")
    
    # API Key input
    api_key = st.sidebar.text_input(
        "Pinecone API Key:",
        type="password",
        help="Get your API key from https://pinecone.io"
    )
    
    if not api_key:
        st.sidebar.warning("⚠️ Please enter your Pinecone API key")
        st.sidebar.info("Sign up at https://pinecone.io to get a free API key")
        return None
    
    # Environment and index settings
    with st.sidebar.expander("⚙️ Advanced Settings"):
        environment = st.selectbox(
            "Environment:",
            ["us-east-1-aws", "us-west-2-aws", "eu-west-1-aws"],
            help="Choose your Pinecone environment"
        )
        
        index_name = st.text_input(
            "Index Name:",
            value="resume-parser",
            help="Name for your Pinecone index"
        )
        
        dimension = st.selectbox(
            "Embedding Dimension:",
            [384, 512, 768, 1536],
            index=0,
            help="384 for all-MiniLM-L6-v2, 1536 for OpenAI"
        )
    
    # Initialize Pinecone
    if st.sidebar.button("🔧 Connect to Pinecone"):
        with st.spinner("Connecting to Pinecone..."):
            pinecone_store = PineconeVectorStore(
                api_key=api_key,
                environment=environment,
                index_name=index_name,
                dimension=dimension
            )
            
            if pinecone_store._initialize_pinecone():
                pinecone_store.setup_vectorstore()
                st.session_state.pinecone_store = pinecone_store
                st.sidebar.success("✅ Connected to Pinecone!")
            else:
                st.sidebar.error("❌ Failed to connect to Pinecone")
                return None
    
    return st.session_state.get('pinecone_store')

def display_pinecone_stats(pinecone_store: PineconeVectorStore):
    """Display Pinecone index statistics."""
    if not pinecone_store:
        return
    
    stats = pinecone_store.get_index_stats()
    
    if "error" not in stats:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Vectors", stats.get("total_vectors", 0))
        with col2:
            st.metric("Dimension", stats.get("dimension", 0))
        with col3:
            fullness = stats.get("index_fullness", 0)
            st.metric("Index Fullness", f"{fullness:.1%}")
        
        if stats.get("namespaces"):
            st.write("**Namespaces:**", ", ".join(stats["namespaces"]))
    else:
        st.error(f"Could not fetch stats: {stats['error']}")