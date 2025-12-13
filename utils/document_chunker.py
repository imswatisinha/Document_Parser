"""
Document Chunking Utility for Better Embeddings and Retrieval
Intelligent text splitting and processing for LLM-based document analysis
"""

import re
from typing import List, Dict, Any, Tuple
import streamlit as st

class DocumentChunker:
    """Intelligent document chunking for better embeddings and retrieval."""
    
    def __init__(self, 
                 chunk_size: int = 300, 
                 overlap: int = 100,
                 min_chunk_size: int = 100):
        """
        Initialize the document chunker.
        
        Args:
            chunk_size (int): Target size for each chunk in tokens/characters
            overlap (int): Number of characters to overlap between chunks
            min_chunk_size (int): Minimum chunk size to avoid too small chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        
    def chunk_by_sections(self, text: str, pages: List[str] = None) -> List[Dict[str, Any]]:
        """
        Chunk document by logical sections (education, experience, etc.).
        
        Args:
            text (str): Full document text
            pages (List[str]): Optional list of per-page texts
            
        Returns:
            List[Dict]: List of chunks with metadata
        """
        chunks = []
        
        # Define section patterns - more precise to avoid merging
        # section_patterns = {
        #     'personal_info': r'(?i)^(?:name|contact|phone|email|address|linkedin|portfolio|github)[\s:]*([^\n]+)',
        #     'objective': r'(?i)^(?:professional\s+)?(?:summary|objective|profile)[\s:]*\n((?:.*\n?)*?)(?=\n\s*(?:experience|education|work|skill|project|certification|achievement|$)|\n\s*[A-Z][^:]*:)',
        #     'education': r'(?i)^(?:education|academic|qualification)[\s:]*\n((?:.*\n?)*?)(?=\n\s*(?:experience|work|skill|project|certification|achievement|technical|professional|$)|\n\s*[A-Z][^:]*:)',
        #     'experience': r'(?i)^(?:professional\s+)?(?:experience|employment|work|career)[\s:]*\n((?:.*\n?)*?)(?=\n\s*(?:education|skill|project|certification|achievement|technical|$)|\n\s*[A-Z][^:]*:)',
        #     'skills': r'(?i)^(?:technical\s+)?(?:skills?|technologies?|competenc|proficienc)[\s:]*\n((?:.*\n?)*?)(?=\n\s*(?:experience|education|project|certification|achievement|$)|\n\s*[A-Z][^:]*:)',
        #     'projects': r'(?i)^(?:projects?|portfolio)[\s:]*\n((?:.*\n?)*?)(?=\n\s*(?:experience|education|skill|certification|achievement|$)|\n\s*[A-Z][^:]*:)',
        #     'certifications': r'(?i)^(?:certification|certificate|license)[\s:]*\n((?:.*\n?)*?)(?=\n\s*(?:experience|education|skill|project|achievement|$)|\n\s*[A-Z][^:]*:)',
        #     'achievements': r'(?i)^(?:achievements?|awards?|honors?)[\s:]*\n((?:.*\n?)*?)(?=\n\s*(?:experience|education|skill|project|certification|$)|\n\s*[A-Z][^:]*:)'
        # }
        
        section_patterns = {
    'personal_info': r'(?i)^(?:name|contact|phone|email|address|linkedin|portfolio|github)[\s:]*([^\n]+)',

    'objective': r'(?i)^(?:professional\s+)?(?:summary|objective|profile)[\s:]*\n((?:.*\n?)*?)(?=\n\s*(?:experience|education|work|skill|project|certification|achievement|award|honor|publication|$)|\n\s*[A-Z][^:]*:)',

    'education': r'(?i)^(?:education|academic|qualification|academics)[\s:]*\n((?:.*\n?)*?)(?=\n\s*(?:experience|work|skill|project|certification|achievement|award|honor|publication|$)|\n\s*[A-Z][^:]*:)',

    'experience': r'(?i)^(?:professional\s+)?(?:experience|employment|work|career|internship)[\s:]*\n((?:.*\n?)*?)(?=\n\s*(?:education|skill|project|certification|achievement|award|honor|publication|$)|\n\s*[A-Z][^:]*:)',

    'skills': r'(?i)^(?:technical\s+)?(?:skills?|technologies?|competenc|proficienc)[\s:]*\n((?:.*\n?)*?)(?=\n\s*(?:experience|education|project|certification|achievement|award|honor|publication|$)|\n\s*[A-Z][^:]*:)',

    'projects': r'(?i)^(?:projects?|portfolio|personal\s+projects)[\s:]*\n((?:.*\n?)*?)(?=\n\s*(?:experience|education|skill|certification|achievement|award|honor|publication|$)|\n\s*[A-Z][^:]*:)',

    'certifications': r'(?i)^(?:certification|certificate|certifications|licenses?|training)[\s:]*\n((?:.*\n?)*?)(?=\n\s*(?:experience|education|skill|project|achievement|award|honor|publication|$)|\n\s*[A-Z][^:]*:)',

    'achievements': r'(?i)^(?:achievements?|accomplishment|accomplishments)[\s:]*\n((?:.*\n?)*?)(?=\n\s*(?:experience|education|skill|project|certification|award|honor|publication|$)|\n\s*[A-Z][^:]*:)',

    'awards': r'(?i)^(?:awards?|recognitions?|distinctions?)[\s:]*\n((?:.*\n?)*?)(?=\n\s*(?:experience|education|skill|project|certification|achievement|honor|publication|$)|\n\s*[A-Z][^:]*:)',

    'honors': r'(?i)^(?:honors?|honours|titles)[\s:]*\n((?:.*\n?)*?)(?=\n\s*(?:experience|education|skill|project|certification|achievement|award|publication|$)|\n\s*[A-Z][^:]*:)',

    'publications': r'(?i)^(?:publications?|research\s+papers?|papers?|articles?|journals?|conference\s+publications?)\s*:\s*\n?((?:.*\n?)*?)(?=\n\s*(?:experience|education|skill|project|achievement|award|honor|certification|$)|\n\s*[A-Z][^:]*:)'
}

        for section_name, pattern in section_patterns.items():
            matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)
            
            for match in matches:
                section_text = match.group(0).strip()
                if len(section_text) > self.min_chunk_size:
                    
                    # Find page number if pages provided
                    page_num = self._find_page_number(section_text, pages) if pages else None
                    # Heuristic start/end positions
                    start_idx = text.find(section_text)
                    end_idx = start_idx + len(section_text) if start_idx != -1 else None
                    # Simple summary: first 2 sentences or 200 chars
                    summary = self._simple_summary(section_text)
                    
                    chunk = {
                        'content': section_text,
                        'section_type': section_name,
                        'chunk_id': f"{section_name}_{len(chunks)}",
                        'page': page_num,
                        'start_char': start_idx if start_idx != -1 else None,
                        'end_char': end_idx,
                        'summary': summary,
                        'metadata': {
                            'section': section_name,
                            'length': len(section_text),
                            'type': 'section_based'
                        }
                    }
                    chunks.append(chunk)
        
        # If no sections found, fall back to sliding window chunking
        if not chunks:
            chunks = self.chunk_by_sliding_window(text, pages)
            
        return chunks
    
    def chunk_by_sliding_window(self, text: str, pages: List[str] = None) -> List[Dict[str, Any]]:
        """
        Chunk document using sliding window approach.
        
        Args:
            text (str): Full document text
            pages (List[str]): Optional list of per-page texts
            
        Returns:
            List[Dict]: List of chunks with metadata
        """
        chunks = []
        
        # Split by sentences first for better chunk boundaries
        sentences = self._split_into_sentences(text)
        
        current_chunk = ""
        current_sentences = []
        
        for i, sentence in enumerate(sentences):
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk + sentence) > self.chunk_size and current_chunk:
                # Create chunk from current content
                page_num = self._find_page_number(current_chunk, pages) if pages else None
                
                summary = self._simple_summary(current_chunk)
                chunk = {
                    'content': current_chunk.strip(),
                    'section_type': 'general',
                    'chunk_id': f"chunk_{len(chunks)}",
                    'page': page_num,
                    'start_char': None,
                    'end_char': None,
                    'summary': summary,
                    'metadata': {
                        'sentence_count': len(current_sentences),
                        'length': len(current_chunk),
                        'type': 'sliding_window'
                    }
                }
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_sentences = current_sentences[-2:] if len(current_sentences) >= 2 else current_sentences
                current_chunk = " ".join(overlap_sentences)
                current_sentences = overlap_sentences.copy()
            
            # Add current sentence
            current_chunk += " " + sentence if current_chunk else sentence
            current_sentences.append(sentence)
        
        # Add final chunk if there's remaining content
        if current_chunk.strip() and len(current_chunk.strip()) > self.min_chunk_size:
            page_num = self._find_page_number(current_chunk, pages) if pages else None
            
            summary = self._simple_summary(current_chunk)
            chunk = {
                'content': current_chunk.strip(),
                'section_type': 'general',
                'chunk_id': f"chunk_{len(chunks)}",
                'page': page_num,
                'start_char': None,
                'end_char': None,
                'summary': summary,
                'metadata': {
                    'sentence_count': len(current_sentences),
                    'length': len(current_chunk),
                    'type': 'sliding_window'
                }
            }
            chunks.append(chunk)
        
        return chunks
    
    def chunk_by_pages(self, pages: List[str]) -> List[Dict[str, Any]]:
        """
        Chunk document by pages.
        
        Args:
            pages (List[str]): List of per-page texts
            
        Returns:
            List[Dict]: List of chunks with metadata
        """
        chunks = []
        
        for page_num, page_text in enumerate(pages, 1):
            if len(page_text.strip()) > self.min_chunk_size:
                summary = self._simple_summary(page_text)
                chunk = {
                    'content': page_text.strip(),
                    'section_type': 'page',
                    'chunk_id': f"page_{page_num}",
                    'page': page_num,
                    'start_char': None,
                    'end_char': None,
                    'summary': summary,
                    'metadata': {
                        'page_number': page_num,
                        'length': len(page_text),
                        'type': 'page_based'
                    }
                }
                chunks.append(chunk)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex patterns."""
        # Pattern to split on sentence endings
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        
        # Clean and filter sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _find_page_number(self, text: str, pages: List[str]) -> int:
        """Find which page a text chunk belongs to."""
        if not pages:
            return None
            
        for page_num, page_text in enumerate(pages, 1):
            # Check if a significant portion of the chunk appears in this page
            overlap = len(set(text.split()) & set(page_text.split()))
            if overlap > len(text.split()) * 0.5:  # 50% overlap threshold
                return page_num
        
        return 1  # Default to first page

    def _simple_summary(self, text: str) -> str:
        """Return a simple summary: first 2 sentences or first 200 chars."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        candidate = " ".join(sentences[:2]).strip()
        if candidate:
            return candidate[:400]
        return text[:200]
    
    def get_chunking_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the chunking process."""
        if not chunks:
            return {}
        
        total_chunks = len(chunks)
        avg_length = sum(len(chunk['content']) for chunk in chunks) / total_chunks
        
        section_types = {}
        for chunk in chunks:
            section_type = chunk.get('section_type', 'unknown')
            section_types[section_type] = section_types.get(section_type, 0) + 1
        
        return {
            'total_chunks': total_chunks,
            'average_length': int(avg_length),
            'section_distribution': section_types,
            'chunks_with_pages': sum(1 for chunk in chunks if chunk.get('page')),
            'total_content_length': sum(len(chunk['content']) for chunk in chunks)
        }

def create_smart_chunks(text: str, pages: List[str] = None, strategy: str = "sections") -> List[Dict[str, Any]]:
    """
    Create smart chunks from document text.
    
    Args:
        text (str): Full document text
        pages (List[str]): Optional list of per-page texts
        strategy (str): Chunking strategy ("sections", "sliding", "pages")
        
    Returns:
        List[Dict]: List of chunks with metadata
    """
    chunker = DocumentChunker()
    
    if strategy == "sections":
        chunks = chunker.chunk_by_sections(text, pages)
    elif strategy == "sliding":
        chunks = chunker.chunk_by_sliding_window(text, pages)
    elif strategy == "pages" and pages:
        chunks = chunker.chunk_by_pages(pages)
    else:
        # Default to sections with sliding window fallback
        chunks = chunker.chunk_by_sections(text, pages)
    
    return chunks

def display_chunking_results(chunks: List[Dict[str, Any]]):
    """Display chunking results in Streamlit."""
    
    if not chunks:
        st.warning("No chunks created")
        return
    
    # Get stats
    chunker = DocumentChunker()
    stats = chunker.get_chunking_stats(chunks)
    
    # Display stats
    st.subheader("📊 Chunking Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Chunks", stats.get('total_chunks', 0))
    with col2:
        st.metric("Avg Length", f"{stats.get('average_length', 0)} chars")
    with col3:
        st.metric("With Pages", stats.get('chunks_with_pages', 0))
    
    # Display section distribution
    if stats.get('section_distribution'):
        st.write("**Section Distribution:**")
        for section, count in stats['section_distribution'].items():
            st.write(f"• {section.title()}: {count} chunks")
    
    # Display chunks
    st.subheader("📝 Document Chunks")
    
    for i, chunk in enumerate(chunks):
        with st.expander(f"Chunk {i+1}: {chunk['section_type'].title()} (Page {chunk.get('page', '?')})"):
            st.write(f"**Chunk ID:** {chunk['chunk_id']}")
            st.write(f"**Length:** {len(chunk['content'])} characters")
            st.write(f"**Type:** {chunk['metadata'].get('type', 'unknown')}")
            
            if chunk.get('page'):
                st.write(f"**Page:** {chunk['page']}")
            
            st.text_area(
                "Content:",
                value=chunk['content'][:500] + "..." if len(chunk['content']) > 500 else chunk['content'],
                height=150,
                disabled=True
            )