"""
Document Processing Service
Handles PDF parsing, text extraction, and document preprocessing.
"""

import fitz  # PyMuPDF
import io
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import hashlib
import re
from datetime import datetime

from config.settings import get_config
from core.exceptions import DocumentProcessingError, ValidationError
from core.logging_system import get_logger, log_performance, log_function_calls
from models.domain_models import DocumentAnalysis

logger = get_logger(__name__)

class DocumentProcessingService:
    """Service for processing and extracting text from documents."""
    
    def __init__(self):
        self.config = get_config()
        self._processing_stats = {
            "documents_processed": 0,
            "total_pages_processed": 0,
            "extraction_failures": 0,
            "average_pages_per_doc": 0.0
        }
        
        logger.info("Document processing service initialized")
    
    @log_function_calls(include_args=False)
    @log_performance(threshold_seconds=10.0)
    def extract_text_from_pdf(self, file_content: bytes, filename: str = None) -> Tuple[str, List[str], DocumentAnalysis]:
        """
        Extract text from PDF file content.
        
        Args:
            file_content: PDF file bytes
            filename: Optional filename for logging
            
        Returns:
            Tuple of (combined_text, page_texts, document_analysis)
        """
        if not file_content:
            raise ValidationError("Empty file content provided")
        
        try:
            # Open PDF from bytes
            pdf_document = fitz.open(stream=file_content, filetype="pdf")
            
            page_texts = []
            combined_text = ""
            
            # Extract text from each page
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                page_text = page.get_text()
                
                # Clean and normalize page text
                cleaned_text = self._clean_text(page_text)
                page_texts.append(cleaned_text)
                combined_text += cleaned_text + "\n\n"
            
            pdf_document.close()
            
            # Analyze document
            doc_analysis = self._analyze_document(combined_text, page_texts, filename)
            
            # Update statistics
            self._update_processing_stats(len(page_texts))
            
            logger.info(f"Successfully extracted text from PDF: {len(page_texts)} pages, "
                       f"{len(combined_text)} characters")
            
            return combined_text.strip(), page_texts, doc_analysis
            
        except Exception as e:
            self._processing_stats["extraction_failures"] += 1
            logger.error(f"PDF text extraction failed for {filename}: {str(e)}")
            raise DocumentProcessingError(f"Failed to extract text from PDF: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', text)
        
        # Normalize line breaks around section headers
        text = re.sub(r'\n\s*([A-Z][A-Z\s]{3,})\s*\n', r'\n\n\1\n\n', text)
        
        # Clean up email formatting
        text = re.sub(r'(\S+)@(\S+)\.(\S+)', r'\1@\2.\3', text)
        
        # Fix phone number formatting
        text = re.sub(r'(\d{3})\s*[-\.\s]\s*(\d{3})\s*[-\.\s]\s*(\d{4})', r'\1-\2-\3', text)
        
        return text.strip()
    
    def _analyze_document(self, text: str, pages: List[str], filename: str = None) -> DocumentAnalysis:
        """Analyze document characteristics for processing optimization."""
        
        total_chars = len(text)
        total_words = len(text.split()) if text else 0
        
        # Analyze complexity
        complexity_score = self._calculate_complexity_score(text)
        
        # Detect document language (simple heuristic)
        language = self._detect_language(text)
        
        # Analyze structure
        structure_analysis = self._analyze_structure(text)
        
        # Generate document hash for deduplication
        doc_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        return DocumentAnalysis(
            page_count=len(pages),
            total_characters=total_chars,
            total_words=total_words,
            complexity_score=complexity_score,
            estimated_processing_time=self._estimate_processing_time(complexity_score, total_words),
            language=language,
            structure_analysis=structure_analysis,
            document_hash=doc_hash,
            filename=filename,
            analysis_timestamp=datetime.now()
        )
    
    def _calculate_complexity_score(self, text: str) -> float:
        """Calculate document complexity score (0.0 to 1.0)."""
        if not text:
            return 0.0
        
        complexity = 0.0
        
        # Length factor (normalized to 0.3)
        word_count = len(text.split())
        length_score = min(word_count / 1000, 1.0) * 0.3
        complexity += length_score
        
        # Structure complexity (0.3)
        sections = len(re.findall(r'\n\s*[A-Z][A-Z\s]{2,}\n', text))
        bullets = len(re.findall(r'[•\-\*]\s+', text))
        structure_score = min((sections * 0.1 + bullets * 0.01), 1.0) * 0.3
        complexity += structure_score
        
        # Technical content (0.2)
        technical_keywords = [
            'python', 'javascript', 'java', 'sql', 'api', 'framework',
            'database', 'machine learning', 'ai', 'cloud', 'aws', 'azure',
            'docker', 'kubernetes', 'react', 'angular', 'vue'
        ]
        technical_matches = sum(1 for keyword in technical_keywords if keyword.lower() in text.lower())
        technical_score = min(technical_matches / len(technical_keywords), 1.0) * 0.2
        complexity += technical_score
        
        # Formatting complexity (0.2)
        special_chars = len(re.findall(r'[^\w\s]', text))
        formatting_score = min(special_chars / len(text) * 10, 1.0) * 0.2
        complexity += formatting_score
        
        return min(complexity, 1.0)
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection (placeholder for more sophisticated detection)."""
        if not text:
            return "unknown"
        
        # Simple heuristics - could be replaced with langdetect or similar
        english_indicators = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'with', 'as']
        text_lower = text.lower()
        
        english_score = sum(1 for indicator in english_indicators if indicator in text_lower)
        
        if english_score >= 3:
            return "english"
        else:
            return "unknown"
    
    def _analyze_structure(self, text: str) -> Dict[str, Any]:
        """Analyze document structure."""
        if not text:
            return {}
        
        # Count different structural elements
        sections = len(re.findall(r'\n\s*([A-Z][A-Z\s]{2,})\n', text))
        bullet_points = len(re.findall(r'[•\-\*]\s+', text))
        phone_numbers = len(re.findall(r'[\+]?[1-9][\d\s\-\(\)]{7,15}', text))
        emails = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
        urls = len(re.findall(r'https?://[^\s]+', text))
        
        # Detect common resume sections
        resume_sections = {
            'experience': bool(re.search(r'\b(experience|employment|work history)\b', text, re.IGNORECASE)),
            'education': bool(re.search(r'\b(education|academic|degree)\b', text, re.IGNORECASE)),
            'skills': bool(re.search(r'\b(skills|competencies|technologies)\b', text, re.IGNORECASE)),
            'projects': bool(re.search(r'\b(projects|portfolio)\b', text, re.IGNORECASE)),
            'contact': bool(emails > 0 or phone_numbers > 0)
        }
        
        return {
            'sections_count': sections,
            'bullet_points': bullet_points,
            'contact_elements': {
                'emails': emails,
                'phones': phone_numbers,
                'urls': urls
            },
            'resume_sections': resume_sections,
            'estimated_format': 'resume' if sum(resume_sections.values()) >= 3 else 'document'
        }
    
    def _estimate_processing_time(self, complexity_score: float, word_count: int) -> float:
        """Estimate processing time in seconds."""
        base_time = 2.0  # Base processing time
        complexity_multiplier = 1.0 + (complexity_score * 2.0)  # 1.0 to 3.0
        word_factor = word_count / 500.0  # Additional time per 500 words
        
        estimated_time = base_time * complexity_multiplier + word_factor
        return min(estimated_time, 60.0)  # Cap at 60 seconds
    
    def preprocess_for_chunking(self, text: str, max_chunk_size: int = 2000) -> List[Dict[str, Any]]:
        """
        Preprocess text and create chunks for processing.
        
        Args:
            text: Input text to chunk
            max_chunk_size: Maximum characters per chunk
            
        Returns:
            List of chunk dictionaries with metadata
        """
        if not text:
            return []
        
        # Split text into logical sections first
        sections = self._split_into_sections(text)
        
        chunks = []
        chunk_id = 1
        
        for section_name, section_text in sections.items():
            if len(section_text) <= max_chunk_size:
                # Section fits in one chunk
                chunks.append({
                    'id': chunk_id,
                    'text': section_text,
                    'section': section_name,
                    'character_count': len(section_text),
                    'word_count': len(section_text.split()),
                    'is_complete_section': True
                })
                chunk_id += 1
            else:
                # Split large section into smaller chunks
                section_chunks = self._split_large_section(section_text, max_chunk_size)
                for i, chunk_text in enumerate(section_chunks):
                    chunks.append({
                        'id': chunk_id,
                        'text': chunk_text,
                        'section': f"{section_name}_part_{i+1}",
                        'character_count': len(chunk_text),
                        'word_count': len(chunk_text.split()),
                        'is_complete_section': False,
                        'part_number': i + 1,
                        'total_parts': len(section_chunks)
                    })
                    chunk_id += 1
        
        logger.info(f"Created {len(chunks)} chunks from {len(sections)} sections")
        return chunks
    
    def _split_into_sections(self, text: str) -> Dict[str, str]:
        """Split text into logical sections."""
        sections = {}
        
        # Common resume section headers
        section_patterns = [
            (r'\b(summary|profile|objective)\b', 'summary'),
            (r'\b(experience|employment|work\s+history)\b', 'experience'),
            (r'\b(education|academic|qualifications)\b', 'education'),
            (r'\b(skills|competencies|technologies)\b', 'skills'),
            (r'\b(projects|portfolio)\b', 'projects'),
            (r'\b(certifications?|licenses?)\b', 'certifications'),
            (r'\b(awards?|honors?|achievements?)\b', 'awards'),
        ]
        
        # Find section boundaries
        section_starts = []
        for pattern, section_name in section_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
            for match in matches:
                section_starts.append((match.start(), section_name, match.group()))
        
        # Sort by position
        section_starts.sort()
        
        if not section_starts:
            # No clear sections found, return entire text as one section
            return {'content': text}
        
        # Extract sections
        for i, (start_pos, section_name, header) in enumerate(section_starts):
            end_pos = section_starts[i + 1][0] if i + 1 < len(section_starts) else len(text)
            section_text = text[start_pos:end_pos].strip()
            
            if section_text and len(section_text) > 10:  # Skip very short sections
                sections[section_name] = section_text
        
        # If we have leftover text at the beginning, add it as header section
        if section_starts and section_starts[0][0] > 50:
            header_text = text[:section_starts[0][0]].strip()
            if header_text:
                sections = {'header': header_text, **sections}
        
        return sections if sections else {'content': text}
    
    def _split_large_section(self, text: str, max_size: int) -> List[str]:
        """Split a large section into smaller chunks while preserving meaning."""
        if len(text) <= max_size:
            return [text]
        
        chunks = []
        
        # Try to split on paragraphs first
        paragraphs = text.split('\n\n')
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) <= max_size:
                current_chunk += paragraph + '\n\n'
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # If single paragraph is too large, split on sentences
                if len(paragraph) > max_size:
                    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                    temp_chunk = ""
                    
                    for sentence in sentences:
                        if len(temp_chunk) + len(sentence) <= max_size:
                            temp_chunk += sentence + ' '
                        else:
                            if temp_chunk.strip():
                                chunks.append(temp_chunk.strip())
                            temp_chunk = sentence + ' '
                    
                    current_chunk = temp_chunk
                else:
                    current_chunk = paragraph + '\n\n'
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def validate_file_upload(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Validate uploaded file."""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'file_info': {}
        }
        
        try:
            # Check file size
            file_size = len(file_content)
            max_size = self.config.document_processing.max_file_size_mb * 1024 * 1024
            
            if file_size > max_size:
                validation_result['valid'] = False
                validation_result['errors'].append(f"File size ({file_size/1024/1024:.1f}MB) exceeds limit ({self.config.document_processing.max_file_size_mb}MB)")
            
            # Check if it's a valid PDF
            try:
                pdf_doc = fitz.open(stream=file_content, filetype="pdf")
                page_count = pdf_doc.page_count
                pdf_doc.close()
                
                validation_result['file_info'] = {
                    'size_mb': file_size / 1024 / 1024,
                    'page_count': page_count,
                    'format': 'PDF'
                }
                
                # Warn about large documents
                if page_count > 10:
                    validation_result['warnings'].append(f"Large document ({page_count} pages) may take longer to process")
                
            except Exception as e:
                validation_result['valid'] = False
                validation_result['errors'].append("Invalid PDF file or corrupted content")
        
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"File validation failed: {str(e)}")
        
        return validation_result
    
    def _update_processing_stats(self, page_count: int):
        """Update processing statistics."""
        self._processing_stats["documents_processed"] += 1
        self._processing_stats["total_pages_processed"] += page_count
        
        # Update average pages per document
        docs_processed = self._processing_stats["documents_processed"]
        total_pages = self._processing_stats["total_pages_processed"]
        self._processing_stats["average_pages_per_doc"] = total_pages / docs_processed if docs_processed > 0 else 0.0
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {**self._processing_stats}
    
    def health_check(self) -> Dict[str, bool]:
        """Perform health check."""
        try:
            # Test basic PDF processing capability
            test_pdf = self._create_test_pdf()
            text, pages, analysis = self.extract_text_from_pdf(test_pdf, "health_check.pdf")
            
            return {
                "healthy": True,
                "pdf_processing": len(text) > 0,
                "text_extraction": len(pages) > 0,
                "analysis_working": analysis is not None
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "healthy": False,
                "error": str(e)
            }
    
    def _create_test_pdf(self) -> bytes:
        """Create a minimal test PDF for health checks."""
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Test Document\n\nName: Test User\nEmail: test@example.com")
        
        pdf_bytes = doc.tobytes()
        doc.close()
        
        return pdf_bytes

# Global service instance
_document_service: Optional[DocumentProcessingService] = None

def get_document_service() -> DocumentProcessingService:
    """Get the global document processing service."""
    global _document_service
    if _document_service is None:
        _document_service = DocumentProcessingService()
    return _document_service