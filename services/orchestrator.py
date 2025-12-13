"""
Main Application Orchestrator
Coordinates all services and handles the complete resume processing workflow.
"""

from typing import Dict, List, Any, Optional, Tuple
import time
from datetime import datetime

from config.settings import get_config
from core.exceptions import handle_errors, ResumeParserException
from core.logging_system import get_logger, log_performance, log_function_calls
from models.domain_models import ParsedResumeData, ParsingResult, DocumentAnalysis

# Service imports
from services.document_service import get_document_service
from services.parsing_service import get_parsing_service
from services.model_service import get_model_service
from services.rag_service import get_rag_service

logger = get_logger(__name__)

class ResumeParserOrchestrator:
    """Main orchestrator for the resume parsing application."""
    
    def __init__(self):
        self.config = get_config()
        
        # Initialize services
        self.document_service = get_document_service()
        self.parsing_service = get_parsing_service()
        self.model_service = get_model_service()
        self.rag_service = get_rag_service()
        
        # Application state
        self._processing_history = []
        self._session_stats = {
            "documents_processed": 0,
            "successful_parses": 0,
            "failed_parses": 0,
            "total_processing_time": 0.0,
            "session_start": datetime.now()
        }
        
        logger.info("Resume parser orchestrator initialized")
    
    @log_function_calls(include_args=False, include_result=False)
    @log_performance(threshold_seconds=60.0)
    def process_resume(
        self,
        file_content: bytes,
        filename: str,
        options: Dict[str, Any] = None,
        progress_callback: Optional[callable] = None
    ) -> ParsingResult:
        """
        Complete resume processing workflow.
        
        Args:
            file_content: PDF file bytes
            filename: Original filename
            options: Processing options (model selection, etc.)
            progress_callback: Optional callback for progress updates
            
        Returns:
            ParsingResult with parsed data or error information
        """
        start_time = time.time()
        processing_id = f"proc_{int(time.time())}"
        
        # Initialize options
        options = options or {}
        preferred_model = options.get('selected_model')
        use_rag = options.get('use_rag', False)
        enable_caching = options.get('enable_caching', True)
        confidence_threshold = options.get('confidence_threshold', 0.7)
        
        try:
            logger.info(f"Starting resume processing: {filename} (ID: {processing_id})")
            
            # Update progress
            self._update_progress(progress_callback, "Validating file", 0.1)
            
            # Step 1: Validate file
            validation_result = self.document_service.validate_file_upload(file_content, filename)
            if not validation_result['valid']:
                return ParsingResult(
                    success=False,
                    error_code="ValidationError",
                    error_message="; ".join(validation_result['errors']),
                    processing_time=time.time() - start_time
                )
            
            # Update progress
            self._update_progress(progress_callback, "Extracting text from PDF", 0.2)
            
            # Step 2: Extract text from PDF
            try:
                combined_text, page_texts, doc_analysis = self.document_service.extract_text_from_pdf(
                    file_content, filename
                )
                
                logger.info(f"Text extraction completed: {len(combined_text)} chars, {len(page_texts)} pages")
                
            except Exception as e:
                logger.error(f"Text extraction failed: {str(e)}")
                return ParsingResult(
                    success=False,
                    error_code="TextExtractionError",
                    error_message=f"Failed to extract text from PDF: {str(e)}",
                    processing_time=time.time() - start_time
                )
            
            # Update progress
            self._update_progress(progress_callback, "Preparing document for analysis", 0.3)
            
            # Step 3: Prepare document chunks (optimized - only chunk if needed)
            chunk_data = None
            # Only chunk if document is large enough to benefit from chunking
            if len(combined_text) > 5000 and doc_analysis.complexity_score > 0.7:
                chunk_data = self.document_service.preprocess_for_chunking(
                    combined_text, 
                    max_chunk_size=options.get('chunk_size', 2000)
                )
                logger.info(f"Document chunked into {len(chunk_data)} segments")
            else:
                logger.debug(f"Skipping chunking for small document ({len(combined_text)} chars)")
            
            # Update progress
            self._update_progress(progress_callback, "Running AI analysis", 0.4)
            
            # Step 4: Parse with AI model
            parsing_result = self.parsing_service.parse_resume(
                text=combined_text,
                pages=page_texts,
                preferred_model=preferred_model,
                chunk_data=chunk_data
            )
            
            if not parsing_result.success:
                logger.error(f"Parsing failed: {parsing_result.error_message}")
                return parsing_result
            
            # Update progress
            self._update_progress(progress_callback, "Enhancing with retrieval data", 0.7)
            
            # Step 5: Enhance with RAG (if enabled)
            if use_rag and parsing_result.data:
                try:
                    enhanced_data = self._enhance_with_rag(parsing_result.data, combined_text)
                    if enhanced_data:
                        parsing_result.data = enhanced_data
                        logger.info("Results enhanced with RAG")
                except Exception as e:
                    logger.warning(f"RAG enhancement failed: {str(e)}")
                    # Continue without RAG enhancement
            
            # Update progress
            self._update_progress(progress_callback, "Finalizing results", 0.9)
            
            # Step 6: Post-process and validate results
            final_result = self._post_process_results(
                parsing_result, 
                doc_analysis, 
                filename,
                confidence_threshold
            )
            
            # Step 7: Store results (if caching enabled)
            if enable_caching and final_result.success:
                self._store_processing_result(processing_id, final_result, combined_text)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_session_stats(final_result.success, processing_time)
            
            # Add processing metadata
            final_result.processing_time = processing_time
            final_result.processing_id = processing_id
            
            # Update progress
            self._update_progress(progress_callback, "Complete", 1.0)
            
            logger.info(f"Resume processing completed: {filename} "
                       f"(success: {final_result.success}, time: {processing_time:.2f}s)")
            
            return final_result
            
        except Exception as e:
            # Handle unexpected errors
            processing_time = time.time() - start_time
            self._update_session_stats(False, processing_time)
            
            logger.error(f"Unexpected error in resume processing: {str(e)}")
            
            return ParsingResult(
                success=False,
                error_code=type(e).__name__,
                error_message=f"Processing failed: {str(e)}",
                processing_time=processing_time
            )
    
    def _update_progress(self, callback: Optional[callable], message: str, progress: float):
        """Update progress if callback is provided."""
        if callback:
            try:
                callback(message, progress)
            except Exception as e:
                logger.warning(f"Progress callback failed: {str(e)}")
    
    def _enhance_with_rag(self, parsed_data: ParsedResumeData, original_text: str) -> Optional[ParsedResumeData]:
        """Enhance parsed data using RAG retrieval."""
        try:
            # Find similar resumes for context
            similar_docs = self.rag_service.find_similar_documents(
                original_text[:1000],  # Use first 1000 chars for similarity
                top_k=3,
                score_threshold=0.7
            )
            
            if not similar_docs:
                return None
            
            # Extract patterns and enhance data
            enhanced_data = parsed_data
            
            # Example enhancements based on similar documents
            # (This could be more sophisticated)
            
            # Enhance skills based on similar profiles
            if similar_docs:
                similar_skills = []
                for doc in similar_docs:
                    metadata = doc.get('metadata', {})
                    if 'skills' in metadata:
                        similar_skills.extend(metadata.get('skills', []))
                
                # Add missing skills that are common in similar profiles
                if similar_skills and hasattr(enhanced_data.skills, 'technical'):
                    current_skills = [skill.lower() for skill in enhanced_data.skills.technical]
                    for skill in similar_skills:
                        if skill.lower() not in current_skills and skill.lower() in original_text.lower():
                            enhanced_data.skills.technical.append(skill)
            
            logger.info(f"Enhanced data using {len(similar_docs)} similar documents")
            return enhanced_data
            
        except Exception as e:
            logger.error(f"RAG enhancement failed: {str(e)}")
            return None
    
    def _post_process_results(
        self, 
        result: ParsingResult, 
        doc_analysis: DocumentAnalysis,
        filename: str,
        confidence_threshold: float
    ) -> ParsingResult:
        """Post-process and validate parsing results."""
        
        if not result.success:
            return result
        
        # Validate confidence threshold
        if result.confidence_score < confidence_threshold:
            logger.warning(f"Low confidence score: {result.confidence_score:.2f} < {confidence_threshold}")
            result.warnings = result.warnings or []
            result.warnings.append(f"Low confidence score ({result.confidence_score:.2f})")
        
        # Add document analysis to result
        result.document_analysis = doc_analysis
        
        # Add processing metadata
        if result.data:
            if hasattr(result.data, 'processing_metadata'):
                result.data.processing_metadata.update({
                    'filename': filename,
                    'processing_timestamp': datetime.now().isoformat(),
                    'document_analysis': doc_analysis.__dict__ if doc_analysis else None
                })
        
        # Validate data completeness
        completeness_issues = self._validate_data_completeness(result.data)
        if completeness_issues:
            result.warnings = result.warnings or []
            result.warnings.extend(completeness_issues)
        
        return result
    
    def _validate_data_completeness(self, data: ParsedResumeData) -> List[str]:
        """Validate completeness of parsed data."""
        issues = []
        
        if not data:
            return ["No data extracted"]
        
        # Check essential fields
        if not data.name or len(data.name.strip()) < 2:
            issues.append("Name not found or invalid")
        
        if not data.contact or (not data.contact.email and not data.contact.phone):
            issues.append("No contact information found")
        
        if not data.experience:
            issues.append("No work experience found")
        
        if not data.skills or not data.skills.get_all_skills():
            issues.append("No skills information found")
        
        return issues
    
    def _store_processing_result(self, processing_id: str, result: ParsingResult, text: str):
        """Store processing result for caching."""
        try:
            # Store in vector database for future similarity searches
            if result.success and result.data:
                metadata = {
                    'processing_id': processing_id,
                    'name': result.data.name,
                    'skills': result.data.skills.get_all_skills() if result.data.skills else [],
                    'experience_count': len(result.data.experience) if result.data.experience else 0,
                    'education_count': len(result.data.education) if result.data.education else 0,
                    'confidence_score': result.confidence_score,
                    'model_used': result.model_used
                }
                
                # Store document embedding
                self.rag_service.store_document_embedding(
                    document_id=processing_id,
                    text=text,
                    metadata=metadata
                )
                
                logger.info(f"Stored processing result: {processing_id}")
            
        except Exception as e:
            logger.warning(f"Failed to store processing result: {str(e)}")
    
    def _update_session_stats(self, success: bool, processing_time: float):
        """Update session statistics."""
        self._session_stats["documents_processed"] += 1
        self._session_stats["total_processing_time"] += processing_time
        
        if success:
            self._session_stats["successful_parses"] += 1
        else:
            self._session_stats["failed_parses"] += 1
    
    def get_available_models(self) -> List[str]:
        """Get list of available AI models."""
        try:
            return self.model_service.get_available_models()
        except Exception as e:
            logger.error(f"Failed to get available models: {str(e)}")
            return []
    
    def get_application_health(self) -> Dict[str, Any]:
        """Get comprehensive application health status."""
        health_status = {
            "healthy": True,
            "timestamp": datetime.now().isoformat(),
            "services": {},
            "session_stats": self._session_stats
        }
        
        try:
            # Check all services
            services_to_check = [
                ("document_service", self.document_service),
                ("parsing_service", self.parsing_service),
                ("model_service", self.model_service),
                ("rag_service", self.rag_service)
            ]
            
            for service_name, service in services_to_check:
                try:
                    service_health = service.health_check()
                    health_status["services"][service_name] = service_health
                    
                    # Update overall health
                    if not service_health.get("healthy", False):
                        health_status["healthy"] = False
                        
                except Exception as e:
                    health_status["services"][service_name] = {
                        "healthy": False,
                        "error": str(e)
                    }
                    health_status["healthy"] = False
            
        except Exception as e:
            health_status["healthy"] = False
            health_status["error"] = str(e)
        
        return health_status
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        stats = {
            "session": self._session_stats,
            "services": {}
        }
        
        # Get statistics from each service
        try:
            stats["services"]["parsing"] = self.parsing_service.get_parsing_statistics()
            stats["services"]["document"] = self.document_service.get_processing_statistics()
            stats["services"]["rag"] = self.rag_service.get_service_statistics()
            stats["services"]["model"] = self.model_service.get_service_statistics()
        except Exception as e:
            logger.error(f"Failed to get service statistics: {str(e)}")
        
        # Calculate derived metrics
        total_docs = stats["session"]["documents_processed"]
        if total_docs > 0:
            stats["session"]["success_rate"] = stats["session"]["successful_parses"] / total_docs
            stats["session"]["average_processing_time"] = stats["session"]["total_processing_time"] / total_docs
        else:
            stats["session"]["success_rate"] = 0.0
            stats["session"]["average_processing_time"] = 0.0
        
        return stats
    
    def clear_caches(self):
        """Clear all service caches."""
        try:
            self.rag_service.clear_cache()
            logger.info("Cleared all service caches")
        except Exception as e:
            logger.error(f"Failed to clear caches: {str(e)}")
    
    def shutdown(self):
        """Graceful shutdown of the orchestrator."""
        logger.info("Shutting down resume parser orchestrator")
        
        # Log final session statistics
        final_stats = self.get_processing_statistics()
        logger.info(f"Session summary: {final_stats['session']}")
        
        # Clear caches
        self.clear_caches()
        
        logger.info("Orchestrator shutdown complete")

# Global orchestrator instance
_orchestrator: Optional[ResumeParserOrchestrator] = None

def get_orchestrator() -> ResumeParserOrchestrator:
    """Get the global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ResumeParserOrchestrator()
    return _orchestrator

# Convenience function for main processing workflow
def process_resume_file(
    file_content: bytes,
    filename: str,
    options: Dict[str, Any] = None,
    progress_callback: Optional[callable] = None
) -> ParsingResult:
    """
    Process a resume file using the complete workflow.
    
    This is the main entry point for resume processing.
    """
    orchestrator = get_orchestrator()
    return orchestrator.process_resume(file_content, filename, options, progress_callback)