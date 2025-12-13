"""
Resume Parsing Service
Coordinates the parsing process using AI models and post-processing.
"""

import json
import re
import requests
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import time

from config.settings import get_config
from core.exceptions import ParseError, ModelNotAvailableError, ExternalAPIError, handle_errors
from core.logging_system import get_logger, log_performance, log_function_calls
from models.domain_models import (
    ParsedResumeData, ParsingResult, DocumentAnalysis, 
    ExperienceItem, EducationItem, ContactInfo, SkillsData,
    create_experience_item, create_education_item, create_contact_info
)
from services.model_service import get_model_service

logger = get_logger(__name__)

class ResumeParsingService:
    """Main service for parsing resumes using AI models."""
    
    def __init__(self):
        self.config = get_config()
        self.model_service = get_model_service()
        self._parsing_stats = {
            "total_parsed": 0,
            "successful_parses": 0,
            "failed_parses": 0,
            "average_processing_time": 0.0,
            "models_used": {}
        }
        
        logger.info("Resume parsing service initialized")
    
    @log_function_calls(include_args=False, include_result=False)
    @log_performance(threshold_seconds=30.0)
    def parse_resume(
        self,
        text: str,
        pages: List[str] = None,
        preferred_model: str = None,
        chunk_data: List[Dict[str, Any]] = None
    ) -> ParsingResult:
        """
        Parse resume text using AI model.
        
        Args:
            text: Resume text to parse
            pages: List of page texts (for provenance)
            preferred_model: Preferred AI model to use
            chunk_data: Pre-processed document chunks
            
        Returns:
            ParsingResult with parsed data or error information
        """
        start_time = time.time()
        
        try:
            # Update stats
            self._parsing_stats["total_parsed"] += 1
            
            # Select optimal model
            selected_model, doc_analysis = self.model_service.select_optimal_model(
                text, preferred_model
            )
            
            # Track model usage
            if selected_model in self._parsing_stats["models_used"]:
                self._parsing_stats["models_used"][selected_model] += 1
            else:
                self._parsing_stats["models_used"][selected_model] = 1
            
            logger.info(f"Starting resume parsing with model: {selected_model}")
            
            # Parse using selected model
            raw_result = self._parse_with_model(text, selected_model)
            
            # Post-process the results
            parsed_data = self._post_process_results(raw_result, text, pages)
            
            # Calculate quality metrics
            confidence_score = self._calculate_confidence(parsed_data, raw_result)
            completeness_score = self._calculate_completeness(parsed_data)
            
            # Create result
            processing_time = time.time() - start_time
            result = ParsingResult(
                success=True,
                data=parsed_data,
                model_used=selected_model,
                processing_time=processing_time,
                document_analysis=doc_analysis,
                confidence_score=confidence_score,
                completeness_score=completeness_score,
                source_pages=list(range(1, len(pages) + 1)) if pages else [1],
                chunks_processed=len(chunk_data) if chunk_data else 1
            )
            
            # Update stats
            self._parsing_stats["successful_parses"] += 1
            self._update_average_processing_time(processing_time)
            
            logger.info(f"Resume parsing completed successfully in {processing_time:.2f}s "
                       f"(confidence: {confidence_score:.2f}, completeness: {completeness_score:.2f})")
            
            return result
            
        except Exception as e:
            # Update error stats
            self._parsing_stats["failed_parses"] += 1
            
            processing_time = time.time() - start_time
            error_result = ParsingResult(
                success=False,
                error_code=type(e).__name__,
                error_message=str(e),
                processing_time=processing_time,
                document_analysis=doc_analysis if 'doc_analysis' in locals() else None
            )
            
            logger.error(f"Resume parsing failed after {processing_time:.2f}s: {str(e)}")
            
            return error_result
    
    def _parse_with_model(self, text: str, model_name: str) -> Dict[str, Any]:
        """Parse text using specified AI model."""
        prompt = self._build_parsing_prompt(text)
        
        try:
            response = requests.post(
                f"{self.config.ollama.base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for consistent extraction
                        "top_p": 0.9,
                        "num_ctx": min(8192, len(text) + 2000)  # Adjust context based on text length
                    }
                },
                timeout=self.config.ollama.timeout
            )
            
            if response.status_code != 200:
                raise ExternalAPIError("Ollama", response.status_code, response.text)
            
            result = response.json()
            response_text = result.get("response", "")
            
            # Parse JSON response
            try:
                parsed_json = json.loads(response_text)
                return parsed_json
            except json.JSONDecodeError as e:
                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group())
                    except json.JSONDecodeError:
                        pass
                
                # Fallback: parse as plain text
                logger.warning(f"Failed to parse JSON response, using fallback parsing: {str(e)}")
                return self._fallback_text_parsing(response_text, text)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to Ollama failed: {str(e)}")
            raise ExternalAPIError("Ollama", None, str(e))
    
    def _build_parsing_prompt(self, text: str) -> str:
        """Build structured prompt for resume parsing."""
        return f"""
Extract structured information from the following resume text. Return ONLY a valid JSON object with these exact fields:

{{
    "name": "Full name",
    "email": "email@example.com or null",
    "phone": "phone number or null", 
    "professional_summary": "professional summary or null",
    "experience": [
        {{
            "title": "Job title",
            "company_name": "Company name",
            "start_time": "Start date",
            "end_time": "End date or present", 
            "summary": "Job description",
            "location": "Location or null"
        }}
    ],
    "education": [
        {{
            "degree": "Degree name",
            "institution": "School name",
            "year": "Year or null",
            "gpa": "GPA or null",
            "location": "Location or null"
        }}
    ],
    "skills": {{
        "technical": ["list of technical skills"],
        "soft_skills": ["list of soft skills"],
        "languages": ["programming languages"],
        "frameworks": ["frameworks and libraries"],
        "tools": ["tools and software"],
        "certifications": ["certifications"]
    }},
    "projects": [
        {{
            "name": "Project name",
            "description": "Project description", 
            "technologies": ["tech1", "tech2"],
            "duration": "Duration or null",
            "url": "URL or null"
        }}
    ]
}}

Important rules:
1. Return ONLY the JSON object, no additional text
2. Use null for missing information, not empty strings
3. Extract all available information accurately
4. Categorize skills appropriately 
5. Include all work experience and education entries
6. Be consistent with date formats

Resume text:
{text}

JSON:"""
    
    def _fallback_text_parsing(self, response_text: str, original_text: str) -> Dict[str, Any]:
        """Fallback parsing when JSON parsing fails."""
        logger.info("Using fallback text parsing")
        
        # Basic regex-based extraction
        name_match = re.search(r'^([A-Z][a-z]+ [A-Z][a-z]+)', original_text, re.MULTILINE)
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', original_text)
        phone_match = re.search(r'[\+]?[1-9][\d\s\-\(\)]{7,15}', original_text)
        
        return {
            "name": name_match.group(1) if name_match else "Unknown",
            "email": email_match.group() if email_match else None,
            "phone": phone_match.group() if phone_match else None,
            "professional_summary": None,
            "experience": [],
            "education": [],
            "skills": {
                "technical": [],
                "soft_skills": [],
                "languages": [],
                "frameworks": [],
                "tools": [],
                "certifications": []
            },
            "projects": []
        }
    
    def _post_process_results(
        self, 
        raw_result: Dict[str, Any], 
        original_text: str,
        pages: List[str] = None
    ) -> ParsedResumeData:
        """Post-process and enhance raw parsing results."""
        
        # Create structured objects
        try:
            # Process contact information
            contact = create_contact_info(
                email=raw_result.get("email"),
                phone=raw_result.get("phone"),
                linkedin=self._extract_linkedin(original_text),
                github=self._extract_github(original_text),
                website=self._extract_website(original_text)
            )
            
            # Process experience
            experience = []
            for exp_data in raw_result.get("experience", []):
                try:
                    exp_item = create_experience_item(
                        title=exp_data.get("title", ""),
                        company=exp_data.get("company_name", ""),
                        start=exp_data.get("start_time", ""),
                        end=exp_data.get("end_time", ""),
                        summary=exp_data.get("summary", ""),
                        location=exp_data.get("location")
                    )
                    experience.append(exp_item)
                except Exception as e:
                    logger.warning(f"Failed to process experience item: {str(e)}")
            
            # Process education
            education = []
            for edu_data in raw_result.get("education", []):
                try:
                    edu_item = create_education_item(
                        degree=edu_data.get("degree", ""),
                        institution=edu_data.get("institution", ""),
                        year=edu_data.get("year"),
                        gpa=edu_data.get("gpa"),
                        location=edu_data.get("location")
                    )
                    education.append(edu_item)
                except Exception as e:
                    logger.warning(f"Failed to process education item: {str(e)}")
            
            # Process skills
            skills_data = raw_result.get("skills", {})
            skills = SkillsData(
                technical=self._clean_skills_list(skills_data.get("technical", [])),
                soft_skills=self._clean_skills_list(skills_data.get("soft_skills", [])),
                languages=self._clean_skills_list(skills_data.get("languages", [])),
                frameworks=self._clean_skills_list(skills_data.get("frameworks", [])),
                tools=self._clean_skills_list(skills_data.get("tools", [])),
                certifications=self._clean_skills_list(skills_data.get("certifications", []))
            )
            
            # Process projects (if any)
            projects = []
            for proj_data in raw_result.get("projects", []):
                try:
                    from models.domain_models import ProjectItem
                    project = ProjectItem(
                        name=proj_data.get("name", ""),
                        description=proj_data.get("description", ""),
                        technologies=proj_data.get("technologies", []),
                        duration=proj_data.get("duration"),
                        url=proj_data.get("url")
                    )
                    projects.append(project)
                except Exception as e:
                    logger.warning(f"Failed to process project item: {str(e)}")
            
            # Create final parsed data
            parsed_data = ParsedResumeData(
                name=raw_result.get("name", "Unknown"),
                contact=contact,
                professional_summary=raw_result.get("professional_summary"),
                experience=experience,
                education=education,
                skills=skills,
                projects=projects,
                parsing_confidence=0.8,  # Will be recalculated
                processing_metadata={
                    "original_text_length": len(original_text),
                    "pages_count": len(pages) if pages else 1,
                    "processing_timestamp": datetime.now().isoformat()
                }
            )
            
            return parsed_data
            
        except Exception as e:
            logger.error(f"Post-processing failed: {str(e)}")
            # Return minimal valid structure
            return ParsedResumeData(
                name=raw_result.get("name", "Unknown"),
                parsing_confidence=0.1
            )
    
    def _extract_linkedin(self, text: str) -> Optional[str]:
        """Extract LinkedIn URL from text."""
        patterns = [
            r'linkedin\.com/in/[a-zA-Z0-9\-]+',
            r'www\.linkedin\.com/in/[a-zA-Z0-9\-]+',
            r'https?://(?:www\.)?linkedin\.com/in/[a-zA-Z0-9\-]+'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                url = match.group()
                if not url.startswith('http'):
                    url = 'https://' + url
                return url
        return None
    
    def _extract_github(self, text: str) -> Optional[str]:
        """Extract GitHub URL from text.""" 
        patterns = [
            r'github\.com/[a-zA-Z0-9\-]+',
            r'www\.github\.com/[a-zA-Z0-9\-]+',
            r'https?://(?:www\.)?github\.com/[a-zA-Z0-9\-]+'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                url = match.group()
                if not url.startswith('http'):
                    url = 'https://' + url
                return url
        return None
    
    def _extract_website(self, text: str) -> Optional[str]:
        """Extract personal website URL from text."""
        # Look for URLs that aren't LinkedIn/GitHub
        url_pattern = r'https?://[^\s/$.?#].[^\s]*'
        matches = re.findall(url_pattern, text, re.IGNORECASE)
        
        for url in matches:
            if 'linkedin.com' not in url.lower() and 'github.com' not in url.lower():
                return url
        
        return None
    
    def _clean_skills_list(self, skills: List[str]) -> List[str]:
        """Clean and deduplicate skills list."""
        if not skills:
            return []
        
        cleaned = []
        seen = set()
        
        for skill in skills:
            if not isinstance(skill, str):
                continue
                
            # Clean the skill
            clean_skill = skill.strip()
            clean_skill = re.sub(r'\s+', ' ', clean_skill)  # Normalize whitespace
            
            # Skip empty or very short skills
            if len(clean_skill) < 2:
                continue
                
            # Skip duplicates (case insensitive)
            if clean_skill.lower() not in seen:
                seen.add(clean_skill.lower())
                cleaned.append(clean_skill)
        
        return cleaned[:20]  # Limit to 20 skills per category
    
    def _calculate_confidence(self, parsed_data: ParsedResumeData, raw_result: Dict[str, Any]) -> float:
        """Calculate parsing confidence score."""
        confidence = 0.0
        max_score = 10.0
        
        # Name extraction (2 points)
        if parsed_data.name and len(parsed_data.name.strip()) > 1:
            confidence += 2.0
        
        # Contact information (2 points)
        if parsed_data.contact.email:
            confidence += 1.0
        if parsed_data.contact.phone:
            confidence += 1.0
        
        # Experience (3 points)
        if parsed_data.experience:
            confidence += min(len(parsed_data.experience) * 0.5, 3.0)
        
        # Skills (2 points)
        total_skills = len(parsed_data.skills.get_all_skills())
        if total_skills > 0:
            confidence += min(total_skills * 0.1, 2.0)
        
        # Education (1 point)
        if parsed_data.education:
            confidence += 1.0
        
        return min(confidence / max_score, 1.0)
    
    def _calculate_completeness(self, parsed_data: ParsedResumeData) -> float:
        """Calculate data completeness score."""
        completeness_factors = [
            bool(parsed_data.name and len(parsed_data.name.strip()) > 1),
            bool(parsed_data.contact.email or parsed_data.contact.phone),
            bool(parsed_data.experience),
            bool(parsed_data.skills.get_all_skills()),
            bool(parsed_data.education),
            bool(parsed_data.professional_summary)
        ]
        
        return sum(completeness_factors) / len(completeness_factors)
    
    def _update_average_processing_time(self, new_time: float):
        """Update running average of processing time."""
        current_avg = self._parsing_stats["average_processing_time"]
        total_successful = self._parsing_stats["successful_parses"]
        
        if total_successful == 1:
            self._parsing_stats["average_processing_time"] = new_time
        else:
            # Running average calculation
            new_avg = ((current_avg * (total_successful - 1)) + new_time) / total_successful
            self._parsing_stats["average_processing_time"] = new_avg
    
    def get_parsing_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""
        success_rate = 0.0
        if self._parsing_stats["total_parsed"] > 0:
            success_rate = self._parsing_stats["successful_parses"] / self._parsing_stats["total_parsed"]
        
        return {
            **self._parsing_stats,
            "success_rate": success_rate,
            "error_rate": 1.0 - success_rate
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check of parsing service."""
        try:
            # Check if models are available
            models = self.model_service.get_available_models()
            
            # Test connection to Ollama
            response = requests.get(
                f"{self.config.ollama.base_url}/api/tags",
                timeout=5
            )
            
            ollama_healthy = response.status_code == 200
            
            return {
                "healthy": ollama_healthy and len(models) > 0,
                "ollama_connection": ollama_healthy,
                "available_models": len(models),
                "models": models[:3],  # Show first 3 models
                "service_stats": self.get_parsing_statistics()
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "service_stats": self.get_parsing_statistics()
            }

# Global service instance
_parsing_service: Optional[ResumeParsingService] = None

def get_parsing_service() -> ResumeParsingService:
    """Get the global resume parsing service."""
    global _parsing_service
    if _parsing_service is None:
        _parsing_service = ResumeParsingService()
    return _parsing_service

# Convenience function
def parse_resume_text(
    text: str, 
    pages: List[str] = None,
    preferred_model: str = None
) -> ParsingResult:
    """Parse resume text using the service."""
    return get_parsing_service().parse_resume(text, pages, preferred_model)