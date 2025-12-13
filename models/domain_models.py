"""
Domain Models for Resume Parser
Provides type-safe data structures using Pydantic for validation.
"""

from pydantic import BaseModel, Field, validator, EmailStr
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

class DocumentType(str, Enum):
    """Supported document types."""
    RESUME = "resume"
    CV = "cv" 
    COVER_LETTER = "cover_letter"
    PORTFOLIO = "portfolio"

class ComplexityLevel(str, Enum):
    """Document complexity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class SkillCategory(str, Enum):
    """Skill categories for classification."""
    TECHNICAL = "technical"
    SOFT_SKILLS = "soft_skills"
    LANGUAGES = "languages"
    CERTIFICATIONS = "certifications"
    TOOLS = "tools"
    FRAMEWORKS = "frameworks"

class ExperienceItem(BaseModel):
    """Professional experience item."""
    title: str = Field(..., min_length=1, max_length=200, description="Job title")
    company_name: str = Field(..., min_length=1, max_length=200, description="Company name")
    start_time: str = Field(..., description="Start date (flexible format)")
    end_time: str = Field(..., description="End date or 'present'")
    summary: str = Field("", max_length=2000, description="Experience summary")
    location: Optional[str] = Field(None, max_length=200, description="Location")
    
    @validator('end_time')
    def validate_end_time(cls, v, values):
        """Validate end time format."""
        if v.lower() == 'present':
            return v
        # Add more validation logic as needed
        return v
    
    @validator('summary')
    def clean_summary(cls, v):
        """Clean and validate summary text."""
        if v:
            # Remove excessive whitespace
            return ' '.join(v.split())
        return v

class EducationItem(BaseModel):
    """Education item."""
    degree: str = Field(..., min_length=1, max_length=200, description="Degree name")
    institution: str = Field(..., min_length=1, max_length=200, description="Institution name")
    year: Optional[str] = Field(None, max_length=20, description="Graduation year")
    gpa: Optional[str] = Field(None, max_length=10, description="GPA if mentioned")
    location: Optional[str] = Field(None, max_length=200, description="Location")
    
    @validator('gpa')
    def validate_gpa(cls, v):
        """Validate GPA format."""
        if v and v.replace('.', '').replace('/', '').isdigit():
            return v
        return None

class ProjectItem(BaseModel):
    """Project item."""
    name: str = Field(..., min_length=1, max_length=200, description="Project name")
    description: str = Field("", max_length=1000, description="Project description")
    technologies: List[str] = Field(default_factory=list, description="Technologies used")
    duration: Optional[str] = Field(None, max_length=50, description="Project duration")
    url: Optional[str] = Field(None, max_length=500, description="Project URL")
    
    @validator('url')
    def validate_url(cls, v):
        """Validate URL format."""
        if v and not (v.startswith('http://') or v.startswith('https://')):
            return f"https://{v}"
        return v

class ContactInfo(BaseModel):
    """Contact information."""
    email: Optional[str] = Field(None, max_length=254, description="Email address")
    phone: Optional[str] = Field(None, max_length=20, description="Phone number")
    linkedin: Optional[str] = Field(None, max_length=500, description="LinkedIn URL")
    github: Optional[str] = Field(None, max_length=500, description="GitHub URL")
    website: Optional[str] = Field(None, max_length=500, description="Personal website")
    address: Optional[str] = Field(None, max_length=500, description="Address")
    
    @validator('email')
    def validate_email(cls, v):
        """Validate email format."""
        if v and '@' not in v:
            return None
        return v
    
    @validator('linkedin', 'github', 'website')
    def normalize_urls(cls, v):
        """Normalize URL formats."""
        if v and not (v.startswith('http://') or v.startswith('https://')):
            return f"https://{v}"
        return v

class SkillsData(BaseModel):
    """Skills information with categorization."""
    technical: List[str] = Field(default_factory=list, description="Technical skills")
    soft_skills: List[str] = Field(default_factory=list, description="Soft skills") 
    languages: List[str] = Field(default_factory=list, description="Programming languages")
    frameworks: List[str] = Field(default_factory=list, description="Frameworks and libraries")
    tools: List[str] = Field(default_factory=list, description="Tools and software")
    certifications: List[str] = Field(default_factory=list, description="Certifications")
    
    def get_all_skills(self) -> List[str]:
        """Get all skills as a flat list."""
        all_skills = []
        for skill_list in [self.technical, self.soft_skills, self.languages, 
                          self.frameworks, self.tools, self.certifications]:
            all_skills.extend(skill_list)
        return list(set(all_skills))  # Remove duplicates
    
    def get_skills_by_category(self) -> Dict[str, List[str]]:
        """Get skills organized by category."""
        return {
            "technical": self.technical,
            "soft_skills": self.soft_skills,
            "languages": self.languages,
            "frameworks": self.frameworks,
            "tools": self.tools,
            "certifications": self.certifications
        }

class ParsedResumeData(BaseModel):
    """Main resume data structure."""
    # Basic information
    name: str = Field(..., min_length=1, max_length=200, description="Full name")
    contact: ContactInfo = Field(default_factory=ContactInfo, description="Contact information")
    
    # Professional information
    professional_summary: Optional[str] = Field(None, max_length=2000, description="Professional summary")
    experience: List[ExperienceItem] = Field(default_factory=list, description="Work experience")
    education: List[EducationItem] = Field(default_factory=list, description="Education background")
    skills: SkillsData = Field(default_factory=SkillsData, description="Skills information")
    projects: List[ProjectItem] = Field(default_factory=list, description="Projects")
    
    # Metadata
    document_type: DocumentType = Field(DocumentType.RESUME, description="Document type")
    parsing_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Parsing confidence score")
    
    # Processing information
    processing_metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata")
    parsed_at: datetime = Field(default_factory=datetime.now, description="Parse timestamp")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }
        
    def to_legacy_format(self) -> Dict[str, Any]:
        """Convert to legacy format for backward compatibility."""
        return {
            "name": self.name,
            "email": self.contact.email,
            "phone": self.contact.phone,
            "professional_summary": self.professional_summary,
            "experience": [exp.dict() for exp in self.experience],
            "education": [edu.dict() for edu in self.education],
            "skills": self.skills.get_all_skills(),
            "projects": [proj.dict() for proj in self.projects],
            "parsing_confidence": self.parsing_confidence,
            "parsed_at": self.parsed_at.isoformat()
        }
    
    def get_experience_summary(self) -> Dict[str, Any]:
        """Get experience summary statistics."""
        if not self.experience:
            return {"total_positions": 0, "total_companies": 0}
        
        companies = set(exp.company_name for exp in self.experience)
        current_position = any(exp.end_time.lower() == "present" for exp in self.experience)
        
        return {
            "total_positions": len(self.experience),
            "total_companies": len(companies),
            "companies": list(companies),
            "currently_employed": current_position,
            "most_recent_title": self.experience[0].title if self.experience else None
        }
    
    def get_skill_summary(self) -> Dict[str, Any]:
        """Get skills summary."""
        skills_by_category = self.skills.get_skills_by_category()
        total_skills = len(self.skills.get_all_skills())
        
        return {
            "total_skills": total_skills,
            "by_category": {k: len(v) for k, v in skills_by_category.items()},
            "technical_focus": len(self.skills.technical) > len(self.skills.soft_skills),
            "has_certifications": len(self.skills.certifications) > 0
        }

class DocumentAnalysis(BaseModel):
    """Document complexity analysis results."""
    complexity_level: ComplexityLevel = Field(..., description="Overall complexity level")
    character_count: int = Field(..., ge=0, description="Total character count")
    word_count: int = Field(..., ge=0, description="Total word count")
    line_count: int = Field(..., ge=0, description="Total line count")
    
    # Content indicators
    has_tables: bool = Field(False, description="Contains table structures")
    has_code: bool = Field(False, description="Contains code snippets")
    has_math: bool = Field(False, description="Contains mathematical expressions")
    has_lists: bool = Field(False, description="Contains structured lists")
    
    # Technical indicators
    technical_terms_count: int = Field(0, ge=0, description="Number of technical terms")
    complexity_score: float = Field(0.0, ge=0.0, description="Calculated complexity score")
    
    # Recommendations
    recommended_models: List[str] = Field(default_factory=list, description="Recommended AI models")
    processing_time_estimate: float = Field(0.0, ge=0.0, description="Estimated processing time")
    
    def get_size_category(self) -> str:
        """Get document size category."""
        if self.character_count < 3000:
            return "small"
        elif self.character_count < 8000:
            return "medium"
        elif self.character_count < 15000:
            return "large"
        else:
            return "very_large"

class ParsingResult(BaseModel):
    """Complete parsing result with metadata."""
    success: bool = Field(..., description="Whether parsing was successful")
    data: Optional[ParsedResumeData] = Field(None, description="Parsed data if successful")
    
    # Error information
    error_code: Optional[str] = Field(None, description="Error code if failed")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    
    # Processing metadata
    model_used: Optional[str] = Field(None, description="AI model used for parsing")
    processing_time: Optional[float] = Field(None, ge=0.0, description="Processing time in seconds")
    document_analysis: Optional[DocumentAnalysis] = Field(None, description="Document analysis")
    
    # Quality metrics
    confidence_score: float = Field(0.0, ge=0.0, le=1.0, description="Overall confidence")
    completeness_score: float = Field(0.0, ge=0.0, le=1.0, description="Data completeness")
    
    # Provenance
    source_pages: List[int] = Field(default_factory=list, description="Source page numbers")
    chunks_processed: int = Field(0, ge=0, description="Number of chunks processed")
    
    created_at: datetime = Field(default_factory=datetime.now, description="Result timestamp")
    
    def is_high_quality(self) -> bool:
        """Check if result meets high quality thresholds."""
        return (
            self.success and
            self.confidence_score >= 0.7 and
            self.completeness_score >= 0.6 and
            self.data is not None
        )
    
    def get_quality_report(self) -> Dict[str, Any]:
        """Get detailed quality report."""
        if not self.success or not self.data:
            return {"quality": "failed", "reason": self.error_message}
        
        # Calculate quality metrics
        has_name = bool(self.data.name and len(self.data.name.strip()) > 1)
        has_contact = bool(self.data.contact.email or self.data.contact.phone)
        has_experience = len(self.data.experience) > 0
        has_skills = len(self.data.skills.get_all_skills()) > 0
        
        completeness_factors = [has_name, has_contact, has_experience, has_skills]
        completeness = sum(completeness_factors) / len(completeness_factors)
        
        quality_level = "high" if completeness >= 0.75 else "medium" if completeness >= 0.5 else "low"
        
        return {
            "quality": quality_level,
            "confidence_score": self.confidence_score,
            "completeness_score": completeness,
            "completeness_factors": {
                "has_name": has_name,
                "has_contact": has_contact,
                "has_experience": has_experience,
                "has_skills": has_skills
            },
            "data_points": {
                "experience_items": len(self.data.experience),
                "education_items": len(self.data.education),
                "total_skills": len(self.data.skills.get_all_skills()),
                "projects": len(self.data.projects)
            }
        }

# Utility functions for model creation
def create_experience_item(
    title: str, 
    company: str, 
    start: str, 
    end: str, 
    summary: str = "",
    location: str = None
) -> ExperienceItem:
    """Create an experience item with validation."""
    return ExperienceItem(
        title=title.strip(),
        company_name=company.strip(),
        start_time=start.strip(),
        end_time=end.strip(),
        summary=summary.strip(),
        location=location.strip() if location else None
    )

def create_education_item(
    degree: str,
    institution: str,
    year: str = None,
    gpa: str = None,
    location: str = None
) -> EducationItem:
    """Create an education item with validation."""
    return EducationItem(
        degree=degree.strip(),
        institution=institution.strip(),
        year=year.strip() if year else None,
        gpa=gpa.strip() if gpa else None,
        location=location.strip() if location else None
    )

def create_contact_info(
    email: str = None,
    phone: str = None,
    linkedin: str = None,
    github: str = None,
    website: str = None,
    address: str = None
) -> ContactInfo:
    """Create contact info with validation."""
    return ContactInfo(
        email=email.strip() if email else None,
        phone=phone.strip() if phone else None,
        linkedin=linkedin.strip() if linkedin else None,
        github=github.strip() if github else None,
        website=website.strip() if website else None,
        address=address.strip() if address else None
    )

# Validation helpers
def validate_resume_data(data: Dict[str, Any]) -> ParsedResumeData:
    """Validate and convert dictionary to ParsedResumeData."""
    try:
        return ParsedResumeData(**data)
    except Exception as e:
        # Log validation error and return with minimal data
        from core.logging_system import get_logger
        logger = get_logger(__name__)
        logger.error(f"Resume data validation failed: {str(e)}")
        
        # Create minimal valid structure
        return ParsedResumeData(
            name=data.get("name", "Unknown"),
            parsing_confidence=0.1
        )