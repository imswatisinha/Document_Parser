"""
UI Service
Handles Streamlit interface components and user interactions.
"""

import streamlit as st
from typing import Dict, List, Any, Optional, Callable
import time
from datetime import datetime
import json

from config.settings import get_config
from core.logging_system import get_logger
from models.domain_models import ParsedResumeData, ParsingResult, DocumentAnalysis

logger = get_logger(__name__)

class UIService:
    """Service for managing Streamlit UI components."""
    
    def __init__(self):
        self.config = get_config()
        logger.info("UI service initialized")
    
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Resume Parser with LLM",
            page_icon="📄",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Load custom CSS if available
        self._load_custom_css()
    
    def _load_custom_css(self):
        """Load custom CSS styling."""
        try:
            css_path = self.config.app.static_path / "style.css"
            if css_path.exists():
                with open(css_path) as f:
                    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
                logger.debug("Custom CSS loaded")
        except Exception as e:
            logger.warning(f"Failed to load custom CSS: {str(e)}")
    
    def render_header(self):
        """Render application header."""
        st.markdown("""
        # 📄 Resume Parser with LLM
        
        Upload your resume and get AI-powered parsing with intelligent analysis.
        """)
        
        # Add divider
        st.markdown("---")
    
    def render_sidebar(self) -> Dict[str, Any]:
        """Render sidebar with configuration options."""
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Model selection
            model_options = self._get_available_models()
            selected_model = st.selectbox(
                "AI Model",
                options=model_options,
                help="Choose the AI model for parsing"
            )
            
            # Processing options
            st.subheader("Processing Options")
            
            use_rag = st.checkbox(
                "Enable RAG (Retrieval)",
                value=True,
                help="Use retrieval augmented generation for better context"
            )
            
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Minimum confidence score for results"
            )
            
            # Advanced options
            with st.expander("🔧 Advanced Options"):
                chunk_size = st.number_input(
                    "Chunk Size",
                    min_value=500,
                    max_value=5000,
                    value=2000,
                    step=100,
                    help="Maximum characters per processing chunk"
                )
                
                enable_caching = st.checkbox(
                    "Enable Caching",
                    value=True,
                    help="Cache results for faster repeated processing"
                )
                
                debug_mode = st.checkbox(
                    "Debug Mode",
                    value=False,
                    help="Show detailed processing information"
                )
            
            # Service status
            self._render_service_status()
            
            return {
                "selected_model": selected_model,
                "use_rag": use_rag,
                "confidence_threshold": confidence_threshold,
                "chunk_size": chunk_size,
                "enable_caching": enable_caching,
                "debug_mode": debug_mode
            }
    
    def _get_available_models(self) -> List[str]:
        """Get list of available AI models."""
        return ["phi3:mini", "llama3.2:3b"]
    
    def _render_service_status(self):
        """Render service health status in sidebar."""
        st.subheader("🔧 Service Status")
        
        try:
            # Get service health status
            from services.parsing_service import get_parsing_service
            from services.rag_service import get_rag_service
            
            parsing_service = get_parsing_service()
            rag_service = get_rag_service()
            
            # Parsing service health
            parsing_health = parsing_service.health_check()
            self._render_status_indicator("Parsing Service", parsing_health.get("healthy", False))
            
            # RAG service health
            rag_health = rag_service.health_check()
            self._render_status_indicator("RAG Service", rag_health.get("healthy", False))
            
            # Model availability
            models_available = len(self._get_available_models()) > 0
            self._render_status_indicator("AI Models", models_available)
            
        except Exception as e:
            st.error(f"Failed to check service status: {str(e)}")
    
    def _render_status_indicator(self, service_name: str, healthy: bool):
        """Render a status indicator for a service."""
        icon = "🟢" if healthy else "🔴"
        status = "Healthy" if healthy else "Issues"
        st.markdown(f"{icon} **{service_name}**: {status}")
    
    def render_file_upload(self) -> Optional[bytes]:
        """Render file upload interface."""
        st.subheader("📤 Upload Resume")
        
        uploaded_file = st.file_uploader(
            "Choose your resume (PDF only)",
            type=['pdf'],
            help="Upload a PDF file of your resume for parsing",
            accept_multiple_files=False
        )
        
        if uploaded_file is not None:
            # Show file info
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📄 File", uploaded_file.name)
            with col2:
                st.metric("📊 Size", f"{file_size_mb:.1f} MB")
            with col3:
                st.metric("📋 Type", "PDF")
            
            return uploaded_file.getvalue()
        
        return None
    
    def render_processing_progress(self, stages: List[str]) -> Any:
        """Render processing progress indicator."""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        class ProgressUpdater:
            def __init__(self, progress_bar, status_text, stages):
                self.progress_bar = progress_bar
                self.status_text = status_text
                self.stages = stages
                self.current_stage = 0
            
            def update(self, stage_name: str, progress: float = None):
                if stage_name in self.stages:
                    self.current_stage = self.stages.index(stage_name)
                
                # Calculate overall progress
                stage_progress = progress or 1.0
                overall_progress = (self.current_stage + stage_progress) / len(self.stages)
                
                self.progress_bar.progress(min(overall_progress, 1.0))
                self.status_text.text(f"🔄 {stage_name}...")
            
            def complete(self):
                self.progress_bar.progress(1.0)
                self.status_text.text("✅ Processing completed!")
            
            def error(self, message: str):
                self.status_text.text(f"❌ Error: {message}")
        
        return ProgressUpdater(progress_bar, status_text, stages)
    
    def render_document_analysis(self, analysis: DocumentAnalysis):
        """Render document analysis results."""
        if not analysis:
            return
        
        st.subheader("📊 Document Analysis")
        
        # Create metrics columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📄 Pages", analysis.page_count)
        
        with col2:
            st.metric("📝 Words", f"{analysis.total_words:,}")
        
        with col3:
            st.metric("🧠 Complexity", f"{analysis.complexity_score:.2f}")
        
        with col4:
            est_time = analysis.estimated_processing_time
            st.metric("⏱️ Est. Time", f"{est_time:.1f}s")
        
        # Additional analysis details
        if hasattr(analysis, 'structure_analysis') and analysis.structure_analysis:
            with st.expander("🔍 Detailed Analysis"):
                structure = analysis.structure_analysis
                
                # Resume sections detected
                if 'resume_sections' in structure:
                    sections = structure['resume_sections']
                    st.markdown("**Resume Sections Detected:**")
                    
                    section_cols = st.columns(len(sections))
                    for i, (section, detected) in enumerate(sections.items()):
                        with section_cols[i % len(section_cols)]:
                            icon = "✅" if detected else "❌"
                            st.markdown(f"{icon} {section.title()}")
                
                # Structure metrics
                if 'contact_elements' in structure:
                    contact = structure['contact_elements']
                    st.markdown("**Contact Information:**")
                    st.markdown(f"- Emails: {contact.get('emails', 0)}")
                    st.markdown(f"- Phone numbers: {contact.get('phones', 0)}")
                    st.markdown(f"- URLs: {contact.get('urls', 0)}")
    
    def render_parsing_results(self, result: ParsingResult, config: Dict[str, Any]):
        """Render parsing results."""
        if not result.success:
            # Show error
            st.error(f"❌ Parsing failed: {result.error_message}")
            if config.get("debug_mode", False):
                with st.expander("🐛 Debug Information"):
                    st.json({
                        "error_code": result.error_code,
                        "processing_time": result.processing_time,
                        "model_used": result.model_used
                    })
            return
        
        # Show success metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("✅ Success", "Yes")
        
        with col2:
            confidence = result.confidence_score
            st.metric("🎯 Confidence", f"{confidence:.2f}", delta=f"{confidence - 0.5:.2f}")
        
        with col3:
            completeness = result.completeness_score
            st.metric("📊 Completeness", f"{completeness:.2f}", delta=f"{completeness - 0.5:.2f}")
        
        with col4:
            st.metric("⏱️ Time", f"{result.processing_time:.1f}s")
        
        # Render parsed data
        self._render_parsed_data(result.data, config)
        
        # Debug information
        if config.get("debug_mode", False):
            with st.expander("🐛 Debug Information"):
                st.json({
                    "model_used": result.model_used,
                    "chunks_processed": result.chunks_processed,
                    "source_pages": result.source_pages,
                    "document_analysis": result.document_analysis.__dict__ if result.document_analysis else None
                })
    
    def _render_parsed_data(self, data: ParsedResumeData, config: Dict[str, Any]):
        """Render structured parsed resume data."""
        st.subheader("📋 Parsed Resume Data")
        
        # Create tabs for different sections
        tabs = st.tabs(["👤 Personal", "💼 Experience", "🎓 Education", "💡 Skills", "🚀 Projects"])
        
        # Personal Information Tab
        with tabs[0]:
            self._render_personal_info(data)
        
        # Experience Tab
        with tabs[1]:
            self._render_experience(data.experience)
        
        # Education Tab
        with tabs[2]:
            self._render_education(data.education)
        
        # Skills Tab
        with tabs[3]:
            self._render_skills(data.skills)
        
        # Projects Tab
        with tabs[4]:
            self._render_projects(data.projects)
    
    def _render_personal_info(self, data: ParsedResumeData):
        """Render personal information section."""
        st.markdown("### 👤 Personal Information")
        
        # Name
        st.markdown(f"**Name:** {data.name}")
        
        # Contact information
        if data.contact:
            contact = data.contact
            
            col1, col2 = st.columns(2)
            with col1:
                if contact.email:
                    st.markdown(f"**Email:** {contact.email}")
                if contact.phone:
                    st.markdown(f"**Phone:** {contact.phone}")
            
            with col2:
                if contact.linkedin:
                    st.markdown(f"**LinkedIn:** [{contact.linkedin}]({contact.linkedin})")
                if contact.github:
                    st.markdown(f"**GitHub:** [{contact.github}]({contact.github})")
                if contact.website:
                    st.markdown(f"**Website:** [{contact.website}]({contact.website})")
        
        # Professional summary
        if data.professional_summary:
            st.markdown("**Professional Summary:**")
            st.markdown(data.professional_summary)
    
    def _render_experience(self, experience: List):
        """Render work experience section."""
        st.markdown("### 💼 Work Experience")
        
        if not experience:
            st.info("No work experience found in the resume.")
            return
        
        for i, exp in enumerate(experience):
            with st.container():
                st.markdown(f"#### {exp.title} at {exp.company}")
                
                # Duration and location
                duration = f"{exp.start_date} - {exp.end_date}"
                if exp.location:
                    duration += f" | {exp.location}"
                st.markdown(f"*{duration}*")
                
                # Description
                if exp.description:
                    st.markdown(exp.description)
                
                if i < len(experience) - 1:
                    st.markdown("---")
    
    def _render_education(self, education: List):
        """Render education section."""
        st.markdown("### 🎓 Education")
        
        if not education:
            st.info("No education information found in the resume.")
            return
        
        for i, edu in enumerate(education):
            with st.container():
                st.markdown(f"#### {edu.degree}")
                st.markdown(f"**{edu.institution}**")
                
                details = []
                if edu.graduation_year:
                    details.append(f"Year: {edu.graduation_year}")
                if edu.gpa:
                    details.append(f"GPA: {edu.gpa}")
                if edu.location:
                    details.append(f"Location: {edu.location}")
                
                if details:
                    st.markdown(" | ".join(details))
                
                if i < len(education) - 1:
                    st.markdown("---")
    
    def _render_skills(self, skills):
        """Render skills section."""
        st.markdown("### 💡 Skills")
        
        if not skills:
            st.info("No skills information found in the resume.")
            return
        
        # Get all skills
        all_skills = skills.get_all_skills() if hasattr(skills, 'get_all_skills') else []
        
        if not all_skills:
            st.info("No skills extracted from the resume.")
            return
        
        # Create skill categories
        skill_categories = {
            "Technical Skills": getattr(skills, 'technical', []),
            "Programming Languages": getattr(skills, 'languages', []),
            "Frameworks": getattr(skills, 'frameworks', []),
            "Tools": getattr(skills, 'tools', []),
            "Soft Skills": getattr(skills, 'soft_skills', []),
            "Certifications": getattr(skills, 'certifications', [])
        }
        
        # Render each category
        for category, skill_list in skill_categories.items():
            if skill_list:
                st.markdown(f"**{category}:**")
                # Display as badges
                cols = st.columns(min(len(skill_list), 4))
                for i, skill in enumerate(skill_list):
                    with cols[i % len(cols)]:
                        st.markdown(f"`{skill}`")
                st.markdown("")
    
    def _render_projects(self, projects: List):
        """Render projects section."""
        st.markdown("### 🚀 Projects")
        
        if not projects:
            st.info("No projects found in the resume.")
            return
        
        for i, project in enumerate(projects):
            with st.container():
                project_title = project.name
                if hasattr(project, 'url') and project.url:
                    project_title = f"[{project_title}]({project.url})"
                
                st.markdown(f"#### {project_title}")
                
                if hasattr(project, 'description') and project.description:
                    st.markdown(project.description)
                
                # Technologies used
                if hasattr(project, 'technologies') and project.technologies:
                    st.markdown("**Technologies:**")
                    tech_cols = st.columns(min(len(project.technologies), 5))
                    for j, tech in enumerate(project.technologies):
                        with tech_cols[j % len(tech_cols)]:
                            st.markdown(f"`{tech}`")
                
                # Duration
                if hasattr(project, 'duration') and project.duration:
                    st.markdown(f"**Duration:** {project.duration}")
                
                if i < len(projects) - 1:
                    st.markdown("---")
    
    def render_export_options(self, data: ParsedResumeData):
        """Render data export options."""
        st.subheader("💾 Export Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Export as JSON"):
                json_data = self._convert_to_json(data)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"resume_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📊 Export Summary"):
                summary = self._generate_summary(data)
                st.download_button(
                    label="Download Summary",
                    data=summary,
                    file_name=f"resume_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        with col3:
            if st.button("🔄 Show Raw Data"):
                with st.expander("Raw Parsed Data"):
                    st.json(self._convert_to_dict(data))
    
    def _convert_to_json(self, data: ParsedResumeData) -> str:
        """Convert parsed data to JSON."""
        try:
            data_dict = self._convert_to_dict(data)
            return json.dumps(data_dict, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"JSON conversion failed: {str(e)}")
            return json.dumps({"error": "Failed to convert data to JSON"})
    
    def _convert_to_dict(self, data: ParsedResumeData) -> Dict[str, Any]:
        """Convert parsed data to dictionary."""
        if hasattr(data, 'dict'):
            return data.dict()
        else:
            # Fallback conversion
            return {
                "name": getattr(data, 'name', 'Unknown'),
                "contact": getattr(data, 'contact', {}).__dict__ if hasattr(getattr(data, 'contact', {}), '__dict__') else {},
                "professional_summary": getattr(data, 'professional_summary', None),
                "experience": [exp.__dict__ if hasattr(exp, '__dict__') else exp for exp in getattr(data, 'experience', [])],
                "education": [edu.__dict__ if hasattr(edu, '__dict__') else edu for edu in getattr(data, 'education', [])],
                "skills": getattr(data, 'skills', {}).__dict__ if hasattr(getattr(data, 'skills', {}), '__dict__') else {},
                "projects": [proj.__dict__ if hasattr(proj, '__dict__') else proj for proj in getattr(data, 'projects', [])]
            }
    
    def _generate_summary(self, data: ParsedResumeData) -> str:
        """Generate text summary of parsed data."""
        lines = [
            f"Resume Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 50,
            "",
            f"Name: {data.name}",
            ""
        ]
        
        # Contact info
        if data.contact:
            lines.append("CONTACT INFORMATION:")
            if data.contact.email:
                lines.append(f"Email: {data.contact.email}")
            if data.contact.phone:
                lines.append(f"Phone: {data.contact.phone}")
            lines.append("")
        
        # Experience summary
        if data.experience:
            lines.append(f"WORK EXPERIENCE ({len(data.experience)} positions):")
            for exp in data.experience:
                lines.append(f"- {exp.title} at {exp.company}")
            lines.append("")
        
        # Education summary
        if data.education:
            lines.append(f"EDUCATION ({len(data.education)} entries):")
            for edu in data.education:
                lines.append(f"- {edu.degree} from {edu.institution}")
            lines.append("")
        
        # Skills summary
        if data.skills:
            all_skills = data.skills.get_all_skills() if hasattr(data.skills, 'get_all_skills') else []
            if all_skills:
                lines.append(f"SKILLS ({len(all_skills)} total):")
                for skill in all_skills[:10]:  # Top 10 skills
                    lines.append(f"- {skill}")
                lines.append("")
        
        return "\n".join(lines)
    
    def show_error(self, message: str, details: str = None):
        """Display error message."""
        st.error(f"❌ {message}")
        if details:
            with st.expander("Error Details"):
                st.code(details)
    
    def show_warning(self, message: str):
        """Display warning message."""
        st.warning(f"⚠️ {message}")
    
    def show_success(self, message: str):
        """Display success message."""
        st.success(f"✅ {message}")
    
    def show_info(self, message: str):
        """Display info message."""
        st.info(f"ℹ️ {message}")

# Global service instance
_ui_service: Optional[UIService] = None

def get_ui_service() -> UIService:
    """Get the global UI service."""
    global _ui_service
    if _ui_service is None:
        _ui_service = UIService()
    return _ui_service