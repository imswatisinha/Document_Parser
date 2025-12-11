import streamlit as st
import re
import json
import requests
from typing import Dict, List, Tuple, Optional

def detect_document_complexity(text: str) -> Dict[str, any]:
    """
    Analyze document to determine its complexity and processing requirements.
    
    Args:
        text (str): Document text to analyze
        
    Returns:
        dict: Analysis results with size category and recommendations
    """
    # Basic metrics
    char_count = len(text)
    word_count = len(text.split())
    line_count = len(text.split('\n'))
    
    # Complex content indicators
    has_tables = bool(re.search(r'\|.*\|.*\|', text))  # Table-like structures
    has_code = bool(re.search(r'```|def |class |function|import |#include', text, re.IGNORECASE))
    has_math = bool(re.search(r'\$.*\$|\\frac|\\sum|\\int', text))
    has_lists = len(re.findall(r'^\s*[•·\-\*]\s+', text, re.MULTILINE))
    
    # Technical content complexity
    technical_terms = len(re.findall(r'(?i)(python|java|javascript|sql|api|database|algorithm|machine learning|ai|data science|backend|frontend)', text))
    
    # Determine complexity level
    complexity_score = 0
    
    # Size-based scoring
    if char_count > 15000:
        complexity_score += 3
    elif char_count > 8000:
        complexity_score += 2
    elif char_count > 3000:
        complexity_score += 1
    
    # Content-based scoring
    if has_tables:
        complexity_score += 1
    if has_code:
        complexity_score += 2
    if has_math:
        complexity_score += 1
    if has_lists > 10:
        complexity_score += 1
    if technical_terms > 5:
        complexity_score += 1
    
    # Categorize document
    if complexity_score >= 6:
        category = "very_complex"
    elif complexity_score >= 4:
        category = "complex"
    elif complexity_score >= 2:
        category = "medium"
    else:
        category = "simple"
    
    return {
        "char_count": char_count,
        "word_count": word_count,
        "line_count": line_count,
        "complexity_score": complexity_score,
        "category": category,
        "features": {
            "has_tables": has_tables,
            "has_code": has_code,
            "has_math": has_math,
            "list_count": has_lists,
            "technical_terms": technical_terms
        }
    }

def get_available_ollama_models() -> List[str]:
    """
    Get list of available Ollama models from the API.
    Filtered to only supported models: phi3:mini and llama3.2:3b
    
    Returns:
        list: Available model names
    """
    supported_models = ["phi3:mini", "llama3.2:3b"]
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            all_models = [model['name'] for model in models_data.get('models', [])]
            return [m for m in all_models if m in supported_models]
    except Exception:
        pass
    return []

def get_model_memory_requirements() -> Dict[str, Dict[str, any]]:
    """
    Get memory requirements and characteristics for supported models.
    Simplified to only two models: llama3.2:3b and phi3:mini.
    
    Returns:
        dict: Model characteristics including memory requirements
    """
    return {
        "phi3:mini": {
            "ram_required": 2,
            "vram_required": 2,
            "model_size": "2.2GB",
            "performance": "fast",
            "quality": "good",
            "reliability": "high"
        },
        "llama3.2:3b": {
            "ram_required": 4,
            "vram_required": 3,
            "model_size": "2.0GB",
            "performance": "medium",
            "quality": "very_good",
            "reliability": "high"
        }
    }

def select_optimal_model(document_analysis: Dict, available_models: List[str]) -> Tuple[str, str]:
    """
    Select the best Ollama model based on document complexity and system capabilities.
    Prioritizes memory-safe models to avoid out-of-memory errors.
    
    Args:
        document_analysis (dict): Document complexity analysis
        available_models (list): List of available Ollama models
        
    Returns:
        tuple: (selected_model, reasoning)
    """
    if not available_models:
        return None, "No Ollama models available"
    
    char_count = document_analysis["char_count"]
    category = document_analysis["category"]
    features = document_analysis["features"]
    
    # Get model characteristics
    model_specs = get_model_memory_requirements()
    
    # Simplified model preferences for two models
    memory_safe_preferences = {
        "very_complex": ["llama3.2:3b", "phi3:mini"],
        "complex": ["llama3.2:3b", "phi3:mini"],
        "medium": ["llama3.2:3b", "phi3:mini"],
        "simple": ["phi3:mini", "llama3.2:3b"]
    }
    
    # Get preferred models for this category
    preferred_models = memory_safe_preferences.get(category, memory_safe_preferences["medium"])
    
    # Find the first available preferred model
    selected_model = None
    selected_specs = None
    
    for model in preferred_models:
        if model in available_models:
            selected_model = model
            selected_specs = model_specs.get(model, {})
            break
    
    # Emergency fallback - choose the smallest available model
    if not selected_model:
        # Sort available models by memory requirement (safest first)
        safe_models = []
        for model in available_models:
            specs = model_specs.get(model, {"ram_required": 10})  # Default to medium requirement
            safe_models.append((model, specs.get("ram_required", 10)))
        
        # Sort by RAM requirement (ascending)
        safe_models.sort(key=lambda x: x[1])
        selected_model = safe_models[0][0]
        selected_specs = model_specs.get(selected_model, {})
    
    # Generate reasoning with memory awareness
    reasoning_parts = [
        f"Document: {char_count:,} chars ({category} complexity)"
    ]
    
    if selected_specs:
        reasoning_parts.append(f"Memory: ~{selected_specs.get('ram_required', 'Unknown')}GB RAM")
        reasoning_parts.append(f"Size: {selected_specs.get('model_size', 'Unknown')}")
        reasoning_parts.append(f"Performance: {selected_specs.get('performance', 'Unknown')}")
    
    if features["has_code"]:
        reasoning_parts.append("Code detected")
    if features["has_tables"]:
        reasoning_parts.append("Tables detected")
    if features["technical_terms"] > 5:
        reasoning_parts.append("High technical content")
    
    reasoning = " | ".join(reasoning_parts)
    
    return selected_model, reasoning

def auto_select_model_for_document(text: str) -> Dict[str, any]:
    """
    Automatically analyze document and select the optimal Ollama model.
    
    Args:
        text (str): Document text to analyze
        
    Returns:
        dict: Selection results with model and analysis
    """
    # Analyze document complexity
    analysis = detect_document_complexity(text)
    
    # Get available models
    available_models = get_available_ollama_models()
    
    # Select optimal model
    selected_model, reasoning = select_optimal_model(analysis, available_models)
    
    return {
        "analysis": analysis,
        "available_models": available_models,
        "selected_model": selected_model,
        "reasoning": reasoning,
        "success": selected_model is not None
    }

def _enhance_education_extraction(text: str, existing_education) -> list:
    """
    Use regex patterns to find additional education entries that might have been missed.
    Handles both list and other formats for existing education.
    """
    # Handle different input formats
    if isinstance(existing_education, list):
        enhanced_education = existing_education.copy()
    elif existing_education is None:
        enhanced_education = []
    else:
        # If it's not a list, convert to list format
        enhanced_education = []
    
    # Common education patterns
    education_patterns = [
        r'(?i)(bachelor|master|phd|diploma|certificate|b\.?tech|m\.?tech|b\.?sc|m\.?sc|mba|b\.?com|m\.?com|b\.?a|m\.?a|b\.?e|m\.?e)[\s\w]*(?:in|of)?\s*([\w\s,&-]+?)(?:from|at|-)?\s*([\w\s,&.-]+?)(?:\(|,|\.|$|\n)',
        r'(?i)([\w\s,&.-]+?)\s*(?:university|college|institute|school|academy)\s*(?:\(|,|\.|$|\n)',
        r'(?i)(university|college|institute|school|academy)\s+of\s+([\w\s,&-]+)',
        r'(?i)([\d]{4})\s*[-–—]\s*([\d]{4})?\s*[:;]?\s*([\w\s,&.-]+?)(?:university|college|institute|school)',
    ]
    
    found_institutions = set()
    for edu in enhanced_education:
        if edu.get('institution'):
            inst_val = edu.get('institution')
            if not isinstance(inst_val, str):
                inst_val = str(inst_val)
            found_institutions.add(inst_val.lower())
    
    for pattern in education_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            groups = match.groups()
            
            # Extract potential degree, institution, and year
            degree = None
            institution = None
            year = None
            raw_text = match.group(0).strip()
            
            # Heuristic to assign groups based on pattern
            if len(groups) >= 3:
                if re.search(r'\d{4}', groups[0]):  # First group is year
                    year = groups[0]
                    institution = groups[2] if groups[2] else groups[1]
                else:
                    degree = groups[0]
                    institution = groups[2] if groups[2] else groups[1]
                    
            elif len(groups) >= 2:
                g1 = groups[1] if isinstance(groups[1], str) else str(groups[1])
                if 'university' in g1.lower() or 'college' in g1.lower():
                    institution = g1
                    degree = groups[0]
                else:
                    institution = groups[0]
                    
            # Clean up extracted data
            if institution:
                institution = re.sub(r'[(),\.]', '', str(institution)).strip()
                if institution.lower() not in found_institutions and len(institution) > 3:
                    enhanced_education.append({
                        "degree": degree,
                        "institution": institution,
                        "year": year,
                        "raw_text": raw_text
                    })
                    found_institutions.add(institution.lower())
    
    return enhanced_education

def _enhance_skills_extraction_structured(text: str, existing_skills_dict: dict) -> dict:
    """
    Enhanced skills extraction that maintains the structured format from the new JSON schema.
    
    Args:
        text (str): Resume text
        existing_skills_dict (dict): Structured skills dictionary
        
    Returns:
        dict: Enhanced structured skills
    """
    # Initialize with existing structure or create default
    enhanced_skills = existing_skills_dict.copy() if existing_skills_dict else {}
    
    # Ensure all required categories exist
    categories = ["technical", "soft", "programming_languages", "tools_and_technologies", "domains"]
    for category in categories:
        if category not in enhanced_skills:
            enhanced_skills[category] = []
    
    # Extract additional skills by category
    technical_patterns = [
        r'(?i)(?:technical\s+skills?|technologies?)\s*[:;]?\s*([^\n]+)',
        r'(?i)(?:proficient\s+in|experienced\s+with|knowledge\s+of)\s*[:;]?\s*([^\n]+)',
        r'(?i)(?:skills?\s*include|expertise\s+in|specializing\s+in)\s*[:;]?\s*([^\n]+)'
    ]
    
    programming_patterns = [
        r'(?i)(?:programming\s+languages?|languages?)\s*[:;]?\s*([^\n]+)',
        r'(?i)(?:coded?\s+in|developed?\s+with|using)\s*[:;]?\s*([^\n]+)'
    ]
    
    tools_patterns = [
        r'(?i)(?:tools?|frameworks?|platforms?|software)\s*[:;]?\s*([^\n]+)',
        r'(?i)(?:experience\s+with|worked\s+with|used)\s*[:;]?\s*([^\n]+)'
    ]
    
    # Extract skills for each category
    skill_extractions = [
        ("technical", technical_patterns),
        ("programming_languages", programming_patterns),
        ("tools_and_technologies", tools_patterns)
    ]
    
    for category, patterns in skill_extractions:
        existing_lower = [skill.lower() for skill in enhanced_skills[category] if isinstance(skill, str)]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                skills_text = match.group(1)
                # Split by common delimiters and clean
                skills = re.split(r'[,;|•·\n]+', skills_text)
                for skill in skills:
                    skill = skill.strip(' •·-').strip()
                    if skill and len(skill) > 1 and skill.lower() not in existing_lower:
                        enhanced_skills[category].append(skill)
                        existing_lower.append(skill.lower())
    
    # Extract domain expertise
    domain_patterns = [
        r'(?i)(?:domain\s+expertise|industry\s+experience|specialization)\s*[:;]?\s*([^\n]+)',
        r'(?i)(?:worked\s+in|experience\s+in)\s+([^\n,]+?)(?:industry|domain|field)'
    ]
    
    existing_domains = [d.lower() for d in enhanced_skills.get("domains", [])]
    for pattern in domain_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            domain_text = match.group(1)
            domains = re.split(r'[,;|•·\n]+', domain_text)
            for domain in domains:
                domain = domain.strip(' •·-').strip()
                if domain and len(domain) > 2 and domain.lower() not in existing_domains:
                    enhanced_skills["domains"].append(domain)
                    existing_domains.append(domain.lower())
    
    return enhanced_skills

def _enhance_skills_extraction(text: str, existing_skills) -> list:
    """
    Extract additional skills using patterns for common skill sections.
    Handles both list and dictionary formats for existing skills.
    """
    # Handle different input formats
    if isinstance(existing_skills, dict):
        # If skills is a dictionary (e.g., {"technical": [...], "soft": [...]})
        enhanced_skills = []
        for skill_category, skill_list in existing_skills.items():
            if isinstance(skill_list, list):
                enhanced_skills.extend(skill_list)
            elif isinstance(skill_list, str):
                enhanced_skills.append(skill_list)
    elif isinstance(existing_skills, list):
        enhanced_skills = existing_skills.copy()
    else:
        enhanced_skills = []
    
    existing_lower = [skill.lower() for skill in enhanced_skills if isinstance(skill, str)]
    
    # Skills section patterns
    skills_patterns = [
        r'(?i)(?:skills?|technologies?|tools?|languages?|frameworks?|platforms?)\s*[:;]?\s*([^\n]+)',
        r'(?i)(?:technical\s+skills?|programming\s+languages?|software\s+skills?)\s*[:;]?\s*([^\n]+)',
        r'(?i)(?:proficient\s+in|experienced\s+with|knowledge\s+of)\s*[:;]?\s*([^\n]+)',
    ]
    
    # Common skill delimiters
    skill_delimiters = r'[,;|•·\n]+'
    
    for pattern in skills_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            skills_text = match.group(1)
            # Split by common delimiters and clean
            skills = re.split(skill_delimiters, skills_text)
            for skill in skills:
                skill = skill.strip(' •·-')
                if skill and len(skill) > 1 and skill.lower() not in existing_lower:
                    enhanced_skills.append(skill)
                    existing_lower.append(skill.lower())
    
    return enhanced_skills

def _enhance_projects_extraction(text: str, existing_projects) -> list:
    """
    Find additional project entries with descriptions.
    Handles both list and other formats for existing projects.
    """
    # Handle different input formats
    if isinstance(existing_projects, list):
        enhanced_projects = existing_projects.copy()
    elif existing_projects is None:
        enhanced_projects = []
    else:
        # If it's not a list, convert to list format
        enhanced_projects = []
    
    # Project section patterns
    project_patterns = [
        r'(?i)(?:projects?)\s*[:;]?\s*\n((?:.*\n?)*?)(?=\n\s*(?:experience|education|skills|certifications?|$))',
        r'(?i)(?:project\s+\d+|project\s+name)\s*[:;]?\s*([^\n]+)(?:\n((?:.*\n?)*?))?',
        r'(?i)([A-Z][^\n]+?(?:project|application|system|website|app))\s*[:;]?\s*\n((?:.*\n?)*?)(?=\n\s*[A-Z]|\n\s*$)',
    ]
    
    existing_names = set()
    for proj in enhanced_projects:
        if proj.get('name'):
            existing_names.add(proj['name'].lower())
    
    for pattern in project_patterns:
        matches = re.finditer(pattern, text, re.MULTILINE)
        for match in matches:
            if len(match.groups()) >= 2:
                name = match.group(1).strip()
                description = match.group(2).strip() if match.group(2) else ""
            else:
                # Single group - treat as description block, extract name from first line
                content = match.group(1).strip()
                lines = content.split('\n')
                name = lines[0].strip() if lines else ""
                description = '\n'.join(lines[1:]).strip() if len(lines) > 1 else ""
            
            # Clean up name
            name = re.sub(r'^[•·-]\s*', '', name)
            
            if name and len(name) > 3 and name.lower() not in existing_names:
                # Extract technologies from description
                tech_patterns = [
                    r'(?i)(?:using|with|technologies?|tech\s+stack|built\s+with)\s*[:;]?\s*([^\n.]+)',
                    r'(?i)(?:python|java|javascript|react|node|html|css|sql|mongodb|mysql|aws|docker|git|django|flask|spring)',
                ]
                
                technologies = []
                for tech_pattern in tech_patterns:
                    tech_matches = re.findall(tech_pattern, description)
                    for tech_match in tech_matches:
                        techs = re.split(r'[,;|•·\n]+', tech_match)
                        technologies.extend([t.strip() for t in techs if t.strip()])
                
                enhanced_projects.append({
                    "name": name,
                    "description": description,
                    "technologies": list(set(technologies)),
                    "raw_text": match.group(0)
                })
                existing_names.add(name.lower())
    
    return enhanced_projects

def _add_page_provenance(parsed_data: dict, pages: list) -> dict:
    """
    Add page numbers to parsed entries by finding where they appear in the original pages.
    """
    if not pages or not parsed_data:
        return parsed_data
    
    enhanced_data = parsed_data.copy()
    
    # Add page provenance to education
    if enhanced_data.get('education'):
        for edu in enhanced_data['education']:
            if edu.get('raw_text') or edu.get('institution'):
                search_text = edu.get('raw_text', edu.get('institution', ''))
                for page_num, page_text in enumerate(pages, 1):
                    stx = search_text if isinstance(search_text, str) else str(search_text)
                    ptx = page_text if isinstance(page_text, str) else str(page_text)
                    if stx.lower() in ptx.lower():
                        edu['page'] = page_num
                        break
    
    # Add page provenance to experience
    if enhanced_data.get('experience'):
        for exp in enhanced_data['experience']:
            if exp.get('raw_text') or exp.get('company'):
                search_text = exp.get('raw_text', exp.get('company', ''))
                for page_num, page_text in enumerate(pages, 1):
                    stx = search_text if isinstance(search_text, str) else str(search_text)
                    ptx = page_text if isinstance(page_text, str) else str(page_text)
                    if stx.lower() in ptx.lower():
                        exp['page'] = page_num
                        break
    
    # Add page provenance to projects
    if enhanced_data.get('projects'):
        for proj in enhanced_data['projects']:
            if proj.get('raw_text') or proj.get('name'):
                search_text = proj.get('raw_text', proj.get('name', ''))
                for page_num, page_text in enumerate(pages, 1):
                    stx = search_text if isinstance(search_text, str) else str(search_text)
                    ptx = page_text if isinstance(page_text, str) else str(page_text)
                    if stx.lower() in ptx.lower():
                        proj['page'] = page_num
                        break
    
    return enhanced_data

def post_process_parsed_data(parsed_data: dict, full_text: str, pages: list = None) -> dict:
    """
    Post-process LLM output to fill missing details, add keyword classification, and add page provenance.
    Handles various data formats returned by different LLM parsers.
    """
    if "error" in parsed_data or "raw_output" in parsed_data:
        return parsed_data
    
    enhanced_data = parsed_data.copy()
    
    try:
        # Enhance education extraction - handle any format
        enhanced_data['education'] = _enhance_education_extraction(
            full_text, enhanced_data.get('education')
        )
    except Exception as e:
        st.warning(f"Education enhancement failed: {str(e)}")
        # Keep original education data if enhancement fails
        enhanced_data['education'] = enhanced_data.get('education', [])
    
    try:
        # Enhance skills extraction - handle dict/list formats
        skills_data = enhanced_data.get('skills')
        if isinstance(skills_data, dict):
            # Keep the structured skills format from the new JSON structure
            enhanced_skills = _enhance_skills_extraction_structured(full_text, skills_data)
            enhanced_data['skills'] = enhanced_skills
        else:
            # Fallback for old format
            enhanced_data['skills'] = _enhance_skills_extraction(full_text, skills_data)
    except Exception as e:
        st.warning(f"Skills enhancement failed: {str(e)}")
        # Keep original skills data if enhancement fails
        enhanced_data['skills'] = enhanced_data.get('skills', {})
    
    try:
        # Enhance projects extraction - handle any format
        enhanced_data['projects'] = _enhance_projects_extraction(
            full_text, enhanced_data.get('projects')
        )
    except Exception as e:
        st.warning(f"Projects enhancement failed: {str(e)}")
        # Keep original projects data if enhancement fails
        enhanced_data['projects'] = enhanced_data.get('projects', [])
    
    try:
        # Add intelligent keyword extraction and classification
        from .keyword_classifier import ResumeKeywordClassifier
        classifier = ResumeKeywordClassifier()
        classification_result = classifier.generate_classification_vector(full_text)
        
        # Merge AI-extracted keywords with classifier keywords
        ai_keywords = enhanced_data.get('keywords_extracted', [])
        classifier_keywords = classification_result['keywords_extracted']
        
        # Combine and deduplicate keywords
        all_keywords = list(set(ai_keywords + classifier_keywords))
        
        # Update enhanced data with comprehensive classification
        enhanced_data['keywords_extracted'] = all_keywords
        enhanced_data['classification_tags'] = classification_result['classification_tags']
        enhanced_data['category_scores'] = classification_result['category_scores']
        enhanced_data['primary_classification'] = classification_result['primary_classification']
        enhanced_data['tech_subcategories'] = classification_result['tech_subcategories']
        enhanced_data['classification_confidence'] = classification_result['confidence']
        
    except Exception as e:
        st.warning(f"Keyword classification failed: {str(e)}")
        # Ensure basic structure exists even if classification fails
        enhanced_data['keywords_extracted'] = enhanced_data.get('keywords_extracted', [])
        enhanced_data['classification_tags'] = enhanced_data.get('classification_tags', [])
    
    # Add page provenance if pages are available
    if pages:
        enhanced_data = _add_page_provenance(enhanced_data, pages)
    
    enhanced_data['enhanced'] = True
    return enhanced_data

def show_model_recommendations(available_models: List[str]) -> None:
    """
    Display model recommendations in Streamlit sidebar.
    
    Args:
        available_models (list): List of available Ollama models
    """
    if not available_models:
        return
        
    st.sidebar.markdown("### 🎯 Model Recommendations")
    
    # Categorize available models - simplified to two models
    model_categories = {
        "⚡ Fast Models": ["phi3:mini"],
        "⚖️ Balanced Models": ["llama3.2:3b"]
    }
    
    for category, models in model_categories.items():
        available_in_category = [m for m in models if m in available_models]
        if available_in_category:
            st.sidebar.write(f"**{category}**")
            for model in available_in_category:
                st.sidebar.write(f"  ✅ {model}")
    
    # Show missing recommended models
    all_recommended = []
    for models in model_categories.values():
        all_recommended.extend(models)
    
    missing_models = [m for m in all_recommended if m not in available_models]
    if missing_models:
        with st.sidebar.expander("📥 Install More Models"):
            st.write("**Recommended models to install:**")
            for model in missing_models[:3]:  # Show top 3 missing
                if st.button(f"Install {model}", key=f"install_{model}"):
                    st.info(f"Run: `ollama pull {model}`")

def get_model_performance_info(model_name: str) -> Dict[str, any]:
    """
    Get performance characteristics for a specific model.
    
    Args:
        model_name (str): Name of the Ollama model
        
    Returns:
        dict: Performance information
    """
    # Model performance database - simplified to two models
    model_info = {
        "phi3:mini": {
            "size": "2.2GB",
            "speed": "⚡⚡⚡",
            "quality": "⭐⭐",
            "best_for": "Fast processing, small documents",
            "context_length": "4K tokens",
            "ram_requirement": "2GB"
        },
        "llama3.2:3b": {
            "size": "2.0GB",
            "speed": "⚡⚡",
            "quality": "⭐⭐⭐",
            "best_for": "Balanced speed and quality",
            "context_length": "8K tokens", 
            "ram_requirement": "4GB"
        }
    }
    
    return model_info.get(model_name, {
        "size": "Unknown",
        "speed": "⚡",
        "quality": "⭐⭐",
        "best_for": "General processing",
        "context_length": "4K tokens",
        "ram_requirement": "4GB+"
    })

def parse_resume_with_ollama(text, pages=None, model_name=None, use_expanders=True):
    """
    Parse resume text using Ollama/Llama models with intelligent model selection.
    
    Args:
        text (str): Raw resume/document text
        pages (list): Optional list of per-page texts for provenance tracking
        model_name (str): Override model selection (optional - will auto-select if None)
        use_expanders (bool): Whether to use st.expander for detailed info (default True)
        
    Returns:
        dict: Structured document data
    """
    
    try:
        from .ollama_parser import OllamaParser
        
        # Initialize doc_size early to avoid scope issues
        doc_size = len(text)
        
        parser = OllamaParser()
        if not parser.is_available():
            st.error("❌ Ollama not available")
            with st.expander("🔧 Setup Instructions", expanded=True):
                st.markdown("""
                **To fix this:**
                1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai)
                2. **Start Ollama**: Run `ollama serve` in terminal
                3. **Install a model**: Run `ollama pull llama3.2:3b`
                4. **Verify**: Run `ollama list` to see installed models
                """)
            return {"error": "Ollama not available. Please install and start Ollama service."}
        
        # Step 1: Analyze document and auto-select model (if not specified)
        if not model_name:
            st.info("🔍 Analyzing document to select optimal AI model...")
            
            selection_result = auto_select_model_for_document(text)
            
            if not selection_result["success"]:
                st.error("❌ No suitable models available")
                return {"error": "No Ollama models available for processing"}
            
            model_name = selection_result["selected_model"]
            analysis = selection_result["analysis"]
            reasoning = selection_result["reasoning"]
            
            # Display analysis results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.success(f"🎯 **Selected Model:** `{model_name}`")
                st.info(f"📊 **Analysis:** {reasoning}")
            
            with col2:
                complexity = analysis["category"].replace("_", " ").title()
                st.metric("Document Complexity", complexity)
                st.metric("Size", f"{analysis['char_count']:,} chars")
            
            # Show detailed analysis conditionally
            if use_expanders:
                with st.expander("🔬 Detailed Document Analysis", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**📏 Size Metrics:**")
                        st.write(f"• Characters: {analysis['char_count']:,}")
                        st.write(f"• Words: {analysis['word_count']:,}")
                        st.write(f"• Lines: {analysis['line_count']:,}")
                    
                    with col2:
                        st.write("**🔧 Content Features:**")
                        features = analysis['features']
                        st.write(f"• Tables: {'✅' if features['has_tables'] else '❌'}")
                        st.write(f"• Code: {'✅' if features['has_code'] else '❌'}")
                        st.write(f"• Math: {'✅' if features['has_math'] else '❌'}")
                        st.write(f"• Lists: {features['list_count']}")
                    
                    with col3:
                        st.write("**🎯 Model Selection:**")
                        st.write(f"• Complexity Score: {analysis['complexity_score']}/8")
                        st.write(f"• Technical Terms: {features['technical_terms']}")
                        st.write(f"• Available Models: {len(selection_result['available_models'])}")
            else:
                # Show key metrics inline without expander
                st.write(f"📊 **Analysis:** Complexity: {analysis['category']}, Size: {analysis['char_count']:,} chars, Features: {', '.join([k for k, v in analysis['features'].items() if v and k != 'technical_terms'])}")
        
        else:
            # Manual model selection
            st.info(f"🦙 Using specified model: {model_name} - Document size: {doc_size:,} characters")
        
        # Parse using Ollama with automatic fallback for memory errors
        with st.spinner("🤖 AI is processing your document..."):
            # Use fallback parsing to handle memory errors automatically
            ollama_result = parser.parse_resume_with_fallback(text, model_name)
        
        if "error" in ollama_result:
            # Handle different types of errors with specific help
            error_msg = ollama_result.get("error", "Unknown error")
            
            if "timed out" in error_msg.lower() or "timeout" in error_msg.lower():
                # Timeout-specific error handling
                st.error("⏰ **Processing Timeout**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Document Info:**")
                    st.write(f"• Size: {doc_size:,} characters")
                    st.write(f"• Model: {model_name}")
                    st.write(f"• Timeout: 5 minutes")
                
                with col2:
                    st.markdown("**💡 Solutions:**")
                    suggestions = ollama_result.get("suggestions", [])
                    for suggestion in suggestions[:3]:  # Show top 3 suggestions
                        st.write(f"• {suggestion}")
                
                # Show troubleshooting info if available
                if "troubleshooting" in ollama_result:
                    if use_expanders:
                        with st.expander("🔍 Troubleshooting Details"):
                            trouble_info = ollama_result["troubleshooting"]
                            for key, value in trouble_info.items():
                                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                    else:
                        st.write("**🔍 Troubleshooting Details:**")
                        trouble_info = ollama_result["troubleshooting"]
                        for key, value in trouble_info.items():
                            st.write(f"• **{key.replace('_', ' ').title()}:** {value}")
                
                return ollama_result
            
            elif "all available models failed" in error_msg.lower():
                # Memory/model failure error handling
                st.error("💾 **Memory/Model Error**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 System Info:**")
                    st.write(f"• Document Size: {doc_size:,} characters")
                    available_models = ollama_result.get("available_models", [])
                    st.write(f"• Available Models: {len(available_models)}")
                    if available_models:
                        st.write(f"• Models: {', '.join(available_models[:3])}")
                
                with col2:
                    st.markdown("**💡 Solutions:**")
                    suggestions = ollama_result.get("suggestions", [])
                    for suggestion in suggestions[:4]:
                        st.write(f"• {suggestion}")
                
                # Show attempts made
                attempts = ollama_result.get("attempts", [])
                if attempts:
                    if use_expanders:
                        with st.expander("🔍 Attempted Models"):
                            for attempt in attempts:
                                st.write(f"**{attempt['attempt']}.** {attempt['model']}: {attempt['error']}")
                    else:
                        st.write("**🔍 Attempted Models:**")
                        for attempt in attempts:
                            st.write(f"• **{attempt['attempt']}.** {attempt['model']}: {attempt['error']}")
                
                # Show quick fix options
                st.markdown("### 🚀 Quick Fixes:")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Install Fast Model:**")
                    st.code("ollama pull phi3:mini", language="bash")
                    
                with col2:
                    st.markdown("**Restart Ollama:**")
                    st.code("ollama serve", language="bash")
                
                with col3:
                    st.markdown("**Check Memory:**")
                    st.code("ollama ps", language="bash")
                
                return ollama_result
            else:
                # Other errors
                st.error(f"❌ Ollama Error: {error_msg}")
                
                # Show last error details if available
                if ollama_result.get("last_error"):
                    if use_expanders:
                        with st.expander("🔍 Error Details"):
                            st.json(ollama_result["last_error"])
                    else:
                        st.write("**🔍 Error Details:**")
                        st.json(ollama_result["last_error"])
                
                # Automatic fallback chain: OpenAI then Gemini
                try:
                    st.warning("⚠️ Ollama failed, switching to OpenAI...")
                    from .llm_router import parse_with_fallback_min_schema
                    from .normalizers import validate_and_normalize
                    fb = parse_with_fallback_min_schema(text, model_name)
                    if "error" not in fb:
                        provider = fb.get("_provider", "openai")
                        if provider == "openai":
                            st.success("✅ Successfully parsed using OpenAI fallback")
                        elif provider == "gemini":
                            st.success("✅ Successfully parsed using Gemini fallback")
                        maybe_chunks = None
                        if isinstance(pages, list) and pages and isinstance(pages[0], dict) and 'content' in pages[0]:
                            maybe_chunks = pages
                        normalized = validate_and_normalize(
                            fb,
                            chunks=maybe_chunks,
                            provider=provider,
                            model=fb.get("_model", "-"),
                            parsing_method="model",
                            raw_text=text,
                        )
                        return normalized
                    else:
                        st.error(f"❌ Fallbacks failed: {fb.get('error')}")
                        # Offline minimal structure
                        from .normalizers import validate_and_normalize
                        maybe_chunks = None
                        if isinstance(pages, list) and pages and isinstance(pages[0], dict) and 'content' in pages[0]:
                            maybe_chunks = pages
                        offline = validate_and_normalize({}, chunks=maybe_chunks, provider="offline", model=None, parsing_method="offline_basic", raw_text=text)
                        return offline
                except Exception as e:
                    st.error(f"❌ Fallbacks failed: {e}")
                    from .normalizers import validate_and_normalize
                    maybe_chunks = None
                    if isinstance(pages, list) and pages and isinstance(pages[0], dict) and 'content' in pages[0]:
                        maybe_chunks = pages
                    offline = validate_and_normalize({}, chunks=maybe_chunks, provider="offline", model=None, parsing_method="offline_basic", raw_text=text)
                    return offline
        
        st.success(f"✅ Successfully parsed with Ollama ({model_name})!")
        
        # Apply post-processing enhancements
        enhanced_result = post_process_parsed_data(ollama_result, text, pages)
        
        # Normalize to standard schema and attach chunks
        try:
            from .normalizers import validate_and_normalize
            # Build normalized chunks from provided 'pages' param if it is actually chunks
            maybe_chunks = None
            if isinstance(pages, list) and pages and isinstance(pages[0], dict) and 'content' in pages[0]:
                maybe_chunks = pages
            normalized = validate_and_normalize(
                enhanced_result,
                chunks=maybe_chunks,
                provider="ollama",
                model=model_name,
                parsing_method="model",
                raw_text=text,
            )
            return normalized
        except Exception:
            # Fallback minimal display
            enhanced_result["ai_provider"] = "ollama"
            enhanced_result["model"] = model_name
            return enhanced_result
        
    except ImportError:
        st.error("❌ Ollama parser module not found")
        return {"error": "Ollama parser not available. Please check ollama_parser.py"}
    except Exception as e:
        st.error(f"❌ Unexpected error during Ollama parsing: {str(e)}")
        return {"error": f"Ollama parsing failed: {str(e)}"}

def format_document_display(parsed_data):
    """
    Format parsed document data for better display in Streamlit with new JSON structure.
    
    Args:
        parsed_data (dict): Parsed document data
        
    Returns:
        None: Displays formatted data in Streamlit
    """
    if "error" in parsed_data:
        st.error(parsed_data["error"])
        return
    
    if "raw_output" in parsed_data:
        st.warning("Could not parse as structured data. Raw output:")
        st.text(parsed_data["raw_output"])
        return
    
    # Show parsing info
    enhanced = parsed_data.get("enhanced", False)
    enhancement_text = " + Enhanced" if enhanced else ""
    
    # Get AI provider info (fallback to normalized keys)
    ai_provider = parsed_data.get("ai_provider") or parsed_data.get("provider", "unknown")
    model = parsed_data.get("model") or parsed_data.get("_model", "unknown")
    
    provider_emojis = {
        "ollama": "🦙",
        "llama": "🦙"
    }
    
    emoji = provider_emojis.get(ai_provider, "🤖")
    provider_name = ai_provider.title()
    st.info(f"{emoji} Parsed using: {provider_name} ({model}){enhancement_text}")
    
    # Display classification results if available
    if parsed_data.get("primary_classification"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Primary Category", parsed_data["primary_classification"].title())
        with col2:
            confidence = parsed_data.get("classification_confidence", 0)
            st.metric("Confidence", f"{confidence:.1%}")
        with col3:
            tags = parsed_data.get("classification_tags", [])
            st.metric("Categories", len(tags))
    
    # Display personal info
    personal_info = parsed_data.get("personal_info", {})
    if personal_info.get("name"):
        st.subheader(f"👤 {personal_info['name']}")
    
    # Contact Information
    col1, col2, col3 = st.columns(3)
    with col1:
        if personal_info.get("email"):
            st.write(f"📧 **Email:** {personal_info['email']}")
        if personal_info.get("phone"):
            st.write(f"📱 **Phone:** {personal_info['phone']}")
    with col2:
        if personal_info.get("location"):
            st.write(f"📍 **Location:** {personal_info['location']}")
    with col3:
        if personal_info.get("linkedin"):
            st.write(f"🔗 **LinkedIn:** {personal_info['linkedin']}")
        if personal_info.get("github"):
            st.write(f"💻 **GitHub:** {personal_info['github']}")
        if personal_info.get("portfolio"):
            st.write(f"🌐 **Portfolio:** {personal_info['portfolio']}")
    
    # Professional Summary
    if parsed_data.get("summary"):
        st.subheader("📋 Professional Summary")
        st.write(parsed_data["summary"])
    
    # Experience (New Structure)
    if parsed_data.get("experience"):
        st.subheader("💼 Experience")
        for exp in parsed_data["experience"]:
            page_info = f" (Page {exp['page']})" if exp.get('page') else ""
            
            # Display title and company
            title = exp.get('title', 'N/A')
            company = exp.get('company_name', 'N/A')
            location = exp.get('location', '')
            start_time = exp.get('start_time', '')
            end_time = exp.get('end_time', '')
            
            st.write(f"**{title}** at **{company}**{page_info}")
            
            # Display time and location
            col1, col2 = st.columns(2)
            with col1:
                if start_time or end_time:
                    duration = f"{start_time} - {end_time}" if start_time and end_time else start_time or end_time
                    st.write(f"🗓️ **Duration:** {duration}")
            with col2:
                if location:
                    st.write(f"📍 **Location:** {location}")
            
            # Display summary
            if exp.get("summary"):
                st.write(f"**Summary:** {exp['summary']}")
            
            if exp.get("raw_text"):
                with st.expander("📝 Raw text"):
                    st.text(exp["raw_text"])
            st.write("---")
    
    # Education (filter invalid rows for display)
    if parsed_data.get("education"):
        st.subheader("🎓 Education")
        def _edu_valid(name: str) -> bool:
            if not name:
                return False
            n = str(name).lower()
            if len(n) < 3:
                return False
            keyw = ("university", "college", "institute", "school", "academy")
            return any(k in n for k in keyw)

        edu_list = [e for e in parsed_data["education"] if _edu_valid(e.get('institution'))]
        for edu in edu_list:
            page_info = f" (Page {edu['page']})" if edu.get('page') else ""
            st.write(f"**{edu.get('degree', 'N/A')}**{page_info}")
            
            col1, col2 = st.columns(2)
            with col1:
                if edu.get('institution'):
                    st.write(f"🏫 **Institution:** {edu['institution']}")
            with col2:
                if edu.get("year"):
                    st.write(f"📅 **Year:** {edu['year']}")
            
            if edu.get("location"):
                st.write(f"📍 **Location:** {edu['location']}")
            if edu.get("gpa"):
                gpa_val = edu['gpa']
                # Show as GPA or Percentage based on value
                if "%" in str(gpa_val) or (isinstance(gpa_val, str) and "percent" in gpa_val.lower()):
                    st.write(f"📊 **Percentage:** {gpa_val}")
                else:
                    st.write(f"📊 **GPA:** {gpa_val}")
            
            if edu.get("raw_text"):
                with st.expander("📝 Raw text"):
                    st.text(edu["raw_text"])
            st.write("---")
    
    # Skills (Structured or build categories from flat lists)
    if parsed_data.get("skills") or parsed_data.get("technologies"):
        st.subheader("🛠️ Skills")
        skills = parsed_data.get("skills", [])
        techs = parsed_data.get("technologies", [])

        if isinstance(skills, dict):
            # Display structured skills
            skill_categories = {
                "technical": "💻 Technical Skills",
                "programming_languages": "🐍 Programming Languages", 
                "tools_and_technologies": "🔧 Tools & Technologies",
                "soft": "🤝 Soft Skills",
                "domains": "🏢 Domain Expertise"
            }
            
            cols = st.columns(2)
            col_idx = 0
            
            for category, title in skill_categories.items():
                if skills.get(category):
                    with cols[col_idx % 2]:
                        st.write(f"**{title}:**")
                        skill_list = skills[category]
                        if isinstance(skill_list, list):
                            for skill in skill_list:
                                st.write(f"• {skill}")
                        st.write("")
                    col_idx += 1
        else:
            # Build categorized view from flat lists
            def _cat(name):
                return name.lower()
            langs = []
            markup = []
            frameworks = []
            database = []
            tools = []
            others = []
            # Combine all skills and techs
            from_list = (skills if isinstance(skills, list) else []) + (techs if isinstance(techs, list) else [])
            # Deduplicate while preserving order
            seen = set()
            deduped = []
            for s in from_list:
                if not s:
                    continue
                sl = str(s).lower().strip()
                if sl and sl not in seen:
                    seen.add(sl)
                    deduped.append(str(s).strip())
            
            for s in deduped:
                sl = s.lower()
                if sl in {"javascript","typescript","python","c++","java","c","sql"}:
                    langs.append(s)
                elif sl in {"html","css","tailwind css","bootstrap"}:
                    markup.append(s)
                elif sl in {"react","react.js","next.js","node","node.js","express","express.js"}:
                    frameworks.append(s)
                elif sl in {"mongodb","mysql","postgresql","appwrite","sql"}:
                    database.append(s)
                elif sl in {"git","postman","vs code","vscode","spline","docker"}:
                    tools.append(s)
                else:
                    others.append(s)

            cols = st.columns(2)
            with cols[0]:
                if langs:
                    st.write("**🐍 Programming Languages:**")
                    st.write(", ".join(sorted(set(langs), key=lambda x: str(x).lower())))
                if markup:
                    st.write("**🎨 Markup & Styling:**")
                    st.write(", ".join(sorted(set(markup), key=lambda x: str(x).lower())))
                if frameworks:
                    st.write("**🧩 Frameworks & Libraries:**")
                    st.write(", ".join(sorted(set(frameworks), key=lambda x: str(x).lower())))
            with cols[1]:
                if database:
                    st.write("**🗄️ Database:**")
                    st.write(", ".join(sorted(set(database), key=lambda x: str(x).lower())))
                if tools:
                    st.write("**🔧 Tools:**")
                    st.write(", ".join(sorted(set(tools), key=lambda x: str(x).lower())))
                if others:
                    st.write("**📦 Others:**")
                    st.write(", ".join(sorted(set(others), key=lambda x: str(x).lower())))
    
    # Projects
    if parsed_data.get("projects"):
        st.subheader("🚀 Projects")
        for proj in parsed_data["projects"]:
            page_info = f" (Page {proj['page']})" if proj.get('page') else ""
            st.write(f"**{proj.get('name', 'N/A')}**{page_info}")
            
            if proj.get("description"):
                st.write(f"📝 **Description:** {proj['description']}")
            if proj.get("technologies"):
                st.write(f"⚙️ **Technologies:** {', '.join(proj['technologies'])}")
            if proj.get("duration"):
                st.write(f"🗓️ **Duration:** {proj['duration']}")
            if proj.get("url"):
                st.write(f"🔗 **URL:** {proj['url']}")
            
            if proj.get("raw_text"):
                with st.expander("📝 Raw text"):
                    st.text(proj["raw_text"])
            st.write("---")
    
    # Certifications (New Structure)
    if parsed_data.get("certifications"):
        st.subheader("🏆 Certifications")
        for cert in parsed_data["certifications"]:
            if isinstance(cert, dict):
                st.write(f"**{cert.get('name', 'N/A')}**")
                if cert.get('issuer'):
                    st.write(f"📋 **Issuer:** {cert['issuer']}")
                if cert.get('date'):
                    st.write(f"📅 **Date:** {cert['date']}")
                if cert.get('url'):
                    st.write(f"🔗 **URL:** {cert['url']}")
                st.write("---")
            else:
                st.write(f"• {cert}")
    
    # Keywords and Classification
    if parsed_data.get("keywords_extracted") or parsed_data.get("classification_tags"):
        st.subheader("🔍 Analysis & Classification")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if parsed_data.get("keywords_extracted"):
                st.write("**📊 Extracted Keywords:**")
                keywords = parsed_data["keywords_extracted"][:20]  # Show first 20 keywords
                st.write(", ".join(keywords))
                if len(parsed_data["keywords_extracted"]) > 20:
                    with st.expander("View all keywords"):
                        st.write(", ".join(parsed_data["keywords_extracted"]))
        
        with col2:
            if parsed_data.get("classification_tags"):
                st.write("**🏷️ Classification Tags:**")
                for tag in parsed_data["classification_tags"]:
                    st.write(f"• {tag.title()}")
        
        # Show tech subcategories if available
        if parsed_data.get("tech_subcategories"):
            st.write("**⚡ Tech Specializations:**")
            tech_cols = st.columns(len(parsed_data["tech_subcategories"]))
            for i, subcat in enumerate(parsed_data["tech_subcategories"]):
                with tech_cols[i]:
                    st.write(f"🔸 {subcat.replace('_', ' ').title()}")
    
    # Raw JSON option
    with st.expander("🔧 View Raw JSON Data"):
        st.json(parsed_data)

def show_ollama_status():
    """Show status of Ollama service in sidebar"""
    
    st.sidebar.header("� Ollama Status")
    
    try:
        from .ollama_parser import OllamaParser
        parser = OllamaParser()
        
        if parser.is_available():
            st.sidebar.success("🦙 **Ollama**: ✅ Available")
            
            # Show available models
            models = parser.available_models  # Use the available_models property
            if models:
                st.sidebar.write("**Available Models:**")
                for model in models[:5]:  # Show first 5 models
                    st.sidebar.write(f"• {model}")
                if len(models) > 5:
                    st.sidebar.write(f"... and {len(models) - 5} more")
            else:
                st.sidebar.warning("No models installed")
        else:
            st.sidebar.error("� **Ollama**: ❌ Not available")
            st.sidebar.write("Please install and start Ollama service")
    except ImportError:
        st.sidebar.error("🦙 **Ollama**: ❌ Parser not found")
    
    return True