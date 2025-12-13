"""
Ollama Integration for Resume Parsing
Local AI processing without API keys or internet requirements
"""

import requests
import json
import streamlit as st
import os
from typing import Dict, Any, List, Optional
from datetime import datetime

class OllamaParser:
    """Local AI resume parser using Ollama."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.available_models = []
        self.check_connection()
    
    def check_connection(self) -> bool:
        """Check if Ollama is running and get available models."""
        supported_models = ["phi3:mini", "llama3.2:3b"]
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                all_models = [model['name'] for model in models_data.get('models', [])]
                self.available_models = [m for m in all_models if m in supported_models]
                return True
            return False
        except Exception as e:
            return False
    
    def is_available(self) -> bool:
        """Check if Ollama service is available."""
        return len(self.available_models) > 0
    
    def get_recommended_models(self) -> List[str]:
        """Get list of recommended models for resume parsing - simplified to two models."""
        recommended = [
            "llama3.2:3b",      # High quality, good for most documents
            "phi3:mini",        # Fast, good for simple documents
        ]
        
        # Return only models that are available
        available_recommended = [model for model in recommended if model in self.available_models]
        
        # If no recommended models, return first available
        if not available_recommended and self.available_models:
            available_recommended = self.available_models[:1]
        
        return available_recommended
    
    def install_model(self, model_name: str) -> bool:
        """Install a model via Ollama API."""
        try:
            st.info(f"🔄 Installing {model_name}... This may take a few minutes.")
            
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=600  # 10 minutes timeout
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if 'status' in data:
                                status_text.text(f"Status: {data['status']}")
                            if 'completed' in data and 'total' in data:
                                progress = data['completed'] / data['total']
                                progress_bar.progress(progress)
                        except:
                            continue
                
                progress_bar.progress(1.0)
                status_text.text("✅ Installation complete!")
                self.check_connection()  # Refresh available models
                return True
            
            return False
            
        except Exception as e:
            st.error(f"Failed to install {model_name}: {str(e)}")
            return False
    
    def parse_resume(self, resume_text: str, model: str = None) -> Dict[str, Any]:
        """Parse resume using Ollama model. Returns raw model JSON text or error."""
        
        if not self.available_models:
            return {
                "error": "No Ollama models available. Please install a model first.",
                "setup_required": True
            }
        
        # Use first available model if specified model not available
        if not model or model not in self.available_models:
            model = self.available_models[0]
        
        prompt = f"""
        You are an expert resume parser. Extract the following information from this resume and return ONLY a valid JSON object with this EXACT structure:

        {{
            "personal_info": {{
                "name": "Full name",
                "email": "email@example.com",
                "phone": "phone number",
                "location": "city, state/country",
                "linkedin": "LinkedIn URL if mentioned",
                "github": "GitHub URL if mentioned"
            }},
            "summary": "Professional summary in 2-3 sentences or null if not available",
            "experience": [
                {{
                    "title": "Job title/position name",
                    "company_name": "Company name",
                    "location": "Work location (city, state/country)",
                    "start_time": "Start date (format: MMM YYYY like 'Dec 2017')",
                    "end_time": "End date (format: MMM YYYY or 'present')",
                    "summary": "Brief description of role and key achievements"
                }}
            ],
            "education": [
                {{
                    "institution": "School/University name",
                    "degree": "Degree type and field",
                    "year": "Graduation year",
                    "location": "Institution location",
                    "gpa": "GPA if mentioned"
                }}
            ],
            "skills": {{
                "technical": ["Technical skill 1", "Technical skill 2"],
                "soft": ["Soft skill 1", "Soft skill 2"],
                "programming_languages": ["Programming language 1", "Programming language 2"],
                "tools_and_technologies": ["Tool 1", "Tool 2", "Framework 1"],
                "domains": ["Domain expertise 1", "Domain expertise 2"]
            }},
            "projects": [
                {{
                    "name": "Project name",
                    "description": "Brief description",
                    "technologies": ["Tech 1", "Tech 2"],
                    "url": "Project URL if mentioned",
                    "duration": "Project duration if mentioned"
                }}
            ],
            "certifications": [
                {{
                    "name": "Certification name",
                    "issuer": "Issuing organization",
                    "date": "Issue date",
                    "url": "Certification URL if available"
                }}
            ],
            "achievements": ["Achievement 1", "Achievement 2"],
            "languages": ["Language 1", "Language 2"],
            "keywords_extracted": [
                "keyword1", "keyword2", "technology1", "skill1", "domain1"
            ],
            "classification_tags": [
                "technology", "software", "data_science", "AI", "backend", "frontend", "mobile", "devops", "hr", "finance", "marketing", "sales"
            ]
        }}

        IMPORTANT INSTRUCTIONS:
        1. For experience section, use the EXACT field names: title, company_name, location, start_time, end_time, summary
        2. Extract ALL relevant keywords from the resume into keywords_extracted array - include job titles, technologies, skills, tools, companies, domains
        3. For classification_tags, analyze the resume and include relevant categories from: technology, software, data_science, AI, machine_learning, backend, frontend, mobile, devops, cloud, database, hr, finance, marketing, sales, operations, consulting, research, academic
        4. summary field should contain the professional summary if found, otherwise set to null
        5. Make sure all dates are in format "MMM YYYY" (e.g., "Dec 2017", "Jan 2020") or "present"

        Resume text:
        {resume_text}

        Return only the JSON object, no additional text or formatting:
        """
        
        try:
            # Show progress and model being used
            progress_container = st.empty()
            progress_container.info(f"🦙 Processing with Ollama model: {model}")
            
            # Show timeout warning for large documents
            if len(resume_text) > 5000:
                st.warning("⏱️ Large document detected. Processing may take 2-3 minutes...")
            
            # Use a progress bar to show activity
            progress_bar = st.progress(0)
            progress_bar.progress(0.1, "Starting AI processing...")
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for consistent output
                        "top_p": 0.9,
                        "num_ctx": 8192,  # Increased context length for large documents
                        "stop": ["Human:", "Assistant:"]
                    }
                },
                timeout=300  # Increased to 5 minutes for large documents
            )
            
            progress_bar.progress(1.0, "AI processing complete!")
            progress_container.empty()
            progress_bar.empty()
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')
                
                # Try to extract JSON from response
                try:
                    # Find JSON in response (handle cases where model adds extra text)
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    
                    if json_start != -1 and json_end > json_start:
                        json_text = response_text[json_start:json_end]
                        parsed_data = json.loads(json_text)
                        parsed_data['_model_used'] = model
                        parsed_data['_source'] = 'ollama'
                        parsed_data['_timestamp'] = datetime.now().isoformat()
                        return parsed_data
                    else:
                        # If no valid JSON found, return structured fallback
                        return {
                            "error": "Could not extract valid JSON from model response",
                            "raw_response": response_text[:500],
                            "_model_used": model,
                            "_source": "ollama"
                        }
                        
                except json.JSONDecodeError as e:
                    return {
                        "error": f"JSON parsing failed: {str(e)}",
                        "raw_response": response_text[:500],
                        "_model_used": model,
                        "_source": "ollama"
                    }
            else:
                return {
                    "error": f"Ollama API error: {response.status_code}",
                    "message": response.text,
                    "_source": "ollama"
                }
                
        except requests.exceptions.Timeout:
            # Clear progress indicators
            if 'progress_bar' in locals():
                progress_bar.empty()
            if 'progress_container' in locals():
                progress_container.empty()
                
            return {
                "error": "⏰ Processing timed out after 5 minutes. This can happen with very large documents or complex resumes.",
                "suggestions": [
                    "Try a smaller/simpler document first to test the system",
                    "Use a faster model like 'phi3:mini'", 
                    "Check if Ollama is running properly: run 'ollama list' in terminal",
                    "Make sure your system has enough memory for the model"
                ],
                "troubleshooting": {
                    "document_size": f"{len(resume_text)} characters",
                    "model_used": model,
                    "timeout_duration": "300 seconds (5 minutes)"
                },
                "_source": "ollama"
            }
        except Exception as e:
            return {
                "error": f"Unexpected error: {str(e)}",
                "_source": "ollama"
            }

    def parse_resume_normalized(self, resume_text: str, model: str = None) -> Dict[str, Any]:
        """Parse resume and return normalized schema for downstream use."""
        try:
            raw = self.parse_resume(resume_text, model)
            from .normalizers import validate_and_normalize
            if isinstance(raw, dict) and "error" in raw:
                # Bubble up error; caller can fallback
                return raw
            provider = "ollama"
            model_used = raw.get("_model_used") if isinstance(raw, dict) else model
            normalized = validate_and_normalize(raw, provider=provider, model=model_used, parsing_method="model")
            return normalized
        except Exception as e:
            return {"error": f"Normalization failed: {e}", "_source": "ollama"}
    
    def parse_resume_with_fallback(self, resume_text: str, preferred_model: str = None) -> Dict[str, Any]:
        """
        Parse resume with automatic fallback to smaller models if memory errors occur.
        
        Args:
            resume_text (str): Resume text to parse
            preferred_model (str): Preferred model to try first
            
        Returns:
            dict: Parsing result or error with fallback attempts
        """
        
        if not self.available_models:
            return {
                "error": "No Ollama models available. Please install a model first.",
                "setup_required": True
            }
        
        # Order models by memory safety (safest first) - simplified to two models
        model_safety_order = ["phi3:mini", "llama3.2:3b"]
        
        # Create ordered list of models to try
        models_to_try = []
        
        # Add preferred model first if specified and available
        if preferred_model and preferred_model in self.available_models:
            models_to_try.append(preferred_model)
        
        # Add other models in safety order
        for safe_model in model_safety_order:
            if safe_model in self.available_models and safe_model not in models_to_try:
                models_to_try.append(safe_model)
        
        # Add any remaining models
        for model in self.available_models:
            if model not in models_to_try:
                models_to_try.append(model)
        
        last_error = None
        attempts = []
        
        for attempt, model in enumerate(models_to_try, 1):
            st.info(f"🔄 Attempt {attempt}: Trying model '{model}'...")
            
            try:
                result = self.parse_resume(resume_text, model)
                
                # Check if parsing was successful
                if "error" not in result:
                    if attempts:
                        st.success(f"✅ Successfully parsed with '{model}' after {attempt} attempts")
                    return result
                
                # Check for memory-related errors
                error_msg = result.get("error", "").lower()
                if any(keyword in error_msg for keyword in ["memory", "gpu", "system", "load", "500"]):
                    attempts.append({
                        "model": model,
                        "error": "Memory/GPU error",
                        "attempt": attempt
                    })
                    last_error = result
                    st.warning(f"⚠️ Model '{model}' failed due to memory constraints, trying smaller model...")
                    continue
                else:
                    # Non-memory error, return immediately
                    return result
                    
            except Exception as e:
                attempts.append({
                    "model": model,
                    "error": str(e),
                    "attempt": attempt
                })
                last_error = {"error": str(e), "_source": "ollama"}
                st.warning(f"⚠️ Model '{model}' failed: {str(e)}")
        
        # If we get here, all models failed
        return {
            "error": "❌ All available models failed to process the document",
            "last_error": last_error,
            "attempts": attempts,
            "suggestions": [
                "Your system may need more RAM to run these models",
                "Try installing phi3:mini: ollama pull phi3:mini",
                "Close other applications to free up memory",
                "Restart Ollama service: ollama serve",
                "Consider splitting large documents into smaller sections"
            ],
            "available_models": self.available_models,
            "_source": "ollama"
        }
    
    def generate_insights(self, resume_text: str, model: str = None) -> Dict[str, Any]:
        """Generate insights about the resume."""
        
        if not model and self.available_models:
            model = self.available_models[0]
        
        prompt = f"""
        Analyze this resume and provide insights. Return only a JSON object:

        {{
            "experience_summary": "2-3 sentences about work experience and career level",
            "skill_analysis": "Analysis of technical and soft skills, strengths and gaps",
            "career_progression": "How the career has developed over time",
            "strengths": "Key strengths and competitive advantages of the candidate",
            "recommendations": "2-3 specific suggestions for improvement",
            "experience_level": "Entry/Mid/Senior level assessment",
            "industry_fit": "Which industries or roles would be a good fit"
        }}

        Resume: {resume_text}
        
        Return only valid JSON:
        """
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_ctx": 2048
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')
                
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_text = response_text[json_start:json_end]
                    insights = json.loads(json_text)
                    insights['_model_used'] = model
                    insights['_source'] = 'ollama'
                    return insights
            
            return {"error": "Could not generate insights", "_source": "ollama"}
            
        except Exception as e:
            return {"error": f"Failed to generate insights: {str(e)}", "_source": "ollama"}

    def ask_question(self, question: str, context: str = "", max_tokens: int = 500) -> Dict[str, Any]:
        """
        Ask a question to Ollama with improved timeout handling.
        
        Args:
            question (str): The question to ask
            context (str): Additional context for the question
            max_tokens (int): Maximum tokens for response
            
        Returns:
            dict: Response with answer or error information
        """
        if not self.is_connected:
            return {"error": "Ollama is not connected"}
        
        if not self.selected_model:
            return {"error": "No model selected"}
        
        # Create context-aware prompt
        if context:
            prompt = f"""Context: {context}

Question: {question}

Please provide a clear, concise answer based on the context above."""
        else:
            prompt = question
        
        try:
            # Increased timeout and better error handling
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.selected_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "max_tokens": max_tokens,
                        "top_p": 0.9
                    }
                },
                timeout=180  # Increased to 3 minutes
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                
                return {
                    "answer": answer,
                    "model_used": self.selected_model,
                    "tokens_used": len(answer.split()),
                    "success": True
                }
            else:
                return {
                    "error": f"Ollama API error: {response.status_code} - {response.text}",
                    "success": False
                }
                
        except requests.exceptions.Timeout:
            return {
                "error": "Request timed out. The question might be too complex or the document too large. Try asking a more specific question.",
                "suggestion": "Break down complex questions into simpler parts, or try asking about specific sections of the document.",
                "success": False
            }
        except requests.exceptions.ConnectionError:
            return {
                "error": "Cannot connect to Ollama. Please make sure Ollama is running on localhost:11434",
                "success": False
            }
        except Exception as e:
            return {
                "error": f"Failed to get answer from Ollama: {str(e)}",
                "success": False
            }
    
def create_ollama_setup_guide():
    """Create Ollama setup guide in Streamlit."""
    
    st.sidebar.header("🦙 Ollama Local AI")
    
    # Check Ollama status first
    parser = OllamaParser()
    
    if parser.available_models:
        st.sidebar.success(f"✅ Ollama running with {len(parser.available_models)} models")
        
        # Model selection
        selected_model = st.sidebar.selectbox(
            "Select Model:",
            parser.available_models,
            help="Choose which local AI model to use"
        )
        
        # Model info
        with st.sidebar.expander("📊 Model Information"):
            for model in parser.available_models:
                st.sidebar.write(f"• {model}")
        
        return parser, selected_model
    else:
        st.sidebar.error("❌ Ollama not running or no models installed")
        
        with st.sidebar.expander("📋 Setup Instructions", expanded=True):
            st.markdown("""
            ### 🚀 Quick Setup:
            
            **1. Install Ollama:**
            - Windows: Download from [ollama.ai](https://ollama.ai/download)
            - Mac/Linux: `curl -fsSL https://ollama.ai/install.sh | sh`
            
            **2. Install a model:**
            ```bash
            ollama pull llama3.2:3b
            ```
            
            **3. Verify installation:**
            ```bash
            ollama list
            ```
            """)
        
        # Quick install buttons for recommended models
        st.sidebar.write("**Quick Install (if Ollama is running):**")
        recommended = ["llama3.2:3b", "phi3:mini"]
        
        for model in recommended:
            if st.sidebar.button(f"📥 Install {model}"):
                if parser.install_model(model):
                    st.success(f"✅ {model} installed!")
                    st.rerun()
        
        # Check connection button
        if st.sidebar.button("🔄 Check Connection"):
            st.rerun()
        
        return None, None

def integrate_ollama_parser():
    """Integrate Ollama into the main parsing flow."""
    
    parser, selected_model = create_ollama_setup_guide()
    
    if parser and selected_model:
        st.session_state.ollama_parser = parser
        st.session_state.ollama_model = selected_model
        return True
    
    return False

def test_ollama_connection():
    """Test Ollama connection and available models."""
    
    parser = OllamaParser()
    
    if parser.available_models:
        return {
            "status": "success",
            "message": f"Ollama running with {len(parser.available_models)} models",
            "models": parser.available_models
        }
    else:
        return {
            "status": "error",
            "message": "Ollama not available or no models installed",
            "models": []
        }

if __name__ == "__main__":
    # Test Ollama connection
    result = test_ollama_connection()
    print(f"Ollama Status: {result}")
    
    if result["status"] == "success":
        print("Available models:")
        for model in result["models"]:
            print(f"  - {model}")
    else:
        print("Setup required. Please install Ollama and pull a model.")