"""
Keyword Extraction and Resume Classification System
Analyzes resumes to extract relevant keywords and classify them into categories like Technology, HR, etc.
"""

import re
from typing import List, Dict, Tuple
from collections import Counter

class ResumeKeywordClassifier:
    """Extract keywords and classify resumes into categories."""
    
    def __init__(self):
        # Define keyword categories for classification
        self.category_keywords = {
            "technology": [
                "python", "java", "javascript", "c++", "c#", "golang", "rust", "kotlin", "swift",
                "react", "angular", "vue", "node.js", "express", "django", "flask", "spring",
                "html", "css", "sass", "typescript", "php", "ruby", "scala", "r",
                "sql", "nosql", "mongodb", "postgresql", "mysql", "redis", "elasticsearch",
                "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "ci/cd", "devops",
                "git", "github", "gitlab", "linux", "unix", "bash", "shell",
                "api", "rest", "graphql", "microservices", "cloud", "serverless",
                "machine learning", "ai", "deep learning", "tensorflow", "pytorch", "opencv",
                "data science", "analytics", "big data", "hadoop", "spark", "kafka",
                "frontend", "backend", "full-stack", "mobile", "ios", "android",
                "agile", "scrum", "kanban", "jira", "confluence"
            ],
            "data_science": [
                "data science", "machine learning", "deep learning", "ai", "artificial intelligence",
                "python", "r", "sql", "statistics", "analytics", "big data", "data mining",
                "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy", "matplotlib",
                "tableau", "power bi", "spark", "hadoop", "hive", "kafka", "airflow",
                "nlp", "computer vision", "predictive modeling", "regression", "classification",
                "clustering", "neural networks", "data visualization", "etl", "data warehouse"
            ],
            "hr": [
                "human resources", "hr", "recruitment", "talent acquisition", "hiring",
                "employee relations", "performance management", "compensation", "benefits",
                "training", "development", "onboarding", "offboarding", "hr policies",
                "compliance", "labor law", "payroll", "hris", "workday", "successfactors",
                "diversity", "inclusion", "engagement", "retention", "culture",
                "organizational development", "change management", "leadership development"
            ],
            "finance": [
                "finance", "accounting", "financial analysis", "budgeting", "forecasting",
                "financial modeling", "valuation", "investment", "portfolio management",
                "risk management", "compliance", "audit", "tax", "treasury", "banking",
                "excel", "sap", "oracle", "quickbooks", "financial reporting", "gaap",
                "ifrs", "cfa", "cpa", "frm", "financial planning", "corporate finance"
            ],
            "marketing": [
                "marketing", "digital marketing", "seo", "sem", "social media", "content marketing",
                "email marketing", "campaign management", "brand management", "market research",
                "analytics", "google analytics", "facebook ads", "google ads", "hubspot",
                "salesforce", "crm", "lead generation", "conversion optimization", "a/b testing",
                "marketing automation", "growth hacking", "product marketing", "marketing strategy"
            ],
            "sales": [
                "sales", "business development", "account management", "client relations",
                "lead generation", "prospecting", "closing", "negotiation", "crm", "salesforce",
                "pipeline management", "quota", "revenue", "territory management", "b2b", "b2c",
                "inside sales", "outside sales", "enterprise sales", "saas sales", "cold calling"
            ],
            "operations": [
                "operations", "project management", "process improvement", "lean", "six sigma",
                "supply chain", "logistics", "inventory management", "quality assurance",
                "pmp", "agile", "scrum", "operational excellence", "business analysis",
                "workflow optimization", "vendor management", "cost reduction"
            ],
            "consulting": [
                "consulting", "strategy", "business strategy", "management consulting",
                "implementation", "change management", "process optimization", "advisory",
                "client management", "stakeholder management", "business transformation",
                "organizational design", "market analysis", "competitive analysis"
            ],
            "research": [
                "research", "r&d", "analysis", "statistical analysis", "experimental design",
                "methodology", "publications", "papers", "phd", "postdoc", "academic",
                "literature review", "hypothesis", "data collection", "survey", "interview"
            ]
        }
        
        # Technology-specific keywords for more granular classification
        self.tech_subcategories = {
            "frontend": ["react", "angular", "vue", "html", "css", "javascript", "typescript", "sass", "bootstrap", "responsive"],
            "backend": ["python", "java", "node.js", "django", "flask", "spring", "express", "api", "rest", "graphql"],
            "mobile": ["ios", "android", "swift", "kotlin", "react native", "flutter", "xamarin", "mobile app"],
            "devops": ["docker", "kubernetes", "jenkins", "ci/cd", "aws", "azure", "gcp", "terraform", "ansible"],
            "ai_ml": ["machine learning", "ai", "deep learning", "tensorflow", "pytorch", "nlp", "computer vision", "neural networks"],
            "database": ["sql", "nosql", "mongodb", "postgresql", "mysql", "redis", "elasticsearch", "database design"],
            "cloud": ["aws", "azure", "gcp", "cloud computing", "serverless", "lambda", "s3", "ec2"]
        }
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract relevant keywords from resume text.
        
        Args:
            text (str): Resume text
            
        Returns:
            list: Extracted keywords
        """
        # Normalize text
        text_lower = text.lower()
        
        # Remove common stopwords and irrelevant text
        stopwords = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
            'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'among', 'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'can', 'shall', 'experience', 'work', 'working', 'years', 'year', 'months', 'month'
        }
        
        extracted_keywords = set()
        
        # Extract keywords from all categories
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                # Use word boundaries for exact matches
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                if re.search(pattern, text_lower):
                    extracted_keywords.add(keyword)
        
        # Extract tech subcategory keywords
        for subcat, keywords in self.tech_subcategories.items():
            for keyword in keywords:
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                if re.search(pattern, text_lower):
                    extracted_keywords.add(keyword)
        
        # Extract common technical terms using patterns
        tech_patterns = [
            r'\b([A-Z][a-z]+(?:\.[a-z]+)+)\b',  # Framework names like React.js, Node.js
            r'\b([A-Z]{2,})\b',  # Acronyms like API, SQL, AWS
            r'\b(\w+(?:-\w+)+)\b',  # Hyphenated terms like full-stack, end-to-end
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) > 2 and match.lower() not in stopwords:
                    extracted_keywords.add(match.lower())
        
        # Extract programming languages and tools
        prog_langs = [
            "python", "java", "javascript", "c++", "c#", "go", "rust", "kotlin", "swift",
            "php", "ruby", "scala", "r", "matlab", "perl", "shell", "bash", "powershell"
        ]
        
        for lang in prog_langs:
            if re.search(r'\b' + re.escape(lang) + r'\b', text_lower):
                extracted_keywords.add(lang)
        
        return list(extracted_keywords)
    
    def classify_resume(self, keywords: List[str], text: str = "") -> Dict[str, float]:
        """
        Classify resume into categories based on keywords.
        
        Args:
            keywords (list): Extracted keywords
            text (str): Optional full text for additional analysis
            
        Returns:
            dict: Category scores (0-1 scale)
        """
        keyword_set = {kw.lower() for kw in keywords}
        category_scores = {}
        
        for category, category_keywords in self.category_keywords.items():
            # Count matching keywords
            matches = sum(1 for kw in category_keywords if kw.lower() in keyword_set)
            
            # Calculate score based on matches and keyword density
            total_keywords = len(category_keywords)
            score = matches / total_keywords if total_keywords > 0 else 0
            
            # Boost score based on keyword frequency in text
            if text:
                text_lower = text.lower()
                keyword_frequency = sum(
                    len(re.findall(r'\b' + re.escape(kw) + r'\b', text_lower))
                    for kw in category_keywords
                )
                # Normalize by text length and add as bonus
                if len(text) > 0:
                    frequency_bonus = min(keyword_frequency / (len(text) / 1000), 0.3)
                    score += frequency_bonus
            
            category_scores[category] = min(score, 1.0)  # Cap at 1.0
        
        return category_scores
    
    def get_top_categories(self, category_scores: Dict[str, float], threshold: float = 0.1, top_n: int = 3) -> List[str]:
        """
        Get top categories above threshold.
        
        Args:
            category_scores (dict): Category scores from classify_resume
            threshold (float): Minimum score threshold
            top_n (int): Maximum number of categories to return
            
        Returns:
            list: Top category names
        """
        # Filter by threshold and sort by score
        filtered_categories = {k: v for k, v in category_scores.items() if v >= threshold}
        sorted_categories = sorted(filtered_categories.items(), key=lambda x: x[1], reverse=True)
        
        return [category for category, score in sorted_categories[:top_n]]
    
    def generate_classification_vector(self, text: str) -> Dict[str, any]:
        """
        Generate complete classification analysis for a resume.
        
        Args:
            text (str): Resume text
            
        Returns:
            dict: Complete classification results
        """
        # Extract keywords
        keywords = self.extract_keywords(text)
        
        # Classify into categories
        category_scores = self.classify_resume(keywords, text)
        
        # Get top categories
        top_categories = self.get_top_categories(category_scores)
        
        # Determine primary classification
        if category_scores.get("technology", 0) > 0.2:
            primary_classification = "technology"
        elif category_scores.get("data_science", 0) > 0.15:
            primary_classification = "data_science"
        elif category_scores.get("hr", 0) > 0.1:
            primary_classification = "hr"
        elif category_scores.get("finance", 0) > 0.1:
            primary_classification = "finance"
        elif category_scores.get("marketing", 0) > 0.1:
            primary_classification = "marketing"
        else:
            primary_classification = "general"
        
        # Get technology subcategories if it's a tech resume
        tech_subcategories = []
        if "technology" in top_categories:
            for subcat, subcat_keywords in self.tech_subcategories.items():
                matches = sum(1 for kw in subcat_keywords if kw.lower() in [k.lower() for k in keywords])
                if matches >= 2:  # At least 2 matching keywords
                    tech_subcategories.append(subcat)
        
        return {
            "keywords_extracted": keywords,
            "classification_tags": top_categories,
            "category_scores": category_scores,
            "primary_classification": primary_classification,
            "tech_subcategories": tech_subcategories,
            "confidence": max(category_scores.values()) if category_scores else 0.0
        }

def create_resume_classifier():
    """Factory function to create a resume classifier."""
    return ResumeKeywordClassifier()

# Example usage and testing
if __name__ == "__main__":
    classifier = ResumeKeywordClassifier()
    
    # Test with sample resume text
    sample_text = """
    Senior Software Engineer with 5 years of experience in Python, JavaScript, and React.
    Expertise in machine learning, data science, and cloud computing using AWS.
    Led development of web applications using Django, Node.js, and MongoDB.
    Experience with Docker, Kubernetes, and CI/CD pipelines.
    """
    
    result = classifier.generate_classification_vector(sample_text)
    
    print("Keywords extracted:", result["keywords_extracted"])
    print("Classification tags:", result["classification_tags"])
    print("Primary classification:", result["primary_classification"])
    print("Tech subcategories:", result["tech_subcategories"])
    print("Category scores:", result["category_scores"])