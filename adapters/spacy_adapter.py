import spacy
import yaml
import os
import re
from collections import defaultdict
from .base import ResumeAnalyzer

class SpacyResumeAnalyzer(ResumeAnalyzer):
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        
        # Load skills dictionary with categories
        self._load_skills_database()
        
        # Add custom patterns for better entity recognition
        self._add_custom_patterns()
    
    def _load_skills_database(self):
        """Load enhanced skills database with categories."""
        skills_path = os.path.join(os.path.dirname(__file__), "enhanced_skills.yml")
        
        # If enhanced skills file doesn't exist, create it from basic one
        if not os.path.exists(skills_path):
            self._create_enhanced_skills_file(skills_path)
        
        with open(skills_path, "r") as f:
            skills_data = yaml.safe_load(f)
        
        self.skills_by_category = skills_data.get("skills_by_category", {})
        self.all_skills = set()
        
        for category, skills in self.skills_by_category.items():
            self.all_skills.update(s.lower() for s in skills)
    
    def _create_enhanced_skills_file(self, path: str):
        """Create enhanced skills file with categories."""
        enhanced_skills = {
            "skills_by_category": {
                "programming_languages": [
                    "python", "java", "c++", "c#", "javascript", "typescript", 
                    "go", "rust", "kotlin", "swift", "php", "ruby", "scala", "r"
                ],
                "web_technologies": [
                    "react", "angular", "vue.js", "node.js", "express.js", 
                    "django", "flask", "spring", "html", "css", "sass", "less"
                ],
                "databases": [
                    "sql", "mysql", "postgresql", "mongodb", "redis", 
                    "elasticsearch", "sqlite", "oracle", "sql server"
                ],
                "cloud_platforms": [
                    "aws", "azure", "gcp", "google cloud", "docker", 
                    "kubernetes", "terraform", "ansible"
                ],
                "data_science": [
                    "machine learning", "deep learning", "nlp", "pandas", 
                    "numpy", "pytorch", "tensorflow", "scikit-learn", "spark"
                ],
                "tools": [
                    "git", "jenkins", "jira", "confluence", "linux", 
                    "unix", "bash", "powershell"
                ],
                "frameworks": [
                    "spring boot", "laravel", "rails", "fastapi", 
                    "nextjs", "gatsby", "nuxt"
                ]
            }
        }
        
        with open(path, "w") as f:
            yaml.dump(enhanced_skills, f, default_flow_style=False)
    
    def _add_custom_patterns(self):
        """Add custom patterns for better entity recognition."""
        from spacy.matcher import Matcher
        
        self.matcher = Matcher(self.nlp.vocab)
        
        # Pattern for years of experience
        exp_pattern = [
            {"LIKE_NUM": True},
            {"LOWER": {"IN": ["years", "year", "yrs", "yr"]}},
            {"LOWER": {"IN": ["of", "experience", "exp"]}, "OP": "*"}
        ]
        self.matcher.add("EXPERIENCE_YEARS", [exp_pattern])
        
        # Pattern for education degrees
        degree_patterns = [
            [{"LOWER": {"IN": ["bachelor", "bachelors", "bs", "ba", "bsc"]}},
             {"LOWER": "of", "OP": "?"}, {"LOWER": "science", "OP": "?"}, {"LOWER": "in", "OP": "?"}],
            [{"LOWER": {"IN": ["master", "masters", "ms", "ma", "msc"]}},
             {"LOWER": "of", "OP": "?"}, {"LOWER": "science", "OP": "?"}, {"LOWER": "in", "OP": "?"}],
            [{"LOWER": {"IN": ["phd", "ph.d", "doctorate", "doctoral"]}},
             {"LOWER": "in", "OP": "?"}]
        ]
        
        for i, pattern in enumerate(degree_patterns):
            self.matcher.add(f"DEGREE_{i}", [pattern])
    
    def analyze(self, resume_text: str, jd_text: str = None) -> dict:
        doc = self.nlp(resume_text)
        
        # Extract standard entities
        entities = self._extract_standard_entities(doc)
        
        # Extract skills with categories
        skills_analysis = self._extract_skills_with_categories(resume_text, jd_text)
        entities["SKILLS"] = skills_analysis["all_skills"]
        entities["SKILLS_BY_CATEGORY"] = skills_analysis["by_category"]
        
        # Extract custom patterns
        custom_entities = self._extract_custom_patterns(doc)
        entities.update(custom_entities)
        
        # Calculate experience level
        experience_info = self._calculate_experience_level(resume_text)
        entities["EXPERIENCE_INFO"] = experience_info
        
        return {
            "type": "spacy",
            "entities": entities,
            "skills_match_score": skills_analysis.get("match_score", 0),
            "recommendations": self._generate_skill_recommendations(skills_analysis, jd_text)
        }
    
    def _extract_standard_entities(self, doc) -> dict:
        """Extract standard spaCy entities."""
        entities = defaultdict(list)
        
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE"]:
                entities[ent.label_].append(ent.text)
        
        return dict(entities)
    
    def _extract_skills_with_categories(self, resume_text: str, jd_text: str = None) -> dict:
        """Extract skills organized by categories."""
        text_lower = resume_text.lower()
        found_skills_by_category = defaultdict(list)
        all_found_skills = []
        
        for category, skills in self.skills_by_category.items():
            for skill in skills:
                # Use word boundaries to avoid partial matches
                if re.search(rf'\b{re.escape(skill.lower())}\b', text_lower):
                    found_skills_by_category[category].append(skill)
                    all_found_skills.append(skill)
        
        # Calculate match score against JD if provided
        match_score = 0
        if jd_text:
            jd_lower = jd_text.lower()
            jd_skills = []
            
            # Extract skills from JD
            for skill in self.all_skills:
                if re.search(rf'\b{re.escape(skill)}\b', jd_lower):
                    jd_skills.append(skill)
            
            if jd_skills:
                matched_skills = [s for s in all_found_skills if s.lower() in [js.lower() for js in jd_skills]]
                match_score = (len(matched_skills) / len(jd_skills)) * 100
        
        return {
            "by_category": dict(found_skills_by_category),
            "all_skills": list(set(all_found_skills)),
            "match_score": round(match_score, 2)
        }
    
    def _extract_custom_patterns(self, doc) -> dict:
        """Extract custom patterns using matcher."""
        matches = self.matcher(doc)
        custom_entities = defaultdict(list)
        
        for match_id, start, end in matches:
            label = self.nlp.vocab.strings[match_id]
            span = doc[start:end]
            custom_entities[label].append(span.text)
        
        return dict(custom_entities)
    
    def _calculate_experience_level(self, resume_text: str) -> dict:
        """Calculate experience level from resume."""
        # Look for years of experience mentions
        exp_pattern = r'(\d+)[\s\-+]*(years?|yrs?)\s*(of\s*)?(experience|exp)'
        matches = re.findall(exp_pattern, resume_text.lower())
        
        total_years = 0
        if matches:
            years_mentioned = [int(match[0]) for match in matches]
            total_years = max(years_mentioned)  # Take the highest mentioned
        
        # Classify experience level
        if total_years == 0:
            level = "entry"
        elif total_years <= 2:
            level = "junior"
        elif total_years <= 5:
            level = "mid"
        elif total_years <= 10:
            level = "senior"
        else:
            level = "expert"
        
        return {
            "total_years": total_years,
            "level": level
        }
    
    def _generate_skill_recommendations(self, skills_analysis: dict, jd_text: str = None) -> list:
        """Generate recommendations for skill improvements."""
        recommendations = []
        
        if jd_text and skills_analysis.get("match_score", 0) < 70:
            recommendations.append("Consider adding more relevant skills mentioned in the job description")
        
        # Check for balanced skill distribution
        by_category = skills_analysis.get("by_category", {})
        if len(by_category.get("programming_languages", [])) == 0:
            recommendations.append("Consider adding programming languages to strengthen your technical profile")
        
        if len(by_category.get("cloud_platforms", [])) == 0:
            recommendations.append("Cloud platform experience is valuable in today's market")
        
        return recommendations