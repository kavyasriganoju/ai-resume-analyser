from adapters.base import ResumeAnalyzer
from sentence_transformers import SentenceTransformer, util
import numpy as np
import hashlib
from functools import lru_cache
import torch

class BertResumeAnalyzer(ResumeAnalyzer):
    def __init__(self, use_gpu=False):
        # Use a more efficient model or keep the current lightweight one
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Move to GPU if available and requested
        if use_gpu and torch.cuda.is_available():
            self.model = self.model.cuda()
        
        # Cache for embeddings
        self._embedding_cache = {}
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text to use as cache key."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _get_embedding(self, text: str):
        """Get embedding with caching."""
        text_hash = self._get_text_hash(text)
        
        if text_hash not in self._embedding_cache:
            embedding = self.model.encode(text, convert_to_tensor=True)
            self._embedding_cache[text_hash] = embedding
        
        return self._embedding_cache[text_hash]
    
    def analyze(self, resume_text: str, jd_text: str = None) -> dict:
        if not jd_text:
            return {"type": "bert", "ats_score": None, "section_scores": {}}

        # Get embeddings with caching
        emb_resume = self._get_embedding(resume_text)
        emb_jd = self._get_embedding(jd_text)
        
        # Overall similarity
        overall_score = util.pytorch_cos_sim(emb_resume, emb_jd).item()
        
        # Section-wise analysis for more detailed feedback
        section_scores = self._analyze_sections(resume_text, jd_text)
        
        return {
            "type": "bert",
            "ats_score": round(overall_score * 100, 2),
            "section_scores": section_scores,
            "recommendations": self._generate_recommendations(section_scores)
        }
    
    def _analyze_sections(self, resume_text: str, jd_text: str) -> dict:
        """Analyze different sections of resume against JD."""
        # Simple section extraction (can be  with better parsing)
        sections = self._extract_sections(resume_text)
        jd_embedding = self._get_embedding(jd_text)
        
        section_scores = {}
        for section_name, section_text in sections.items():
            if section_text.strip():
                section_emb = self._get_embedding(section_text)
                score = util.pytorch_cos_sim(section_emb, jd_embedding).item()
                section_scores[section_name] = round(score * 100, 2)
        
        return section_scores
    
    def _extract_sections(self, resume_text: str) -> dict:
        """Basic section extraction - can be improved further."""
        import re
        
        sections = {
            "experience": "",
            "skills": "",
            "education": "",
            "summary": ""
        }
        
        # Simple regex-based section extraction
        text_lower = resume_text.lower()
        
        # Extract experience section
        exp_match = re.search(r'(experience|work history).*?(?=education|skills|$)', text_lower, re.DOTALL)
        if exp_match:
            sections["experience"] = exp_match.group(0)
        
        # Extract skills section
        skills_match = re.search(r'(skills|technical skills).*?(?=experience|education|$)', text_lower, re.DOTALL)
        if skills_match:
            sections["skills"] = skills_match.group(0)
        
        # Extract education section
        edu_match = re.search(r'education.*?(?=experience|skills|$)', text_lower, re.DOTALL)
        if edu_match:
            sections["education"] = edu_match.group(0)
            
        return sections
    
    def _generate_recommendations(self, section_scores: dict) -> list:
        """Generate recommendations based on section scores."""
        recommendations = []
        
        for section, score in section_scores.items():
            if score < 30:
                recommendations.append(f"Consider enhancing your {section} section to better match the job requirements")
            elif score < 50:
                recommendations.append(f"Your {section} section could be improved to better align with the role")
        
        return recommendations