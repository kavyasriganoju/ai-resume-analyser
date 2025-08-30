import re
from typing import Dict, List, Tuple
from collections import Counter
import math

class ResumeScorer:
    """Production-grade resume scoring with multiple metrics."""
    
    def __init__(self):
        # Weights for different scoring components
        self.weights = {
            "keyword_match": 0.25,
            "semantic_similarity": 0.35,
            "skills_relevance": 0.20,
            "experience_match": 0.15,
            "format_quality": 0.05
        }
    
    def compute_comprehensive_score(self, resume_data: dict, jd_text: str, 
                                  semantic_score: float) -> dict:
        """Compute comprehensive ATS score with detailed breakdown."""
        
        # Extract components
        resume_skills = resume_data.get("entities", {}).get("SKILLS", [])
        skills_by_category = resume_data.get("entities", {}).get("SKILLS_BY_CATEGORY", {})
        experience_info = resume_data.get("entities", {}).get("EXPERIENCE_INFO", {})
        
        # Calculate individual scores
        keyword_score = self._compute_keyword_score(resume_skills, jd_text)
        skills_score = self._compute_skills_relevance_score(skills_by_category, jd_text)
        experience_score = self._compute_experience_match_score(experience_info, jd_text)
        format_score = self._compute_format_quality_score(resume_data)
        
        # Calculate weighted final score
        final_score = (
            keyword_score * self.weights["keyword_match"] +
            semantic_score * self.weights["semantic_similarity"] +
            skills_score * self.weights["skills_relevance"] +
            experience_score * self.weights["experience_match"] +
            format_score * self.weights["format_quality"]
        )
        
        return {
            "final_score": round(final_score, 2),
            "breakdown": {
                "keyword_match": round(keyword_score, 2),
                "semantic_similarity": round(semantic_score, 2),
                "skills_relevance": round(skills_score, 2),
                "experience_match": round(experience_score, 2),
                "format_quality": round(format_score, 2)
            },
            "grade": self._get_grade(final_score),
            "recommendations": self._generate_detailed_recommendations(
                keyword_score, semantic_score, skills_score, 
                experience_score, format_score
            )
        }
    
    def _compute_keyword_score(self, resume_skills: List[str], jd_text: str) -> float:
        """Enhanced keyword matching with TF-IDF weighting."""
        if not resume_skills or not jd_text:
            return 0.0
        
        jd_lower = jd_text.lower()
        
        # Simple keyword matching
        matched_skills = [skill for skill in resume_skills 
                         if skill.lower() in jd_lower]
        
        if not resume_skills:
            return 0.0
        
        base_score = (len(matched_skills) / len(resume_skills)) * 100
        
        # Bonus for exact phrase matches
        exact_matches = 0
        for skill in matched_skills:
            if re.search(rf'\b{re.escape(skill.lower())}\b', jd_lower):
                exact_matches += 1
        
        bonus = min(20, exact_matches * 2)  # Cap bonus at 20%
        return min(100, base_score + bonus)
    
    def _compute_skills_relevance_score(self, skills_by_category: Dict, jd_text: str) -> float:
        """Score based on skill category relevance to job description."""
        if not skills_by_category or not jd_text:
            return 0.0
        
        jd_lower = jd_text.lower()
        
        # Category importance weights based on common job requirements
        category_weights = {
            "programming_languages": 1.2,
            "web_technologies": 1.1,
            "cloud_platforms": 1.3,
            "databases": 1.0,
            "data_science": 1.2,
            "tools": 0.8,
            "frameworks": 1.0
        }
        
        total_score = 0
        total_weight = 0
        
        for category, skills in skills_by_category.items():
            if skills:
                weight = category_weights.get(category, 1.0)
                category_score = self._score_category_relevance(skills, jd_lower)
                total_score += category_score * weight
                total_weight += weight
        
        return (total_score / total_weight) if total_weight > 0 else 0
    
    def _score_category_relevance(self, skills: List[str], jd_lower: str) -> float:
        """Score relevance of skills in a category."""
        if not skills:
            return 0
        
        relevant_skills = 0
        for skill in skills:
            if skill.lower() in jd_lower:
                relevant_skills += 1
        
        return (relevant_skills / len(skills)) * 100
    
    def _compute_experience_match_score(self, experience_info: Dict, jd_text: str) -> float:
        """Score based on experience level match with JD requirements."""
        if not experience_info or not jd_text:
            return 50  # Neutral score if no data
        
        candidate_years = experience_info.get("total_years", 0)
        candidate_level = experience_info.get("level", "entry")
        
        # Extract experience requirements from JD
        required_years = self._extract_required_experience(jd_text)
        
        if required_years is None:
            return 75  # Good score if no specific requirement mentioned
        
        # Score based on how well experience matches
        if candidate_years >= required_years:
            # Bonus for exceeding requirements (but not too much)
            excess = candidate_years - required_years
            if excess <= 2:
                return 100
            elif excess <= 5:
                return 95
            else:
                return 90  # Too overqualified might be a concern
        else:
            # Penalty for not meeting requirements
            shortage = required_years - candidate_years
            if shortage <= 1:
                return 80
            elif shortage <= 2:
                return 60
            else:
                return 30
    
    def _extract_required_experience(self, jd_text: str) -> int:
        """Extract required years of experience from job description."""
        # Look for patterns like "3+ years", "5-7 years", "minimum 3 years"
        patterns = [
            r'(\d+)\+?\s*years?\s*of\s*experience',
            r'minimum\s*(\d+)\s*years?',
            r'at\s*least\s*(\d+)\s*years?',
            r'(\d+)-\d+\s*years?\s*experience',
            r'(\d+)\s*to\s*\d+\s*years?'
        ]
        
        jd_lower = jd_text.lower()
        for pattern in patterns:
            match = re.search(pattern, jd_lower)
            if match:
                return int(match.group(1))
        
        return None
    
    def _compute_format_quality_score(self, resume_data: Dict) -> float:
        """Score resume format and structure quality."""
        score = 100
        
        # Check if key sections are present
        entities = resume_data.get("entities", {})
        
        if not entities.get("SKILLS"):
            score -= 20
        if not entities.get("EXPERIENCE_YEARS") and not entities.get("EXPERIENCE_INFO"):
            score -= 15
        if not entities.get("PERSON"):
            score -= 10
        
        return max(0, score)
    
    def _get_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def _generate_detailed_recommendations(self, keyword_score: float, 
                                         semantic_score: float, skills_score: float,
                                         experience_score: float, format_score: float) -> List[str]:
        """Generate specific recommendations based on individual scores."""
        recommendations = []
        
        if keyword_score < 60:
            recommendations.append(
                "Include more keywords from the job description in your resume"
            )
        
        if semantic_score < 70:
            recommendations.append(
                "Improve the overall relevance of your resume content to the job description"
            )
        
        if skills_score < 70:
            recommendations.append(
                "Add more technical skills that are relevant to this role"
            )
        
        if experience_score < 70:
            recommendations.append(
                "Highlight your relevant work experience more prominently"
            )
        
        if format_score < 80:
            recommendations.append(
                "Improve your resume structure and ensure all key sections are present"
            )
        
        # Positive reinforcement for strong areas
        if max(keyword_score, semantic_score, skills_score) >= 85:
            strong_area = ["keyword matching", "content relevance", "skills alignment"][
                [keyword_score, semantic_score, skills_score].index(
                    max(keyword_score, semantic_score, skills_score)
                )
            ]
            recommendations.append(f"Great job on {strong_area}!")
        
        return recommendations

# Legacy function for backward compatibility
def compute_keyword_score(resume_skills: List[str], jd_text: str) -> float:
    """Legacy function for backward compatibility."""
    scorer = EnhancedResumeScorer()
    return scorer._compute_keyword_score(resume_skills, jd_text)

def blend_scores(keyword_score: float, semantic_score: float, 
                kw_weight=0.4, sem_weight=0.6) -> float:
    """Legacy function for backward compatibility."""
    final = (keyword_score * kw_weight) + (semantic_score * sem_weight)
    return round(final, 2)