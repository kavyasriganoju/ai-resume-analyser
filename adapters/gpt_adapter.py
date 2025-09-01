"""
Project: Resume Analyser Critic (GPT-powered Resume Analyzer)
File: gpt_adapter.py
Author: Kavyasri Ganoju
Email: kavyasriganoju@gmail.com
Created: 2025-08-30
Description:
    Implements GPT-based resume analyzer using OpenAI API.
    Provides scoring, strengths/weaknesses, and ATS optimization tips.

"""
from openai import OpenAI
from adapters.base import ResumeAnalyzer
import json
import tiktoken
import logging
import re

logger = logging.getLogger(__name__)

class GPTResumeAnalyzer(ResumeAnalyzer):
    def __init__(self, api_key: str):
        
        logger.info(f"About to create OpenAI client with api_key: {api_key[:8]}...")
    
        try:
            self.client = OpenAI(api_key=api_key)
            logger.info("OpenAI client created successfully")
        except Exception as e:
            logger.error(f"OpenAI client creation failed: {e}")
            raise

        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
            logger.info("Tokenizer initialized successfully with cl100k_base")
        except Exception as e:
            logger.error(f"Failed to initialize tokenizer: {str(e)}")
            raise
        self.max_tokens = 4000

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text

        keep_tokens = max_tokens - 100
        start_tokens = tokens[:keep_tokens // 2]
        end_tokens = tokens[-keep_tokens // 2:]
        return self.encoding.decode(start_tokens) + "\n[...truncated...]\n" + self.encoding.decode(end_tokens)

    def analyze(self, resume_text: str, jd_text: str = None) -> dict:
        resume_text = self._truncate_text(resume_text, 2000)
        jd_text = self._truncate_text(jd_text, 1500) if jd_text else "N/A"

        prompt = self._create_structured_prompt(resume_text, jd_text)

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert ATS resume analyzer and career coach. Always respond with valid JSON in the exact format requested. Be specific, actionable, and user-friendly in your analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            # --- NEW: extract numeric score from GPT output ---
            fit_score_raw = result.get("overall_fit_score", "")
            numeric_score = self._extract_numeric_score(fit_score_raw)

            # Validate and clean the response
            cleaned_result = self._validate_and_clean_response(result)
            
            return {
                "type": "gpt",
                "analysis": cleaned_result,
                "token_usage": response.usage.total_tokens if response.usage else 0,
                "final_score": numeric_score,
                "grade": _score_to_grade(numeric_score) if numeric_score is not None else None
            }

        except Exception as e:
            logger.error(e)
            return {
                "type": "gpt",
                "error": "Oops error please contact administrator",
                "analysis": {"strengths": [], "weaknesses": [], "suggestions": []},
                "final_score": None,
                "grade": None
            }

    def _extract_numeric_score(self, raw) -> float:
        """Extract number from GPT's overall_fit_score field."""
        if isinstance(raw, (int, float)):
            # Convert 1-10 scale to 0-100 scale for consistency
            return float(raw) * 10 if raw <= 10 else float(raw)
        if isinstance(raw, str):
            # Extract first number found, handle decimals
            match = re.search(r'(\d+(?:\.\d+)?)', raw)
            if match:
                val = float(match.group(1))
                # Convert 1-10 scale to 0-100 scale
                return val * 10 if val <= 10 else val
        return None

    def _validate_and_clean_response(self, result: dict) -> dict:
        """Validate and clean GPT response to ensure consistent format."""
        cleaned = {
            "overall_fit_score": result.get("overall_fit_score", 0),
            "strengths": self._clean_list_items(result.get("strengths", [])),
            "missing_skills": self._clean_list_items(result.get("missing_skills", [])),
            "suggestions": self._clean_list_items(result.get("suggestions", [])),
            "ats_optimization_tips": self._clean_list_items(result.get("ats_optimization_tips", []))
        }
        
        # Ensure score is numeric
        if isinstance(cleaned["overall_fit_score"], str):
            match = re.search(r'(\d+(?:\.\d+)?)', cleaned["overall_fit_score"])
            if match:
                cleaned["overall_fit_score"] = float(match.group(1))
            else:
                cleaned["overall_fit_score"] = 0
        
        return cleaned
    
    def _clean_list_items(self, items: list) -> list:
        """Clean individual list items to remove formatting artifacts."""
        if not items:
            return []
        
        cleaned_items = []
        for item in items:
            if isinstance(item, str):
                # Remove any score patterns like "8/10" or "5 - text"
                clean_item = re.sub(r'^\d+\s*[-/]\s*', '', item.strip())
                clean_item = re.sub(r'\s*\d+/\d+\s*$', '', clean_item)
                clean_item = clean_item.strip()
                if clean_item:
                    cleaned_items.append(clean_item)
            else:
                cleaned_items.append(str(item))
        
        return cleaned_items

    def _create_structured_prompt(self, resume_text: str, jd_text: str) -> str:
        return f"""
You are an expert ATS resume analyzer and career coach. Analyze this resume against the job description and provide actionable feedback.

RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}

Provide your analysis in the following JSON format. Be specific, actionable, and user-friendly:

{{
    "overall_fit_score": 7,
    "strengths": [
        "Strong technical background with relevant programming languages",
        "Clear project descriptions with quantifiable results",
        "Good educational foundation for the role"
    ],
    "missing_skills": [
        "Cloud platforms (AWS, Azure) mentioned in job requirements",
        "Specific framework experience (React, Angular)",
        "DevOps tools and CI/CD pipeline experience"
    ],
    "suggestions": [
        "Add a professional summary highlighting your key achievements",
        "Quantify your project impacts with specific metrics and numbers",
        "Include relevant certifications or online courses completed",
        "Use action verbs to start each bullet point in experience section"
    ],
    "ats_optimization_tips": [
        "Use exact keywords from the job description throughout your resume",
        "Ensure your resume is in a simple, ATS-friendly format",
        "Include a skills section with both hard and soft skills"
    ]
}}

IMPORTANT GUIDELINES:
- overall_fit_score: Provide ONLY a number from 1-10 (no explanation text)
- strengths: Focus on what the candidate does well relative to the job
- missing_skills: Identify specific skills/technologies from the JD that aren't in the resume
- suggestions: Provide 3-4 concrete, actionable improvement recommendations
- ats_optimization_tips: Give 2-3 specific tips to improve ATS compatibility
- Keep all text clear, professional, and jargon-free
- Focus on actionable advice rather than generic statements
"""

def _score_to_grade(score: float) -> str:
    if score is None:
        return None
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
