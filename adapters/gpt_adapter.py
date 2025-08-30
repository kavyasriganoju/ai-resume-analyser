# gpt_adapter.py
import openai
from adapters.base import ResumeAnalyzer
import json
import tiktoken
import logging
import re

logger = logging.getLogger(__name__)

class GPTResumeAnalyzer(ResumeAnalyzer):
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
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
                    {"role": "system", "content": "You are an expert ATS resume analyzer. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            # --- NEW: extract numeric score from GPT output ---
            fit_score_raw = result.get("overall_fit_score", "")
            numeric_score = self._extract_numeric_score(fit_score_raw)

            return {
                "type": "gpt",
                "analysis": result,
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
            return float(raw) * 10 if raw <= 10 else float(raw)
        if isinstance(raw, str):
            match = re.search(r'(\d+(\.\d+)?)', raw)
            if match:
                val = float(match.group(1))
                return val * 10 if val <= 10 else val
        return None

    def _create_structured_prompt(self, resume_text: str, jd_text: str) -> str:
        return f"""
Analyze this resume against the job description and provide feedback in JSON format.

RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}

Please analyze and respond with JSON containing:
{{
    "strengths": ["list of 3-4 key strengths"],
    "weaknesses": ["list of 3-4 areas for improvement"],
    "suggestions": ["list of 3-4 specific improvement suggestions"],
    "missing_skills": ["list of important skills mentioned in JD but missing from resume"],
    "ats_optimization_tips": ["list of 2-3 ATS-specific tips"],
    "overall_fit_score": "score from 1-10 with brief explanation"
}}
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
