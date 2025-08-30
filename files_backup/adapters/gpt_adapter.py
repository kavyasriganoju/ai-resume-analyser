
import openai
from adapters.base import ResumeAnalyzer

class GPTResumeAnalyzer(ResumeAnalyzer):
    def __init__(self, api_key: str):
        openai.api_key = api_key

    def analyze(self, resume_text: str, jd_text: str = None) -> dict:
        prompt = f"""
        You are an ATS resume reviewer.
        Resume:
        {resume_text}

        Job Description:
        {jd_text if jd_text else "N/A"}

        1. List strengths.
        2. List weaknesses.
        3. Give 3 improvement suggestions.
        4. If possible, rewrite one section to be stronger.
        Return JSON.
        """
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}]
        )
        return {"type": "gpt", "review": resp.choices[0].message["content"]}
