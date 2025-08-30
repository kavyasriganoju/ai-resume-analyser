
"""
Project: Resume Analyser Critic (GPT-powered Resume Analyzer)
File: base.py
Author: Kavyasri Ganoju
Email: kavyasriganoju@gmail.com
Created: 2025-08-30
Description:
    Implements GPT-based resume analyzer using OpenAI API.
    Provides scoring, strengths/weaknesses, and ATS optimization tips.

"""
from abc import ABC, abstractmethod

class ResumeAnalyzer(ABC):
    @abstractmethod
    def analyze(self, resume_text: str, jd_text: str = None) -> dict:
        """Analyze resume (and optionally compare with JD)."""
        pass
