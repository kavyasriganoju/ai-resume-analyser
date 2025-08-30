
from abc import ABC, abstractmethod

class ResumeAnalyzer(ABC):
    @abstractmethod
    def analyze(self, resume_text: str, jd_text: str = None) -> dict:
        """Analyze resume (and optionally compare with JD)."""
        pass
