
from adapters.base import ResumeAnalyzer
from sentence_transformers import SentenceTransformer, util

class BertResumeAnalyzer(ResumeAnalyzer):
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def analyze(self, resume_text: str, jd_text: str = None) -> dict:
        if not jd_text:
            return {"type": "bert", "ats_score": None}

        emb_resume = self.model.encode(resume_text, convert_to_tensor=True)
        emb_jd = self.model.encode(jd_text, convert_to_tensor=True)
        score = util.pytorch_cos_sim(emb_resume, emb_jd).item()

        return {"type": "bert", "ats_score": round(score * 100, 2)}
