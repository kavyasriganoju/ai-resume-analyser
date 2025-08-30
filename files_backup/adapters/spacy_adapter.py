# spacy_adapter.py
import spacy
import yaml
import os
import re
from .base import ResumeAnalyzer

class SpacyResumeAnalyzer(ResumeAnalyzer):
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

        # --- Load skills dictionary ---
        skills_path = os.path.join(os.path.dirname(__file__), "skills.yml")
        with open(skills_path, "r") as f:
            self.skills_dict = set(
                s.lower() for s in yaml.safe_load(f)["skills"]
            )

    def analyze(self, resume_text: str, jd_text: str = None) -> dict:
        doc = self.nlp(resume_text)
        entities = {
            "PERSON": [], "ORG": [], "GPE": [],
            "EDUCATION": [], "SKILLS": []
        }

        # --- Extract default entities ---
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)

        # --- Dictionary-based SKILL extraction ---
        text_lower = resume_text.lower()
        found_skills = []
        for skill in self.skills_dict:
            # whole word match (avoid partials like "C" in "account")
            if re.search(rf"\b{re.escape(skill)}\b", text_lower):
                found_skills.append(skill)

        entities["SKILLS"].extend(found_skills)
        entities["SKILLS"] = list(set(entities["SKILLS"]))  # dedupe

        return {
            "type": "spacy",
            "entities": entities
        }
