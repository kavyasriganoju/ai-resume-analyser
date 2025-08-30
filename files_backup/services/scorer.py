

def compute_keyword_score(resume_skills: list, jd_text: str) -> float:
    """Compute % of resume skills that appear in JD."""
    jd_lower = jd_text.lower()
    matched = sum(1 for skill in resume_skills if skill.lower() in jd_lower)
    if not resume_skills:
        return 0.0
    return (matched / len(resume_skills)) * 100


def blend_scores(keyword_score: float, semantic_score: float, kw_weight=0.4, sem_weight=0.6) -> float:
    """Blend keyword + semantic scores into a final ATS score."""
    final = (keyword_score * kw_weight) + (semantic_score * sem_weight)
    return round(final, 2)
