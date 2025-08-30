
# 📄 Resume Analysis & ATS Scoring System

An end-to-end system that analyzes resumes against job descriptions using **SpaCy**, **BERT**, and **GPT**, then produces a comprehensive ATS-style score and actionable feedback.

---

## 🚀 Project Overview

This project allows candidates or recruiters to:

- Upload resumes (`.pdf`, `.docx`, `.txt`) and job descriptions.
- Extract structured entities (skills, experience, education) with **SpaCy**.
- Compute **semantic similarity** between resumes & JDs using **BERT embeddings**.
- Get **human-style insights** via **GPT** (strengths, weaknesses, suggestions).
- Produce a **comprehensive ATS score** blending multiple signals.

**Architecture**:

![System Architecture](system_architecture.png)

---

## ⚙️ Setup Instructions

### 1. Clone repo & install dependencies
```bash
git clone https://github.com/your-org/resume-ats-analyzer.git
cd resume-ats-analyzer

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Environment variables
Create a `.env` file:
```bash
OPENAI_API_KEY=your_openai_api_key_here
FLASK_ENV=development
PORT=5000
```

### 3. Run the app
```bash
python app.py
```

Visit: [http://localhost:5000](http://localhost:5000)

---

## 🧩 Module Breakdown

### **1. `extractor.py`**
- Extracts text from:
  - **PDFs** (PyMuPDF)
  - **DOCX** (python-docx)
  - **TXT** files
- Cleans text: removes headers/footers, normalizes whitespace, strips page numbers.

### **2. `spacy_adapter.py`**
- Uses **SpaCy** (`en_core_web_sm`) for entity extraction.
- Detects:
  - **Skills** (from `enhanced_skills.yml`)
  - **Experience years**
  - **Degrees (BSc, MSc, PhD)**
- Organizes skills by categories (cloud, web, programming, etc.).

### **3. `bert_adapter.py`**
- Uses **SentenceTransformer** (`all-MiniLM-L6-v2`).
- Computes:
  - Overall **semantic similarity** between resume & JD.
  - Section-level scores (skills, education, experience).
- Returns an **ATS similarity score** (0–100).

### **4. `gpt_adapter.py`**
- Calls **OpenAI GPT (gpt-4o-mini)**.
- Produces:
  - Strengths & weaknesses
  - Suggestions
  - Missing skills
  - ATS tips
  - Fit score (1–10)
- Returns JSON for easy downstream use.

### **5. `scorer.py`**
- Aggregates all scores into a **Comprehensive ATS Score**.
- Components:
  - Keyword Match
  - Semantic Similarity
  - Skills Relevance
  - Experience Match
  - Resume Format Quality
- Produces **final score + grade (A–F)**.

### **6. `factory.py`**
- Loads analyzers dynamically:
  - Always → SpaCy + BERT
  - Optional → GPT (if `OPENAI_API_KEY` present).

### **7. `app.py` (Flask App)**
Endpoints:
- `/` → Home page (HTML UI)
- `/health` → Health check
- `/analyze` → Analyze **1 resume + JD**
- `/analyze-batch` → Analyze **multiple resumes** (GPT skipped for cost reasons).

---

## 📡 Example API Usage

### Analyze a single resume
```bash
curl -X POST http://localhost:5000/analyze   -F "resume=@resume.pdf"   -F "jd_file=@job_desc.txt"
```

Response (JSON):
```json
{
  "SpacyResumeAnalyzer": {...},
  "BertResumeAnalyzer": {...},
  "GPTResumeAnalyzer": {...},
  "ComprehensiveScore": {
    "final_score": 82.5,
    "grade": "B",
    "breakdown": {...}
  },
  "Summary": {
    "overall_assessment": "Good match with room for improvement.",
    "key_strengths": [...],
    "priority_improvements": [...],
    "quick_wins": [...]
  }
}
```

### Batch analysis (no GPT)
```bash
curl -X POST http://localhost:5000/analyze-batch   -F "resumes=@cv1.pdf"   -F "resumes=@cv2.pdf"   -F "job_desc=Backend developer with Python, Django, AWS..."
```

Response (JSON):
```json
{
  "batch_results": [
    {"filename": "cv1.pdf", "QuickScore": {"final_score": 78, "grade": "C"}},
    {"filename": "cv2.pdf", "QuickScore": {"final_score": 92, "grade": "A"}}
  ],
  "summary": {"total_processed": 2, "successful": 2, "failed": 0}
}
```

---

## 📊 Scoring Logic

| Metric               | Weight |
|-----------------------|--------|
| Keyword Match         | 25%    |
| Semantic Similarity   | 35%    |
| Skills Relevance      | 20%    |
| Experience Match      | 15%    |
| Format Quality        | 5%     |

---

## 🛠 Future Improvements
- Add **fine-grained section parsing** (education, projects, certifications).
- Extend **skills.yml** with more categories.
- Add **frontend dashboard** with visualization.
- Optimize for **batch processing** (async jobs, Redis queue).

---
