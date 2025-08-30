"""
Project: Resume Analyser Critic (GPT-powered Resume Analyzer)
File: app.py
Author: Kavyasri Ganoju
Email: kavyasriganoju@gmail.com
Created: 2025-08-30
Description:
    Implements GPT-based resume analyzer using OpenAI API.
    Provides scoring, strengths/weaknesses, and ATS optimization tips.

"""

from flask import Flask, render_template, request, jsonify
from services.extractor import extract_text
from adapters.factory import AnalyzerFactory
from services.scorer import ResumeScorer
import tempfile
import os
import logging
from functools import wraps
import time
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize components
api_key = os.getenv("OPENAI_API_KEY", "")  # Load from environment
analyzers = AnalyzerFactory.get_analyzers(api_key)
scorer = ResumeScorer()

def timing_decorator(f):
    """Decorator to measure execution time."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        logger.info(f"{f.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper

def validate_file(file):
    """Validate uploaded file."""
    if not file or not file.filename:
        return False, "No file selected"
    
    allowed_extensions = {'.pdf', '.docx', '.txt'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        return False, f"Unsupported file type: {file_ext}. Please use PDF, DOCX, or TXT files."
    
    return True, "Valid"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health_check():
    """Health check endpoint for production monitoring."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "analyzers_loaded": len(analyzers)
    })

@app.route("/analyze", methods=["POST"])
@timing_decorator
def analyze():
    try:
        # Get inputs
        resume_file = request.files.get("resume")
        jd_file = request.files.get("jd_file")
        jd_text = (request.form.get("job_desc") or "").strip()
        
        # Validate resume file
        is_valid, message = validate_file(resume_file)
        if not is_valid:
            return jsonify({"error": message}), 400
        
        # Extract resume text
        resume_text = extract_resume_text(resume_file)
        if not resume_text:
            return jsonify({"error": "Could not extract text from resume"}), 400
        
        # Extract JD text
        jd_text = extract_jd_text(jd_file, jd_text)
        if not jd_text:
            return jsonify({"error": "Job description is required"}), 400
        
        # Run analysis
        analysis_results = run_comprehensive_analysis(resume_text, jd_text)
        
        # Add metadata
        analysis_results["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "resume_length": len(resume_text),
            "jd_length": len(jd_text),
            "analyzers_used": [analyzer.__class__.__name__ for analyzer in analyzers]
        }
        
        logger.info(f"Analysis completed successfully")
        return jsonify(analysis_results)
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return jsonify({"error": "Internal server error occurred"}), 500

def extract_resume_text(resume_file):
    """Extract text from resume file with error handling."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(resume_file.filename)[1]) as tmp:
            resume_file.save(tmp.name)
            text = extract_text(tmp.name)
            os.unlink(tmp.name)
            return text
    except Exception as e:
        logger.error(f"Failed to extract resume text: {str(e)}")
        return None

def extract_jd_text(jd_file, jd_text):
    """Extract JD text from file or form input."""
    if jd_text:
        return jd_text
    
    if jd_file:
        try:
            is_valid, message = validate_file(jd_file)
            if not is_valid:
                logger.warning(f"Invalid JD file: {message}")
                return None
                
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(jd_file.filename)[1]) as tmp:
                jd_file.save(tmp.name)
                text = extract_text(tmp.name)
                os.unlink(tmp.name)
                return text
        except Exception as e:
            logger.error(f"Failed to extract JD text: {str(e)}")
            return None
    
    return None

@timing_decorator
def run_comprehensive_analysis(resume_text, jd_text):
    """Run analysis with whichever analyzers are enabled (GPT produces its own score)."""
    results = {}

    for analyzer in analyzers:
        try:
            analyzer_name = analyzer.__class__.__name__
            logger.info(f"Running {analyzer_name}")
            analysis = analyzer.analyze(resume_text, jd_text)
            results[analyzer_name] = analysis
        except Exception as e:
            logger.error(f"Analyzer {analyzer_name} failed: {str(e)}")
            results[analyzer_name] = {"error": str(e)}

    # If GPT analyzer returned a score, treat it as ComprehensiveScore
    gpt_data = results.get("GPTResumeAnalyzer")
    if gpt_data and gpt_data.get("final_score") is not None:
        results["ComprehensiveScore"] = {
            "final_score": gpt_data["final_score"],
            "grade": gpt_data["grade"],
            "breakdown": {"gpt_fit": gpt_data["final_score"]},
            "source": "GPT"
        }
    else:
        logger.info("No GPT score available")

    # Always add a summary
    results["Summary"] = generate_summary_insights(results)

    return results

def generate_summary_insights(results):
    """Generate high-level summary insights."""
    insights = {
        "key_strengths": [],
        "priority_improvements": [],
        "quick_wins": [],
        "overall_assessment": ""
    }

    comp_score = results.get("ComprehensiveScore", {})
    final_score = comp_score.get("final_score")
    breakdown = comp_score.get("breakdown", {})

    # If no numeric score (GPT-only mode), fallback to GPT analysis text
    if final_score is None:
        if "GPTResumeAnalyzer" in results:
            insights["overall_assessment"] = "GPT analysis available — see details in GPT output"
        else:
            insights["overall_assessment"] = "No scoring available"
        return insights

    # Else, use scoring logic (SpaCy + BERT present)
    if final_score >= 85:
        insights["overall_assessment"] = "Excellent match! Your resume is well-optimized for this role."
    elif final_score >= 75:
        insights["overall_assessment"] = "Good match with room for improvement."
    elif final_score >= 65:
        insights["overall_assessment"] = "Moderate match - several areas need attention."
    else:
        insights["overall_assessment"] = "Significant improvements needed to match this role."

    for metric, score in breakdown.items():
        if score > 80:
            insights["key_strengths"].append(f"Strong {metric.replace('_', ' ')}")
        elif score < 60:
            insights["priority_improvements"].append(f"Improve {metric.replace('_', ' ')}")
        else:
            insights["quick_wins"].append(f"Enhance {metric.replace('_', ' ')}")

    return insights


@app.route("/analyze-batch", methods=["POST"])
@timing_decorator
def analyze_batch():
    """Batch analysis endpoint for multiple resumes."""
    try:
        files = request.files.getlist("resumes")
        jd_text = request.form.get("job_desc", "").strip()
        
        if not files or len(files) == 0:
            return jsonify({"error": "No resume files provided"}), 400
        
        if not jd_text:
            return jsonify({"error": "Job description is required"}), 400
        
        if len(files) > 10:  # Limit batch size
            return jsonify({"error": "Maximum 10 resumes allowed per batch"}), 400
        
        results = []
        for i, file in enumerate(files):
            try:
                # Validate and extract
                is_valid, message = validate_file(file)
                if not is_valid:
                    results.append({
                        "filename": file.filename,
                        "error": message
                    })
                    continue
                
                resume_text = extract_resume_text(file)
                if not resume_text:
                    results.append({
                        "filename": file.filename,
                        "error": "Could not extract text"
                    })
                    continue
                
                # Quick analysis (skip GPT for batch to save costs)
                batch_result = run_quick_analysis(resume_text, jd_text)
                batch_result["filename"] = file.filename
                results.append(batch_result)
                
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "error": str(e)
                })
        
        return jsonify({
            "batch_results": results,
            "summary": {
                "total_processed": len(results),
                "successful": len([r for r in results if "error" not in r]),
                "failed": len([r for r in results if "error" in r])
            }
        })
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {str(e)}")
        return jsonify({"error": "Batch analysis failed"}), 500

def run_quick_analysis(resume_text, jd_text):
    """Quick analysis using whatever analyzers are available (skip GPT if you want to save costs)."""
    results = {}

    for analyzer in analyzers:
        name = analyzer.__class__.__name__
        # Example: skip GPT in batch to save tokens, or include if ANALYZERS=GPT only
        if name == "GPTResumeAnalyzer":
            continue
        try:
            results[name] = analyzer.analyze(resume_text, jd_text)
        except Exception as e:
            results[name] = {"error": str(e)}

    return results


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({"error": "File too large. Maximum size is 16MB."}), 413

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    # Production configuration
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV") == "development"
    
    app.run(host="0.0.0.0", port=port, debug=debug)