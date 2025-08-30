from flask import Flask, render_template, request, jsonify
from services.extractor import extract_text
from adapters.factory import AnalyzerFactory
from services.scorer import compute_keyword_score, blend_scores
import tempfile
import os

app = Flask(__name__)
api_key = ""  # Load from env in production
analyzers = AnalyzerFactory.get_analyzers(api_key)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    resume_file = request.files.get("resume")
    jd_file = request.files.get("jd_file")
    jd_text = (request.form.get("job_desc") or "").strip()  # pasted in textarea
    print("Form keys:", request.form.keys())

    if not resume_file:
        return jsonify({"error": "Resume file is required"}), 400

    # --- Extract Resume Text ---
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(resume_file.filename)[1]) as tmp:
        resume_file.save(tmp.name)
        resume_text = extract_text(tmp.name)
        os.unlink(tmp.name)  # cleanup

    # --- Extract JD Text ---
    if not jd_text and jd_file:  # only fallback to file if textarea is empty
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(jd_file.filename)[1]) as tmp:
            jd_file.save(tmp.name)
            jd_text = extract_text(tmp.name)
            os.unlink(tmp.name)

    if not jd_text:
        return jsonify({"error": "Job description is required"}), 400

    # --- Run Analyzers ---
    results = {}
    spacy_data, bert_data = None, None

    for analyzer in analyzers:
        analysis = analyzer.analyze(resume_text, jd_text)
        results[analyzer.__class__.__name__] = analysis

        if analysis.get("type") == "spacy":
            spacy_data = analysis
        elif analysis.get("type") == "bert":
            bert_data = analysis

    # --- Final ATS Score ---
    if spacy_data and bert_data:
        keyword_score = compute_keyword_score(spacy_data["entities"]["SKILLS"], jd_text)
        semantic_score = bert_data["ats_score"]
        final_score = blend_scores(keyword_score, semantic_score)

        results["FinalATS"] = {
            "keyword_score": keyword_score,
            "semantic_score": semantic_score,
            "final_score": final_score,
        }

    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)
