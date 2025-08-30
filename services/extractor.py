# extractor.py
import fitz  # PyMuPDF
import docx
import re
import unicodedata

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF using PyMuPDF."""
    text_chunks = []
    doc = fitz.open(file_path)
    for page in doc:
        text_chunks.append(page.get_text("text"))
    doc.close()
    raw_text = "\n".join(text_chunks)
    return clean_text(raw_text)

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from Word .docx files."""
    doc = docx.Document(file_path)
    raw_text = "\n".join([para.text for para in doc.paragraphs])
    return clean_text(raw_text)

def extract_text_from_txt(file_path: str) -> str:
    """Extract text from plain .txt files."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()
    return clean_text(raw_text)

def clean_text(text: str) -> str:
    """Clean resume text for NLP pipeline."""
    # Normalize unicode (e.g., smart quotes → ascii)
    text = unicodedata.normalize("NFKD", text)

    # Remove common headers/footers (detected as repeated lines)
    lines = text.splitlines()
    line_freq = {}
    for line in lines:
        line_freq[line.strip()] = line_freq.get(line.strip(), 0) + 1
    
    # Drop lines repeated on many pages (like headers/footers)
    cleaned_lines = [
        l for l in lines 
        if line_freq.get(l.strip(), 0) < 3 and l.strip() != ""
    ]

    text = "\n".join(cleaned_lines)

    # Remove page numbers (standalone digits)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # Normalize bullets/dashes
    text = re.sub(r'[•▪●◦]', '-', text)

    # Collapse multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def extract_text(file_path: str) -> str:
    """Main entry point: detect file type and extract text."""
    if file_path.lower().endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.lower().endswith(".docx"):
        return extract_text_from_docx(file_path)
    elif file_path.lower().endswith(".txt"):
        return extract_text_from_txt(file_path)
    else:
        raise ValueError("Unsupported file format. Use PDF, DOCX, or TXT.")
