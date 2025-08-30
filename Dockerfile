# Dockerfile for deploying a Flask app with with lean requirements-lean.txt
# This helps to run the app on small AWS EC2 instances (t2.micro, 1 CPU, 1 GB RAM)
# This image will not include heavy libraries like torch
# It uses Gunicorn with 2 workers to handle requests in production
FROM python:3.11-slim

LABEL org.opencontainers.image.title="AI Resume analyser"
LABEL org.opencontainers.image.description="Flask-based Resume Analyzer base on spaCy + OpenAI API"
LABEL org.opencontainers.image.authors="Kavyasri Ganoju"
LABEL org.opencontainers.image.source="https://github.com/kavyasriganoju/ai-resume-anayser.git"
LABEL org.opencontainers.image.version="1.0.0"

# Set working dir
WORKDIR /app

# Avoid writing .pyc files, keep logs unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker layer caching)
COPY requirements-lean.txt requirements.txt

# Install Python dependencies (lean set, no torch)
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model (small English model)
RUN python -m spacy download en_core_web_sm

# Copy only the application code (exclude venv, .git, logs via .dockerignore)
COPY . .

# Create logs directory
RUN mkdir -p /app/logs

# Add non-root user for security
RUN adduser --disabled-password --gecos '' appuser \
    && chown -R appuser:appuser /app
USER appuser

# Expose Flask/Gunicorn port
EXPOSE 5000

# Use Gunicorn for production
# Reduce workers so t2.micro (1 CPU, 1 GB RAM) doesn’t OOM
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "--max-requests", "500", "app:app"]
