"""
Project: Resume Analyser Critic (GPT-powered Resume Analyzer)
File: factory.py
Author: Kavyasri Ganoju
Email: kavyasriganoju@gmail.com
Created: 2025-08-30
Description:
    Implements GPT-based resume analyzer using OpenAI API.
    Provides scoring, strengths/weaknesses, and ATS optimization tips.

"""
import logging
from adapters.gpt_adapter import GPTResumeAnalyzer
import os

logger = logging.getLogger(__name__)

class AnalyzerFactory:
    """Factory for creating and configuring resume analyzers."""
    

    @staticmethod
    def get_analyzers(api_key: str = None, config: dict = None):
        analyzers = []
        config = config or {}

        # Read from env or config
        enabled = os.getenv("ANALYZERS", "GPT").upper().split(",")
        if not enabled:
            raise RuntimeError("ANALYZERS not set. Please configure it in your environment. GPT") 
        
        logger.info(f"Enabled analyzers: {enabled}")

        if "GPT" in enabled and api_key:
            try:
                gpt_analyzer = GPTResumeAnalyzer(api_key)
                analyzers.append(gpt_analyzer)
                logger.info("GPT analyzer loaded")
            except Exception as e:
                logger.error(f"Failed to load GPT analyzer: {str(e)}")

        if not analyzers:
            raise RuntimeError("No analyzers could be loaded")

        return analyzers
    