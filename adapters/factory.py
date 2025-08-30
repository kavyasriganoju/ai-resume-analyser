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
from adapters.spacy_adapter import SpacyResumeAnalyzer
from adapters.bert_adapter import BertResumeAnalyzer
from adapters.gpt_adapter import GPTResumeAnalyzer
import os

logger = logging.getLogger(__name__)

class AnalyzerFactory:
    """Factory for creating and configuring resume analyzers."""
    
    @staticmethod
    @staticmethod
    def get_analyzers(api_key: str = None, config: dict = None):
        analyzers = []
        config = config or {}

        # Read from env or config
        enabled = os.getenv("ANALYZERS", "GPT").upper().split(",")
        logger.info(f"Enabled analyzers: {enabled}")

        if "SPACY" in enabled:
            try:
                spacy_analyzer = SpacyResumeAnalyzer()
                analyzers.append(spacy_analyzer)
                logger.info("SpaCy analyzer loaded")
            except Exception as e:
                logger.error(f"Failed to load SpaCy analyzer: {str(e)}")

        if "BERT" in enabled:
            try:
                use_gpu = config.get("use_gpu", False)
                bert_analyzer = BertResumeAnalyzer(use_gpu=use_gpu)
                analyzers.append(bert_analyzer)
                logger.info("BERT analyzer loaded")
            except Exception as e:
                logger.error(f"Failed to load BERT analyzer: {str(e)}")

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
    
    @staticmethod
    def get_lightweight_analyzers():
        """Get only lightweight analyzers for resource-constrained environments."""
        analyzers = []
        
        try:
            spacy_analyzer = SpacyResumeAnalyzer()
            analyzers.append(spacy_analyzer)
        except Exception as e:
            logger.error(f"Failed to load SpaCy analyzer: {str(e)}")
        
        return analyzers
    
    @staticmethod
    def get_production_analyzers(api_key: str = None):
        """Get production-optimized analyzer configuration."""
        config = {
            'use_gpu': False,  # Use CPU for better resource management
        }
        return AnalyzerFactory.get_analyzers(api_key, config)