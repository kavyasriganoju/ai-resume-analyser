
from adapters.spacy_adapter import SpacyResumeAnalyzer
from adapters.bert_adapter import BertResumeAnalyzer
from adapters.gpt_adapter import GPTResumeAnalyzer

class AnalyzerFactory:
    @staticmethod
    def get_analyzers(api_key: str = None):
        analyzers = [SpacyResumeAnalyzer(), BertResumeAnalyzer()]
        if api_key:
            analyzers.append(GPTResumeAnalyzer(api_key))
        return analyzers
