import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from config import PRIMARY_MODEL, FALLBACK_MODEL, UTILITY_MODEL, GEMINI_API_KEY

logger = logging.getLogger(__name__)

class ModelFactory:
    @staticmethod
    def _create_model(model_name: str, streaming: bool):
        if "gemini" in model_name.lower():
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=GEMINI_API_KEY,
                streaming=streaming,
                temperature=0.0,
                max_retries=1
            )
        return ChatOllama(
            model=model_name,
            streaming=streaming,
            temperature=0.0
        )

    @staticmethod
    def get_primary(streaming=True):
        return ModelFactory._create_model(PRIMARY_MODEL, streaming)
        
    @staticmethod
    def get_fallback(streaming=True):
        return ModelFactory._create_model(FALLBACK_MODEL, streaming)
    
    @staticmethod
    def get_utility(streaming=False):
        return ModelFactory._create_model(UTILITY_MODEL, streaming)