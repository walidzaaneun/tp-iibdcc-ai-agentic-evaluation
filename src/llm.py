"""Initialisation du modèle de langage Ollama."""
from langchain_ollama import ChatOllama

from src.config import LLM_MODEL_NAME


def get_llm() -> ChatOllama:
    """Retourne une instance du LLM Ollama configuré pour le raisonnement médical.

    Température = 0 pour des réponses déterministes et fiables dans un contexte médical.
    """
    return ChatOllama(model=LLM_MODEL_NAME, temperature=0)
