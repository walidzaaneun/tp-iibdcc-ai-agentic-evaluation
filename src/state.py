"""Définition du state partagé dans le graphe LangGraph."""
from operator import add

from langchain_core.messages import AnyMessage
from typing_extensions import Annotated, TypedDict


class AgentState(TypedDict):
    """State persisté à chaque nœud du graphe.

    Attributes:
        messages: Historique complet de la conversation (accumulatif via `add`).
        llm_calls: Compteur du nombre d'appels LLM effectués dans la session.
        rewrite_count: Nombre de reformulations de requête effectuées.
        retrieved_docs: Derniers documents récupérés par l'outil de recherche.
        last_tool_names: Noms des derniers outils appelés par l'agent.
        docs_relevant: Indicateur de pertinence des documents après grading.
        patient_context: Contexte médical éventuel du patient (âge, symptômes).
    """
    messages: Annotated[list[AnyMessage], add]
    llm_calls: int
    rewrite_count: int
    retrieved_docs: list[str]
    last_tool_names: list[str]
    docs_relevant: bool
    patient_context: str
