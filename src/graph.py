"""Architecture du graphe LangGraph pour le RAG médical agentique.

Le graphe suit le pattern Agentic RAG :
    START → agent → tools → grade_documents → rewrite_query → agent → ...
                          ↘ agent (si outil non-RAG)

Nœuds :
    - agent           : Le LLM raisonne et décide d'appeler un outil ou de répondre.
    - tools           : Exécute les outils demandés par l'agent.
    - grade_documents : Évalue la pertinence des documents récupérés.
    - rewrite_query   : Reformule la question si les documents ne sont pas pertinents.

Mémoire : InMemorySaver (checkpointer LangGraph) indexé par thread_id.
"""
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from src.config import MAX_REWRITES
from src.llm import get_llm
from src.state import AgentState
from src.tools import TOOLS, TOOLS_BY_NAME

# ─── Prompt système ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "Tu es un assistant médical intelligent spécialisé dans la santé, les maladies, "
    "les symptômes, les médicaments et la prévention médicale. "
    "Utilise l'outil retrieve_documents pour rechercher dans la base documentaire médicale "
    "avant de répondre à toute question de connaissance médicale. "
    "Utilise calculate_bmi pour calculer l'Indice de Masse Corporelle quand un patient "
    "fournit son poids et sa taille. "
    "Utilise estimate_drug_dosage pour estimer une posologie en fonction du poids corporel. "
    "Réponds en français, de façon claire, précise et empathique. "
    "Rappelle toujours que tes réponses sont informatives et ne remplacent pas "
    "une consultation médicale professionnelle. Cite tes sources documentaires."
)

# ─── Initialisation du modèle avec outils ──────────────────────────────────────
model = get_llm()
model_with_tools = model.bind_tools(TOOLS)


# ─── Nœuds du graphe ───────────────────────────────────────────────────────────

def agent_node(state: AgentState) -> dict:
    """Nœud agent : le LLM analyse le contexte et décide d'appeler un outil ou de répondre.

    Args:
        state: État courant du graphe.

    Returns:
        Mise à jour du state avec la réponse du LLM et le compteur d'appels.
    """
    response = model_with_tools.invoke(
        [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    )
    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def tools_node(state: AgentState) -> dict:
    """Nœud outils : exécute tous les tool calls demandés par l'agent.

    Gère les erreurs d'exécution d'outil en renvoyant un message d'erreur descriptif
    pour que l'agent puisse adapter sa stratégie.

    Args:
        state: État courant avec le dernier message contenant les tool_calls.

    Returns:
        Mise à jour du state avec les résultats des outils.
    """
    last_message = state["messages"][-1]
    results = []
    tool_names = []
    retrieved_docs = state.get("retrieved_docs", [])

    for call in last_message.tool_calls:
        tool = TOOLS_BY_NAME[call["name"]]
        try:
            observation = tool.invoke(call["args"])
        except Exception as exc:
            observation = (
                f"Erreur lors de l'appel à l'outil '{call['name']}' : {exc}. "
                "Vérifiez les arguments fournis et réessayez, ou répondez sans cet outil."
            )

        results.append(
            ToolMessage(content=str(observation), tool_call_id=call["id"])
        )
        tool_names.append(call["name"])

        # Capture des documents récupérés pour le grading
        if call["name"] == "retrieve_documents":
            retrieved_docs = [str(observation)]

    return {
        "messages": results,
        "last_tool_names": tool_names,
        "retrieved_docs": retrieved_docs,
    }


def grade_documents_node(state: AgentState) -> dict:
    """Nœud de grading : évalue la pertinence des documents récupérés par rapport à la question.

    Le LLM agit comme juge de pertinence (oui/non) pour décider si une reformulation
    de la requête est nécessaire.

    Args:
        state: État courant avec les documents récupérés.

    Returns:
        Mise à jour du state avec le flag docs_relevant.
    """
    # Récupération de la dernière question utilisateur
    question = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )
    docs_text = "\n\n".join(state.get("retrieved_docs", []))

    grading_prompt = (
        f"Question médicale posée : {question}\n\n"
        f"Documents récupérés depuis la base médicale :\n{docs_text}\n\n"
        "Ces documents contiennent-ils des informations médicales utiles et pertinentes "
        "pour répondre à cette question ? Réponds uniquement par 'oui' ou 'non'."
    )

    response = model.invoke([HumanMessage(content=grading_prompt)])
    is_relevant = "oui" in response.content.strip().lower()

    return {"docs_relevant": is_relevant}


def rewrite_query_node(state: AgentState) -> dict:
    """Nœud de reformulation : reformule la question médicale pour améliorer la récupération.

    Activé uniquement si les documents récupérés ne sont pas pertinents et que
    le nombre maximum de reformulations (MAX_REWRITES) n'est pas atteint.

    Args:
        state: État courant avec l'historique des messages.

    Returns:
        Mise à jour du state avec la question reformulée et le compteur de reformulations.
    """
    question = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )

    rewrite_prompt = (
        f"La question médicale suivante n'a pas permis de trouver des documents pertinents "
        f"dans la base documentaire : '{question}'.\n"
        "Reformule-la en utilisant des termes médicaux différents, des synonymes cliniques "
        "ou une formulation plus générale, tout en préservant le sens original. "
        "Réponds uniquement avec la question reformulée, sans explication."
    )

    response = model.invoke([HumanMessage(content=rewrite_prompt)])
    reformulated = response.content.strip()

    return {
        "messages": [HumanMessage(content=reformulated)],
        "rewrite_count": state.get("rewrite_count", 0) + 1,
    }


# ─── Fonctions de routage conditionnel ─────────────────────────────────────────

def route_after_agent(state: AgentState) -> Literal["tools", "__end__"]:
    """Après l'agent : va vers les outils si des tool_calls existent, sinon termine."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


def route_after_tools(state: AgentState) -> Literal["grade_documents", "agent"]:
    """Après les outils : grade les documents si retrieve a été appelé, sinon retour à l'agent."""
    if "retrieve_documents" in state.get("last_tool_names", []):
        return "grade_documents"
    return "agent"


def route_after_grading(state: AgentState) -> Literal["agent", "rewrite_query"]:
    """Après le grading : retourne à l'agent si docs pertinents ou max rewrites atteint,
    sinon reformule la requête."""
    if state.get("docs_relevant", True) or state.get("rewrite_count", 0) >= MAX_REWRITES:
        return "agent"
    return "rewrite_query"


# ─── Construction du graphe ────────────────────────────────────────────────────

builder = StateGraph(AgentState)

# Ajout des nœuds
builder.add_node("agent", agent_node)
builder.add_node("tools", tools_node)
builder.add_node("grade_documents", grade_documents_node)
builder.add_node("rewrite_query", rewrite_query_node)

# Définition des arêtes
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route_after_agent, ["tools", END])
builder.add_conditional_edges("tools", route_after_tools, ["grade_documents", "agent"])
builder.add_conditional_edges(
    "grade_documents", route_after_grading, ["agent", "rewrite_query"]
)
builder.add_edge("rewrite_query", "agent")

# Compilation avec mémoire conversationnelle (checkpointer)
graph = builder.compile(checkpointer=InMemorySaver())
