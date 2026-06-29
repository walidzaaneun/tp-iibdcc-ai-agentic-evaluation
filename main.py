"""Assistant médical Agentic RAG — Interface interactive en ligne de commande.

Lancer avec : python main.py
"""
from langchain_core.messages import HumanMessage

from src.config import CHROMA_DIR
from src.graph import graph


def main() -> None:
    if not CHROMA_DIR.exists():
        print("⚠️  Vectorstore introuvable.")
        print("   Étape 1 : python download_docs.py")
        print("   Étape 2 : python ingest.py")
        return

    config = {"configurable": {"thread_id": "session-cli"}}

    print("╔══════════════════════════════════════════════════════╗")
    print("║        Assistant Médical Agentique RAG               ║")
    print("║  Spécialisé : maladies, symptômes, médicaments       ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  ⚠️  Cet assistant est informatif uniquement.         ║")
    print("║     Consultez toujours un professionnel de santé.    ║")
    print("╚══════════════════════════════════════════════════════╝")
    print("\nTapez 'exit' pour quitter.\n")

    while True:
        try:
            user_input = input("Patient : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir !")
            break

        if user_input.lower() in {"exit", "quit", "quitter"}:
            print("Au revoir !")
            break
        if not user_input:
            continue

        print("Assistant : [réflexion en cours...]\n")
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_input)]}, config
        )
        print(f"Assistant : {result['messages'][-1].content}\n")
        print(f"  [Appels LLM : {result.get('llm_calls', 0)} | "
              f"Reformulations : {result.get('rewrite_count', 0)}]\n")


if __name__ == "__main__":
    main()
