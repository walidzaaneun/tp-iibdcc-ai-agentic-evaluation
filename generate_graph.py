"""Génère une visualisation du graphe LangGraph en format Mermaid et PNG.

Lancer avec : python generate_graph.py
"""
from src.graph import graph


def main() -> None:
    print("=== Génération du graphe LangGraph ===")

    # Export Mermaid
    try:
        mermaid_code = graph.get_graph().draw_mermaid()
        with open("graph.mmd", "w", encoding="utf-8") as f:
            f.write(mermaid_code)
        print("✓ graph.mmd généré")
        print("\nCode Mermaid :")
        print("-" * 50)
        print(mermaid_code)
        print("-" * 50)
    except Exception as e:
        print(f"✗ Erreur Mermaid : {e}")

    # Export PNG (nécessite une connexion internet pour l'API Mermaid)
    try:
        png_data = graph.get_graph().draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(png_data)
        print("✓ graph.png généré")
    except Exception as e:
        print(f"✗ PNG non disponible (connexion requise) : {e}")
        print("  Le fichier graph.mmd peut être visualisé sur https://mermaid.live")


if __name__ == "__main__":
    main()
