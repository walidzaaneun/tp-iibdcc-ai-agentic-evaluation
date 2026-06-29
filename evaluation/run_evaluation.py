"""Exécute l'évaluation complète du système RAG médical agentique.

Lance les 20 questions (10 simples + 10 complexes), mesure les performances
et enregistre les résultats dans evaluation/results/results.csv.

Lancer avec : python -m evaluation.run_evaluation
"""
import csv
import time
from pathlib import Path

from langchain_core.messages import HumanMessage

from evaluation.questions import COMPLEX_QUESTIONS, SIMPLE_QUESTIONS
from src.graph import graph

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_question(question_id: str, question_type: str, question: str) -> dict:
    """Exécute une question et collecte les métriques.

    Args:
        question_id: Identifiant de la question (ex: "S1", "C3").
        question_type: Type de question ("simple" ou "complexe").
        question: Texte de la question médicale.

    Returns:
        Dictionnaire avec la réponse, le temps, les sources et les métriques.
    """
    # Chaque question a son propre thread pour éviter les contaminations de mémoire
    config = {"configurable": {"thread_id": f"eval-{question_id}"}}

    start = time.perf_counter()
    result = graph.invoke(
        {"messages": [HumanMessage(content=question)]}, config
    )
    elapsed = time.perf_counter() - start

    answer = result["messages"][-1].content
    # Tronque les sources pour le CSV
    sources = " | ".join(result.get("retrieved_docs", []))[:600]

    return {
        "id": question_id,
        "type": question_type,
        "question": question,
        "answer": answer,
        "time_s": round(elapsed, 2),
        "sources_snippet": sources,
        "llm_calls": result.get("llm_calls", 0),
        "rewrite_count": result.get("rewrite_count", 0),
        "docs_relevant": result.get("docs_relevant", "N/A"),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("╔══════════════════════════════════════════════════════╗")
    print("║   Évaluation du Système RAG Médical Agentique        ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    rows = []

    # Questions simples
    print("── Questions simples (1/10 → 10/10) ──────────────────")
    for i, q in enumerate(SIMPLE_QUESTIONS, start=1):
        print(f"  [S{i}] {q[:70]}...")
        row = run_question(f"S{i}", "simple", q)
        rows.append(row)
        print(f"       ✓ Répondu en {row['time_s']}s | "
              f"LLM calls: {row['llm_calls']} | Rewrites: {row['rewrite_count']}")

    print()

    # Questions complexes
    print("── Questions complexes (1/10 → 10/10) ─────────────────")
    for i, q in enumerate(COMPLEX_QUESTIONS, start=1):
        print(f"  [C{i}] {q[:70]}...")
        row = run_question(f"C{i}", "complexe", q)
        rows.append(row)
        print(f"       ✓ Répondu en {row['time_s']}s | "
              f"LLM calls: {row['llm_calls']} | Rewrites: {row['rewrite_count']}")

    # Écriture du fichier CSV
    csv_path = RESULTS_DIR / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Statistiques globales
    print(f"\n{'═' * 55}")
    simple_times = [r["time_s"] for r in rows if r["type"] == "simple"]
    complex_times = [r["time_s"] for r in rows if r["type"] == "complexe"]
    print(f"Temps moyen - Questions simples  : {sum(simple_times)/len(simple_times):.2f}s")
    print(f"Temps moyen - Questions complexes: {sum(complex_times)/len(complex_times):.2f}s")
    total_rewrites = sum(r["rewrite_count"] for r in rows)
    print(f"Total reformulations effectuées  : {total_rewrites}")
    print(f"{'═' * 55}")
    print(f"\n✓ Résultats enregistrés dans : {csv_path}\n")


if __name__ == "__main__":
    main()
