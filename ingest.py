"""Construit et persiste le vectorstore Chroma à partir des PDF médicaux de data/pdfs/.

À lancer une seule fois (ou après ajout/modification de documents) :
    python ingest.py
"""
from src.vectorstore import build_vectorstore


def main() -> None:
    print("=== Construction du vectorstore médical ===\n")
    vectorstore = build_vectorstore()
    count = vectorstore._collection.count()
    print(f"\n✓ Vectorstore construit et persisté dans data/chroma_db/")
    print(f"  {count} chunks indexés et prêts pour la recherche sémantique.")


if __name__ == "__main__":
    main()
