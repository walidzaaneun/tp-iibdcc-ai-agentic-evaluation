"""Télécharge les ressources documentaires médicales publiques dans data/pdfs/.

Sources utilisées (domaine public / open access) :
    - OMS : guides sur les maladies non transmissibles et la santé mentale
    - INSERM : fiches d'information médicale
    - Santé Publique France : bulletins épidémiologiques

Lancer avec : python download_docs.py
"""
import urllib.request
from pathlib import Path

from src.config import PDF_DIR

# Liste des documents médicaux publics à télécharger
# Format : (URL, nom_fichier_local)
MEDICAL_DOCUMENTS = [
    (
        "https://iris.who.int/bitstream/handle/10665/43894/9789242563931_fre.pdf",
        "OMS_prevention_maladies_chroniques.pdf",
    ),
    (
        "https://iris.who.int/bitstream/handle/10665/274603/WHO-MSD-MER-2017.2-fre.pdf",
        "OMS_sante_mentale_monde.pdf",
    ),
    (
        "https://iris.who.int/bitstream/handle/10665/254610/WHO-HTM-NTD-NZD-2017.04-fre.pdf",
        "OMS_maladies_tropicales_negligees.pdf",
    ),
    (
        "https://www.who.int/docs/default-source/wpro---documents/emergency/surveillance/"
        "covid-19/covid-19-sitrep-2022-08-03.pdf",
        "OMS_covid19_rapport.pdf",
    ),
]


def download_file(url: str, dest_path: Path) -> bool:
    """Télécharge un fichier depuis une URL vers un chemin local.

    Args:
        url: URL du fichier à télécharger.
        dest_path: Chemin de destination local.

    Returns:
        True si le téléchargement a réussi, False sinon.
    """
    try:
        print(f"Téléchargement : {dest_path.name}...")
        urllib.request.urlretrieve(url, dest_path)
        size_kb = dest_path.stat().st_size / 1024
        print(f"  ✓ {dest_path.name} ({size_kb:.0f} KB)")
        return True
    except Exception as e:
        print(f"  ✗ Échec pour {dest_path.name} : {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def main() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Dossier de destination : {PDF_DIR}\n")

    success_count = 0
    for url, filename in MEDICAL_DOCUMENTS:
        dest = PDF_DIR / filename
        if dest.exists():
            print(f"  (déjà présent) {filename}")
            success_count += 1
            continue
        if download_file(url, dest):
            success_count += 1

    print(f"\n{success_count}/{len(MEDICAL_DOCUMENTS)} documents téléchargés.")
    print("Lancez maintenant : python ingest.py")


if __name__ == "__main__":
    main()
