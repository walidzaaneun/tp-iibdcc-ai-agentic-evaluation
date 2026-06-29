"""Outils médicaux disponibles pour l'agent RAG.

Trois outils sont définis :
    - retrieve_documents : recherche sémantique dans la base documentaire médicale.
    - calculate_bmi : calcule l'IMC (Indice de Masse Corporelle) et sa catégorie.
    - estimate_drug_dosage : estime une posologie indicative en fonction du poids.
"""
import math

from langchain_core.tools import tool

from src.config import CHROMA_DIR, RETRIEVAL_K
from src.vectorstore import load_vectorstore

# Cache du vectorstore pour éviter les rechargements répétés
_vectorstore = None


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = load_vectorstore()
    return _vectorstore


@tool
def retrieve_documents(query: str) -> str:
    """Recherche les documents médicaux les plus pertinents pour une requête donnée.

    Args:
        query: Question ou sujet médical à rechercher dans la base documentaire.

    Returns:
        Extraits des documents les plus pertinents avec leurs sources.
    """
    vs = _get_vectorstore()
    docs = vs.similarity_search(query, k=RETRIEVAL_K)
    if not docs:
        return "Aucun document pertinent trouvé pour cette requête."

    results = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Inconnu")
        page = doc.metadata.get("page", "?")
        results.append(f"[Document {i} | Source: {source} | Page: {page}]\n{doc.page_content}")

    return "\n\n---\n\n".join(results)


@tool
def calculate_bmi(weight_kg: float, height_cm: float) -> str:
    """Calcule l'Indice de Masse Corporelle (IMC) et fournit son interprétation clinique.

    Args:
        weight_kg: Poids du patient en kilogrammes.
        height_cm: Taille du patient en centimètres.

    Returns:
        Valeur de l'IMC, catégorie OMS et recommandations générales.
    """
    if weight_kg <= 0 or height_cm <= 0:
        return "Erreur : le poids et la taille doivent être des valeurs positives."

    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m ** 2)
    bmi_rounded = round(bmi, 1)

    # Classification OMS
    if bmi < 16.0:
        category = "Dénutrition sévère"
        advice = "Consultation médicale urgente recommandée."
    elif bmi < 17.0:
        category = "Dénutrition modérée"
        advice = "Un suivi nutritionnel est conseillé."
    elif bmi < 18.5:
        category = "Insuffisance pondérale"
        advice = "Un apport calorique accru peut être nécessaire. Consultez un professionnel."
    elif bmi < 25.0:
        category = "Corpulence normale"
        advice = "Continuez à maintenir une alimentation équilibrée et une activité physique régulière."
    elif bmi < 30.0:
        category = "Surpoids"
        advice = "Un rééquilibrage alimentaire et une activité physique régulière sont recommandés."
    elif bmi < 35.0:
        category = "Obésité modérée (classe I)"
        advice = "Consultez un médecin ou nutritionniste pour un programme adapté."
    elif bmi < 40.0:
        category = "Obésité sévère (classe II)"
        advice = "Un suivi médical est indispensable."
    else:
        category = "Obésité morbide (classe III)"
        advice = "Prise en charge médicale spécialisée recommandée en urgence."

    return (
        f"IMC calculé : {bmi_rounded} kg/m²\n"
        f"Catégorie OMS : {category}\n"
        f"Conseil général : {advice}\n"
        f"⚠️ Ces informations sont indicatives et ne remplacent pas un avis médical."
    )


@tool
def estimate_drug_dosage(
    drug_name: str,
    weight_kg: float,
    dose_per_kg: float,
    frequency_per_day: int,
) -> str:
    """Estime la posologie indicative d'un médicament basée sur le poids corporel.

    Args:
        drug_name: Nom du médicament (ex: paracétamol, ibuprofène).
        weight_kg: Poids du patient en kilogrammes.
        dose_per_kg: Dose recommandée en mg par kg de poids corporel.
        frequency_per_day: Nombre de prises par jour.

    Returns:
        Dose totale journalière, dose par prise et avertissements.
    """
    if weight_kg <= 0 or dose_per_kg <= 0 or frequency_per_day <= 0:
        return "Erreur : tous les paramètres doivent être des valeurs positives."

    daily_dose = weight_kg * dose_per_kg
    dose_per_intake = daily_dose / frequency_per_day

    return (
        f"Médicament : {drug_name}\n"
        f"Poids : {weight_kg} kg | Dose : {dose_per_kg} mg/kg | Fréquence : {frequency_per_day}x/jour\n"
        f"──────────────────────────────\n"
        f"Dose journalière totale estimée : {round(daily_dose, 1)} mg\n"
        f"Dose par prise : {round(dose_per_intake, 1)} mg\n"
        f"⚠️ AVERTISSEMENT : Cette estimation est purement indicative.\n"
        f"   Consultez impérativement un médecin ou pharmacien avant toute prise médicamenteuse."
    )


# Registre des outils disponibles
TOOLS = [retrieve_documents, calculate_bmi, estimate_drug_dosage]
TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}
