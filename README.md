# Système RAG Médical Agentique — LangGraph

Système RAG agentique construit avec **LangGraph** (sans `create_agent`), spécialisé
dans le domaine médical : maladies, symptômes, médicaments, prévention et santé publique.

## Domaine

**Médical / Santé** — Base documentaire composée de guides et rapports publics de l'OMS
sur les maladies chroniques, la santé mentale, les maladies tropicales et les épidémies.

## Stack Technique

| Composant | Technologie |
|-----------|-------------|
| LLM | Ollama local `llama3.2:3b` |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vectorstore | Chroma (persisté dans `data/chroma_db/`) |
| Orchestration | LangGraph `StateGraph` |
| Mémoire | `InMemorySaver` (checkpointer par `thread_id`) |

## Architecture du Graphe

Le graphe suit le pattern **Agentic RAG** avec correction itérative :

```
START → agent → tools → grade_documents → rewrite_query → agent → ...
                      ↘ agent (outil non-RAG)           ↗
```

### Nœuds

- **`agent`** : Le LLM raisonne et décide d'appeler un outil ou de répondre directement.
- **`tools`** : Exécute les outils demandés (retrieval, calcul IMC, posologie).
- **`grade_documents`** : Évalue la pertinence des documents récupérés (juge LLM oui/non).
- **`rewrite_query`** : Reformule la question médicale si les documents ne sont pas pertinents.

### Routage Conditionnel

- `route_after_agent` : `tools` si tool_calls présents, sinon `END`.
- `route_after_tools` : `grade_documents` si retrieve utilisé, sinon `agent`.
- `route_after_grading` : `agent` si docs pertinents ou MAX_REWRITES atteint, sinon `rewrite_query`.

## Outils Médicaux

| Outil | Description |
|-------|-------------|
| `retrieve_documents(query)` | Recherche sémantique dans la base documentaire médicale |
| `calculate_bmi(weight_kg, height_cm)` | Calcule l'IMC et sa catégorie OMS |
| `estimate_drug_dosage(drug, weight, dose_per_kg, frequency)` | Estime la posologie par poids |

## Installation

```bash
# Installer uv si nécessaire
pip install uv

# Installer les dépendances
uv sync --group dev

# S'assurer qu'Ollama est lancé avec le modèle requis
ollama pull llama3.2:3b
```

## Utilisation

### 1. Télécharger les documents médicaux

```bash
python download_docs.py
```

### 2. Construire le vectorstore

```bash
python ingest.py
```

### 3. Lancer l'assistant

```bash
python main.py
```

### 4. Visualiser le graphe

```bash
python generate_graph.py
# Génère graph.mmd et graph.png
# Visualiser .mmd sur https://mermaid.live
```

### 5. Lancer l'évaluation

```bash
python -m evaluation.run_evaluation
# Résultats dans evaluation/results/results.csv
```

## Structure du Projet

```
agentic-rag-medical/
├── src/
│   ├── config.py          # Paramètres globaux (modèles, chemins, hyperparamètres)
│   ├── state.py           # AgentState (TypedDict avec réducteur add pour messages)
│   ├── llm.py             # Initialisation du modèle Ollama
│   ├── vectorstore.py     # Construction et chargement du vectorstore Chroma
│   ├── tools.py           # 3 outils : retrieval, IMC, posologie
│   └── graph.py           # Architecture LangGraph complète
├── evaluation/
│   ├── questions.py       # 10 questions simples + 10 questions complexes
│   ├── run_evaluation.py  # Exécution de l'évaluation et export CSV
│   └── results/           # Résultats CSV générés
├── data/
│   ├── pdfs/              # Documents médicaux sources (OMS, etc.)
│   └── chroma_db/         # Vectorstore persisté
├── download_docs.py       # Téléchargement des PDF médicaux publics
├── ingest.py              # Indexation et vectorisation
├── main.py                # Interface CLI interactive
├── generate_graph.py      # Export Mermaid / PNG du graphe
└── pyproject.toml         # Dépendances du projet
```

## Évaluation

Le système est testé sur **20 questions médicales** :

- **10 questions simples** : définitions médicales, symptômes, concepts généraux
- **10 questions complexes** : calculs (IMC, posologie), comparaisons cliniques, conseils multi-paramètres

Métriques collectées :
- Temps de réponse (secondes)
- Nombre d'appels LLM par question
- Nombre de reformulations de requête
- Pertinence des documents récupérés
- Extrait des sources citées

## Avertissement

> ⚠️ Ce système est **strictement informatif et éducatif**.  
> Il ne remplace en aucun cas une consultation médicale professionnelle.  
> Pour tout problème de santé, consultez un médecin qualifié.
