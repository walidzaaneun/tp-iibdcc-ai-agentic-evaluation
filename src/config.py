"""Configuration centrale du projet RAG médical agentique."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
CHROMA_DIR = DATA_DIR / "chroma_db"

# Modèle de langage local (Ollama)
LLM_MODEL_NAME = "llama3.2:3b"

# Modèle d'embeddings HuggingFace
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Nom de la collection Chroma
COLLECTION_NAME = "medical_knowledge"

# Paramètres de découpage des documents
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Nombre de documents récupérés par requête
RETRIEVAL_K = 4

# Nombre maximum de reformulations de requête
MAX_REWRITES = 2
