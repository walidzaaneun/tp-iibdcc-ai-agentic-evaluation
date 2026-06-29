"""Construction et chargement du vectorstore Chroma pour la base médicale."""
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    PDF_DIR,
)


def get_embeddings() -> HuggingFaceEmbeddings:
    """Retourne le modèle d'embeddings HuggingFace."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def build_vectorstore() -> Chroma:
    """Charge les PDF médicaux, les découpe et construit le vectorstore Chroma.

    Returns:
        Instance Chroma persistée dans CHROMA_DIR.
    """
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"Aucun PDF trouvé dans {PDF_DIR}. Lancez d'abord : python download_docs.py")

    # Chargement et découpage des documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    all_chunks = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        chunks = splitter.split_documents(pages)
        all_chunks.extend(chunks)
        print(f"  {pdf_path.name} : {len(pages)} pages → {len(chunks)} chunks")

    print(f"\nTotal : {len(all_chunks)} chunks à indexer...")

    # Construction du vectorstore
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    return vectorstore


def load_vectorstore() -> Chroma:
    """Charge un vectorstore Chroma déjà persisté."""
    embeddings = get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
