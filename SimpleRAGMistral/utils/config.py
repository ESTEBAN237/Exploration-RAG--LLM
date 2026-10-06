import os
from pathlib import Path
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# --- Chemins Principaux (Pathlib) ---
BASE_DIR = Path(__file__).resolve().parent.parent

# --- Clés API ---
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    print(" Attention: La clé API Mistral (MISTRAL_API_KEY) n'est pas définie dans le fichier .env")

# --- Modèles Mistral AI ---
EMBEDDING_MODEL = "mistral-embed"
MODEL_NAME = "mistral-small-latest"  # Modèle recommandé pour la génération RAG rapide et précise

# --- Configuration de l'Indexation & Vector Store ---
INPUT_DATA_URL = os.getenv("INPUT_DATA_URL", None)
INPUT_DIR = str(BASE_DIR / "inputs")
VECTOR_DB_DIR = str(BASE_DIR / "vector_db")

FAISS_INDEX_FILE = str(Path(VECTOR_DB_DIR) / "faiss_index.idx")
DOCUMENT_CHUNKS_FILE = str(Path(VECTOR_DB_DIR) / "document_chunks.pkl")

# Paramètres du Chunking
CHUNK_SIZE = 1500           # Taille des chunks en caractères (~300-400 tokens)
CHUNK_OVERLAP = 150         # Chevauchement en caractères pour conserver la continuité du contexte
EMBEDDING_BATCH_SIZE = 32   # Taille des lots pour l'API d'embedding Mistral

# --- Configuration de la Recherche (Retrieval) ---
SEARCH_K = 5                # Nombre de passages/chunks pertinents à extraire pour le contexte RAG

# --- Configuration de la Base de Données (Historique / Tracking) ---
DATABASE_DIR = str(BASE_DIR / "database")
DATABASE_FILE = str(Path(DATABASE_DIR) / "interactions.db")
DATABASE_URL = f"sqlite:///{DATABASE_FILE}"

# --- Configuration de l'Application Streamlit ---
APP_TITLE = "Assistant RAG — Knowledge Base"
PROJECT_NAME = "Système RAG - Base de Connaissances"