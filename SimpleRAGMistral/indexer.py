import argparse
import logging
from typing import Optional

from utils.config import INPUT_DATA_URL, INPUT_DIR
from utils.data_loader import download_and_extract_zip, load_and_parse_files
from utils.vector_store import VectorStoreManager

# --- Configuration du Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
)


def run_indexing(input_directory: str, data_url: Optional[str] = None):
    """Exécute le pipeline complet d'indexation vectorielle (FAISS)."""
    logging.info("=== Démarrage du pipeline d'indexation RAG ===")

    # Étape 1 : Téléchargement & extraction des données (si URL spécifiée)
    if data_url:
        logging.info(f"Téléchargement de l'archive source depuis : {data_url}")
        success = download_and_extract_zip(data_url, input_directory)
        if not success:
            logging.error("Échec du téléchargement ou de l'extraction des données. Interruption du processus.")
            return
    else:
        logging.info(f"Utilisation des fichiers locaux du répertoire : {input_directory}")

    # Étape 2 : Chargement et parsing des documents
    logging.info(f"Parsing des documents sources dans : {input_directory}")
    documents = load_and_parse_files(input_directory)

    if not documents:
        logging.warning("Aucun document n'a été extrait. Vérifiez la présence de fichiers valides dans le dossier.")
        logging.info("=== Fin de l'indexation (aucun document traité) ===")
        return

    # Étape 3 : Découpage, génération des embeddings et construction de l'index FAISS
    logging.info("Initialisation du VectorStoreManager...")
    vector_store = VectorStoreManager()

    logging.info("Génération des embeddings et construction de l'index FAISS...")
    vector_store.build_index(documents)

    # Étape 4 : Bilan de l'indexation
    logging.info("=== Indexation terminée avec succès ===")
    logging.info(f"Documents sources traités : {len(documents)}")
    if vector_store.index:
        logging.info(f"Total de chunks vectorisés dans FAISS : {vector_store.index.ntotal}")
    else:
        logging.warning("L'index FAISS n'a pas pu être instancié ou est vide.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script d'indexation vectorielle pour le système RAG")
    
    parser.add_argument(
        "--input-dir",
        type=str,
        default=INPUT_DIR,
        help=f"Répertoire des documents sources (par défaut : {INPUT_DIR})",
    )
    
    parser.add_argument(
        "--data-url",
        type=str,
        default=INPUT_DATA_URL if 'INPUT_DATA_URL' in globals() else None,
        help="URL optionnelle pour télécharger et extraire un fichier d'archives inputs.zip",
    )

    args = parser.parse_args()

    run_indexing(input_directory=args.input_dir, data_url=args.data_url)