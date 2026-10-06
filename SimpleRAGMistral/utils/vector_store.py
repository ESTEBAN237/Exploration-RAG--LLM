import logging
import os
import pickle
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from mistralai.client import MistralClient
from mistralai.exceptions import MistralAPIException

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENT_CHUNKS_FILE,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    FAISS_INDEX_FILE,
    MISTRAL_API_KEY,
)

# --- Configuration du Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
)


class VectorStoreManager:
    """Gère le chunking, la génération d'embeddings, l'indexation FAISS et la recherche RAG."""

    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.document_chunks: List[Dict[str, Any]] = []
        self.mistral_client = MistralClient(api_key=MISTRAL_API_KEY) if MISTRAL_API_KEY else None
        self._load_index_and_chunks()

    def _load_index_and_chunks(self):
        """Charge l'index FAISS et les métadonnées des chunks s'ils existent sur le disque."""
        if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(DOCUMENT_CHUNKS_FILE):
            try:
                logging.info(f"Chargement de l'index FAISS depuis : {FAISS_INDEX_FILE}")
                self.index = faiss.read_index(FAISS_INDEX_FILE)

                logging.info(f"Chargement des chunks depuis : {DOCUMENT_CHUNKS_FILE}")
                with open(DOCUMENT_CHUNKS_FILE, "rb") as f:
                    self.document_chunks = pickle.load(f)

                logging.info(
                    f"Vector Store initialisé avec succès : {self.index.ntotal} vector(s) et {len(self.document_chunks)} chunk(s)."
                )
            except Exception as e:
                logging.error(f"Erreur lors de la lecture des fichiers vectoriels : {e}")
                self.index = None
                self.document_chunks = []
        else:
            logging.warning("Fichiers FAISS ou chunks non trouvés. Base vectorielle vide.")

    def _split_documents_to_chunks(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Découpe les documents textuels en sous-sections (chunks) avec chevauchement."""
        logging.info(
            f"Découpage de {len(documents)} document(s) (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})..."
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True,
        )

        all_chunks = []
        for doc_counter, doc in enumerate(documents):
            langchain_doc = Document(page_content=doc["page_content"], metadata=doc["metadata"])
            chunks = text_splitter.split_documents([langchain_doc])

            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "id": f"{doc_counter}_{i}",
                    "text": chunk.page_content,
                    "metadata": {
                        **chunk.metadata,
                        "chunk_id_in_doc": i,
                        "start_index": chunk.metadata.get("start_index", -1),
                    },
                })

        logging.info(f"Découpage terminé : {len(all_chunks)} chunks créés au total.")
        return all_chunks

    def _generate_embeddings(self, chunks: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Génère les vecteurs d'embeddings pour une liste de chunks via l'API Mistral."""
        if not self.mistral_client:
            logging.error("MISTRAL_API_KEY non configurée. Génération d'embeddings impossible.")
            return None

        if not chunks:
            logging.warning("Aucun chunk soumis pour l'embedding.")
            return None

        logging.info(f"Génération des embeddings via Mistral AI ({EMBEDDING_MODEL})...")
        all_embeddings = []
        total_batches = (len(chunks) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

        for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch_num = (i // EMBEDDING_BATCH_SIZE) + 1
            batch_chunks = chunks[i : i + EMBEDDING_BATCH_SIZE]
            texts_to_embed = [chunk["text"] for chunk in batch_chunks]

            logging.info(f"Traitement du lot {batch_num}/{total_batches} ({len(texts_to_embed)} chunks)...")

            try:
                response = self.mistral_client.embeddings(
                    model=EMBEDDING_MODEL,
                    input=texts_to_embed,
                )
                batch_vectors = [data.embedding for data in response.data]
                all_embeddings.extend(batch_vectors)

            except MistralAPIException as e:
                logging.error(f"Erreur API Mistral (Lot {batch_num}) [Code {e.status_code}] : {e.message}")
                return None
            except Exception as e:
                logging.error(f"Erreur inattendue pendant l'embedding (Lot {batch_num}) : {e}")
                return None

        if not all_embeddings:
            return None

        embeddings_array = np.array(all_embeddings, dtype="float32")
        logging.info(f"Matrice d'embeddings générée : {embeddings_array.shape}")
        return embeddings_array

    def build_index(self, documents: List[Dict[str, Any]]):
        """Pipeline complet : Splitting -> Embeddings -> Normalisation -> Sauvegarde FAISS."""
        if not documents:
            logging.warning("Aucun document fourni à l'indexeur.")
            return

        # 1. Découpage
        self.document_chunks = self._split_documents_to_chunks(documents)
        if not self.document_chunks:
            logging.error("Aucun chunk produit lors du splitting.")
            return

        # 2. Génération des embeddings
        embeddings = self._generate_embeddings(self.document_chunks)
        if embeddings is None or embeddings.shape[0] != len(self.document_chunks):
            logging.error("Échec de la génération des embeddings. Nettoyage de l'état.")
            self.document_chunks = []
            self.index = None
            return

        # 3. Création de l'index FAISS (Produit Scalaire + Normalisation L2 = Similarité Cosinus)
        dimension = embeddings.shape[1]
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        logging.info(f"Index FAISS créé avec {self.index.ntotal} vecteurs (Dim: {dimension}).")

        # 4. Persistence
        self._save_index_and_chunks()

    def _save_index_and_chunks(self):
        """Sauvegarde l'index FAISS et les chunks sous forme de fichier Pickle."""
        if self.index is None or not self.document_chunks:
            logging.warning("Aucun index à sauvegarder.")
            return

        os.makedirs(os.path.dirname(FAISS_INDEX_FILE), exist_ok=True)
        os.makedirs(os.path.dirname(DOCUMENT_CHUNKS_FILE), exist_ok=True)

        try:
            logging.info(f"Écriture de l'index FAISS : {FAISS_INDEX_FILE}")
            faiss.write_index(self.index, FAISS_INDEX_FILE)

            logging.info(f"Écriture des métadonnées chunks : {DOCUMENT_CHUNKS_FILE}")
            with open(DOCUMENT_CHUNKS_FILE, "wb") as f:
                pickle.dump(self.document_chunks, f)

            logging.info("Sauvegarde de la base vectorielle réussie.")
            
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde sur disque : {e}")

    def search(self, query_text: str, k: int = 5, min_score: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Recherche les k passages les plus proches de la requête.

        Args:
            query_text: Question de l'utilisateur.
            k: Nombre de chunks à retourner.
            min_score: Score minimum (entre 0.0 et 1.0) exigé pour qu'un chunk soit retenu.
        """
        if self.index is None or not self.document_chunks:
            logging.warning("Recherche annulée : Index FAISS non disponible.")
            return []

        if not self.mistral_client:
            logging.error("Recherche annulée : Clé API Mistral absente.")
            return []

        try:
            # 1. Embedding de la requête utilisateur
            response = self.mistral_client.embeddings(
                model=EMBEDDING_MODEL,
                input=[query_text],
            )
            query_vector = np.array([response.data[0].embedding], dtype="float32")
            faiss.normalize_L2(query_vector)

            # 2. Exécution de la recherche vectorielle
            search_k = k * 3 if min_score is not None else k
            scores, indices = self.index.search(query_vector, search_k)

            results = []
            if indices.size > 0:
                for i, idx in enumerate(indices[0]):
                    if 0 <= idx < len(self.document_chunks):
                        raw_score = float(scores[0][i])
                        similarity_pct = raw_score * 100.0

                        if min_score is not None and similarity_pct < (min_score * 100.0):
                            continue

                        chunk = self.document_chunks[idx]
                        results.append({
                            "score": similarity_pct,
                            "raw_score": raw_score,
                            "text": chunk["text"],
                            "metadata": chunk["metadata"],
                        })

            # Trier par similarité décroissante et conserver les k meilleurs résultats
            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:k]

            logging.info(f"{len(results)} passage(s) pertinent(s) retenu(s) pour la recherche.")
            return results

        except MistralAPIException as e:
            logging.error(f"Erreur API Mistral lors de la recherche : [Code {e.status_code}] {e.message}")
            return []
        except Exception as e:
            logging.error(f"Erreur inattendue pendant la recherche vectorielle : {e}")
            return []