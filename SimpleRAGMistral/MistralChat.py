import logging
import os
import streamlit as st
from dotenv import load_dotenv
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# --- Chargement des variables d'environnement ---
load_dotenv()

# --- Importations depuis mes modules ---
try:
    from utils.config import (
        APP_TITLE,
        MISTRAL_API_KEY,
        MODEL_NAME,
        PROJECT_NAME,
        SEARCH_K,
    )
    from utils.vector_store import VectorStoreManager
except ImportError as e:
    st.error(f"Erreur d'importation : {e}. Vérifiez la structure du dossier 'utils'.")
    st.stop()


# --- Configuration du Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(module)s - %(message)s"
)

# --- Configuration de l'API Mistral ---
api_key = MISTRAL_API_KEY or os.getenv("MISTRAL_API_KEY")
model = MODEL_NAME

if not api_key:
    st.error("Erreur : Clé API Mistral introuvable. Définissez MISTRAL_API_KEY dans votre fichier .env.")
    st.stop()

try:
    client = MistralClient(api_key=api_key)
    logging.info("Client Mistral initialisé.")
except Exception as e:
    st.error(f"Erreur lors de l'initialisation du client Mistral : {e}")
    logging.exception("Erreur initialisation client Mistral")
    st.stop()


# --- Chargement du Vector Store (mis en cache) ---
@st.cache_resource
def get_vector_store_manager():
    logging.info("Chargement du VectorStoreManager...")
    try:
        manager = VectorStoreManager()
        if manager.index is None or not manager.document_chunks:
            st.error("L'index FAISS ou les chunks n'ont pas pu être chargés.")
            st.warning("Exécutez 'python indexer.py' après avoir placé vos documents dans 'inputs/'.")
            logging.error("Index FAISS ou chunks non chargés par VectorStoreManager.")
            return None
        logging.info(f"VectorStoreManager chargé avec succès ({manager.index.ntotal} vecteurs).")
        return manager
    except FileNotFoundError:
        st.error("Fichiers d'indexation introuvables.")
        st.warning("Exécutez 'python indexer.py' pour générer la base de connaissances FAISS.")
        logging.error("FileNotFoundError lors de l'initialisation du VectorStoreManager.")
        return None
    except Exception as e:
        st.error(f"Erreur inattendue lors du chargement du VectorStoreManager : {e}")
        logging.exception("Erreur chargement VectorStoreManager")
        return None


vector_store_manager = get_vector_store_manager()

# --- Prompt Système RAG Personalisé ---
SYSTEM_PROMPT = f"""Tu es un assistant virtuel intelligent spécialisé dans la présentation et le support technique pour le projet **{PROJECT_NAME}**, conçu par **Esteban MEZAZEM**.

Ta mission est de répondre aux questions des utilisateurs de manière précise, concise, technique et courtoise, en te basant **exclusivement** sur le CONTEXTE fourni ci-dessous.

Directives strictes :
1. **Règles de vérité :** Réponds UNIQUEMENT à partir du CONTEXTE fourni. N'invente aucun fait, code ou métrique.
2. **Gestion de l'inconnu :** Si le CONTEXTE ne contient pas la réponse, indique clairement : "Je ne trouve pas cette information dans la base de connaissances du projet."
3. **Périmètre :** Ne réponds pas aux questions hors sujet (non liées à {PROJECT_NAME} ou aux technologies associées du document).
4. **Citations :** Indique la source (ex. le nom du fichier ou module) si elle est disponible dans le contexte.
5. **Style :** Reste direct, clair et professionnel (ton d'ingénieur/développeur).

CONTEXTE FOURNI :
---
{{context_str}}
---

QUESTION DE L'UTILISATEUR :
{{question}}

RÉPONSE DE L'ASSISTANT :"""


# --- Configuration de la page Streamlit ---
st.set_page_page_config(page_title=APP_TITLE, page_icon="🤖", layout="wide")

# --- Initialisation de l'historique ---
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": f"Bonjour ! Je suis l'assistant RAG du projet **{PROJECT_NAME}**. Posez-moi vos questions sur l'architecture, la base de connaissances ou le fonctionnement du système.",
    }]

# --- Fonctions de génération ---
def generer_reponse(prompt_messages: list[ChatMessage]) -> str:
    if not prompt_messages:
        return "Je ne peux pas traiter une demande vide."
    try:
        logging.info(f"Appel de l'API Mistral ('{model}') avec {len(prompt_messages)} message(s).")
        response = client.chat(
            model=model,
            messages=prompt_messages,
            temperature=0.1, # Température basse pour des réponses factuelles et précises
        )
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content
        return "Désolé, aucune réponse n'a été renvoyée par le modèle."
    except Exception as e:
        st.error(f"Erreur lors de la génération avec Mistral AI : {e}")
        logging.exception("Erreur API Mistral pendant client.chat")
        return "Une erreur technique s'est produite lors de la génération."


# --- UI Streamlit ---
st.title(APP_TITLE)
st.caption(f"Assistant IA / RAG — Projets & Architecture | Modèle : {model}")

# Affichage des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Entrée de l'utilisateur
if prompt := st.chat_input(f"Posez une question sur {PROJECT_NAME}..."):
    # 1. Message Utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 2. Vérification Vector Store
    if vector_store_manager is None:
        st.error("Le service de recherche vectorielle (FAISS) n'est pas disponible.")
        st.stop()

    # 3. Recherche de Contexte
    try:
        logging.info(f"Recherche FAISS pour : '{prompt}' (k={SEARCH_K})")
        search_results = vector_store_manager.search(prompt, k=SEARCH_K)
    except Exception as e:
        st.error(f"Erreur lors de la recherche vectorielle : {e}")
        logging.exception("Erreur recherche FAISS")
        search_results = []

    # 4. Formattage du Contexte
    if search_results:
        context_str = "\n\n---\n\n".join([
            f"Source : {res['metadata'].get('source', 'Inconnue')} (Pertinence : {res['score']:.1f}%)\nContenu : {res['text']}"
            for res in search_results
        ])
    else:
        context_str = "Aucun contexte pertinent n'a été trouvé dans l'index vectoriel."

    # 5. Construction du Prompt Final
    final_prompt = SYSTEM_PROMPT.format(context_str=context_str, question=prompt)
    messages_for_api = [ChatMessage(role="user", content=final_prompt)]

    # 6. Génération et affichage de la réponse
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.text("Recherche dans le Vector Store et génération...")
        
        response_content = generer_reponse(messages_for_api)
        message_placeholder.write(response_content)

    # 7. Sauvegarde dans l'historique
    st.session_state.messages.append({"role": "assistant", "content": response_content})

# Pied de page
st.markdown("---")
st.caption(f"Conçu par **Esteban MEZAZEM** | Propulsé par **Mistral AI** & **FAISS**")