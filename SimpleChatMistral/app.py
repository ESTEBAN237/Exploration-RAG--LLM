# app.py
import logging
import os
import streamlit as st
from dotenv import load_dotenv
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# --- Chargement des variables d'environnement ---
load_dotenv()

# --- Configuration du Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
)

# --- Configuration de la page Streamlit ---
st.set_page_config(page_title="Mistral AI - Conversational Assistant", page_icon="🤖", layout="wide")

# --- Configuration du Client Mistral AI ---
api_key = os.getenv("MISTRAL_API_KEY")
model = os.getenv("MODEL_NAME", "mistral-large-latest")

if not api_key:
    st.error("⚠️ Clé API Mistral introuvable. Veuillez configurer MISTRAL_API_KEY dans votre fichier .env.")
    st.stop()

try:
    client = MistralClient(api_key=api_key)
    logging.info(f"Client Mistral initialisé avec succès (Modèle : {model}).")
except Exception as e:
    st.error(f"Erreur lors de l'initialisation du client Mistral AI : {e}")
    logging.exception("Erreur d'initialisation du client Mistral.")
    st.stop()

# --- Constantes & Prompts ---
SYSTEM_PROMPT = """Tu es un assistant IA technique, concis et performant, conçu par Esteban MEZAZEM.
Tes réponses doivent être claires, précises et structurées (avec du code formaté si nécessaire).
Adopte un ton professionnel et direct."""

WELCOME_MESSAGE = "Bonjour ! Je suis votre assistant virtuel propulsé par Mistral AI. Comment puis-je vous aider aujourd'hui ?"

# --- Initialisation de la Session ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": WELCOME_MESSAGE}
    ]

# --- Fonctions Utilitaires ---
def construire_prompt_session(messages: list, max_messages: int = 10) -> list[ChatMessage]:
    """
    Formate l'historique de conversation récents et insère le System Prompt.
    """
    recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages

    formatted_messages = [
        ChatMessage(role="system", content=SYSTEM_PROMPT)
    ]

    for msg in recent_messages:
        formatted_messages.append(ChatMessage(role=msg["role"], content=msg["content"]))

    return formatted_messages


def generer_reponse(prompt_messages: list[ChatMessage]) -> str:
    """
    Appelle l'API Mistral AI pour obtenir la réponse du modèle.
    """
    try:
        logging.info(f"Envoi de {len(prompt_messages)} message(s) à l'API Mistral.")
        response = client.chat(
            model=model,
            messages=prompt_messages,
            temperature=0.3,
        )
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content
        return "Désolé, l'API n'a pas retourné de réponse valide."
    except Exception as e:
        logging.exception("Erreur pendant l'exécution de client.chat")
        st.error(f"Erreur lors de la génération de la réponse : {e}")
        return "Une erreur technique s'est produite. Veuillez réessayer."


# --- Interface Utilisateur (UI) ---
st.title("🤖 ChatBot Mistral AI")
st.caption(f"Assistant LLM Direct | Modèle : `{model}`")

# Bouton de réinitialisation dans la barre侧 (Sidebar)
with st.sidebar:
    st.header("Options")
    if st.button("🔄 Effacer la conversation", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]
        st.rerun()

# Affichage de l'historique des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Entrée Utilisateur
if prompt := st.chat_input("Posez votre question..."):
    # 1. Message de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 2. Preparation des messages avec historique et System Prompt
    prompt_formatted = construire_prompt_session(st.session_state.messages)

    # 3. Génération et affichage de la réponse
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.text("Réflexion en cours...")

        response_content = generer_reponse(prompt_formatted)
        message_placeholder.write(response_content)

    # 4. Sauvegarde dans l'historique de session
    st.session_state.messages.append({"role": "assistant", "content": response_content})

# Pied de page
st.markdown("---")
st.caption("Développé par **Esteban MEZAZEM** | Déploiement Streamlit & Mistral AI")