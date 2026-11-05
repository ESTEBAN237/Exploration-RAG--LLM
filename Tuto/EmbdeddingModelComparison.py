import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from mistralai.client import Mistral

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-embed"

client = Mistral(api_key=api_key)

texts = [
    """**1. Demande d'information sur l'entretien de la voirie** Madame, Monsieur,

Je me permets de vous écrire afin d’obtenir des renseignements détaillés sur l’entretien de la voirie dans notre quartier, notamment en ce qui concerne les horaires de passage des équipes techniques et les modalités de ramassage des ordures.  
Je souhaite mieux comprendre le planning d’intervention et les éventuelles perturbations prévues dans la semaine à venir. Pourriez-vous, s’il vous plaît, me communiquer ces informations ou m’indiquer où je pourrais les consulter ?

Dans l’attente de votre réponse, je vous remercie par avance pour votre aide et votre disponibilité.

Cordialement,  
Jean Jean""",
    """Afin  de  maintenir  une  transparence  totale  sur  nos  activités,  des  réunions publiques sont régulièrement organisées. Les dates et lieux de  ces  rencontres sont  affichés  sur  le  panneau  d'information  municipal  ainsi  que  sur  notre  site officiel.""",
    """La commune deTriffouillis-sur-Loire est heureuse d'annoncer l'édition2023de son célèbre Marché de Noël! Cette année encore, petits et grands pourront se plonger dans la magie des fêtes en découvrant de nombreux exposants et animations festives."""
]

# 1. Génération des embeddings via Mistral AI
embeddings_batch_response = client.embeddings.create(
    model=model,
    inputs=texts,
)

print("taille : ",len(embeddings_batch_response.data[0].embedding))

# 2. Extraction des vecteurs sous forme de tableau NumPy
embeddings = np.array([item.embedding for item in embeddings_batch_response.data])

# 3. Calcul de la matrice de similarité cosinus (3x3)
sim_matrix = cosine_similarity(embeddings)

# 4. Affichage sous forme lisible
labels = ["Voirie/Planning", "Réunions publiques", "Marché de Noël"]

print("--- Matrice de similarité cosinus ---")
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        print(f"Similarité entre [{labels[i]}] et [{labels[j]}] : {sim_matrix[i][j]:.4f}")

