
import os
from pathlib import Path
from mistralai.client import Mistral

# Configuration
api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-embed"

client = Mistral(api_key=api_key)

# Dossier contenant les fichiers Markdown générés par Docling
input_dir = Path("Markdown")

# Récupérons tous les fichiers Markdown, y compris dans les sous-dossiers
files = list(input_dir.rglob("*.md"))
print(f"{len(files)} fichiers Markdown trouvés.")

# Lire le contenu des fichiers
documents = []

for file_path in files:
    text = file_path.read_text(encoding="utf-8")

    if text.strip():
        documents.append({
            "filename": str(file_path),
            "text": text
        })


# Créer les embeddings par lots
batch_size = 10
all_embeddings = []

for i in range(0, len(documents), batch_size):
    batch = documents[i:i + batch_size]
    texts = [doc["text"] for doc in batch]

    response = client.embeddings.create(
        model=model,
        inputs=texts
    )

    for doc, embedding in zip(batch, response.data):
        all_embeddings.append({
            "filename": doc["filename"],
            "text": doc["text"],
            "embedding": embedding.embedding
        })

    print(f"{len(all_embeddings)}/{len(documents)} documents traités.")

print("Création des embeddings terminée.")
print("Dimension du premier vecteur :", len(all_embeddings[0]["embedding"])
      if all_embeddings else 0) 




#Reduction de dim pour afficher les données 
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px

def affichage3D(all_embeddings):

    if all_embeddings:
        print("Réduction de dimension en cours...")

        # 1. Extraire les vecteurs et les convertir en tableau NumPy
        embeddings_matrix = np.array([item["embedding"] for item in all_embeddings])
        filenames = [item["filename"] for item in all_embeddings]

        # 2. Ajuster la perplexité
        n_samples = embeddings_matrix.shape[0]
        perplexity_val = min(30, max(1, n_samples - 1))

        # 3. Appliquer t-SNE en 3D
        tsne = TSNE(n_components=3, perplexity=perplexity_val, random_state=42)
        reduced_embeddings = tsne.fit_transform(embeddings_matrix)

        # 4. Créer le DataFrame avec la 3e coordonnée (z)
        df = pd.DataFrame({
            "x": reduced_embeddings[:, 0],
            "y": reduced_embeddings[:, 1],
            "z": reduced_embeddings[:, 2],  
            "filename": filenames
        })

        # 5. Graphique 3D Plotly
        fig = px.scatter_3d(
            df,
            x="x",
            y="y",
            z="z",  
            hover_name="filename",
            title="Visualisation 3D des Embeddings (t-SNE)",
            labels={"x": "Dimension 1", "y": "Dimension 2", "z": "Dimension 3"}
        )

        fig.show()


# Creons l'index FAISS
import faiss 
import numpy as np

def CreationIndexFaiss(all_embeddings):
    #transfomons ce dictionnaire de donnees en tableau  contenant uniquement les valeurs des matrices d'embeddings
    embedding_matrix = np.array([
        item ["embedding"] for item  in all_embeddings
    ])
    print ("taille du tableau numpy: ", embedding_matrix.shape)

    # definissons la taille de la dimension pour l'index faiss
    dimension = embedding_matrix.shape[1]
    print ("dimension : ", dimension)

    #creons l'index
    index = faiss.IndexFlatL2(dimension)

    #ajoutons les vecteurs dans le vecteurs 
    index.add(embedding_matrix)

    #sauvegardons l'index sur le disque 
    faiss.write_index(index, "faiss_index.idx")

    print ("index bien enregistrés !")


#Interrogons l'index faiss    

def Interrogation (question):





CreationIndexFaiss(all_embeddings)

