import os
import subprocess


def convert_documents_to_markdown(input_dir, output_dir):

    if not os.path.isdir(input_dir):
        print(f"Erreur : le répertoire '{input_dir}' n'existe pas.")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"Début de la conversion des documents de '{input_dir}'")
    print(f"vers '{output_dir}'...\n")

    # Parcours récursif de tous les dossiers
    for root, dirs, files in os.walk(input_dir):

        for filename in files:

            # Chemin complet du fichier source
            input_path = os.path.join(root, filename)

            # Récupérer le nom du dossier courant
            relative_dir = os.path.relpath(root, input_dir)

            # Créer le même dossier dans markdown_outputs
            if relative_dir == ".":
                current_output_dir = output_dir
            else:
                current_output_dir = os.path.join(
                    output_dir,
                    relative_dir
                )

            os.makedirs(current_output_dir, exist_ok=True)

            print(f"Traitement de : {input_path}")

            # Commande Docling
            cmd = [
                "docling",
                input_path,
                "--to", "md",
                "--output", current_output_dir
            ]

            try:

                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8"
                )

                print(f"✓ Conversion réussie : {filename}")

            except subprocess.CalledProcessError as e:

                print(f"✗ Erreur lors de la conversion de : {filename}")
                print(e.stderr)

            except FileNotFoundError:

                print("✗ La commande 'docling' n'a pas été trouvée.")
                print("Installe Docling avec :")
                print("python -m pip install docling")
                return

    print("\nConversion terminée.")


# Configuration

INPUT_DIRECTORY = r"C:\Users\Dell\Documents\Projet\RAG - LLM\8532116-mettez-en-place-un-rag-pour-un-llm\inputs"

OUTPUT_DIRECTORY = r"C:\Users\Dell\Documents\Projet\RAG - LLM\markdown_outputs"


# Exécution
if __name__ == "__main__":

    convert_documents_to_markdown(
        INPUT_DIRECTORY,
        OUTPUT_DIRECTORY
    )