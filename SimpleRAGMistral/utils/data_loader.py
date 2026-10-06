import io
import logging
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# --- Configuration du Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
)


# --- Extracteurs de Texte ---

def extract_text_from_pdf(file_path: str) -> Optional[str]:
    """Extrait le texte d'un fichier PDF via PyPDF2 ou pypdf."""
    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(file_path)
        pages_text = [
            page.extract_text()
            for page in reader.pages
            if page.extract_text()
        ]
        text = "\n".join(pages_text).strip()
        logging.info(f"PDF lu : {file_path} ({len(text)} caractères)")
        return text if text else None
    except Exception as e:
        logging.error(f"Erreur d'extraction PDF sur {file_path} : {e}")
        return None


def extract_text_from_docx(file_path: str) -> Optional[str]:
    """Extrait le texte d'un fichier Word DOCX via python-docx."""
    try:
        import docx

        doc = docx.Document(file_path)
        paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
        text = "\n".join(paragraphs)
        logging.info(f"DOCX lu : {file_path} ({len(text)} caractères)")
        return text if text else None
    except Exception as e:
        logging.error(f"Erreur d'extraction DOCX sur {file_path} : {e}")
        return None


def extract_text_from_txt(file_path: str) -> Optional[str]:
    """Extrait le texte d'un fichier texte brut avec détection automatique d'encodage."""
    for encoding in ["utf-8", "latin1", "cp1252"]:
        try:
            with open(file_path, "r", encoding=encoding, errors="replace") as f:
                text = f.read().strip()
            logging.info(f"TXT lu ({encoding}) : {file_path} ({len(text)} caractères)")
            return text if text else None
        except Exception:
            continue
    logging.error(f"Impossible de lire le fichier TXT : {file_path}")
    return None


def extract_text_from_csv(file_path: str) -> Optional[str]:
    """Extrait le texte d'un fichier CSV et le convertit en chaîne lisible."""
    try:
        import pandas as pd

        df = None
        for enc in ["utf-8", "latin1"]:
            for sep in [",", ";", "\t"]:
                try:
                    df = pd.read_csv(file_path, encoding=enc, sep=sep)
                    if not df.empty:
                        break
                except Exception:
                    continue
            if df is not None and not df.empty:
                break

        if df is None or df.empty:
            logging.warning(f"Fichier CSV vide ou non parsable : {file_path}")
            return None

        text = df.to_string(index=False)
        logging.info(f"CSV lu : {file_path} ({len(text)} caractères)")
        return text
    except ImportError:
        logging.warning("Pandas non installé. Impossible d'extraire les fichiers CSV.")
        return None
    except Exception as e:
        logging.error(f"Erreur d'extraction CSV sur {file_path} : {e}")
        return None


def extract_text_from_excel(file_path: str) -> Optional[str]:
    """Extrait le texte de toutes les feuilles d'un fichier Excel."""
    try:
        import pandas as pd

        sheets_dict = pd.read_excel(file_path, sheet_name=None)
        extracted_sections = []

        for sheet_name, df in sheets_dict.items():
            if not df.empty:
                sheet_content = df.to_string(index=False)
                extracted_sections.append(f"--- Feuille : {sheet_name} ---\n{sheet_content}")

        text = "\n\n".join(extracted_sections).strip()
        logging.info(f"Excel lu : {file_path} ({len(text)} caractères)")
        return text if text else None
    except ImportError:
        logging.warning("Pandas ou openpyxl non installé. Impossible de lire les fichiers Excel.")
        return None
    except Exception as e:
        logging.error(f"Erreur d'extraction Excel sur {file_path} : {e}")
        return None


# --- Traitement des Fichiers & Répertoires ---

def download_and_extract_zip(url: str, output_dir: str) -> bool:
    """Télécharge un fichier ZIP distant et extrait son contenu dans output_dir."""
    if not url:
        logging.warning("Aucune URL valide fournie pour le téléchargement.")
        return False

    try:
        logging.info(f"Téléchargement du ZIP depuis : {url}")
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            logging.info(f"Extraction des fichiers dans : {output_dir}")
            z.extractall(output_dir)

        logging.info("Téléchargement et extraction réussis.")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Erreur réseau lors du téléchargement : {e}")
        return False
    except zipfile.BadZipFile:
        logging.error("Le fichier téléchargé n'est pas une archive ZIP valide.")
        return False
    except Exception as e:
        logging.error(f"Erreur inattendue pendant l'extraction : {e}")
        return False


def load_and_parse_files(input_dir: str) -> List[Dict[str, Any]]:
    """
    Parcourt récursivement input_dir, extrait le contenu textuel de chaque document
    compatible et renvoie une liste de documents prêts pour l'indexation.
    """
    documents = []
    input_path = Path(input_dir)

    if not input_path.is_dir():
        logging.error(f"Le dossier spécifié n'existe pas : '{input_dir}'")
        return []

    logging.info(f"Parcours du dossier source : {input_path.resolve()}")

    for file_path in input_path.rglob("*.*"):
        if not file_path.is_file():
            continue

        relative_path = file_path.relative_to(input_path)
        category = relative_path.parts[0] if len(relative_path.parts) > 1 else "root"
        ext = file_path.suffix.lower()
        text: Optional[str] = None

        if ext == ".pdf":
            text = extract_text_from_pdf(str(file_path))
        elif ext == ".docx":
            text = extract_text_from_docx(str(file_path))
        elif ext == ".txt":
            text = extract_text_from_txt(str(file_path))
        elif ext == ".csv":
            text = extract_text_from_csv(str(file_path))
        elif ext in [".xlsx", ".xls"]:
            text = extract_text_from_excel(str(file_path))
        else:
            logging.debug(f"Extension non prise en charge ignorée : {relative_path}")
            continue

        if text:
            documents.append({
                "page_content": text,
                "metadata": {
                    "source": str(relative_path),
                    "filename": file_path.name,
                    "category": category,
                    "full_path": str(file_path.resolve()),
                },
            })
        else:
            logging.warning(f"Aucun contenu extrait de : {relative_path}")

    logging.info(f"Parsing terminé : {len(documents)} document(s) chargé(s).")
    return documents