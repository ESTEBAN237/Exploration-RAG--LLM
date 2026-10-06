# PyPDF2

import PyPDF2

pdf_path = "Document.pdf"
# Ouvrir le fichier PDF en mode lecture binaire
with open(pdf_path, "rb") as pdf_file:
    # Créer un lecteur PDF
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    
# Extraire le texte de chaque page
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()

# Afficher le texte extrait
print(text)