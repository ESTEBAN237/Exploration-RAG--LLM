# pymupdf4llm
# 
import pymupdf4llm

pdf_path = "Document.pdf"
# Ouvrir le fichier PDF
md_text = pymupdf4llm.to_markdown(pdf_path)
print("=== Contenu Markdown extrait avec PyMuPDF4LLM ===")
print(md_text)