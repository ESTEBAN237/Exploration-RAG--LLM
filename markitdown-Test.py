# markitdown
from markitdown import MarkItDown

pdf_path ="Document.pdf"
# Ouvrir le fichier PDF
md = MarkItDown()
result = md.convert(pdf_path)
print(result.text_content)