
from pathlib import Path
from docling.document_converter import DocumentConverter

# Dossier contenant les documents
input_dir = Path("Documents/")

# Dossier de sortie
output_dir = Path("Markdown")
output_dir.mkdir(parents=True, exist_ok=True)

converter = DocumentConverter()

# Parcourir tous les documents , y compris dans les sous-dossiers
for file_path in input_dir.rglob("*"):
    result = converter.convert(file_path)

    markdown = result.document.export_to_markdown()

    # Conserver la structure des sous-dossiers
    relative_path = file_path.relative_to(input_dir).with_suffix(".md")
    output_file = output_dir / relative_path

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(markdown, encoding="utf-8")

    print(f"Converti : {file_path} -> {output_file}")