import json
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.document import PictureItem

from finrag.services.config import NOM_DU_DOCUMENT


# =========================================================
# PATHS CONFIG
# =========================================================
pdf_path = Path(f"assets/raw_pdfs/{NOM_DU_DOCUMENT}.pdf")

output_dir = Path(f"assets/parsed/{NOM_DU_DOCUMENT}")
images_dir = output_dir / "images"
output_md_path = output_dir / f"{NOM_DU_DOCUMENT}.md"
output_json_path = output_dir / f"{NOM_DU_DOCUMENT}.json"

# création dossiers
output_dir.mkdir(parents=True, exist_ok=True)
images_dir.mkdir(parents=True, exist_ok=True)


# =========================================================
# DOCLING PIPELINE
# =========================================================
print(f"[INFO] Démarrage du pipeline pour : {NOM_DU_DOCUMENT}.pdf")

pipeline_options = PdfPipelineOptions()
pipeline_options.generate_picture_images = True
pipeline_options.images_scale = 2.0

doc_converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)

print("[INFO] Analyse du document en cours...")
result = doc_converter.convert(str(pdf_path))


# =========================================================
# MARKDOWN EXPORT
# =========================================================
print("[INFO] Sauvegarde du Markdown (Pour debug)...")
with open(output_md_path, "w", encoding="utf-8") as f:
    f.write(result.document.export_to_markdown())

print("[INFO] Sauvegarde du JSON natif (Pour le Super Chunking)...")
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(result.document.export_to_dict(), f, ensure_ascii=False, indent=2)


# =========================================================
# IMAGE EXTRACTION ET TRACKING DE METADONNEES
# =========================================================
print("[INFO] Extraction des images et de leur contexte...")

image_count = 0
images_metadata = {}
current_section = "Racine du document"  # Le tracker de titre

# iterate_items() lit le document de haut en bas
for item, _level in result.document.iterate_items():

    # 1. Si on lit un titre, on le mémorise
    if hasattr(item, "label") and item.label in ["title", "section_header"]:
        current_section = item.text.strip()

    # 2. Si on lit une image, on la sauvegarde AVEC son contexte mémorisé
    elif isinstance(item, PictureItem):
        image_count += 1
        page_number = item.prov[0].page_no if item.prov else "Inconnue"
        image_name = f"img_{image_count:02d}_page_{page_number}.png"
        image_filename = images_dir / image_name

        # Sauvegarde physique du PNG
        with image_filename.open("wb") as fp:
            item.get_image(result.document).save(fp, "PNG")

        # Sauvegarde du contexte (Page + Section exacte)
        images_metadata[image_name] = {
            "page": str(page_number),
            "section": current_section,
        }

# 3. On sauvegarde ce dictionnaire dans un fichier JSON à côté des images
meta_path = images_dir / "images_meta.json"
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(images_metadata, f, ensure_ascii=False, indent=2)


# =========================================================
# LOG FINAL
# =========================================================
print("==========================================")
print(f"[SUCCESS] {NOM_DU_DOCUMENT}")
print(f"JSON Structure : {output_json_path}")
print(f"Images : {image_count}")
print("==========================================")
