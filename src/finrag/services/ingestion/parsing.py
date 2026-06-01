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

output_dir = Path(f"assets/parsed_md/{NOM_DU_DOCUMENT}")
images_dir = output_dir / "images"
output_md_path = output_dir / f"{NOM_DU_DOCUMENT}.md"

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
print("[INFO] Sauvegarde du Markdown...")

with open(output_md_path, "w", encoding="utf-8") as f:
    f.write(result.document.export_to_markdown())


# =========================================================
# IMAGE EXTRACTION
# =========================================================
print("[INFO] Extraction des images...")

image_count = 0

for element, _level in result.document.iterate_items():
    if isinstance(element, PictureItem):
        image_count += 1

        page_number = element.prov[0].page_no

        image_filename = images_dir / f"img_{image_count:02d}_page_{page_number}.png"

        with image_filename.open("wb") as fp:
            element.get_image(result.document).save(fp, "PNG")


# =========================================================
# LOG FINAL
# =========================================================
print("==========================================")
print(f"[SUCCESS] {NOM_DU_DOCUMENT}")
print(f"Markdown : {output_md_path}")
print(f"Images : {image_count}")
print("==========================================")
