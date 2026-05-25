from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.document import PictureItem


NOM_DU_DOCUMENT = "CASA_CP_T1-26_0"

# Chemins dynamiques
pdf_path = Path(f"assets/raw_pdfs/{NOM_DU_DOCUMENT}.pdf")

# dossier exclusif pour ce document pour ne rien mélanger
output_dir = Path(f"assets/parsed_md/{NOM_DU_DOCUMENT}")
images_dir = output_dir / "images"
output_md_path = output_dir / f"{NOM_DU_DOCUMENT}.md"

# Création physique des dossiers
output_dir.mkdir(parents=True, exist_ok=True)
images_dir.mkdir(exist_ok=True)


print(f"⚙️ Démarrage du pipeline pour : {NOM_DU_DOCUMENT}.pdf")
pipeline_options = PdfPipelineOptions()
pipeline_options.generate_picture_images = True
pipeline_options.images_scale = 2.0

doc_converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)

print("🧠 Analyse du document en cours (Texte, Tableaux, Images)...")
result = doc_converter.convert(str(pdf_path))

# Sauvegarde du Markdown
print("📝 Sauvegarde du fichier Markdown...")
with open(output_md_path, "w", encoding="utf-8") as f:
    f.write(result.document.export_to_markdown())


print("🖼️ Extraction et Mapping des images...")
image_count = 0

# On parcourt chaque élément du PDF
for element, _level in result.document.iterate_items():
    if isinstance(element, PictureItem):
        image_count += 1

        # On récupère le numéro de la page où se trouve l'image
        page_number = element.prov[0].page_no

        # On nomme le fichier intelligemment pour le RAG
        image_filename = images_dir / f"img_{image_count:02d}_page_{page_number}.png"

        # On sauvegarde l'image
        with image_filename.open("wb") as fp:
            element.get_image(result.document).save(fp, "PNG")

print("==========================================")
print(f"✅ SUCCÈS POUR : {NOM_DU_DOCUMENT}")
print(f"📂 Markdown : {output_md_path}")
print(f"🖼️ Images extraites : {image_count} (sauvegardées avec leur numéro de page)")
print("==========================================")
