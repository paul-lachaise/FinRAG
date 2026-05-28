from pathlib import Path
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# NOM_DU_DOCUMENT = "CASA_CP_T1-26_FR"
NOM_DU_DOCUMENT = "BNPP_CP_T4-25_FR"

md_path = Path(f"assets/parsed_md/{NOM_DU_DOCUMENT}/{NOM_DU_DOCUMENT}.md")

# On charge le contenu de ton fichier Markdown
print(f"Chargement du document : {md_path}")
with open(md_path, "r", encoding="utf-8") as f:
    markdown_content = f.read()

print("Découpage hiérarchique en cours...")

headers_to_split_on = [
    ("#", "Titre_Niveau_1"),
    ("##", "Titre_Niveau_2"),
    ("###", "Titre_Niveau_3"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_chunks = markdown_splitter.split_text(markdown_content)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=300)
final_chunks = text_splitter.split_documents(md_chunks)

print("==========================================")
print(f" Document découpé en {len(final_chunks)} chunks.")
print("==========================================")

for i, chunk in enumerate(final_chunks[:3]):
    print(f"\n---  CHUNK {i + 1} ---")
    print(f" MÉTADONNÉES du chunk : {chunk.metadata}")
    print(f" HUNK SIZE (chars) : {len(chunk.page_content)}")
    print(f" TEXTE (Aperçu) :\n{chunk.page_content[:4000]}...\n")
