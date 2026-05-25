from pathlib import Path
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# ==========================================
# ⚙️ 1. CONFIGURATION
# ==========================================
NOM_DU_DOCUMENT = "CASA_CP_T1-26_0"  # Le nom de ton doc
md_path = Path(f"assets/parsed_md/{NOM_DU_DOCUMENT}/{NOM_DU_DOCUMENT}.md")

# On charge le contenu de ton fichier Markdown
print(f"📖 Chargement du document : {md_path}")
with open(md_path, "r", encoding="utf-8") as f:
    markdown_content = f.read()

# ==========================================
# 🧠 2. LE CHUNKING INTELLIGENT (HIÉRARCHIQUE)
# ==========================================
print("✂️ Découpage hiérarchique en cours...")

# On dit à l'algorithme de couper à chaque titre, et de retenir le nom du titre !
headers_to_split_on = [
    ("#", "Titre_Niveau_1"),
    ("##", "Titre_Niveau_2"),
    ("###", "Titre_Niveau_3"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_chunks = markdown_splitter.split_text(markdown_content)

# 🛡️ Sécurité : Si un chapitre est VRAIMENT trop long, on le recoupe proprement
# sans casser les paragraphes (chunk size de 1000 caractères, overlap de 200 pour le contexte)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=300)
final_chunks = text_splitter.split_documents(md_chunks)

# ==========================================
# 📊 3. RÉSULTATS
# ==========================================
print("==========================================")
print(f"✅ Document découpé en {len(final_chunks)} morceaux (chunks) intelligents.")
print("==========================================")

# On affiche les 2 premiers chunks pour te prouver la magie
for i, chunk in enumerate(final_chunks[:3]):
    print(f"\n--- 🧩 CHUNK {i + 1} ---")
    print(f"📌 MÉTADONNÉES (Ce que l'IA retiendra) : {chunk.metadata}")
    print(f"📏 HUNK SIZE (chars) : {len(chunk.page_content)}")
    print(f"📝 TEXTE (Aperçu) :\n{chunk.page_content[:4000]}...\n")
