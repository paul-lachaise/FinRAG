from pathlib import Path
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
import uuid

# ==========================================
# ⚙️ 1. CONFIGURATION ET CHUNKING
# ==========================================
NOM_DU_DOCUMENT = "CASA_CP_T1-26_0"
md_path = Path(f"assets/parsed_md/{NOM_DU_DOCUMENT}/{NOM_DU_DOCUMENT}.md")

print(f"📖 Chargement du document : {NOM_DU_DOCUMENT}...")
with open(md_path, "r", encoding="utf-8") as f:
    markdown_content = f.read()

print("✂️ Découpage hiérarchique en cours...")
headers_to_split_on = [("#", "Titre_1"), ("##", "Titre_2"), ("###", "Titre_3")]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_chunks = markdown_splitter.split_text(markdown_content)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=300)
final_chunks = text_splitter.split_documents(md_chunks)

# ==========================================
# 🧠 2. CHARGEMENT DE BGE-M3 (Local)
# ==========================================
print("🧠 Chargement du modèle BAAI/bge-m3 (Téléchargement requis au 1er lancement)...")
# On charge le modèle explicitement
model = SentenceTransformer("BAAI/bge-m3")

# On extrait les textes purs pour l'embedding
textes_purs = [chunk.page_content for chunk in final_chunks]

print(f"🔢 Calcul des vecteurs pour {len(textes_purs)} chunks...")
# Le modèle calcule les vecteurs mathématiques
embeddings = model.encode(textes_purs)

# ==========================================
# 🚀 3. CONNEXION QDRANT ET INJECTION
# ==========================================
print("🔌 Connexion à Qdrant local...")
client = QdrantClient(url="http://localhost:6333")
NOM_COLLECTION = "rapports_financiers"

# On vérifie si la collection existe. Si non, on la crée avec la bonne taille (1024 pour BGE-M3)
if not client.collection_exists(NOM_COLLECTION):
    print(f"🏗️ Création de la collection '{NOM_COLLECTION}'...")
    client.create_collection(
        collection_name=NOM_COLLECTION,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

print("📥 Préparation des points de données...")
points = []
for i, chunk in enumerate(final_chunks):
    # On génère un ID unique pour chaque morceau
    point_id = str(uuid.uuid4())

    # On prépare les métadonnées (titres, texte source, nom du doc)
    payload = chunk.metadata.copy()
    payload["source"] = NOM_DU_DOCUMENT
    payload["texte"] = (
        chunk.page_content
    )  # On garde le texte brut pour pouvoir le relire !

    # On crée le "Point" (ID + Vecteur + Métadonnées)
    points.append(
        PointStruct(id=point_id, vector=embeddings[i].tolist(), payload=payload)
    )

print("💾 Sauvegarde dans Qdrant...")
client.upsert(collection_name=NOM_COLLECTION, points=points)

print("==========================================")
print(f"✅ SUCCÈS ! Les données de {NOM_DU_DOCUMENT} sont indexées dans Qdrant.")
print("==========================================")
