import uuid
from pathlib import Path
from typing import List, Dict, Any

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from qdrant_client import QdrantClient
from qdrant_client import models
from FlagEmbedding import BGEM3FlagModel

# =====================================================================
# 1. CONFIGURATION SYSTEME
# =====================================================================
# NOM_DU_DOCUMENT = "BNPP_CP_T4-25_FR"
NOM_DU_DOCUMENT = "CASA_CP_T1-26_FR"
COLLECTION_NAME = "financial_reports"
QDRANT_URL = "http://localhost:6333"

PATH_MARKDOWN = Path(f"assets/parsed_md/{NOM_DU_DOCUMENT}/{NOM_DU_DOCUMENT}.md")

# =====================================================================
# 2. PHASE DE CHUNKING (DECOUPAGE HIERARCHIQUE)
# =====================================================================
if not PATH_MARKDOWN.exists():
    raise FileNotFoundError(f"Le fichier spécifié est introuvable : {PATH_MARKDOWN}")

print(f"[INFO] Lecture du fichier Markdown : {PATH_MARKDOWN}")
with open(PATH_MARKDOWN, "r", encoding="utf-8") as f:
    markdown_content = f.read()

print("[INFO] Initialisation du découpage hiérarchique...")
headers_to_split_on = [("#", "Titre_1"), ("##", "Titre_2"), ("###", "Titre_3")]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_chunks = markdown_splitter.split_text(markdown_content)

# Paramétrage large pour préserver l'intégrité des tableaux financiers
text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=300)
final_chunks = text_splitter.split_documents(md_chunks)
textes_purs = [chunk.page_content for chunk in final_chunks]

print(f"[INFO] Document segmenté en {len(textes_purs)} chunks.")

# =====================================================================
# 3. GENERATION DES EMBEDDINGS MULTI-VECTEURS (BGE-M3)
# =====================================================================
print("[INFO] Chargement du modèle local BAAI/bge-m3...")
# use_fp16=True peut être activé si l'exécution se fait sur GPU compatible
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)

print("[INFO] Calcul des matrices vectorielles (Dense, Sparse, ColBERT)...")
embeddings = model.encode(
    textes_purs, return_dense=True, return_sparse=True, return_colbert_vecs=True
)

# =====================================================================
# 4. INITIALISATION DE LA COLLECTION QDRANT
# =====================================================================
print(f"[INFO] Connexion à l'instance Qdrant locale ({QDRANT_URL})...")
client = QdrantClient(url=QDRANT_URL)

if not client.collection_exists(COLLECTION_NAME):
    print(
        f"[INFO] Création de la collection '{COLLECTION_NAME}' avec topologie hybride et ColBERT..."
    )
    client.create_collection(
        collection_name=COLLECTION_NAME,
        # Configuration des espaces vectoriels denses et late interaction
        vectors_config={
            "dense": models.VectorParams(size=1024, distance=models.Distance.COSINE),
            "colbert": models.VectorParams(
                size=1024,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            ),
        },
        # Configuration de l'index de recherche lexicale inverse (Sparse)
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )

# =====================================================================
# 5. STRUCTURATION DES DONNEES ET PERSISTANCE (BATCH UPSERT)
# =====================================================================
print("[INFO] Préparation et alignement des structures de données...")
points: List[models.PointStruct] = []

for i, chunk in enumerate(final_chunks):
    point_id = str(uuid.uuid4())

    payload: Dict[str, Any] = {"source": NOM_DU_DOCUMENT, "texte": chunk.page_content}
    payload.update(chunk.metadata)

    lexical_weights = embeddings["lexical_weights"][i]
    sparse_dict: Dict[int, float] = {}

    for token_str, weight in lexical_weights.items():
        # Résolution de la chaîne en ID entier
        token_id = model.tokenizer.convert_tokens_to_ids(token_str)

        if token_id is not None:
            # Si l'ID a déjà été rencontré (collision), on garde le poids maximum
            if token_id in sparse_dict:
                sparse_dict[token_id] = max(sparse_dict[token_id], float(weight))
            else:
                sparse_dict[token_id] = float(weight)

    # Conversion du dictionnaire dédupliqué en listes parallèles pour Qdrant
    sparse_indices = list(sparse_dict.keys())
    sparse_values = list(sparse_dict.values())

    points.append(
        models.PointStruct(
            id=point_id,
            payload=payload,
            vector={
                "dense": embeddings["dense_vecs"][i].tolist(),
                "colbert": embeddings["colbert_vecs"][i].tolist(),
                "sparse": models.SparseVector(
                    indices=sparse_indices, values=sparse_values
                ),
            },
        )
    )

print(f"[INFO] Payload global généré : {len(points)} points.")

BATCH_SIZE = 2
print(f"[INFO] Début de l'injection par lots de {BATCH_SIZE} points...")

for batch_start in range(0, len(points), BATCH_SIZE):
    batch = points[batch_start : batch_start + BATCH_SIZE]
    print(
        f"  -> Envoi du lot [{batch_start} à {batch_start + len(batch) - 1}] / {len(points)} points ..."
    )

    client.upsert(collection_name=COLLECTION_NAME, points=batch)

print(
    "[SUCCESS] Indexation terminée. Les données sont persistées localement dans Qdrant."
)
