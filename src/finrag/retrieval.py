from typing import Dict
from qdrant_client import QdrantClient
from qdrant_client import models
from FlagEmbedding import BGEM3FlagModel

# =====================================================================
# 1. CONFIGURATION
# =====================================================================
COLLECTION_NAME = "financial_reports"
QDRANT_URL = "http://localhost:6333"

# La question cible
# USER_QUERY = "Quels sont les revenus et le coût du risque pour le groupe ?"
# USER_QUERY = "Cout du risque en 2025 de la banque BNP Paribas"
USER_QUERY = "quels sont les frais de gestion de BNP CIB pour le trimestre 4 de 2025"


# =====================================================================
# 2. ENCODAGE DE LA REQUÊTE
# =====================================================================
print("[INFO] Chargement du modèle BAAI/bge-m3...")
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)

print(f"[INFO] Traitement de la requête : '{USER_QUERY}'")
query_embeddings = model.encode(
    USER_QUERY, return_dense=True, return_sparse=True, return_colbert_vecs=True
)

# Préparation Dense
dense_vec = query_embeddings["dense_vecs"].tolist()

# Préparation ColBERT
colbert_vec = query_embeddings["colbert_vecs"].tolist()

# Préparation Sparse (Mapping Token String -> Token ID avec pooling)
lexical_weights = query_embeddings["lexical_weights"]
sparse_dict: Dict[int, float] = {}

for token_str, weight in lexical_weights.items():
    token_id = model.tokenizer.convert_tokens_to_ids(token_str)
    if token_id is not None:
        if token_id in sparse_dict:
            sparse_dict[token_id] = max(sparse_dict[token_id], float(weight))
        else:
            sparse_dict[token_id] = float(weight)

sparse_vec = models.SparseVector(
    indices=list(sparse_dict.keys()), values=list(sparse_dict.values())
)

# =====================================================================
# 3. INTERROGATION DE QDRANT (TWO-STAGE RETRIEVAL)
# =====================================================================
print("[INFO] Connexion au cluster Qdrant et exécution de la recherche...")
client = QdrantClient(url=QDRANT_URL)

search_results = client.query_points(
    collection_name=COLLECTION_NAME,
    # --- NIVEAU 1 & 2 : PRÉSÉLECTION HYBRIDE ET FUSION RRF ---
    prefetch=[
        models.Prefetch(
            # Niveau 1 : Ratissage large (Top 20 Dense + Top 20 Sparse)
            prefetch=[
                models.Prefetch(
                    query=dense_vec,
                    using="dense",
                    limit=20,
                ),
                models.Prefetch(
                    query=sparse_vec,
                    using="sparse",
                    limit=20,
                ),
            ],
            # Niveau 2 : Fusion RRF et conservation du Top 15 hybride
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=15,
        )
    ],
    # --- NIVEAU 3 : RERANKING FINAL (LATE INTERACTION) ---
    query=colbert_vec,
    using="colbert",
    limit=3,  # Récupération du Top 3 final
    with_payload=True,
)

# =====================================================================
# 4. ANALYSE DES RÉSULTATS
# =====================================================================
print("\n" + "=" * 50)
print("RÉSULTATS DE LA RECHERCHE (TOP 3)")
print("=" * 50)

if not search_results.points:
    print("[WARN] Aucun document pertinent trouvé.")
else:
    for i, point in enumerate(search_results.points):
        source = point.payload.get("source", "Inconnue")
        score = round(point.score, 4)
        texte = point.payload.get("texte", "")

        # Extraction dynamique de la hiérarchie des titres
        t1 = point.payload.get("Titre_1") or ""
        t2 = point.payload.get("Titre_2")
        t3 = point.payload.get("Titre_3")

        hierarchie = [t for t in [t1, t2, t3] if t is not None]
        chemin_contexte = " > ".join(hierarchie) if hierarchie else "Racine du document"

        print(
            f"\n[{i+1}] Source : {source} | Titre : {chemin_contexte} | Score MaxSim : {score}"
        )
        print("-" * 50)
        print(f"{texte[:2000]}...\n")
