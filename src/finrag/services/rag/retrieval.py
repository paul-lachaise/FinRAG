from typing import Dict, List, Any

import torch
from qdrant_client import QdrantClient, models
from FlagEmbedding import BGEM3FlagModel

from finrag.services.config import (
    COLLECTION_NAME,
    QDRANT_URL,
    USER_QUERY,
)


# =========================================================
# DEVICE
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

print("[INFO] CUDA:", torch.cuda.is_available())
print("[INFO] Device:", device)


# =========================================================
# MODEL
# =========================================================
print("[INFO] Loading BGE-M3 model...")
model = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=(device == "cuda"),
)


# =========================================================
# QDRANT CLIENT
# =========================================================
client = QdrantClient(url=QDRANT_URL)


# =========================================================
# ENCODING QUERY
# =========================================================
def encode_query(query: str):

    print(f"[INFO] Encoding query: {query}")

    emb = model.encode(
        query,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )

    # -------- dense --------
    dense_vec = emb["dense_vecs"].tolist()

    # -------- colbert --------
    colbert_vec = emb["colbert_vecs"].tolist()

    # -------- sparse --------
    lexical_weights = emb["lexical_weights"]
    sparse_dict: Dict[int, float] = {}

    for token_str, weight in lexical_weights.items():
        token_id = model.tokenizer.convert_tokens_to_ids(token_str)

        if token_id is None:
            continue

        sparse_dict[token_id] = max(
            sparse_dict.get(token_id, 0.0),
            float(weight),
        )

    sparse_vec = models.SparseVector(
        indices=list(sparse_dict.keys()),
        values=list(sparse_dict.values()),
    )

    return dense_vec, sparse_vec, colbert_vec


# =========================================================
# RETRIEVAL QDRANT
# =========================================================
def retrieve(query: str, top_k: int = 3):

    dense_vec, sparse_vec, colbert_vec = encode_query(query)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
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
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=15,
            )
        ],
        query=colbert_vec,
        using="colbert",
        limit=top_k,
        with_payload=True,
    )

    return results.points


# =========================================================
# DEBUG PRINT
# =========================================================
def pretty_print_results(points: List[Any]):

    print("\n" + "=" * 50)
    print("RÉSULTATS RAG")
    print("=" * 50)

    if not points:
        print("[WARN] Aucun résultat.")
        return

    for i, point in enumerate(points):

        payload = point.payload or {}

        source = payload.get("source", "unknown")
        score = round(point.score, 4)

        t1 = payload.get("Titre_1", "")
        t2 = payload.get("Titre_2")
        t3 = payload.get("Titre_3")

        hierarchy = [t for t in [t1, t2, t3] if t]
        context = " > ".join(hierarchy) if hierarchy else "root"

        text = payload.get("texte", "")

        print(f"\n[{i+1}] {source} | {context} | score={score}")
        print("-" * 50)
        print(text[:1500] + "...\n")


# =========================================================
# TEST LOCAL
# =========================================================
if __name__ == "__main__":

    test_query = USER_QUERY

    docs = retrieve(test_query)

    pretty_print_results(docs)
