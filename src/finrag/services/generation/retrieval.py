from typing import Dict, List, Any, Optional

import torch
from qdrant_client import QdrantClient, models
from FlagEmbedding import BGEM3FlagModel

from finrag.services.config import (
    COLLECTION_NAME,
    QDRANT_URL,
    USER_QUERY,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("[INFO] CUDA:", torch.cuda.is_available())
print("[INFO] Device:", device)

print("[INFO] Loading BGE-M3 model...")
model = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=(device == "cuda"),
)

client = QdrantClient(url=QDRANT_URL)


def encode_query(query: str):
    print(f"[INFO] Encoding query: {query}")

    emb = model.encode(
        query,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )

    dense_vec = emb["dense_vecs"].tolist()
    colbert_vec = emb["colbert_vecs"].tolist()

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


def retrieve(
    query: str,
    top_k: int = 3,
    entite: Optional[str] = None,
    annee_exacte: Optional[int] = None,
    annee_gte: Optional[int] = None,
    annee_lte: Optional[int] = None,
    trimestre: Optional[str] = None,
    doc_type: Optional[str] = None,
    format: Optional[str] = None,
    langue: Optional[str] = None,
):
    dense_vec, sparse_vec, colbert_vec = encode_query(query)

    must_conditions = []

    if entite:
        must_conditions.append(
            models.FieldCondition(key="entite", match=models.MatchValue(value=entite))
        )

    if trimestre:
        must_conditions.append(
            models.FieldCondition(
                key="trimestre", match=models.MatchValue(value=trimestre)
            )
        )

    if doc_type:
        must_conditions.append(
            models.FieldCondition(key="type", match=models.MatchValue(value=doc_type))
        )

    if format:
        must_conditions.append(
            models.FieldCondition(key="format", match=models.MatchValue(value=format))
        )

    if langue:
        must_conditions.append(
            models.FieldCondition(key="langue", match=models.MatchValue(value=langue))
        )

    if annee_exacte is not None:
        must_conditions.append(
            models.FieldCondition(
                key="annee", match=models.MatchValue(value=annee_exacte)
            )
        )
    elif annee_gte is not None or annee_lte is not None:
        range_params = {}
        if annee_gte is not None:
            range_params["gte"] = annee_gte
        if annee_lte is not None:
            range_params["lte"] = annee_lte
        must_conditions.append(
            models.FieldCondition(key="annee", range=models.Range(**range_params))
        )

    query_filter = models.Filter(must=must_conditions) if must_conditions else None

    if query_filter:
        print(f"[INFO] Filtres appliqués : {len(must_conditions)} condition(s)")

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query_filter=query_filter,
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


def pretty_print_results(points: List[Any]):
    print("\n" + "=" * 50)
    print("RÉSULTATS RAG")
    print("=" * 50)

    if not points:
        print("[WARN] Aucun résultat ne correspond à la requête ou aux filtres.")
        return

    for i, point in enumerate(points):
        payload = point.payload or {}

        entite = payload.get("entite", "Inconnue")
        annee = payload.get("annee", "????")
        trimestre = payload.get("trimestre", "")
        page = payload.get("page", "?")
        section = payload.get("section", "Racine")
        doc_type = payload.get("type", "Doc").upper()
        format_doc = payload.get("format", "texte").upper()
        langue = payload.get("langue", "?")

        score = round(point.score, 4)
        text = payload.get("texte", "")

        header = f"[{doc_type}|{format_doc}]({langue}) {entite} {annee} {trimestre} | Page {page} | {section}"

        print(f"\n[{i+1}] {header} | Score: {score}")
        print("-" * 50)
        print(text[:1500] + "...\n")


if __name__ == "__main__":
    test_query = USER_QUERY
    # docs = retrieve(test_query) # <-- Aucun filtre, cherche partout
    # docs = retrieve(test_query, entite="CASA", doc_type="texte") # <-- Filtre exact
    docs = retrieve(
        test_query, entite="BNP Paribas", annee_gte=2025
    )  # <-- Filtre avec intervalle
    pretty_print_results(docs)
