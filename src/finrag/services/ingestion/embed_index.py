from typing import List, Dict, Any
import uuid
import torch

from qdrant_client import QdrantClient
from qdrant_client import models
from FlagEmbedding import BGEM3FlagModel

from finrag.services.config import (
    COLLECTION_NAME,
    QDRANT_URL,
    NOM_DU_DOCUMENT,
    INDEX_VERSION,
    BATCH_SIZE,
)

# utiliation du GPU si disponible
device = "cuda" if torch.cuda.is_available() else "cpu"
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


# =========================
# ID DETERMINISTE (ANTI DOUBLONS)
# =========================
def make_point_id(doc_name: str, chunk_index: int, text: str) -> str:
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_DNS, f"{doc_name}-{INDEX_VERSION}-{chunk_index}-{text}"
        )
    )


# =========================
# EMBEDDING MODEL
# =========================
def load_model():
    return BGEM3FlagModel("BAAI/bge-m3", use_fp16=True if device == "cuda" else False)


# =========================
# QDRANT INIT
# =========================
def get_qdrant_client():
    return QdrantClient(url=QDRANT_URL)


def ensure_collection(client: QdrantClient):
    if client.collection_exists(COLLECTION_NAME):
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(
                size=1024,
                distance=models.Distance.COSINE,
            ),
            "colbert": models.VectorParams(
                size=1024,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            ),
        },
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )


# =========================
# ENCODING + UPSERT
# =========================
def build_points(chunks: List[Dict[str, Any]], embeddings, model):
    points = []

    for i, chunk_dict in enumerate(chunks):

        # -------- payload --------
        payload = {
            "source": NOM_DU_DOCUMENT,
            "version": INDEX_VERSION,
            "texte": chunk_dict["texte"],
            **chunk_dict["metadata"],
        }

        # -------- sparse --------
        lexical_weights = embeddings["lexical_weights"][i]
        sparse_dict = {}

        for token_str, weight in lexical_weights.items():
            token_id = model.tokenizer.convert_tokens_to_ids(token_str)

            if token_id is None:
                continue

            if token_id in sparse_dict:
                sparse_dict[token_id] = max(
                    sparse_dict[token_id],
                    float(weight),
                )
            else:
                sparse_dict[token_id] = float(weight)

        sparse_indices = list(sparse_dict.keys())
        sparse_values = list(sparse_dict.values())

        # -------- ID stable --------
        point_id = make_point_id(
            NOM_DU_DOCUMENT,
            i,
            chunk_dict["texte"],
        )

        # -------- Qdrant point --------
        points.append(
            models.PointStruct(
                id=point_id,
                payload=payload,
                vector={
                    "dense": embeddings["dense_vecs"][i].tolist(),
                    "colbert": embeddings["colbert_vecs"][i].tolist(),
                    "sparse": models.SparseVector(
                        indices=sparse_indices,
                        values=sparse_values,
                    ),
                },
            )
        )

    return points


def upsert_points(client, points):
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i : i + BATCH_SIZE]

        print(f"[UPSERT] batch {i} -> {i+len(batch)-1}")

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch,
        )
