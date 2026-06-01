from finrag.services.config import PATH_MARKDOWN
from chunking import load_markdown, chunk_markdown
from embed_index import (
    load_model,
    get_qdrant_client,
    ensure_collection,
    build_points,
    upsert_points,
)


def main():
    print("[1] Loading markdown...")
    md = load_markdown(PATH_MARKDOWN)

    print("[2] Chunking...")
    chunks = chunk_markdown(md)
    print(f"Chunks: {len(chunks)}")

    print("[3] Loading embedding model...")
    model = load_model()

    print("[4] Encoding...")
    embeddings = model.encode(
        [c.page_content for c in chunks],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )

    print("[5] Qdrant init...")
    client = get_qdrant_client()
    ensure_collection(client)

    print("[6] Building points...")

    # 1. Conversion des objets LangChain 'Document' en dictionnaires Python
    chunks_as_dicts = [
        {"texte": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks
    ]

    # 2. Appel de ta nouvelle fonction prête pour la production
    points = build_points(chunks_as_dicts, embeddings, model)

    print(f"[INFO] {len(points)} points ready")

    print("[7] Upserting...")
    upsert_points(client, points)

    print("[DONE] Ingestion finished")


if __name__ == "__main__":
    main()
