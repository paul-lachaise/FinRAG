from finrag.services.config import PATH_JSON, PATH_IMAGES_DIR
from chunking import chunk_document
from vision_processor import process_images_to_chunks
from embed_index import (
    load_model,
    get_qdrant_client,
    ensure_collection,
    build_points,
    upsert_points,
)


def main():
    print("==========================================")
    print("[1] Découpage du texte (Docling JSON)...")
    chunks_texte = chunk_document(PATH_JSON)
    print(f" -> {len(chunks_texte)} chunks de texte créés.")

    print("\n[2] Analyse des images (LLM Vision)...")
    chunks_images = process_images_to_chunks(PATH_IMAGES_DIR)
    print(f" -> {len(chunks_images)} images analysées et transformées en texte.")

    # FUSION MAGIQUE
    chunks_globaux = chunks_texte + chunks_images
    print(f"\n[INFO] Total à indexer : {len(chunks_globaux)} éléments hybrides.")

    print("\n[3] Chargement du modèle BGE-M3...")
    model = load_model()

    print("[4] Encodage Vectoriel Multi-Vecteurs (ColBERT inclus)...")
    # On donne l'ensemble des textes (texte pur + descriptions d'images) au modèle
    textes_purs = [chunk["texte"] for chunk in chunks_globaux]
    embeddings = model.encode(
        textes_purs,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )

    print("[5] Initialisation Qdrant...")
    client = get_qdrant_client()
    ensure_collection(client)

    print("[6] Construction des payloads et des identifiants (UUID5)...")
    # Ta fonction build_points actuelle n'a pas besoin de changer,
    # elle ingère parfaitement nos nouveaux dictionnaires unifiés.
    points = build_points(chunks_globaux, embeddings, model)

    print("[7] Injection dans Qdrant (Upsert)...")
    upsert_points(client, points)

    print("\n==========================================")
    print("[SUCCESS] Ingestion Multimodale (Texte + Images) terminée !")
    print("==========================================")


if __name__ == "__main__":
    main()
