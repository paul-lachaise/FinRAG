from fastapi import FastAPI, HTTPException
from typing import List

from finrag.api.schemas import QueryRequest, DocumentResponse
from finrag.services.generation.retrieval import retrieve, client
from qdrant_client import models
from finrag.services.config import COLLECTION_NAME

# from finrag.services.generation import generate_rag_answer

app = FastAPI(
    title="FinRAG API",
    description="""
    Moteur de recherche hybride (Texte et Images) dédié à l'analyse financière.

    Cette API permet de :
    - Filtrer des documents avec précision (Entité, Année, Trimestre, Langue).
    - Effectuer des recherches sémantiques via BGE-M3 et ColBERT.
    - Générer des réponses sourcées via un modèle de langage (LLM).
    """,
    version="1.0.0",
)


@app.get(
    "/health",
    tags=["Système"],
    summary="Vérification profonde de l'état de l'API et de la collection Qdrant",
)
def health_check():
    api_status = "opérationnel"
    qdrant_status = "indisponible"

    try:
        if client.collection_exists(COLLECTION_NAME):
            collection_info = client.get_collection(COLLECTION_NAME)
            qdrant_status = str(collection_info.status).upper()
        else:
            qdrant_status = "COLLECTION_INTROUVABLE"
    except Exception as e:
        qdrant_status = f"ERREUR_CONNEXION: {str(e)}"

    return {
        "api": api_status,
        "base_de_donnees_qdrant": qdrant_status,
        "version": "1.0.0",
    }


@app.et(
    "/api/v1/stats",
    tags=["Système"],
    summary="Statistiques volumétriques de la base de connaissances",
)
def get_database_stats():
    try:
        # Vérification si la collection existe pour éviter un crash
        if not client.collection_exists(COLLECTION_NAME):
            raise HTTPException(
                status_code=404,
                detail=f"La collection '{COLLECTION_NAME}' n'existe pas.",
            )

        # Comptage des chunks de texte
        count_texte = client.count(
            collection_name=COLLECTION_NAME,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="format", match=models.MatchValue(value="texte")
                    )
                ]
            ),
        ).count

        # Comptage des chunks issus d'images
        count_image = client.count(
            collection_name=COLLECTION_NAME,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="format", match=models.MatchValue(value="image")
                    )
                ]
            ),
        ).count

        return {
            "nom_collection": COLLECTION_NAME,
            "chunks_texte": count_texte,
            "chunks_image": count_image,
            "total_hybrid_elements": count_texte + count_image,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/v1/search",
    response_model=List[DocumentResponse],
    tags=["Moteur de Recherche"],
    summary="Récupérer les contextes bruts depuis Qdrant",
)
def search_documents(request: QueryRequest):
    try:
        raw_points = retrieve(**request.model_dump())

        contextes = []
        for point in raw_points:
            payload = point.payload or {}
            doc = DocumentResponse(
                source=payload.get("source", "Inconnue"),
                entite=payload.get("entite"),
                type=payload.get("type"),
                annee=payload.get("annee"),
                langue=payload.get("langue"),
                trimestre=payload.get("trimestre"),
                format=payload.get("format"),
                page=payload.get("page"),
                section=payload.get("section"),
                texte=payload.get("texte", ""),
                score=round(point.score, 4),
            )
            contextes.append(doc)

        return contextes

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# @app.post(
#     "/api/v1/ask",
#     response_model=RAGFinalAnswer,
#     tags=["Intelligence Artificielle"],
#     summary="Poser une question au RAG (Recherche vectorielle + Génération LLM)"
# )
# def ask_question(request: QueryRequest):
#     try:
#         raw_points = retrieve(**request.model_dump())

#         contextes = []
#         for point in raw_points:
#             payload = point.payload or {}
#             doc = DocumentResponse(
#                 source=payload.get("source", "Inconnue"),
#                 entite=payload.get("entite"),
#                 type=payload.get("type"),
#                 annee=payload.get("annee"),
#                 langue=payload.get("langue"),
#                 trimestre=payload.get("trimestre"),
#                 format=payload.get("format"),
#                 page=payload.get("page"),
#                 section=payload.get("section"),
#                 texte=payload.get("texte", ""),
#                 score=round(point.score, 4)
#             )
#             contextes.append(doc)

#         if not contextes:
#             return RAGFinalAnswer(
#                 reponse_texte="Je n'ai trouvé aucun document correspondant à votre recherche.",
#                 sources_utilisees=[]
#             )

#         reponse_finale = generate_rag_answer(request.query, contextes)

#         return reponse_finale

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
