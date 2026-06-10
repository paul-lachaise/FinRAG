from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class TrimestreEnum(str, Enum):
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"
    T4 = "T4"


class FormatEnum(str, Enum):
    texte = "texte"
    image = "image"


class LangueEnum(str, Enum):
    ANGLAIS = "EN"
    FRANCAIS = "FR"
    ALLEMAND = "DE"


# ==========================================
# SCHÉMA POUR L'INGESTION
# ==========================================
class IngestionPayload(BaseModel):
    source: str = Field(
        ..., description="Nom du document brut d'origine (ex: 'BNPP_CP_T4-25_FR')"
    )
    version: str = Field("v1", description="Version de l'indexation")
    entite: Optional[str] = Field(
        None, description="Nom de l'entité bancaire (ex: 'BNP Paribas')"
    )
    type: Optional[str] = Field(
        None, description="Type de document (ex: 'Communiqué de presse', 'DEU')"
    )
    annee: Optional[int] = Field(None, description="Année de l'exercice (ex: 2025)")
    langue: Optional[LangueEnum] = Field(
        None, description="Langue du document (FR, EN, DE)"
    )
    trimestre: Optional[TrimestreEnum] = Field(
        None, description="Trimestre concerné (T1 à T4)"
    )
    format: Optional[FormatEnum] = Field(
        None, description="Format du chunk extrait ('texte' ou 'image')"
    )
    page: Optional[str] = Field(
        None, description="Numéro de la page physique dans le PDF"
    )
    section: Optional[str] = Field(
        None, description="Chemin hiérarchique de la section (ex: 'Titre 1 > Titre 2')"
    )
    texte: Optional[str] = Field(
        None,
        description="Contenu textuel de l'extrait ou description détaillée de l'image",
    )
    image_path: Optional[str] = Field(
        "", description="Chemin local de l'image si le format est 'image'"
    )


# ==========================================
# SCHÉMA POUR LA REQUÊTE UTILISATEUR
# ==========================================
class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="La question posée au RAG (ex: 'Quel est le résultat net ?')",
    )
    top_k: int = Field(
        3, ge=1, le=10, description="Nombre maximum de documents pertinents à récupérer"
    )

    # Filtres optionnels
    entite: Optional[str] = Field(None, description="Filtre : Entité bancaire exacte")
    annee_exacte: Optional[int] = Field(
        None, description="Filtre : Année exacte de publication"
    )
    annee_gte: Optional[int] = Field(
        None, description="Filtre : Année minimum (inclusive)"
    )
    annee_lte: Optional[int] = Field(
        None, description="Filtre : Année maximum (inclusive)"
    )
    trimestre: Optional[TrimestreEnum] = Field(
        None, description="Filtre : Trimestre spécifique (T1, T2, T3, T4)"
    )
    doc_type: Optional[str] = Field(
        None, description="Filtre : Type de document (ex: 'Communiqué de presse')"
    )
    format: Optional[FormatEnum] = Field(
        None, description="Filtre : Uniquement du texte ou uniquement des images"
    )
    langue: Optional[LangueEnum] = Field(
        None, description="Filtre : Restreindre à une langue spécifique"
    )


# ==========================================
# SCHÉMA POUR LA RÉPONSE DE L'API
# ==========================================
class DocumentResponse(BaseModel):
    source: str = Field(..., description="Fichier source du document trouvé")
    entite: Optional[str] = Field(None, description="Entité bancaire concernée")
    type: Optional[str] = Field(None, description="Type du document analysé")
    annee: Optional[int] = Field(None, description="Année de l'exercice financier")
    langue: Optional[str] = Field(None, description="Langue du document")
    trimestre: Optional[str] = Field(None, description="Trimestre de publication")
    format: Optional[str] = Field(None, description="Format du chunk (texte ou image)")
    page: Optional[str] = Field(
        None, description="Numéro de la page où se trouve l'information"
    )
    section: Optional[str] = Field(
        None, description="Titre de la section correspondante"
    )
    texte: str = Field(..., description="Extrait de texte contenant la réponse")
    score: float = Field(
        ...,
        description="Score de pertinence hybride calculé par le moteur de recherche",
    )


# ==========================================
# SCHÉMA POUR LA RÉPONSE FINALE DU LLM
# ==========================================
class SourceCitation(BaseModel):
    document: str = Field(
        ..., description="Nom du document source (ex: 'BNPP_CP_T4-25_FR')"
    )
    page: str = Field(..., description="Numéro de la page utilisée")
    entite: Optional[str] = Field(None, description="Nom de l'entité concernée")


class RAGFinalAnswer(BaseModel):
    reponse_texte: str = Field(
        ..., description="La réponse complète générée par le LLM, formatée en Markdown."
    )
    sources_utilisees: List[SourceCitation] = Field(
        ...,
        description="Liste stricte et structurée des sources utilisées pour rédiger la réponse.",
    )
