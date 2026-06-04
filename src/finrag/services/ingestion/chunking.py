import json
from typing import List, Dict, Any
from pathlib import Path

from docling.chunking import HierarchicalChunker
from docling.datamodel.document import DoclingDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from finrag.services.config import CHUNK_SIZE, CHUNK_OVERLAP, NOM_DU_DOCUMENT


def load_docling_json(path: Path) -> DoclingDocument:
    """Recharge le document complet avec son intelligence (pages, tables) depuis le JSON."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        doc_dict = json.load(f)

    return DoclingDocument.model_validate(doc_dict)


def chunk_document(json_path: Path) -> List[Dict[str, Any]]:
    print("[INFO] Chargement de l'arbre sémantique du document...")
    docling_doc = load_docling_json(json_path)

    # 1. Chunking Sémantique Natif (Préserve les tableaux et la hiérarchie)
    chunker = HierarchicalChunker()
    semantic_chunks = chunker.chunk(docling_doc)

    # 2. Filet de stabilisation
    # Si un paragraphe narratif dépasse CHUNK_SIZE, on le coupe proprement.
    # Les tableaux extraits par Docling ne sont pas affectés et restent entiers.
    text_stabilizer = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )

    final_structured_chunks: List[Dict[str, Any]] = []

    for chunk in semantic_chunks:
        # --- A. Extraction robuste de la page ---
        pages = set()
        image_refs = []

        if hasattr(chunk.meta, "doc_items"):
            for item in chunk.meta.doc_items:
                # Récupération de la page
                if hasattr(item, "prov") and item.prov:
                    for prov in item.prov:
                        pages.add(prov.page_no)

                # Récupération des liens images (Si un chunk parle d'une image)
                if type(item).__name__ == "PictureItem":
                    # On pourrait lier dynamiquement le nom du fichier image généré ici
                    image_refs.append(f"page_{item.prov[0].page_no}")

        liste_pages = sorted(list(pages))
        page_ref = "Inconnue"
        if len(liste_pages) == 1:
            page_ref = str(liste_pages[0])
        elif len(liste_pages) > 1:
            page_ref = f"{liste_pages[0]}-{liste_pages[-1]}"

        # --- B. Extraction de la Section (Breadcrumb) ---
        headings = (
            chunk.meta.headings
            if hasattr(chunk.meta, "headings") and chunk.meta.headings
            else []
        )
        section_complete = " > ".join(headings) if headings else "Racine du document"

        # --- C. Normalisation et Stabilisation ---
        # Le chunk.text par défaut de Docling normalise déjà très bien les tableaux (en Markdown enrichi)
        texte_brut = chunk.text

        # Split uniquement si le texte est gigantesque ET n'est pas un tableau
        if len(texte_brut) > CHUNK_SIZE and "|-" not in texte_brut[:500]:
            sub_texts = text_stabilizer.split_text(texte_brut)
        else:
            sub_texts = [texte_brut]

        for sub_text in sub_texts:
            # Rejet des micro-morceaux (bruit de parsing)
            if len(sub_text.strip()) < 50:
                continue

            final_structured_chunks.append(
                {
                    "texte": sub_text,
                    "metadata": {
                        "source": NOM_DU_DOCUMENT,
                        "type": "texte",  # <-- Clé unifiée
                        "section": section_complete,
                        "page": page_ref,
                        "image_path": "",  # <-- Vide pour le texte, mais la clé existe !
                    },
                }
            )
    print(f"[SUCCESS] {len(final_structured_chunks)} chunks stabilisés générés.")
    return final_structured_chunks
