import json
from pathlib import Path
from typing import List, Dict, Any

from finrag.services.config import NOM_DU_DOCUMENT, INDEX_VERSION


def generer_description_vision(image_path: Path) -> str:
    """
    Fonction pour appeler ton LLM Vision (Ollama, etc.).
    """
    return f"[DESCRIPTION AUTO] Graphique ou tableau visuel extrait du document. (Fichier: {image_path.name})"


def process_images_to_chunks(images_dir: Path) -> List[Dict[str, Any]]:
    image_chunks = []

    if not images_dir.exists():
        print(f"[WARN] Le dossier {images_dir} n'existe pas.")
        return image_chunks

    # 1. Chargement du fichier de mapping (Pages & Sections)
    meta_path = images_dir / "images_meta.json"
    images_metadata = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            images_metadata = json.load(f)
    else:
        print(
            "[WARN] Fichier images_meta.json introuvable, contexte par défaut utilisé."
        )

    print(f"[INFO] Traitement LLM des images dans {images_dir}...")

    # 2. Boucle sur les images
    for image_file in images_dir.glob("*.png"):

        # Récupération du contexte exact (ou valeurs par défaut si bug)
        contexte = images_metadata.get(
            image_file.name, {"page": "Inconnue", "section": "Support Visuel"}
        )

        description = generer_description_vision(image_file)

        chunk_dict = {
            "texte": description,
            "metadata": {
                "source": NOM_DU_DOCUMENT,
                "version": INDEX_VERSION,
                "format": "image",
                "section": contexte["section"],
                "page": contexte["page"],
                "image_path": str(image_file),
            },
        }

        image_chunks.append(chunk_dict)
        print(f"  -> {image_file.name} | Sec: {contexte['section'][:30]}...")

    return image_chunks
