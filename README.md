# 🏦 FinRAG : Stack RAG Multimodal & Souverain

Ce projet est PoC visant à tester et mettre en place une architecture **RAG (Retrieval-Augmented Generation) Multimodale** spécialisée dans l'ingestion de données bancaires et financières complexes (rapports annuels, bilans comptables, etc.).

L'objectif principal est la **souveraineté des données** : l'intégralité du pipeline (de l'extraction du PDF jusqu'à l'indexation vectorielle) est conçue pour fonctionner **100 % en local (Air-Gapped)**, sans aucune fuite de données vers des API cloud externes.

## 🧠 Modèles d'IA Embarqués (100% Locaux)
Pour garantir cette souveraineté, le pipeline s'appuie sur **6 modèles d'Intelligence Artificielle** exécutés localement sur la machine :

1. **Modèles de Vision & OCR (Via RapidOCR) :**
   * `ch_PP-OCRv4_det_mobile` : Détection des zones de texte.
   * `ch_ptocr_mobile_v2.0_cls_mobile` : Classification et orientation du texte.
   * `ch_PP-OCRv4_rec_mobile` : Reconnaissance optique des caractères (lecture).
2. **Modèles d'Architecture Visuelle (Via Docling / Hugging Face) :**
   * `docling-layout-heron` : Analyse avancée de la mise en page (titres, paragraphes).
   * `docling-models` (*TableFormer*) : Reconstitution structurelle des tableaux complexes.
3. **Modèle de Vectorisation (Via SentenceTransformers) :**
   * `BAAI/bge-m3` : Transformation du texte en embeddings (vecteurs de 1024 dimensions, support multilingue et multi-granularité jusqu'à 8192 tokens).

## ⚙️ La Stack Technique

Cette stack a été pensée pour préserver l'intégrité des tableaux financiers et préparer le terrain pour une analyse IA poussée.

### 1. Extraction & Parsing (Multimodal)
* **[Docling](https://github.com/DS4SD/docling) :** Cœur du parsing. Transforme les PDF financiers complexes en format Markdown propre tout en préservant la structure stricte des tableaux.
* **Extraction des images :** Isolation physique des graphiques avec un système de métadonnées (mapping par numéro de page) pour une analyse ultérieure par un LLM visuel.

### 2. Chunking Intelligent (Sémantique & Hiérarchique)
* **[LangChain](https://python.langchain.com/) :** Utilisation des text splitters (`MarkdownHeaderTextSplitter` et `RecursiveCharacterTextSplitter`).
* **Stratégie :** Découpage basé sur les balises Markdown (`#`, `##`) pour conserver le contexte des titres, avec de grandes fenêtres de caractères (jusqu'à 4000) pour ne jamais scinder un tableau comptable en deux.

### 3. Base de Données Vectorielle
* **[Qdrant](https://qdrant.tech/) :** Moteur de recherche vectorielle ultra-rapide.
* **Déploiement :** Hébergé en local via conteneur **Docker** (`localhost:6333`).
* **Stockage :** Les vecteurs, métadonnées (titres, source) et le texte brut sont sauvegardés de manière persistante sur le disque via un *Bind Mount* Docker.
