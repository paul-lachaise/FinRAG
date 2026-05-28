# FinRAG : Multimodal RAG for Financial Documents

This project is a Proof of Concept aimed at testing and implementing a **Multimodal Retrieval-Augmented Generation (RAG) architecture** specialized in the ingestion of complex banking and financial data (annual reports, financial statements, balance sheets, etc.).

The main objective is **data sovereignty**: the entire pipeline (from PDF extraction to vector indexing) is designed to run **100% locally (air-gapped)**, without any leakage to external cloud APIs.

## Embedded AI Models (100% Local)
To ensure sovereignty, the pipeline relies on **6 AI models** executed locally on the machine:

1. **Vision & OCR Models (via RapidOCR):**
   * `ch_PP-OCRv4_det_mobile`: Text region detection.
   * `ch_ptocr_mobile_v2.0_cls_mobile`: Text classification and orientation.
   * `ch_PP-OCRv4_rec_mobile`: Optical Character Recognition (text reading).

2. **Visual Architecture Models (via Docling / Hugging Face):**
   * `docling-layout-heron`: Advanced layout analysis (titles, paragraphs, structure).
   * `docling-models` (*TableFormer*): Structural reconstruction of complex tables.

3. **Multi-Space Embedding Model (via FlagEmbedding):**
   * `BAAI/bge-m3`: all-in-one model executed via the official BAAI library, generating three vector spaces in a single forward pass:
      * **Dense (Semantic):** 1024-dimensional global embeddings for semantic understanding.
      * **Sparse (Lexical):** sparse token-weighted vectors for exact keyword matching (financial codes, numbers).
      * **ColBERT (Late Interaction):** 1024-dimensional token-level embeddings enabling fine-grained retrieval via Qdrant `MAX_SIM`.

## Technical Stack

This stack is designed to preserve the integrity of financial tables while enabling downstream AI analysis.

### 1. Extraction & Parsing (Multimodal)
* **[Docling](https://github.com/DS4SD/docling):** Core parsing engine. Converts complex financial PDFs into clean Markdown while preserving strict table structure.
* **Image extraction:** physical extraction of graphs with metadata (page-level mapping system) for later multimodal analysis using vision LLMs.

### 2. Intelligent Chunking (Semantic & Hierarchical)
* **[LangChain](https://python.langchain.com/):** usage of text splitters (`MarkdownHeaderTextSplitter`, `RecursiveCharacterTextSplitter`).
* **Strategy:** Markdown-based splitting using headers (`#`, `##`) to preserve context, with large chunk sizes (up to 4000 characters) to avoid splitting financial tables.

### 3. Vector Database
* **[Qdrant](https://qdrant.tech/):** high-performance vector search engine for semantic retrieval.
* **Deployment:** fully local via Docker container (`localhost:6333`).
* **Persistence:** embeddings, metadata (titles, sources), and raw text are persisted on disk using Docker bind mounts.
