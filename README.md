# FinRAG : Multimodal RAG for Financial Documents

> **Project Context:** This repository is a **personal sandbox and Proof of Concept (PoC)** developed in parallel with my internship. It serves as an independent testing ground to experiment with cutting-edge RAG architectures, validate complex ideas, and evaluate solutions that can later be securely reimplemented in a corporate environment.

The ultimate objective of this project is to build an **autonomous Multimodal RAG Agent** capable of reasoning over complex, highly sensitive banking and financial data (annual reports, financial statements, balance sheets, tables, and embedded graphs).

### The Core Need: Absolute Confidentiality
Because financial analysis often involves **highly confidential and proprietary documents**, data sovereignty is the absolute priority. This specific requirement dictated the choice of the entire technical stack: the pipeline is designed to run **100% locally (air-gapped)**, ensuring zero data leakage to external cloud providers or third-party APIs.

---

## Embedded AI Models (100% Local)
To guarantee data sovereignty and privacy, the pipeline relies on **6 AI models** executed locally on the machine:

1. **Vision & OCR Models (via RapidOCR):**
   * `ch_PP-OCRv4_det_mobile`: Text region detection.
   * `ch_ptocr_mobile_v2.0_cls_mobile`: Text classification and orientation.
   * `ch_PP-OCRv4_rec_mobile`: Optical Character Recognition (text reading).

2. **Visual Architecture Models (via Docling / Hugging Face):**
   * `docling-layout-heron`: Advanced layout analysis (titles, paragraphs, structure).
   * `docling-models` (*TableFormer*): Structural reconstruction of complex financial tables.

3. **Multi-Space Embedding Model (via FlagEmbedding):**
   * `BAAI/bge-m3`: All-in-one model executed via the official BAAI library, generating three complementary vector spaces in a single forward pass:
      * **Dense (Semantic):** 1024-dimensional global embeddings for deep semantic understanding.
      * **Sparse (Lexical):** Sparse token-weighted vectors for exact keyword and numerical matching (financial codes, precise percentages).
      * **ColBERT (Late Interaction):** 1024-dimensional token-level embeddings enabling fine-grained token alignment via Qdrant `MAX_SIM`.

---

## Technical Stack

The selected stack balances complex document parsing with strict data security boundaries.

### 1. Extraction & Parsing (Multimodal Ingestion)
* **[Docling](https://github.com/DS4SD/docling):** Core document layout analysis engine. Converts complex financial PDFs into clean Markdown while preserving strict multi-page table structures.
* **Visual Asset Tracking:** Automatic physical extraction of charts, graphs, and images with an associated metadata page-level mapping system. This prepares the ground for multimodal analysis using local vision LLMs.

### 2. Intelligent Chunking (Semantic & Hierarchical)
* **[LangChain](https://python.langchain.com/):** Orchestration of contextual text splitters (`MarkdownHeaderTextSplitter`, `RecursiveCharacterTextSplitter`).
* **Strategy:** Document slicing based on Markdown headers (`#`, `##`) to preserve local context. Chunk sizes are adapted (up to 4000 characters) to ensure that tightly coupled financial tables are never broken apart mid-page.

### 3. Vector Database & Storage
* **[Qdrant](https://qdrant.tech/):** High-performance vector search engine optimized for hybrid and multi-vector search (Dense + Sparse + ColBERT).
* **Deployment:** Fully local deployment via Docker container (`localhost:6333`).
* **Persistence:** Embeddings, raw payloads, and business metadata (entity, year, quarter, format) are fully persisted on disk using Docker bind mounts.
