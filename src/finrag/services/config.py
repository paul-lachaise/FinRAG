from pathlib import Path

# =========================
# CONFIG GLOBAL
# =========================

NOM_DU_DOCUMENT = "BNPP_CP_T4-25_FR"
# NOM_DU_DOCUMENT = "CASA_CP_T1-26_FR"

COLLECTION_NAME = "financial_reports"
QDRANT_URL = "http://localhost:6333"

PATH_MARKDOWN = Path(f"assets/parsed/{NOM_DU_DOCUMENT}/{NOM_DU_DOCUMENT}.md")
PATH_JSON = Path(f"assets/parsed/{NOM_DU_DOCUMENT}/{NOM_DU_DOCUMENT}.json")
PATH_IMAGES_DIR = Path(f"assets/parsed/{NOM_DU_DOCUMENT}/images")

CHUNK_SIZE = 4000
CHUNK_OVERLAP = 300

BATCH_SIZE = 2

# USER_QUERY = "Quels sont les revenus et le coût du risque pour le groupe ?"
# USER_QUERY = "Cout du risque en 2025 de la banque BNP Paribas"
USER_QUERY = "quels sont les frais de gestion de BNP CIB pour le trimestre 4 de 2025"

# versioning
INDEX_VERSION = "v1"
