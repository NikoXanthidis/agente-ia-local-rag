from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "documentos"
CHROMA_DIR = BASE_DIR / "vector_db"
DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "conversas.db"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# O Modelo llama3.2  foi escolhido pois eu já tinha testado no meu pc, mas pode ser outro.
# Se quiser testar um modelo maior depois, altere esta variável.
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

COLLECTION_NAME = "base_conhecimento"

# Configuração otimizada para computadores comuns.
TOP_K = 4
CHUNK_SIZE = 1100
CHUNK_OVERLAP = 150
HISTORY_MESSAGES = 6
MAX_SEMANTIC_DISTANCE = 0.75
MAX_CONTEXT_CHARS = 5200
MAX_HISTORY_ITEM_CHARS = 900

# Limita memória/KV cache do modelo.
NUM_CTX = 4096
NUM_PREDICT = 700
