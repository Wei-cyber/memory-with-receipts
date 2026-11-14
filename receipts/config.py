import os

from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/?directConnection=true")
DB_NAME = os.getenv("RECEIPTS_DB", "receipts")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or None
MODEL = os.getenv("RECEIPTS_MODEL", "claude-opus-5-5")
EFFORT = os.getenv("RECEIPTS_EFFORT", "low")
BRAIN = os.getenv("RECEIPTS_BRAIN", "auto")

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY") or None
VOYAGE_MODEL = os.getenv("VOYAGE_MODEL", "voyage-3.5")
VOYAGE_BASE_URL = os.getenv("VOYAGE_BASE_URL", "https://api.voyageai.com/v1").rstrip("/")

EMBED_DIMS = 1024
VECTOR_INDEX = "memory_vec"
TOP_K = 5

# A memory that fails this many times is pulled from retrieval until a human reviews it.
QUARANTINE_STRIKES = 2
