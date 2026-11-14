"""Collections, schema validation and indexes. All of it lives in Atlas."""
import time
from datetime import datetime, timezone
from functools import lru_cache

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.database import Database
from pymongo.operations import SearchIndexModel

from . import config

FOREVER = datetime(9999, 1, 1, tzinfo=timezone.utc)

MEMORY_STATUSES = ["active", "superseded", "quarantined", "expired", "forgotten"]

MEMORY_SCHEMA = {
    "bsonType": "object",
    "required": [
        "_id", "lineage_id", "version", "audience", "team", "status",
        "content_hash", "created_at", "created_by", "expires_at", "stats",
    ],
    "properties": {
        "_id": {"bsonType": "string"},
        "lineage_id": {"bsonType": "string"},
        "version": {"bsonType": "int", "minimum": 1},
        # Who may read it. Every memory must name at least one scope - there is
        # no such thing as an unscoped memory.
        "audience": {
            "bsonType": "array", "minItems": 1,
            "items": {"bsonType": "string", "pattern": "^(org|team:[a-z0-9-]+)$"},
        },
        "team": {"bsonType": "string"},
        "status": {"enum": MEMORY_STATUSES},
        "kind": {"enum": ["fix", "caveat", "runbook"]},
        "text": {"bsonType": "string", "minLength": 1},
        "embedding": {"bsonType": "array"},
        "content_hash": {"bsonType": "string"},
        # Every memory expires. "Keep until told otherwise" is db.FOREVER, set on purpose.
        "expires_at": {"bsonType": "date"},
        "ttl_days": {"bsonType": ["int", "null"]},
        "created_at": {"bsonType": "date"},
        "created_by": {"bsonType": "string"},
        "stats": {"bsonType": "object"},
    },
}

RECEIPT_SCHEMA = {
    "bsonType": "object",
    "required": [
        "_id", "seq", "at", "principal", "alert", "retrieval", "decision",
        "counterfactual", "influence", "prev_hash", "hash", "outcome",
    ],
    "properties": {
        "principal": {"bsonType": "object", "required": ["id", "team", "scopes"]},
        "retrieval": {"bsonType": "object", "required": ["filter", "shown", "withheld"]},
        "decision": {"bsonType": "object", "required": ["action", "target", "cited"]},
        "influence": {"bsonType": "object", "required": ["changed_action", "causal_memory_ids"]},
        "outcome": {"bsonType": "object", "required": ["status"]},
    },
}

VECTOR_INDEX_DEFINITION = {
    "fields": [
        {"type": "vector", "path": "embedding", "numDimensions": config.EMBED_DIMS, "similarity": "cosine"},
        # Governance is enforced inside the search, not after it: these three
        # fields are pre-filters on every query the agent can issue.
        {"type": "filter", "path": "audience"},
        {"type": "filter", "path": "status"},
        {"type": "filter", "path": "expires_at"},
    ]
}


@lru_cache
def client() -> MongoClient:
    return MongoClient(config.MONGODB_URI, tz_aware=True, appname="receipts")


def get_db() -> Database:
    return client()[config.DB_NAME]


def _ensure_collection(db: Database, name: str, schema: dict | None = None) -> None:
    opts = {"validator": {"$jsonSchema": schema}, "validationLevel": "strict"} if schema else {}
    if name in db.list_collection_names():
        if schema:
            db.command("collMod", name, **opts)
    else:
        db.create_collection(name, **opts)


def ensure_schema(db: Database | None = None, wait: bool = True) -> None:
    db = db if db is not None else get_db()
    _ensure_collection(db, "memories", MEMORY_SCHEMA)
    _ensure_collection(db, "receipts", RECEIPT_SCHEMA)
    for name in ("memory_events", "principals", "actions"):
        _ensure_collection(db, name)

    db.memories.create_index([("lineage_id", ASCENDING), ("version", DESCENDING)])
    db.memories.create_index([("audience", ASCENDING), ("status", ASCENDING)])
    db.receipts.create_index([("seq", ASCENDING)], unique=True)
    db.receipts.create_index([("principal.team", ASCENDING), ("at", DESCENDING)])
    db.memory_events.create_index([("at", DESCENDING)])
    db.memory_events.create_index([("lineage_id", ASCENDING), ("at", ASCENDING)])

    existing = {ix["name"] for ix in db.memories.list_search_indexes()}
    if config.VECTOR_INDEX not in existing:
        db.memories.create_search_index(
            SearchIndexModel(definition=VECTOR_INDEX_DEFINITION, name=config.VECTOR_INDEX, type="vectorSearch")
        )
    if wait:
        wait_vector_index(db)


def wait_vector_index(db: Database, timeout: float = 120) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        ix = next(iter(db.memories.list_search_indexes(config.VECTOR_INDEX)), None)
        if ix and ix.get("queryable"):
            return
        time.sleep(1)
    raise TimeoutError(f"vector index {config.VECTOR_INDEX} not queryable after {timeout}s")
