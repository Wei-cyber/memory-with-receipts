"""Governed memory: scoped reads, versioned corrections, expiry, forgetting.

Rules this module enforces:
  * Every read is a $vectorSearch whose pre-filter is derived from the caller's
    Principal. There is no retrieval path that omits it.
  * Memories are never edited in place. A correction is a new version; the old
    one stays (status "superseded") so past receipts still resolve to exactly
    what the agent saw.
  * Forgetting is the one destructive operation: text and embedding are removed
    from every version in the lineage. A tombstone with the content hash stays
    so receipts can still say "a memory was used here, and it was forgotten".
  * Every change is written to memory_events with the actor.
"""
import hashlib
import secrets
import time
from datetime import datetime, timedelta, timezone

from . import config, embed
from .db import FOREVER, get_db
from .identity import Forbidden, Principal


def now() -> datetime:
    # Mongo stores millisecond precision; truncate so round-trips are exact.
    t = datetime.now(timezone.utc)
    return t.replace(microsecond=t.microsecond // 1000 * 1000)


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(4)}"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def log_event(type_: str, memory: dict, actor: str, **detail) -> None:
    get_db().memory_events.insert_one({
        "at": now(), "type": type_, "memory_id": memory["_id"], "lineage_id": memory["lineage_id"],
        "team": memory["team"], "audience": memory["audience"], "actor": actor, "detail": detail,
    })


def embed_text(text: str, service: str | None) -> str:
    return f"[{service}] {text}" if service else text


# ---------------------------------------------------------------- write

def write(p: Principal, text: str, *, audience: list[str] | None = None, service: str | None = None,
          kind: str = "fix", ttl_days: int | None = 90, suggested_action: dict | None = None,
          avoid_action: dict | None = None, source_receipt: str | None = None,
          _lineage: tuple[str, int] | None = None, _event: str = "created", _wait: bool = True) -> dict:
    audience = audience or [f"team:{p.team}"]
    if not p.can_write(audience):
        raise Forbidden(f"{p.id} may not write memory for {audience}")
    created = now()
    mem_id = new_id("mem")
    lineage_id, version = _lineage or (mem_id, 1)
    doc = {
        "_id": mem_id, "lineage_id": lineage_id, "version": version,
        "audience": audience, "team": p.team or "org", "status": "active", "kind": kind,
        "service": service, "text": text, "content_hash": content_hash(text),
        "embedding": embed.embed([embed_text(text, service)], "document")[0], "embedder": embed.name(),
        "suggested_action": suggested_action, "avoid_action": avoid_action,
        "created_at": created, "created_by": p.id, "source_receipt": source_receipt,
        "ttl_days": ttl_days, "expires_at": created + timedelta(days=ttl_days) if ttl_days else FOREVER,
        "stats": {"shown": 0, "cited": 0, "causal": 0, "wins": 0, "strikes": 0},
    }
    get_db().memories.insert_one(doc)
    log_event(_event, doc, p.id, version=version, ttl_days=ttl_days, source_receipt=source_receipt)
    if _wait:
        wait_indexed(doc)
    return doc


def wait_indexed(doc: dict, timeout: float = 8.0) -> bool:
    """Atlas vector indexes are eventually consistent. Block (briefly) until a
    new memory is searchable so 'write, then act' behaves as people expect."""
    deadline = time.time() + timeout
    pipeline = [
        {"$vectorSearch": {"index": config.VECTOR_INDEX, "path": "embedding", "queryVector": doc["embedding"],
                           "numCandidates": 20, "limit": 5,
                           "filter": {"audience": {"$in": doc["audience"]}}}},
        {"$match": {"_id": doc["_id"]}}, {"$project": {"_id": 1}},
    ]
    while time.time() < deadline:
        if list(get_db().memories.aggregate(pipeline)):
            return True
        time.sleep(0.25)
    return False


# ---------------------------------------------------------------- read

def sweep_expired() -> int:
    """Mark lapsed memories. Retrieval never depends on this having run - the
    expires_at pre-filter is evaluated at query time - it keeps the vault honest."""
    db = get_db()
    n = 0
    for doc in db.memories.find({"status": "active", "expires_at": {"$lte": now()}}):
        db.memories.update_one({"_id": doc["_id"], "status": "active"}, {"$set": {"status": "expired"}})
        log_event("expired", doc, "system:ttl", expired_at=doc["expires_at"])
        n += 1
    return n


def read_filter(p: Principal, at: datetime) -> dict:
    """The only filter the agent can search with: what this principal may read,
    that is still in force, right now."""
    return {"$and": [
        {"audience": {"$in": p.scopes}},
        {"status": "active"},
        {"expires_at": {"$gt": at}},
    ]}


def retrieve(p: Principal, query: str, k: int = config.TOP_K) -> dict:
    db = get_db()
    sweep_expired()
    at = now()
    qvec = embed.embed([query], "query")[0]
    flt = read_filter(p, at)
    shown = list(db.memories.aggregate([
        {"$vectorSearch": {"index": config.VECTOR_INDEX, "path": "embedding", "queryVector": qvec,
                           "numCandidates": 200, "limit": k, "filter": flt}},
        # The index lags writes by a moment. Re-check the same rule against the
        # live document so a memory corrected or forgotten a second ago cannot
        # slip through on a stale index entry.
        {"$match": flt},
        {"$set": {"score": {"$meta": "vectorSearchScore"}}},
        {"$project": {"embedding": 0}},
    ]))

    # What the scope filter kept out: of the k most relevant live memories
    # org-wide, how many was this principal not allowed to see? Counts and
    # owning scopes only - content never leaves the database.
    live = {"$and": [{"status": "active"}, {"expires_at": {"$gt": at}}]}
    unscoped = list(db.memories.aggregate([
        {"$vectorSearch": {"index": config.VECTOR_INDEX, "path": "embedding", "queryVector": qvec,
                           "numCandidates": 200, "limit": k, "filter": live}},
        {"$match": live},
        {"$project": {"audience": 1}},
    ]))
    by_scope: dict[str, int] = {}
    for m in unscoped:
        if not set(m["audience"]) & set(p.scopes):
            for scope in m["audience"]:
                by_scope[scope] = by_scope.get(scope, 0) + 1
    return {
        "at": at, "query": query, "k": k, "embedder": embed.name(), "index": config.VECTOR_INDEX,
        "filter": {"audience": {"$in": p.scopes}, "status": "active", "expires_at": {"$gt": at.isoformat()}},
        "shown": shown,
        "withheld": {"count": sum(1 for m in unscoped if not set(m["audience"]) & set(p.scopes)),
                     "of_top": len(unscoped), "by_scope": by_scope},
    }


def list_visible(p: Principal, include_inactive: bool = True) -> list[dict]:
    sweep_expired()
    q: dict = {"audience": {"$in": p.scopes}}
    if not include_inactive:
        q["status"] = "active"
    return list(get_db().memories.find(q, {"embedding": 0}).sort("created_at", -1))


def get_visible(p: Principal, memory_id: str) -> dict | None:
    return get_db().memories.find_one({"_id": memory_id, "audience": {"$in": p.scopes}}, {"embedding": 0})


# ---------------------------------------------------------------- correct / forget

def _curatable(p: Principal, memory_id: str) -> dict:
    doc = get_db().memories.find_one({"_id": memory_id}, {"embedding": 0})
    # Same answer for "does not exist" and "not yours": no existence oracle.
    if not doc or not set(doc["audience"]) & set(p.scopes):
        raise Forbidden("memory not found")
    if not p.can_curate(doc["audience"]):
        raise Forbidden(f"{p.id} may not curate memory for {doc['audience']}")
    return doc


def correct(p: Principal, memory_id: str, new_text: str, reason: str, *,
            suggested_action: dict | None = None, avoid_action: dict | None = None,
            ttl_days: int | None = None) -> dict:
    old = _curatable(p, memory_id)
    if old["status"] not in ("active", "quarantined"):
        raise ValueError(f"cannot correct a memory that is {old['status']}")
    db = get_db()
    db.memories.update_one({"_id": old["_id"]}, {"$set": {"status": "superseded", "superseded_at": now()}})
    new = write(p, new_text, audience=old["audience"], service=old.get("service"), kind=old.get("kind", "fix"),
                ttl_days=ttl_days if ttl_days is not None else old.get("ttl_days"),
                suggested_action=suggested_action, avoid_action=avoid_action,
                _lineage=(old["lineage_id"], old["version"] + 1), _event="corrected")
    db.memories.update_one({"_id": old["_id"]}, {"$set": {"superseded_by": new["_id"]}})
    log_event("superseded", old, p.id, by=new["_id"], reason=reason)
    return new


def forget(p: Principal, memory_id: str, reason: str) -> int:
    doc = _curatable(p, memory_id)
    db = get_db()
    at = now()
    n = 0
    for version in db.memories.find({"lineage_id": doc["lineage_id"]}, {"embedding": 0}):
        db.memories.update_one({"_id": version["_id"]}, {
            "$set": {"status": "forgotten", "forgotten_at": at, "forgotten_by": p.id, "expires_at": at},
            "$unset": {"text": "", "embedding": "", "suggested_action": "", "avoid_action": ""},
        })
        log_event("forgotten", version, p.id, reason=reason)
        n += 1
    return n


def release(p: Principal, memory_id: str, reason: str) -> None:
    doc = _curatable(p, memory_id)
    if doc["status"] != "quarantined":
        raise ValueError("memory is not quarantined")
    get_db().memories.update_one({"_id": doc["_id"]}, {"$set": {"status": "active", "stats.strikes": 0}})
    log_event("released", doc, p.id, reason=reason)


# ---------------------------------------------------------------- feedback from outcomes

def record_use(shown: list[str], cited: list[str], causal: list[str]) -> None:
    db = get_db()
    for ids, field in ((shown, "shown"), (cited, "cited"), (causal, "causal")):
        if ids:
            db.memories.update_many({"_id": {"$in": ids}}, {"$inc": {f"stats.{field}": 1}})


def reinforce(memory_ids: list[str], receipt_id: str, actor: str) -> None:
    """A memory that just helped resolve an incident earns a fresh lease."""
    db = get_db()
    for doc in db.memories.find({"_id": {"$in": memory_ids}, "status": "active"}, {"embedding": 0}):
        update: dict = {"$inc": {"stats.wins": 1}}
        if doc.get("ttl_days"):
            update["$set"] = {"expires_at": now() + timedelta(days=doc["ttl_days"])}
        db.memories.update_one({"_id": doc["_id"]}, update)
        log_event("reinforced", doc, actor, receipt=receipt_id)


def strike(memory_ids: list[str], receipt_id: str, actor: str) -> list[str]:
    """A memory the agent relied on, and the action failed. Enough strikes and
    it is pulled from retrieval until a human reviews it."""
    db = get_db()
    quarantined = []
    for doc in db.memories.find({"_id": {"$in": memory_ids}, "status": "active"}, {"embedding": 0}):
        strikes = doc["stats"].get("strikes", 0) + 1
        update: dict = {"$inc": {"stats.strikes": 1}}
        if strikes >= config.QUARANTINE_STRIKES:
            update["$set"] = {"status": "quarantined"}
            quarantined.append(doc["_id"])
        db.memories.update_one({"_id": doc["_id"]}, update)
        log_event("quarantined" if doc["_id"] in quarantined else "struck", doc, actor,
                  receipt=receipt_id, strikes=strikes)
    return quarantined
