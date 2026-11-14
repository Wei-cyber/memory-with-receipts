"""The agent loop, and the receipt it leaves behind.

For every alert the agent:
  1. retrieves memory through the scoped vector search,
  2. decides with that memory,
  3. decides again with no memory            (the counterfactual),
  4. decides again with each cited memory removed in turn (leave-one-out),
  5. acts, and writes a receipt holding all of the above, hash-chained to the
     previous receipt.

Steps 3-4 are the proof that a memory changed behaviour rather than just
occupying the prompt: a memory is *causal* for a receipt if removing it, and
nothing else, changes the action.
"""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from pymongo import DESCENDING

from . import brain, memory
from .db import get_db
from .identity import Forbidden, Principal, agent_for

GENESIS = "0" * 64
_pool = ThreadPoolExecutor(max_workers=8)


def _same(a: dict, b: dict) -> bool:
    return (a["action"], a["target"]) == (b["action"], b["target"])


def _slim(d: dict) -> dict:
    return {"action": d["action"], "target": d["target"], "rationale": d["rationale"]}


def _canonical(obj) -> str:
    def default(o):
        if isinstance(o, datetime):
            return int(o.timestamp() * 1000)
        raise TypeError(type(o))
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=default)


def _sealed(receipt: dict) -> dict:
    """The part of a receipt that must never change. The outcome is recorded
    later by a human, so it sits outside the seal."""
    return {k: v for k, v in receipt.items() if k not in ("hash", "outcome")}


def _hash(receipt: dict) -> str:
    return hashlib.sha256(_canonical(_sealed(receipt)).encode()).hexdigest()


def handle_alert(agent: Principal, alert: dict, *, replay_of: str | None = None,
                 requested_by: str | None = None) -> dict:
    db = get_db()
    alert = {k: alert.get(k) for k in ("service", "title", "body", "severity")}
    query = f"[{alert['service']}] {alert['title']}. {alert.get('body') or ''}"

    r = memory.retrieve(agent, query)
    shown = r["shown"]

    with_memory_f = _pool.submit(brain.decide, agent.team, alert, shown)
    without_f = _pool.submit(brain.decide, agent.team, alert, []) if shown else with_memory_f
    decision, baseline = with_memory_f.result(), without_f.result()

    loo_fs = {mid: _pool.submit(brain.decide, agent.team, alert, [m for m in shown if m["_id"] != mid])
              for mid in decision["cited"]}
    leave_one_out = []
    for mid, f in loo_fs.items():
        alt = f.result()
        leave_one_out.append({"memory_id": mid, **_slim(alt), "flips": not _same(alt, decision)})

    changed = not _same(baseline, decision)
    causal = [x["memory_id"] for x in leave_one_out if x["flips"]]
    joint = changed and not causal and bool(decision["cited"])
    if joint:  # redundant memories: no single one is necessary, together they are
        causal = list(decision["cited"])

    last = db.receipts.find_one(sort=[("seq", DESCENDING)], projection={"seq": 1, "hash": 1})
    receipt = {
        "_id": memory.new_id("rcpt"),
        "seq": (last["seq"] + 1) if last else 1,
        "at": r["at"],
        "principal": {"id": agent.id, "team": agent.team, "scopes": agent.scopes},
        "requested_by": requested_by,
        "replay_of": replay_of,
        "alert": alert,
        "retrieval": {
            "query": r["query"], "index": r["index"], "embedder": r["embedder"], "k": r["k"],
            "filter": r["filter"],
            # Pointers and hashes, not text: forgetting a memory must not leave
            # a copy of it behind in the audit trail.
            "shown": [{"memory_id": m["_id"], "lineage_id": m["lineage_id"], "version": m["version"],
                       "score": round(m["score"], 4), "content_hash": m["content_hash"],
                       "audience": m["audience"]} for m in shown],
            "withheld": r["withheld"],
        },
        "decision": {**_slim(decision), "cited": decision["cited"], "brain": brain.name()},
        "counterfactual": {"without_memory": _slim(baseline), "leave_one_out": leave_one_out},
        "influence": {"changed_action": changed, "causal_memory_ids": causal, "joint": joint},
        "executed": replay_of is None,
        "prev_hash": last["hash"] if last else GENESIS,
        "outcome": {"status": "pending" if replay_of is None else "dry_run"},
    }
    receipt["hash"] = _hash(receipt)
    db.receipts.insert_one(receipt)

    if replay_of is None:
        # The action itself. In this demo the "infrastructure" is a collection.
        db.actions.insert_one({"at": receipt["at"], "receipt_id": receipt["_id"], "by": agent.id,
                               "action": decision["action"], "target": decision["target"]})
        memory.record_use([m["_id"] for m in shown], decision["cited"], causal)
    return receipt


def get_receipt(viewer: Principal, receipt_id: str) -> dict:
    receipt = get_db().receipts.find_one({"_id": receipt_id})
    if not receipt or not viewer.can_read_receipts_of(receipt["principal"]["team"]):
        raise Forbidden("receipt not found")
    return receipt


def replay(viewer: Principal, receipt_id: str) -> dict:
    """Re-run a past alert against memory as it stands now. Nothing is executed.
    If someone corrected or forgot a memory in between, this is where it shows."""
    original = get_receipt(viewer, receipt_id)
    agent = agent_for(original["principal"]["team"])
    new = handle_alert(agent, original["alert"], replay_of=original["_id"], requested_by=viewer.id)

    db = get_db()
    before = {m["memory_id"]: m for m in original["retrieval"]["shown"]}
    after = {m["memory_id"]: m for m in new["retrieval"]["shown"]}
    dropped = []
    for mid in before.keys() - after.keys():
        doc = db.memories.find_one({"_id": mid}, {"status": 1, "superseded_by": 1})
        dropped.append({"memory_id": mid, "now": doc["status"] if doc else "deleted",
                        "superseded_by": (doc or {}).get("superseded_by")})
    return {
        "original": original, "replay": new,
        "diff": {
            "action_changed": not _same(original["decision"], new["decision"]),
            "before": _slim(original["decision"]), "after": _slim(new["decision"]),
            "memories_dropped": dropped,
            "memories_added": sorted(after.keys() - before.keys()),
        },
    }


def record_outcome(viewer: Principal, receipt_id: str, status: str, note: str = "") -> dict:
    """Close the loop. The outcome is what turns a receipt into memory:
    success renews the memories that drove it (or records a new one), failure
    counts against them and leaves a caveat so the mistake is not repeated."""
    if status not in ("resolved", "failed"):
        raise ValueError("status must be 'resolved' or 'failed'")
    receipt = get_receipt(viewer, receipt_id)
    if viewer.kind != "human" or viewer.team != receipt["principal"]["team"]:
        raise Forbidden("only a human on the owning team can record an outcome")
    if receipt["outcome"]["status"] != "pending":
        raise ValueError(f"outcome already recorded: {receipt['outcome']['status']}")

    agent = agent_for(receipt["principal"]["team"])
    alert, d = receipt["alert"], receipt["decision"]
    act = {"action": d["action"], "target": d["target"]}
    learned, quarantined = None, []
    if status == "resolved":
        memory.reinforce(d["cited"], receipt_id, viewer.id)
        if not receipt["influence"]["causal_memory_ids"]:
            text = (f"{alert['service']}: \"{alert['title']}\" was resolved by {d['action']} on "
                    f"{d['target']}." + (f" {note}" if note else ""))
            learned = memory.write(agent, text, service=alert["service"], kind="fix", ttl_days=90,
                                   suggested_action=act, source_receipt=receipt_id)
    else:
        quarantined = memory.strike(d["cited"], receipt_id, viewer.id)
        text = (f"{alert['service']}: for \"{alert['title']}\", {d['action']} on {d['target']} "
                f"did NOT resolve it." + (f" {note}" if note else ""))
        learned = memory.write(agent, text, service=alert["service"], kind="caveat", ttl_days=30,
                               avoid_action=act, source_receipt=receipt_id)

    outcome = {"status": status, "note": note, "by": viewer.id, "at": memory.now(),
               "learned_memory_id": learned["_id"] if learned else None, "quarantined": quarantined}
    get_db().receipts.update_one({"_id": receipt_id}, {"$set": {"outcome": outcome}})
    return {**receipt, "outcome": outcome}


def verify_chain() -> dict:
    """Recompute every receipt hash in order. Any edit to a sealed field, or a
    removed receipt, breaks the chain from that point on."""
    prev, n = GENESIS, 0
    for receipt in get_db().receipts.find().sort("seq", 1):
        n += 1
        if receipt["prev_hash"] != prev or _hash(receipt) != receipt["hash"] or receipt["seq"] != n:
            return {"ok": False, "checked": n, "broken_at": receipt["_id"], "seq": receipt["seq"]}
        prev = receipt["hash"]
    return {"ok": True, "checked": n, "head": prev}
