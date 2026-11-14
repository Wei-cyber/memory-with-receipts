"""HTTP API + the single-page UI.

    uvicorn receipts.api:app --reload
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from . import agent, brain, embed, memory, voice
from .db import FOREVER, ensure_schema, get_db
from .identity import Forbidden, Principal, agent_for, resolve

STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_schema()
    if get_db().principals.count_documents({}) == 0:
        from scripts.seed import seed
        seed(quiet=True)
    yield


app = FastAPI(title="Receipts", lifespan=lifespan)


@app.exception_handler(Forbidden)
def _forbidden(_: Request, exc: Forbidden):
    return JSONResponse({"detail": str(exc)}, status_code=403)


@app.exception_handler(ValueError)
def _bad(_: Request, exc: ValueError):
    return JSONResponse({"detail": str(exc)}, status_code=400)


def me(x_principal: str | None = Header(default=None)) -> Principal:
    return resolve(x_principal)


def _mem_view(m: dict) -> dict:
    m = dict(m)
    m.pop("embedding", None)
    if m.get("expires_at") and m["expires_at"] >= FOREVER:
        m["expires_at"] = None
    return m


def _receipt_view(viewer: Principal, r: dict) -> dict:
    """Join memory text onto a receipt - but only the memories this viewer may
    read. An auditor can verify what happened without being shown a team's notes."""
    db = get_db()
    ids = [s["memory_id"] for s in r["retrieval"]["shown"]]
    docs = {d["_id"]: d for d in db.memories.find({"_id": {"$in": ids}}, {"embedding": 0})}
    shown = []
    for s in r["retrieval"]["shown"]:
        d = docs.get(s["memory_id"], {})
        readable = bool(set(s["audience"]) & set(viewer.scopes))
        shown.append({
            **s, "status_now": d.get("status", "deleted"), "kind": d.get("kind"),
            "superseded_by": d.get("superseded_by"),
            "text": d.get("text") if readable else None,
            "withheld_from_viewer": not readable,
            "cited": s["memory_id"] in r["decision"]["cited"],
            "causal": s["memory_id"] in r["influence"]["causal_memory_ids"],
        })
    return {**r, "retrieval": {**r["retrieval"], "shown": shown}}


# ---------------------------------------------------------------- meta

@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/meta")
def meta():
    from scripts.seed import SCENARIOS
    db = get_db()
    return {
        "principals": [resolve(p["_id"]).public() for p in db.principals.find().sort("_id", 1)],
        "scenarios": SCENARIOS, "actions": brain.ACTIONS,
        "brain": brain.name(), "embedder": embed.name(), "database": db.name,
        "voice": "elevenlabs" if voice.enabled() else "browser",
    }


@app.post("/api/reset")
def reset():
    from scripts.seed import seed
    seed(quiet=True)
    return {"ok": True}


# ---------------------------------------------------------------- alerts & receipts

class AlertIn(BaseModel):
    team: str
    service: str
    title: str
    body: str = ""
    severity: str = "high"


@app.post("/api/alerts")
def fire_alert(alert: AlertIn, viewer: Principal = Depends(me)):
    receipt = agent.handle_alert(agent_for(alert.team), alert.model_dump(), requested_by=viewer.id)
    if not viewer.can_read_receipts_of(alert.team):
        # Anyone can page a team; only that team (and audit) reads what its agent did.
        return {"_id": receipt["_id"], "routed_to": f"agent:{alert.team}", "readable": False}
    return _receipt_view(viewer, receipt)


@app.get("/api/receipts")
def list_receipts(viewer: Principal = Depends(me)):
    q = {} if viewer.has("auditor") else {"principal.team": viewer.team}
    cur = get_db().receipts.find(q, {"retrieval.shown": 0, "counterfactual": 0}).sort("seq", -1).limit(50)
    return list(cur)


@app.get("/api/receipts/verify")
def verify(viewer: Principal = Depends(me)):
    return agent.verify_chain()


@app.get("/api/receipts/{receipt_id}")
def get_receipt(receipt_id: str, viewer: Principal = Depends(me)):
    return _receipt_view(viewer, agent.get_receipt(viewer, receipt_id))


@app.get("/api/receipts/{receipt_id}/page")
def page_text(receipt_id: str, viewer: Principal = Depends(me)):
    """The spoken on-call page for this receipt, as this viewer may hear it."""
    view = _receipt_view(viewer, agent.get_receipt(viewer, receipt_id))
    return {"text": voice.script(view), "voice": "elevenlabs" if voice.enabled() else "browser"}


@app.get("/api/receipts/{receipt_id}/page/audio")
def page_audio(receipt_id: str, viewer: Principal = Depends(me)):
    if not voice.enabled():
        raise HTTPException(503, "ELEVENLABS_API_KEY is not set")
    view = _receipt_view(viewer, agent.get_receipt(viewer, receipt_id))
    try:
        return Response(voice.speak(voice.script(view)), media_type="audio/mpeg")
    except Exception as exc:  # the browser falls back to its own voice
        raise HTTPException(502, str(exc))


class OutcomeIn(BaseModel):
    status: str
    note: str = ""


@app.post("/api/receipts/{receipt_id}/outcome")
def outcome(receipt_id: str, body: OutcomeIn, viewer: Principal = Depends(me)):
    return _receipt_view(viewer, agent.record_outcome(viewer, receipt_id, body.status, body.note))


@app.post("/api/receipts/{receipt_id}/replay")
def replay(receipt_id: str, viewer: Principal = Depends(me)):
    out = agent.replay(viewer, receipt_id)
    return {"diff": out["diff"], "original": _receipt_view(viewer, out["original"]),
            "replay": _receipt_view(viewer, out["replay"])}


# ---------------------------------------------------------------- memory vault

class ActionRef(BaseModel):
    action: str
    target: str


class MemoryIn(BaseModel):
    text: str
    audience: list[str] | None = None
    service: str | None = None
    kind: str = "fix"
    ttl_days: int | None = 90
    suggested_action: ActionRef | None = None


class CorrectIn(BaseModel):
    text: str
    reason: str
    suggested_action: ActionRef | None = None
    ttl_days: int | None = None


class ReasonIn(BaseModel):
    reason: str


@app.get("/api/memories")
def list_memories(viewer: Principal = Depends(me)):
    return [{**_mem_view(m), "can_curate": viewer.can_curate(m["audience"])} for m in memory.list_visible(viewer)]


@app.post("/api/memories")
def add_memory(body: MemoryIn, viewer: Principal = Depends(me)):
    sa = body.suggested_action.model_dump() if body.suggested_action else None
    return _mem_view(memory.write(viewer, body.text, audience=body.audience, service=body.service,
                                  kind=body.kind, ttl_days=body.ttl_days, suggested_action=sa))


@app.post("/api/memories/{memory_id}/correct")
def correct_memory(memory_id: str, body: CorrectIn, viewer: Principal = Depends(me)):
    sa = body.suggested_action.model_dump() if body.suggested_action else None
    return _mem_view(memory.correct(viewer, memory_id, body.text, body.reason,
                                    suggested_action=sa, ttl_days=body.ttl_days))


@app.post("/api/memories/{memory_id}/forget")
def forget_memory(memory_id: str, body: ReasonIn, viewer: Principal = Depends(me)):
    return {"versions_erased": memory.forget(viewer, memory_id, body.reason)}


@app.post("/api/memories/{memory_id}/release")
def release_memory(memory_id: str, body: ReasonIn, viewer: Principal = Depends(me)):
    memory.release(viewer, memory_id, body.reason)
    return {"ok": True}


@app.get("/api/events")
def events(viewer: Principal = Depends(me)):
    q = {} if viewer.has("auditor") else {"audience": {"$in": viewer.scopes}}
    return list(get_db().memory_events.find(q, {"_id": 0}).sort("at", -1).limit(40))
