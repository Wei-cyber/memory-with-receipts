"""End-to-end tests against a real $vectorSearch (local Atlas or the sandbox).

    docker compose up -d && pytest

Uses the offline stub brain and local embedder so runs are deterministic.
"""
import os

os.environ["RECEIPTS_DB"] = "receipts_test"
os.environ["RECEIPTS_BRAIN"] = "stub"
os.environ["VOYAGE_API_KEY"] = ""

import pytest  # noqa: E402

from receipts import agent, memory  # noqa: E402
from receipts.db import get_db  # noqa: E402
from receipts.identity import Forbidden, agent_for, resolve  # noqa: E402
from scripts.seed import SCENARIOS, seed  # noqa: E402

CHECKOUT = next(s for s in SCENARIOS if s["id"] == "checkout-latency")
SEARCH = next(s for s in SCENARIOS if s["id"] == "search-latency")
CDN = next(s for s in SCENARIOS if s["id"] == "cdn-errors")


@pytest.fixture(autouse=True)
def fresh():
    seed(quiet=True)


def texts(principal_id):
    return {m["_id"]: m for m in memory.list_visible(resolve(principal_id))}


def pool_memory():
    return next(m for m in memory.list_visible(resolve("alice"))
                if m["status"] == "active" and "pgbouncer" in m.get("text", ""))


# --- who can read each memory

def test_same_alert_different_team_sees_different_memory():
    pay = agent.handle_alert(agent_for("payments"), CHECKOUT)
    sea = agent.handle_alert(agent_for("search"), CHECKOUT)
    assert pay["decision"]["action"] == "flush_connection_pool"
    assert sea["decision"]["action"] != "flush_connection_pool"
    for receipt, team in ((pay, "payments"), (sea, "search")):
        for m in receipt["retrieval"]["shown"]:
            assert set(m["audience"]) & {f"team:{team}", "org"}
    # ...and the receipt records that relevant memory existed but was withheld.
    assert sea["retrieval"]["withheld"]["by_scope"].get("team:payments", 0) >= 1


def test_org_memory_reaches_every_team():
    for team in ("payments", "search"):
        r = agent.handle_alert(agent_for(team), CDN)
        assert (r["decision"]["action"], r["decision"]["target"]) == ("failover", "cdn")


def test_vault_and_curation_are_scoped():
    mem = pool_memory()
    assert mem["_id"] not in texts("bob")
    with pytest.raises(Forbidden):
        memory.forget(resolve("bob"), mem["_id"], "not mine")
    with pytest.raises(Forbidden):  # agents write and flag; they never erase
        memory.forget(agent_for("payments"), mem["_id"], "agent")
    with pytest.raises(Forbidden):  # alice cannot publish org-wide
        memory.write(resolve("alice"), "x", audience=["org"])
    with pytest.raises(Forbidden):  # nor into another team
        memory.write(resolve("alice"), "x", audience=["team:search"])


def test_schema_rejects_unscoped_memory():
    from pymongo.errors import WriteError
    doc = get_db().memories.find_one({"status": "active"})
    with pytest.raises(WriteError):
        get_db().memories.update_one({"_id": doc["_id"]}, {"$set": {"audience": []}})


def test_auditor_reads_receipt_but_not_team_memory_text():
    from receipts.api import _receipt_view
    r = agent.handle_alert(agent_for("payments"), CHECKOUT)
    view = _receipt_view(resolve("carol"), agent.get_receipt(resolve("carol"), r["_id"]))
    team_rows = [m for m in view["retrieval"]["shown"] if m["audience"] == ["team:payments"]]
    assert team_rows and all(m["text"] is None and m["content_hash"] for m in team_rows)
    with pytest.raises(Forbidden):
        agent.get_receipt(resolve("bob"), r["_id"])


def test_spoken_page_follows_the_same_scopes():
    from receipts import voice
    from receipts.api import _receipt_view
    r = agent.handle_alert(agent_for("payments"), CHECKOUT)
    alice = voice.script(_receipt_view(resolve("alice"), r))
    carol = voice.script(_receipt_view(resolve("carol"), r))
    assert "flush connection pool on checkout-db-pool" in alice and "pgbouncer" in alice
    assert "Without it, I would have chosen restart service" in alice
    assert "pgbouncer" not in carol and "not cleared to hear" in carol
    assert "withheld from me by scope" in alice and "mem_" not in alice


# --- did the memory do anything

def test_receipt_proves_memory_changed_the_action():
    r = agent.handle_alert(agent_for("payments"), CHECKOUT)
    assert r["counterfactual"]["without_memory"]["action"] == "restart_service"
    assert r["decision"]["action"] == "flush_connection_pool"
    assert r["influence"]["changed_action"] is True
    assert r["influence"]["causal_memory_ids"] == [pool_memory()["_id"]]
    assert r["counterfactual"]["leave_one_out"][0]["flips"] is True
    assert pool_memory()["stats"] == {"shown": 1, "cited": 1, "causal": 1, "wins": 0, "strikes": 0}


def test_memory_that_only_fills_the_prompt_is_not_credited():
    r = agent.handle_alert(agent_for("search"),
                           {"service": "synonym-admin", "severity": "low",
                            "title": "TLS certificate expires in 6 days", "body": "Renewal job has not run."})
    assert r["retrieval"]["shown"]  # memories were in the prompt
    assert r["influence"] == {"changed_action": False, "causal_memory_ids": [], "joint": False}


# --- corrected, expired, forgotten

def test_expired_memory_is_never_retrieved():
    stale = next(m for m in memory.list_visible(resolve("alice")) if "Release freeze" in m.get("text", ""))
    assert stale["status"] == "expired"
    r = agent.handle_alert(agent_for("payments"), CHECKOUT)
    assert stale["_id"] not in [m["memory_id"] for m in r["retrieval"]["shown"]]


def test_correction_changes_the_next_run_and_replay_shows_why():
    first = agent.handle_alert(agent_for("payments"), CHECKOUT)
    old = pool_memory()
    new = memory.correct(resolve("alice"), old["_id"],
                         "checkout-api p99 latency spike with DB connection wait time climbing: since the pgbouncer "
                         "v2 migration, flushing the pool drops live transactions. Fail over to the read replica pool.",
                         "pool flush caused an outage on 11/02",
                         suggested_action={"action": "failover", "target": "checkout-db-replica"})
    assert (new["lineage_id"], new["version"]) == (old["lineage_id"], 2)

    out = agent.replay(resolve("alice"), first["_id"])
    assert out["diff"]["action_changed"] is True
    assert out["diff"]["after"]["action"] == "failover"
    assert out["diff"]["memories_dropped"] == [
        {"memory_id": old["_id"], "now": "superseded", "superseded_by": new["_id"]}]
    assert new["_id"] in out["diff"]["memories_added"]
    assert out["replay"]["executed"] is False
    # the original receipt still resolves to the exact version the agent saw
    kept = get_db().memories.find_one({"_id": old["_id"]})
    assert kept["status"] == "superseded" and kept["content_hash"] == first["retrieval"]["shown"][0]["content_hash"]


def test_forget_erases_every_version_but_receipts_still_resolve():
    first = agent.handle_alert(agent_for("payments"), CHECKOUT)
    old = pool_memory()
    new = memory.correct(resolve("alice"), old["_id"], "pgbouncer pool flush no longer safe.", "v2")
    assert memory.forget(resolve("alice"), new["_id"], "contained customer data") == 2
    for doc in get_db().memories.find({"lineage_id": old["lineage_id"]}):
        assert doc["status"] == "forgotten" and "text" not in doc and "embedding" not in doc
    assert "pgbouncer" not in str(get_db().receipts.find_one({"_id": first["_id"]})["retrieval"])
    again = agent.handle_alert(agent_for("payments"), CHECKOUT)
    assert again["decision"]["action"] == "restart_service"  # back to the no-memory default
    assert agent.verify_chain()["ok"]


def test_failed_outcomes_quarantine_a_memory_and_leave_a_caveat():
    mem = pool_memory()
    r1 = agent.handle_alert(agent_for("payments"), CHECKOUT)
    out = agent.record_outcome(resolve("alice"), r1["_id"], "failed", "pool flush dropped live transactions")
    caveat = get_db().memories.find_one({"_id": out["outcome"]["learned_memory_id"]})
    assert caveat["kind"] == "caveat" and caveat["source_receipt"] == r1["_id"]

    # next run: the caveat vetoes the flush - the agent drops the bad advice on its own
    r2 = agent.handle_alert(agent_for("payments"), CHECKOUT)
    assert r2["decision"]["action"] != "flush_connection_pool"
    assert caveat["_id"] in r2["decision"]["cited"]

    memory.strike([mem["_id"]], r2["_id"], "alice")
    assert get_db().memories.find_one({"_id": mem["_id"]})["status"] == "quarantined"
    r3 = agent.handle_alert(agent_for("payments"), CHECKOUT)
    assert mem["_id"] not in [m["memory_id"] for m in r3["retrieval"]["shown"]]


def test_resolved_outcome_renews_memory_or_learns_a_new_one():
    mem = pool_memory()
    r = agent.handle_alert(agent_for("payments"), CHECKOUT)
    agent.record_outcome(resolve("alice"), r["_id"], "resolved")
    after = get_db().memories.find_one({"_id": mem["_id"]})
    assert after["stats"]["wins"] == 1 and after["expires_at"] > mem["expires_at"]
    with pytest.raises(Forbidden):
        agent.record_outcome(resolve("bob"), r["_id"], "resolved")

    novel = {"team": "payments", "service": "ledger-export", "severity": "high",
             "title": "ledger-export job timeout on nightly batch", "body": "Batch exceeded its 2h window."}
    r1 = agent.handle_alert(agent_for("payments"), novel)
    assert r1["decision"]["cited"] == []
    learned = agent.record_outcome(resolve("alice"), r1["_id"], "resolved", "restart cleared a stuck lock")
    r2 = agent.handle_alert(agent_for("payments"), novel)
    assert learned["outcome"]["learned_memory_id"] in r2["decision"]["cited"]


# --- the receipts themselves

def test_tampering_with_a_receipt_breaks_the_chain():
    a = agent.handle_alert(agent_for("payments"), CHECKOUT)
    agent.handle_alert(agent_for("search"), SEARCH)
    agent.record_outcome(resolve("alice"), a["_id"], "resolved")  # outcomes sit outside the seal
    assert agent.verify_chain() == {"ok": True, "checked": 2, "head": agent.verify_chain()["head"]}
    get_db().receipts.update_one({"_id": a["_id"]}, {"$set": {"decision.action": "page_human"}})
    assert agent.verify_chain()["broken_at"] == a["_id"]
