"""Reset the database to the demo's starting state.

    python -m scripts.seed
"""
from datetime import timedelta

from receipts import embed, memory
from receipts.db import ensure_schema, get_db
from receipts.identity import resolve

PRINCIPALS = [
    {"_id": "agent:payments", "name": "Payments on-call agent", "kind": "agent", "team": "payments", "roles": []},
    {"_id": "agent:search", "name": "Search on-call agent", "kind": "agent", "team": "search", "roles": []},
    {"_id": "alice", "name": "Alice - Payments SRE", "kind": "human", "team": "payments", "roles": ["sre"]},
    {"_id": "bob", "name": "Bob - Search SRE", "kind": "human", "team": "search", "roles": ["sre"]},
    {"_id": "dana", "name": "Dana - Platform lead", "kind": "human", "team": "platform", "roles": ["sre", "org_writer"]},
    {"_id": "carol", "name": "Carol - Compliance auditor", "kind": "human", "team": None, "roles": ["auditor"]},
]

MEMORIES = [
    ("alice", dict(
        text="checkout-api p99 latency spike with DB connection wait time climbing and no recent deploy: "
             "flushing the pgbouncer connection pool cleared it within two minutes. Restarting the pods did not help.",
        service="checkout-api", suggested_action={"action": "flush_connection_pool", "target": "checkout-db-pool"})),
    ("alice", dict(
        text="payments-worker OOM kills that start right after a deploy have been a memory leak in the new build "
             "every time. Roll back the deploy; scaling up only delays the next kill.",
        service="payments-worker", suggested_action={"action": "rollback_deploy", "target": "payments-worker"})),
    ("alice", dict(
        text="card-auth 5xx burst with upstream gateway errors from the primary processor: fail over to the "
             "secondary processor. Do not restart card-auth, it drops in-flight authorizations.",
        service="card-auth", suggested_action={"action": "failover", "target": "card-processor"})),
    ("bob", dict(
        text="search-api p99 latency spike with connection wait time climbing: the Elasticsearch client pool is "
             "exhausted by stuck scroll queries. Restarting the search-api pods releases them.",
        service="search-api", suggested_action={"action": "restart_service", "target": "search-api"})),
    ("bob", dict(
        text="search-api latency spike right after an index rebuild is a cold query cache, not a capacity problem. "
             "Clear and re-warm the cache.",
        service="search-api", suggested_action={"action": "clear_cache", "target": "search-query-cache"})),
    ("bob", dict(
        text="search-indexer consumer lag growing during catalog imports: scale up the indexer workers.",
        service="search-indexer", suggested_action={"action": "scale_up", "target": "search-indexer"})),
    ("dana", dict(
        text="When the CDN reports elevated origin errors across several services at once, it is the edge "
             "provider, not the services. Fail traffic over to the secondary CDN and leave the services alone.",
        audience=["org"], service="cdn", ttl_days=None,
        suggested_action={"action": "failover", "target": "cdn"})),
]

SCENARIOS = [
    {"id": "checkout-latency", "label": "Checkout latency spike", "team": "payments", "service": "checkout-api",
     "severity": "critical", "title": "p99 latency 4.2s, DB connection wait time climbing",
     "body": "checkout-api p99 latency rose from 300ms to 4.2s over ten minutes. Connection pool wait time "
             "is climbing. No deploy in the last 24h."},
    {"id": "search-latency", "label": "Search latency spike", "team": "search", "service": "search-api",
     "severity": "high", "title": "p99 latency 3.8s, connection wait time climbing",
     "body": "search-api p99 latency rose from 200ms to 3.8s over ten minutes. Connection pool wait time "
             "is climbing. No deploy in the last 24h."},
    {"id": "worker-oom", "label": "Worker OOM after deploy", "team": "payments", "service": "payments-worker",
     "severity": "high", "title": "OOM kills on payments-worker after deploy",
     "body": "payments-worker pods are being OOM killed every few minutes. Started 15 minutes after the "
             "14:02 deploy."},
    {"id": "cdn-errors", "label": "CDN origin errors", "team": "search", "service": "cdn",
     "severity": "critical", "title": "CDN reports elevated origin errors across several services",
     "body": "The CDN is reporting elevated origin errors for search-api, checkout-api and the web frontend "
             "at once."},
]


def seed(quiet: bool = False) -> None:
    db = get_db()
    for name in ("memories", "receipts", "memory_events", "principals", "actions"):
        db[name].delete_many({})
    ensure_schema(db)
    db.principals.insert_many(PRINCIPALS)

    docs = [memory.write(resolve(author), **kwargs, _wait=False) for author, kwargs in MEMORIES]

    # One that has already lapsed: a freeze rule from last quarter.
    stale = memory.write(resolve("alice"), _wait=False, service="checkout-api", ttl_days=30,
                         text="Release freeze in effect for checkout-api: do not flush or restart anything "
                              "without the release manager; page a human for any latency alert.",
                         suggested_action={"action": "page_human", "target": "checkout-api"})
    db.memories.update_one({"_id": stale["_id"]}, {"$set": {
        "created_at": stale["created_at"] - timedelta(days=31),
        "expires_at": stale["created_at"] - timedelta(days=1)}})
    memory.sweep_expired()

    for d in docs:
        memory.wait_indexed(d, timeout=30)
    if not quiet:
        print(f"seeded {len(PRINCIPALS)} principals, {len(docs) + 1} memories "
              f"(embedder: {embed.name()}) into {db.name}")


if __name__ == "__main__":
    seed()
