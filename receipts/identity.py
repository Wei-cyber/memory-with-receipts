"""Who is asking. Every read and write of memory starts from a Principal.

Auth here is a demo header (X-Principal); swap `resolve` for your IdP. What
matters is that scopes are derived server-side from the stored principal and
never accepted from the caller.
"""
from dataclasses import dataclass, field

from .db import get_db


class Forbidden(Exception):
    pass


@dataclass(frozen=True)
class Principal:
    id: str
    name: str
    kind: str  # "agent" | "human"
    team: str | None
    roles: tuple[str, ...] = field(default_factory=tuple)

    @property
    def scopes(self) -> list[str]:
        """The audiences this principal may read."""
        return ([f"team:{self.team}"] if self.team else []) + ["org"]

    def has(self, role: str) -> bool:
        return role in self.roles

    def can_write(self, audience: list[str]) -> bool:
        for scope in audience:
            if scope == "org":
                if not self.has("org_writer"):
                    return False
            elif scope != f"team:{self.team}":
                return False
        return self.kind == "agent" or self.has("sre")

    def can_curate(self, audience: list[str]) -> bool:
        """Correct, forget, release from quarantine. Humans only - an agent can
        write what it learned and flag what failed, but never erases memory."""
        return self.kind == "human" and self.can_write(audience)

    def can_read_receipts_of(self, team: str) -> bool:
        return self.has("auditor") or self.team == team

    def public(self) -> dict:
        return {"id": self.id, "name": self.name, "kind": self.kind, "team": self.team,
                "roles": list(self.roles), "scopes": self.scopes}


def resolve(principal_id: str | None) -> Principal:
    doc = get_db().principals.find_one({"_id": principal_id}) if principal_id else None
    if not doc:
        raise Forbidden(f"unknown principal: {principal_id!r}")
    return Principal(id=doc["_id"], name=doc["name"], kind=doc["kind"], team=doc.get("team"),
                     roles=tuple(doc.get("roles", [])))


def agent_for(team: str) -> Principal:
    return resolve(f"agent:{team}")
