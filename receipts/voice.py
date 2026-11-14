"""The on-call page: a receipt, spoken.

The script is built from the receipt *as the listener is allowed to see it*,
so the same access rules hold on the phone as on the screen - an auditor hears
that a team memory drove the action, not what the memory says.
"""
import hashlib
import re

import httpx

from . import config

_audio_cache: dict[str, bytes] = {}


def enabled() -> bool:
    return bool(config.ELEVENLABS_API_KEY)


def _say(action: str) -> str:
    return action.replace("_", " ")


def _first_sentence(text: str, limit: int = 220) -> str:
    sentence = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]
    return sentence if len(sentence) <= limit else sentence[:limit].rsplit(" ", 1)[0] + "."


def script(view: dict) -> str:
    """`view` is a receipt already filtered for the listener (api._receipt_view)."""
    alert, d = view["alert"], view["decision"]
    base = view["counterfactual"]["without_memory"]
    inf, withheld = view["influence"], view["retrieval"]["withheld"]
    team = view["principal"]["team"]
    did = "I ran" if view.get("executed") else "Dry run. I would run"

    parts = [
        f"Page for the {team} on-call.",
        f"{(alert.get('severity') or 'high').capitalize()} alert on {alert['service']}: {alert['title']}.",
        f"{did} {_say(d['action'])} on {d['target']}.",
    ]
    causal = [m for m in view["retrieval"]["shown"] if m["causal"]]
    if causal:
        m = causal[0]
        if m["text"]:
            parts.append(f"That came from memory. {_first_sentence(m['text'])}")
        elif m["withheld_from_viewer"]:
            parts.append(f"That came from a {team} team memory you are not cleared to hear.")
        else:
            parts.append("That came from a memory that has since been erased.")
        if inf["changed_action"]:
            parts.append(f"Without it, I would have chosen {_say(base['action'])} on {base['target']}.")
        else:
            parts.append("Without any memory I would have made the same call.")
    else:
        parts.append("No memory changed this decision. " + _first_sentence(d["rationale"]))
    if withheld["count"]:
        n = withheld["count"]
        parts.append(f"{n} relevant {'memory' if n == 1 else 'memories'} from other teams "
                     f"{'was' if n == 1 else 'were'} withheld from me by scope.")
    if view["outcome"]["status"] == "pending":
        parts.append(f"This is receipt number {view['seq']}. Tell me whether it worked.")
    return " ".join(parts)


def speak(text: str) -> bytes:
    key = hashlib.sha256(f"{config.ELEVENLABS_VOICE_ID}|{config.ELEVENLABS_MODEL}|{text}".encode()).hexdigest()
    if key not in _audio_cache:
        resp = httpx.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{config.ELEVENLABS_VOICE_ID}",
            params={"output_format": "mp3_44100_128"},
            headers={"xi-api-key": config.ELEVENLABS_API_KEY},
            json={"text": text, "model_id": config.ELEVENLABS_MODEL},
            timeout=60,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"ElevenLabs returned {resp.status_code}: {resp.text[:300]}")
        _audio_cache[key] = resp.content
    return _audio_cache[key]
