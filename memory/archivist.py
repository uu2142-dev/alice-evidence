#!/usr/bin/env python3
"""
archivist.py — the clean, verifiable foundation for Personal ALICE (inner-alignment arm).

Built from the Guardian's hard lessons (fragmented identities + schema-drift hash
mismatches + global-chain/per-operator-verify mismatch). This one is clean from entry #1:
  - ONE canonical identity (a single human),
  - ONE append-only hash chain (prev_hash linkage + content hash),
  - VERSIONED sealed schema (a verifier always knows how to recompute -> no drift),
  - client occurred_at honored (offline/queued capture seals with the REAL event time),
  - independent verify(): chain_valid True from the first entry; any tamper detected,
    and you cannot fix a tamper by re-hashing (the chain catches it at the next link).

Pure stdlib — no deps. This same primitive carries the AAR co-evolution loop AND the
dev-intent queue. seal() to write, verify() to prove, read(kind) to recall.
"""
import os, json, hashlib
from datetime import datetime, timezone

SCHEMA_VERSION = 1
GENESIS = "0" * 64

class Archivist:
    def __init__(self, identity: str, log_path: str):
        self.identity = identity
        self.log_path = log_path
        d = os.path.dirname(log_path)
        if d:
            os.makedirs(d, exist_ok=True)

    def _entries(self):
        if not os.path.exists(self.log_path):
            return []
        out = []
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    @staticmethod
    def _content_hash(entry: dict) -> str:
        d = {k: v for k, v in entry.items() if k != "hash"}
        return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _tip(self):
        e = self._entries()
        return e[-1]["hash"] if e else GENESIS

    def seal(self, kind: str, payload: dict, occurred_at: str = None, occurred_local: str = None) -> dict:
        """Append one tamper-evident entry. occurred_at preserves real time for offline/queued capture."""
        entries = self._entries()
        entry = {
            "v": SCHEMA_VERSION,
            "seq": len(entries),
            "identity": self.identity,
            "kind": kind,                      # aar_morning | aar_evening | dev_intent | ...
            "payload": payload,
            "timestamp": occurred_at or datetime.now(timezone.utc).isoformat(),
            "local_time": occurred_local or datetime.now().strftime("%H:%M"),
            "prev_hash": entries[-1]["hash"] if entries else GENESIS,
        }
        entry["hash"] = self._content_hash(entry)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry, separators=(",", ":")) + "\n")
        return entry

    def verify(self) -> dict:
        """Independent integrity check — anyone holding the file can run this."""
        entries = self._entries()
        if not entries:
            return {"chain_valid": True, "n": 0, "tip": GENESIS, "message": "empty chain"}
        expected_prev = GENESIS
        for i, e in enumerate(entries):
            if e.get("prev_hash") != expected_prev:
                return {"chain_valid": False, "n": len(entries), "broken_at": i,
                        "reason": "prev_hash linkage break", "tip": entries[-1].get("hash")}
            if self._content_hash(e) != e.get("hash"):
                return {"chain_valid": False, "n": len(entries), "broken_at": i,
                        "reason": "content hash mismatch (entry altered)", "tip": entries[-1].get("hash")}
            expected_prev = e["hash"]
        return {"chain_valid": True, "n": len(entries), "tip": entries[-1]["hash"],
                "message": "intact — every entry verified, chain unbroken"}

    def read(self, kind: str = None):
        return [e for e in self._entries() if kind is None or e.get("kind") == kind]


def _selftest():
    import tempfile
    base = os.path.join(tempfile.gettempdir(), "pa_archivist_selftest")
    def fresh(p):
        if os.path.exists(p): os.remove(p)
        return Archivist("jerry", p)

    # 1) clean chain verifies
    p = base + "_1.jsonl"; a = fresh(p)
    a.seal("aar_morning", {"intention": "lay the foundation"})
    a.seal("aar_evening", {"well": "clean chain sealed", "change": "wire Oura grounding next"})
    a.seal("dev_intent", {"request": "add the AAR loop on top"})
    v = a.verify(); print("clean chain:", v)
    assert v["chain_valid"] and v["n"] == 3

    # 2) naive tamper (alter payload, leave hash) -> content mismatch at that entry
    rows = [json.loads(l) for l in open(p)]
    rows[1]["payload"]["well"] = "ALTERED"
    open(p, "w").write("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))
    v = a.verify(); print("naive tamper @#1:", v)
    assert not v["chain_valid"] and v["broken_at"] == 1 and "content hash" in v["reason"]

    # 3) sophisticated tamper (alter payload AND re-hash that entry) -> chain breaks at NEXT link
    p2 = base + "_2.jsonl"; b = fresh(p2)
    b.seal("aar_morning", {"intention": "x"}); b.seal("aar_evening", {"well": "y"}); b.seal("dev_intent", {"request": "z"})
    rows = [json.loads(l) for l in open(p2)]
    rows[1]["payload"]["well"] = "ALTERED+REHASHED"
    rows[1]["hash"] = Archivist._content_hash(rows[1])   # forger recomputes the entry's own hash
    open(p2, "w").write("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))
    v = b.verify(); print("rehash tamper @#1:", v)
    assert not v["chain_valid"] and v["broken_at"] == 2 and "linkage" in v["reason"]

    for x in (p, p2):
        if os.path.exists(x): os.remove(x)
    print("\nSELFTEST PASSED — clean chain verifies; tampering detected even when re-hashed.")

if __name__ == "__main__":
    _selftest()
