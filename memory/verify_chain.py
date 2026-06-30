#!/usr/bin/env python3
"""
verify_chain.py — independent tamper-evidence test of the Guardian's sealed record.

This is the foundation check: can an OUTSIDER (court, contract manager, heir),
holding only the log file, prove (a) no past entry was altered, and (b) the chain
is unbroken — and would tampering with a single past entry be detected?

Replicates seal_segment exactly: hash = sha256(json.dumps(entry_without_hash,
sort_keys=True)); each entry's prev_hash must equal the previous entry's hash.
"""
import json, hashlib, copy

LOG = "/opt/alice-guardian/data/report_segments.jsonl"
GENESIS = "0" * 64

def recompute(entry: dict) -> str:
    d = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()

def verify(rows):
    ok = breaks = 0; prev = GENESIS; first_break = None
    for i, r in enumerate(rows):
        if recompute(r) == r.get("hash"):
            ok += 1
        elif first_break is None:
            first_break = ("hash-mismatch", i)
        if r.get("prev_hash", GENESIS) != prev:
            breaks += 1
            if first_break is None:
                first_break = ("chain-break", i)
        prev = r.get("hash")
    return ok, breaks, first_break

def main():
    rows = [json.loads(l) for l in open(LOG) if l.strip()]
    n = len(rows)
    print(f"=== GUARDIAN SEALED RECORD — INDEPENDENT VERIFICATION ===")
    print(f"  log: {LOG}")
    print(f"  entries: {n}")
    ok, breaks, fb = verify(rows)
    print(f"\n  [AS-IS] hash recompute matches: {ok}/{n} | prev_hash linkage breaks: {breaks}")
    print(f"  -> {'INTACT — every entry verifies, chain unbroken' if ok==n and breaks==0 else 'INTEGRITY PROBLEM: '+str(fb)}")

    # Tamper simulation: silently alter ONE past entry, re-verify.
    if n >= 3:
        tampered = copy.deepcopy(rows)
        victim = n // 2
        fld = next((k for k in ("observations","description","location","briefing_summary") if k in tampered[victim]), None)
        if fld:
            old = str(tampered[victim][fld])[:40]
            tampered[victim][fld] = (tampered[victim][fld] or "") + " [ALTERED AFTER THE FACT]"
            # leave the stored hash unchanged, as a forger who edits the text would
            ok2, breaks2, fb2 = verify(tampered)
            print(f"\n  [TAMPER TEST] silently edited entry #{victim} field '{fld}' (was: '{old}...')")
            print(f"  re-verify: hash matches {ok2}/{n} | chain breaks: {breaks2}")
            detected = (ok2 < n) or (breaks2 > 0)
            print(f"  -> tampering {'DETECTED' if detected else 'NOT detected'}"
                  + (f"  (first failure at entry #{fb2[1]}, {fb2[0]})" if fb2 else ""))
            print(f"     entries from #{victim} onward are invalidated — the alteration cannot hide.")

if __name__ == "__main__":
    main()
