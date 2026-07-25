#!/usr/bin/env python3
"""
verify_session.py — independent verifier for rabbitholeai.ai sealed session exports.

Pure Python standard library. No dependency on any RHAI code, and no network access:
a third party can run this against an exported JSON and confirm, for every exchange,

  1. MERKLE   the seal's leaf digests recompute to the claimed root
  2. CHAIN    each exchange links to the previous one (genesis is derived from startedAt)
  3. ED25519  the root was signed by the published public key (origin-attestable)
  4. CONTENT  the QUERY / RESPONSE text in the file recomputes to the sealed leaves

(4) is the one that matters for "was this transcript altered after the fact" — a
signature over a root only proves the root was sealed; it does not, by itself, bind
the human-readable text you are reading to that root.

Usage:  python verify_session.py <export.json> [more.json ...]
"""
import sys, json, hashlib, base64

# ─────────────────────────── Ed25519 (RFC 8032) verify ───────────────────────────
q = 2**255 - 19
L = 2**252 + 27742317777372353535851937790883648493

def _inv(x):     return pow(x, q - 2, q)
d      = -121665 * _inv(121666) % q
I_CONST = pow(2, (q - 1) // 4, q)

def _xrecover(y):
    xx = (y*y - 1) * _inv(d*y*y + 1)
    x = pow(xx, (q + 3) // 8, q)
    if (x*x - xx) % q != 0:
        x = (x * I_CONST) % q
    if x % 2 != 0:
        x = q - x
    return x

By = 4 * _inv(5)
B  = [_xrecover(By) % q, By % q, 1, (_xrecover(By) * By) % q]

def _add(P, Q):
    A = (P[1]-P[0]) * (Q[1]-Q[0]) % q
    Bb = (P[1]+P[0]) * (Q[1]+Q[0]) % q
    C = 2 * P[3] * Q[3] * d % q
    D = 2 * P[2] * Q[2] % q
    E, F, G, H = Bb-A, D-C, D+C, Bb+A
    return [E*F % q, G*H % q, F*G % q, E*H % q]

def _mul(P, e):
    if e == 0:
        return [0, 1, 1, 0]
    Q = _mul(P, e // 2)
    Q = _add(Q, Q)
    if e & 1:
        Q = _add(Q, P)
    return Q

def _decodepoint(s):
    y = int.from_bytes(s, "little") & ((1 << 255) - 1)
    x = _xrecover(y)
    if x & 1 != (s[31] >> 7) & 1:
        x = q - x
    P = [x, y, 1, x*y % q]
    # on-curve check
    x, y, z, t = P
    if (-x*x + y*y - z*z - d*t*t) % q != 0 or (x*y - z*t) % q != 0:
        raise ValueError("point not on curve")
    return P

def _eq(P, Q):
    return (P[0]*Q[2] - Q[0]*P[2]) % q == 0 and (P[1]*Q[2] - Q[1]*P[2]) % q == 0

def ed25519_verify(pubkey32: bytes, msg: bytes, sig64: bytes) -> bool:
    if len(sig64) != 64 or len(pubkey32) != 32:
        return False
    try:
        A = _decodepoint(pubkey32)
    except Exception:
        return False
    Rs, S = sig64[:32], int.from_bytes(sig64[32:], "little")
    if S >= L:
        return False
    try:
        R = _decodepoint(Rs)
    except Exception:
        return False
    h = int.from_bytes(hashlib.sha512(Rs + pubkey32 + msg).digest(), "little") % L
    return _eq(_mul(B, S), _add(R, _mul(A, h)))

# ─────────────────────────────── seal primitives ────────────────────────────────
def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def merkle_root(leaves):
    """Pairs left-to-right, duplicating the last when odd, hashing hex-string concats."""
    if not leaves:
        return None
    level = list(leaves)
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [sha256_hex(level[i] + level[i+1]) for i in range(0, len(level), 2)]
    return level[0]

def spki_to_raw(spki_b64: str) -> bytes:
    raw = base64.b64decode(spki_b64)
    return raw[-32:]           # 12-byte DER prefix + 32-byte key

# ──────────────────────────────────── main ──────────────────────────────────────
def verify_file(path):
    d = json.load(open(path, encoding="utf-8"))
    ex = d.get("exchanges", [])
    pk_info = d.get("sealPublicKey") or {}
    pub = spki_to_raw(pk_info["publicKeySpkiB64"]) if pk_info.get("publicKeySpkiB64") else None

    print(f"\n=== {path.split(chr(92))[-1]} ===")
    print(f"site {d.get('site')}   started {d.get('startedAt')}   exchanges {len(ex)}")
    if pk_info:
        print(f"key  {pk_info.get('alg')} id={pk_info.get('keyId')}  payload={pk_info.get('signedPayloadFormat')}")

    chain = sha256_hex("VERUM_FRONTIER_SESSION_GENESIS" + d["startedAt"])
    tally = {"merkle": [0, 0], "chain": [0, 0], "sig": [0, 0], "query": [0, 0], "resp": [0, 0]}

    def mark(k, ok):
        tally[k][0 if ok else 1] += 1
        return ok

    for i, e in enumerate(ex):
        seal = e.get("seal") or {}
        leaves = seal.get("leaves") or []
        digests = [l["sha256"] for l in leaves]
        labels = {l["label"]: l["sha256"] for l in leaves}
        root = seal.get("root")

        m_ok = mark("merkle", merkle_root(digests) == root)

        chain = sha256_hex(chain + root)
        c_ok = mark("chain", chain == e.get("sessionChainHash"))

        sig = (seal.get("sig") or {})
        if sig.get("signature") and pub:
            payload = f"VF-SEAL-v1|{root}|{seal.get('sealedAt')}".encode()
            s_ok = mark("sig", ed25519_verify(pub, payload, base64.b64decode(sig["signature"])))
        else:
            s_ok = None

        # content binding: recompute the QUERY / RESPONSE leaves from the text in the file
        q_ok = r_ok = None
        if "QUERY" in labels:
            q_ok = mark("query", sha256_hex(json.dumps(
                {"q": e.get("query"), "ts": seal.get("sealedAt")}, separators=(",", ":"), ensure_ascii=False)) == labels["QUERY"])
        if "RESPONSE" in labels:
            r_ok = mark("resp", sha256_hex(json.dumps(
                {"r": e.get("response"), "model": e.get("model")},
                separators=(",", ":"), ensure_ascii=False)) == labels["RESPONSE"])

        def g(v):
            return "—" if v is None else ("ok" if v else "FAIL")
        name = e.get("model")
        print(f"  [{i:02d}] merkle {g(m_ok):4s} chain {g(c_ok):4s} sig {g(s_ok):4s} "
              f"query {g(q_ok):4s} resp {g(r_ok):4s}  {str(name)[:34]:34s} leaves={len(digests)}")

    print(f"  session root claimed  {d.get('sessionChainRoot')}")
    print(f"  session root recomputed {chain}   "
          f"{'MATCH' if chain == d.get('sessionChainRoot') else 'MISMATCH'}")
    for k, (ok, bad) in tally.items():
        if ok or bad:
            print(f"    {k:7s} pass={ok} fail={bad}")
    return tally


if __name__ == "__main__":
    for p in sys.argv[1:]:
        try:
            verify_file(p)
        except Exception as exc:
            print(f"\n=== {p} ===\n  ERROR {type(exc).__name__}: {exc}")
