#!/usr/bin/env python3
"""
verify_session.py — independent verifier for rabbitholeai.ai sealed session exports.

Pure Python standard library. No dependency on any RHAI code, and no network access:
a third party can run this against an exported JSON and confirm, for every exchange,

  1. MERKLE   the seal's leaf digests recompute to the claimed root
  2. CHAIN    each exchange links to the previous one (genesis is derived from startedAt)
  3. ED25519  the root was signed by the published public key (origin-attestable)
  4. CONTENT  the readable text/data in the file recomputes to the sealed leaves —
              QUERY, RESPONSE, RECEIPT, BIAS, SOURCES, MEMORY, TIMING

(4) is the one that matters for "was this transcript altered after the fact" — a
signature over a root only proves the root was sealed; it does not, by itself, bind
the human-readable text you are reading to that root. This verifier recomputes every
content leaf whose preimage is present in the export. DOCUMENT is the exception: the
attachment's full text is deliberately not exported (it is yours), so a DOCUMENT leaf
is verifiable only against your own copy of the file you attached, and is reported
as 'na' here.

Every leaf is SHA-256 over a preimage serialized exactly as the gate's JavaScript
JSON.stringify would produce it: keys in the order given, no whitespace, non-ASCII
left unescaped, hashed as UTF-8. The one place a naive json.dumps diverges from
JSON.stringify is float formatting (e.g. 0.000002203 -> "0.000002203" in JS but
"2.203e-06" in Python), so this file ships a faithful ECMAScript Number::toString.

Usage:  python verify_session.py <export.json> [more.json ...]
"""
import sys, json, hashlib, base64, re

# id -> providerModel. Public data mirrored from the model registry; the TIMING
# leaf is hashed over the provider's model name, not the gate's display id.
PROVIDER_MODEL = {
    "llama-3.3-70b": "llama-3.3-70b-versatile",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "qwen3.6-27b": "qwen/qwen3.6-27b",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "claude-opus-4.8": "claude-opus-4-8",
    "claude-sonnet-5": "claude-sonnet-5",
    "claude-haiku-4.5": "claude-haiku-4-5",
    "claude-fable-5": "claude-fable-5",
    "gpt-5.6-sol": "gpt-5.6-sol",
    "grok-4.5": "grok-4.5",
}

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

# ─────────────────── JS-faithful JSON serialization (for leaves) ──────────────────
def js_number(x: float) -> str:
    """ECMAScript Number::toString for the JSON-relevant range. Produces the exact
    string JSON.stringify would, which Python's json.dumps does NOT for small/large
    magnitudes (it goes exponential earlier and pads the exponent)."""
    if x != x or x in (float("inf"), float("-inf")):
        return "null"                     # JSON.stringify(NaN|Infinity) === null
    if x == 0:
        return "0"                        # JSON.stringify(-0) === "0"
    if x < 0:
        return "-" + js_number(-x)
    if x == int(x) and abs(x) < 1e21:
        return str(int(x))
    r = repr(x)                           # shortest round-tripping digits (== JS digits)
    if "e" in r or "E" in r:
        mant, e = re.split("[eE]", r); e = int(e)
    else:
        mant, e = r, 0
    ip, fp = (mant.split(".") + [""])[:2]
    all_digits = ip + fp
    stripped = all_digits.lstrip("0")
    D = stripped.rstrip("0") or "0"       # significant digits, no leading/trailing zeros
    trail = len(stripped) - len(D)
    k = len(D)
    n = k + trail + e - len(fp)           # decimal-point position (ECMAScript n)
    if k <= n <= 21:
        return D + "0" * (n - k)
    if 0 < n <= 21:
        return D[:n] + "." + D[n:]
    if -6 < n <= 0:
        return "0." + "0" * (-n) + D
    ee = n - 1                            # exponential form
    esign = "+" if ee >= 0 else "-"
    body = D if k == 1 else D[0] + "." + D[1:]
    return body + "e" + esign + str(abs(ee))

def js_json(v) -> str:
    """Reproduces JavaScript JSON.stringify(v) byte-for-byte for the value types that
    appear in a sealed exchange (dict/list/str/int/float/bool/None). Strings reuse
    json.dumps(ensure_ascii=False), whose escaping already matches JSON.stringify;
    only float formatting is overridden."""
    if v is True:  return "true"
    if v is False: return "false"
    if v is None:  return "null"
    if isinstance(v, str):   return json.dumps(v, ensure_ascii=False)
    if isinstance(v, bool):  return "true" if v else "false"   # (bool before int)
    if isinstance(v, int):   return str(v)
    if isinstance(v, float): return js_number(v)
    if isinstance(v, list):  return "[" + ",".join(js_json(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ",".join(json.dumps(k, ensure_ascii=False) + ":" + js_json(val)
                              for k, val in v.items()) + "}"
    raise TypeError(f"cannot serialize {type(v).__name__}")

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

# The preimage the gate hashed for a given leaf label, recomputed from the export.
# Returns None when the leaf is not recomputable from the file alone (DOCUMENT: the
# attachment text is not exported; unknown labels from a future format).
def leaf_preimage(label, e, sealed_at):
    if label == "QUERY":
        return js_json({"q": e.get("query"), "ts": sealed_at})
    if label == "RESPONSE":
        return js_json({"r": e.get("response"), "model": e.get("model")})
    if label == "RECEIPT":
        return js_json(e.get("receipt"))
    if label == "BIAS":
        return js_json(e.get("biasScreen")) if e.get("biasScreen") is not None else None
    if label == "SOURCES":
        g = e.get("grounding")
        return js_json([s.get("uri") for s in g["sources"]]) if g and g.get("sources") is not None else None
    if label == "MEMORY":
        m = e.get("memoryRecall")
        return js_json(m["roots"]) if m and m.get("roots") is not None else None
    if label == "TIMING":
        model = e.get("model")
        return js_json({"model": PROVIDER_MODEL.get(model, model),
                        "llmMs": (e.get("timingMs") or {}).get("llm", 0)})
    return None  # DOCUMENT, or an unrecognized label

# ──────────────────────────────────── main ──────────────────────────────────────
def verify_file(path):
    d = json.load(open(path, encoding="utf-8"))
    ex = d.get("exchanges", [])
    pk_info = d.get("sealPublicKey") or {}
    pub = spki_to_raw(pk_info["publicKeySpkiB64"]) if pk_info.get("publicKeySpkiB64") else None

    print(f"\n=== {path.split(chr(92))[-1].split('/')[-1]} ===")
    print(f"site {d.get('site')}   started {d.get('startedAt')}   exchanges {len(ex)}")
    if pk_info:
        print(f"key  {pk_info.get('alg')} id={pk_info.get('keyId')}  payload={pk_info.get('signedPayloadFormat')}")

    chain = sha256_hex("VERUM_FRONTIER_SESSION_GENESIS" + d["startedAt"])
    tally = {}

    def mark(k, ok):
        t = tally.setdefault(k, [0, 0])
        t[0 if ok else 1] += 1
        return ok

    all_content_ok = True
    for i, e in enumerate(ex):
        seal = e.get("seal") or {}
        leaves = seal.get("leaves") or []
        digests = [l["sha256"] for l in leaves]
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

        # Content binding: recompute every leaf whose preimage is in the export.
        results = []
        for l in leaves:
            label = l["label"]
            pre = leaf_preimage(label, e, seal.get("sealedAt"))
            if pre is None:
                results.append(f"{label.lower()}=na")
                continue
            ok = mark(f"leaf:{label}", sha256_hex(pre) == l["sha256"])
            if not ok:
                all_content_ok = False
            results.append(f"{label.lower()}={'ok' if ok else 'FAIL'}")

        def g(v):
            return "na" if v is None else ("ok" if v else "FAIL")
        name = str(e.get("model"))[:22]
        print(f"  [{i:02d}] {name:22s} merkle {g(m_ok):4s} chain {g(c_ok):4s} sig {g(s_ok):4s}  "
              + " ".join(results))

    root_match = chain == d.get("sessionChainRoot")
    print(f"  session chain root: {'MATCH' if root_match else 'MISMATCH'}")
    for k in sorted(tally):
        ok, bad = tally[k]
        if ok or bad:
            print(f"    {k:16s} pass={ok} fail={bad}")
    overall = root_match and all(bad == 0 for ok, bad in tally.values())
    print(f"  OVERALL: {'VERIFIED' if overall else 'FAILED'}")
    return overall


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    all_ok = True
    for p in sys.argv[1:]:
        try:
            if not verify_file(p):
                all_ok = False
        except Exception as exc:
            print(f"\n=== {p} ===\n  ERROR {type(exc).__name__}: {exc}")
            all_ok = False
    sys.exit(0 if all_ok else 1)
