# Sealed sessions — FY2027 NDAA §219 case study

Two exported sessions from the live gate at rabbitholeai.ai, published so the case study's
claims can be re-checked rather than taken on trust.

Verify them yourself — pure standard library, no network, no dependency on any code in this
repo except the verifier itself:

```bash
python memory/verify_session.py sessions/*.json
```

Expected output: `merkle`, `chain`, `sig`, `query` and `resp` all `fail=0`, and both session
roots `MATCH`.

| File | What it shows |
|---|---|
| `session_2026-07-23_classified-refusal-osint-pivot.json` | Gemini 2.5 Flash declines a request touching classified DIA counterintelligence reporting — *"Such reports, if they exist, would be highly classified and not publicly available"* — then pivots to open sources on its own in the next exchange. 3 exchanges. |
| `session_2026-07-24_ndaa219-multimodel.json` | Three frontier models from three labs (Grok 4.5, GPT-5.6 Sol, Claude Fable 5) on §219 and IAA §622 under identical sealing and screening. 8 exchanges, root `cbb2bc4f…c94eab`, $2.52 receipted. |

## What these artifacts do and do not show

**They show.** Every exchange SHA-256 Merkle-sealed, Ed25519-signed, chained to the previous
one, with a per-answer cost receipt and a bias-screen result. Grounded answers cite primary
documents — clerk.house.gov floor actions, the Senate Intelligence Committee's IAA report,
the Rules Committee bill text.

**They do not show a clean run, and that is deliberate.** Exchanges 0–3 of the §219 session
are `grounded: false`. The gateway's search path was broken during that stretch; the opening
query is the operator saying so. Those answers are labeled ungrounded rather than dressed up,
and the honest label is *how the bug was found*. The unedited record is the point — a curated
one would prove nothing.

**The framing head scored 0.94–0.998** on the Israel-related answers. It is a style triage
signal — toxicity AUROC 0.924, media-framing 0.842, live at `rabbitholeai.ai/api/bias/health`
— **not a truth, accuracy, or hallucination detector.** Reported here rather than suppressed.

**Sessions exported before ~July 15, 2026 carry no `sig` block.** Root signing was not live
yet. Both files here are signed.

## Signatures alone are not enough — read this before quoting a transcript

A signature over a Merkle root proves *the root* was sealed by this gate at that time. It does
**not**, by itself, prove the human-readable text printed beside it is what produced that root.

This was demonstrated against these exact files: changing three words inside a sealed response
and leaving every hash and signature untouched still passes the Merkle check, the chain check,
**and** the Ed25519 signature check. The forgery only surfaces when the content leaves are
recomputed from the visible text:

```
[03] merkle ok  chain ok  sig ok  query ok  resp FAIL   claude-fable-5
```

`verify_session.py` performs that fourth check. Any verifier that stops at the signature will
accept a doctored transcript as authentic. The export's own `verify` field now documents the leaf
preimage format so a third party can reproduce the check independently:

- `QUERY` = `SHA-256(JSON.stringify({q: <query>, ts: <seal.sealedAt>}))`
- `RESPONSE` = `SHA-256(JSON.stringify({r: <response>, model: <model>}))`

Serialization is `JSON.stringify` semantics: keys in the order given, no whitespace, non-ASCII
characters **not** escaped, hashed as UTF-8.

This gap was found in our own instructions and is documented here rather than quietly patched.
