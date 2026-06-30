#!/usr/bin/env python3
"""
corpus_gate.py — the gated ingestion path for ALICE's living memory.

The boundary between "ALICE read something" and "ALICE knows something."
Untrusted input (web via BeautifulSoup, or the AI panel) enters as CLAIMS; nothing
is sealed to the corpus until it survives a skeptical, adversarial, multi-model gate:

  claim -> PANEL (diverse models, each told to be skeptical + generate the strongest
           COUNTER-case) -> aggregate (agreement, mean support, classification) ->
           VERDICT + provenance + preserved counter-case -> SEAL to corpus.

HONEST LIMIT (stated, not hidden): for general knowledge there is no market-returns
ground truth. This gate validates against *multi-model adversarial consensus*, NOT
external truth. So the top auto-tier is "corroborated" (survived the skeptical panel),
NOT "proven." 'invariant' is reserved for human confirmation or real external ground
truth. Claims that need empirical checking are flagged, not asserted. Correctable,
not erasable: re-gating a claim seals a NEW verdict; the old one stays in the chain.
"""
import os, re, json, statistics
from collections import Counter
import requests
from archivist import Archivist

GROQ_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# diverse panel = the cross-check. Two Llamas share a barn (correlated blind spots),
# so Gemini (different family/company) is the independent witness.
PANEL = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"]
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
CORPUS_LOG = os.getenv("PA_CORPUS_LOG", "/opt/alice-personal/data/alice_corpus.jsonl")
corpus = Archivist("alice-corpus", CORPUS_LOG)

_FACT_PROMPT = (
    "You are a rigorous, skeptical fact-checker. Do NOT be agreeable; your job is to stress-test the claim. "
    "Assess the CLAIM and reply with ONLY a JSON object: "
    '{"classification":"fact|false|contested|opinion|temporal",'
    '"support":0.0-1.0,'                       # probability the claim is TRUE
    '"counter_case":"the strongest argument AGAINST the claim",'
    '"needs_external_data":true or false,'      # true if it needs empirical/real-world checking
    '"reasoning":"one sentence"}. '
    "CLAIM: ")

def _extract_json(txt: str):
    try:
        return json.loads(txt)
    except Exception:
        m = re.search(r"\{.*\}", txt, re.S)
        if m:
            try: return json.loads(m.group(0))
            except Exception: return None
    return None

def _ask(model: str, claim: str, key: str):
    try:
        r = requests.post(GROQ_URL, headers={"Authorization": f"Bearer {key}"},
                          json={"model": model, "temperature": 0.2, "max_tokens": 320,
                                "response_format": {"type": "json_object"},
                                "messages": [{"role": "user", "content": _FACT_PROMPT + claim}]},
                          timeout=35)
        if r.status_code != 200:
            return None
        v = _extract_json(r.json()["choices"][0]["message"]["content"])
        if v: v["model"] = model
        return v
    except Exception:
        return None

def _ask_gemini(claim: str, key: str):
    if not key:
        return None
    try:
        r = requests.post(GEMINI_URL + "?key=" + key,
                          json={"contents": [{"parts": [{"text": _FACT_PROMPT + claim}]}],
                                "generationConfig": {"temperature": 0.2, "response_mime_type": "application/json",
                                                     "maxOutputTokens": 1024,
                                                     "thinkingConfig": {"thinkingBudget": 0}}}, timeout=40)
        if r.status_code != 200:
            return None
        v = _extract_json(r.json()["candidates"][0]["content"]["parts"][0]["text"])
        if v: v["model"] = GEMINI_MODEL
        return v
    except Exception:
        return None

def gate_claim(claim: str, source: str = "manual", key: str = None) -> dict:
    key = key or GROQ_KEY
    votes = [v for v in (_ask(m, claim, key) for m in PANEL) if v]
    g = _ask_gemini(claim, GEMINI_KEY)          # independent cross-family witness
    if g:
        votes.append(g)
    if not votes:
        return {"claim": claim, "source": source, "trust_tier": "unverified",
                "error": "no model responses", "basis": "panel unavailable"}
    supports = [float(v["support"]) for v in votes if isinstance(v.get("support"), (int, float))]
    mean_support = round(statistics.mean(supports), 2) if supports else 0.0
    classes = [str(v.get("classification", "")).lower() for v in votes if v.get("classification")]
    cc = Counter(classes)
    top, topn = (cc.most_common(1)[0] if cc else ("unknown", 0))
    agreement = round(topn / len(votes), 2)
    disagree = agreement < 0.6
    if disagree or top in ("contested", "opinion"):
        tier = "contested"
    elif top == "false" or mean_support <= 0.3:
        tier = "rejected"
    elif top == "fact" and mean_support >= 0.8 and not disagree:
        tier = "corroborated"          # survived the skeptical panel — NOT 'proven'
    else:
        tier = "provisional"
    return {
        "claim": claim, "source": source,
        "trust_tier": tier, "classification": top,
        "confidence": mean_support, "agreement": agreement,
        "needs_external_data": any(v.get("needs_external_data") for v in votes),
        "counter_cases": [v["counter_case"] for v in votes if v.get("counter_case")][:2],
        "model_votes": [{"model": v["model"], "classification": v.get("classification"),
                         "support": v.get("support")} for v in votes],
        "basis": "multi-model adversarial cross-check (not external ground truth)",
    }

def ingest_claim(claim: str, source: str = "manual", key: str = None) -> dict:
    """Gate a claim and seal the verdict to the corpus (rejected claims are sealed too —
    knowledge never lost includes 'we checked X and it failed')."""
    verdict = gate_claim(claim, source, key)
    entry = corpus.seal("knowledge", verdict)
    return {"sealed": True, "seq": entry["seq"], "verdict": verdict}

def get_url_text(url: str, max_chars: int = 4000) -> str:
    from bs4 import BeautifulSoup
    r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0 (ALICE)"})
    soup = BeautifulSoup(r.text, "html.parser")
    for t in soup(["script", "style", "nav", "footer", "header", "aside"]):
        t.extract()
    return " ".join(soup.get_text(" ").split())[:max_chars]

def extract_claims(text: str, n: int = 3, key: str = None) -> list:
    key = key or GROQ_KEY
    prompt = (f"Extract the {n} most important, checkable factual CLAIMS from this text as a JSON "
              f'{{"claims":["...","..."]}}. Only checkable statements, not opinions or filler. TEXT: ' + text)
    try:
        r = requests.post(GROQ_URL, headers={"Authorization": f"Bearer {key}"},
                          json={"model": "llama-3.3-70b-versatile", "temperature": 0.1, "max_tokens": 400,
                                "response_format": {"type": "json_object"},
                                "messages": [{"role": "user", "content": prompt}]}, timeout=35)
        d = _extract_json(r.json()["choices"][0]["message"]["content"])
        return (d or {}).get("claims", [])[:n]
    except Exception:
        return []

def _selftest():
    if not GROQ_KEY:
        print("SKIP selftest — GROQ_API_KEY not set"); return
    tests = [
        "Water boils at 100 degrees Celsius at sea-level atmospheric pressure.",  # fact
        "The Earth is flat.",                                                     # false
        "Pineapple belongs on pizza.",                                            # opinion/contested
    ]
    for c in tests:
        v = gate_claim(c, "selftest")
        print(f"\nCLAIM: {c}")
        print(f"  -> tier={v['trust_tier']} class={v['classification']} conf={v['confidence']} "
              f"agree={v['agreement']} needs_ext={v.get('needs_external_data')}")
        print(f"  votes: {[(x['model'].split('-')[0], x['classification'], x['support']) for x in v['model_votes']]}")
        if v.get("counter_cases"):
            print(f"  counter: {v['counter_cases'][0][:90]}")
    print("\nSELFTEST DONE — fact should be 'corroborated', flat-earth 'rejected', pizza 'contested'.")

if __name__ == "__main__":
    _selftest()
