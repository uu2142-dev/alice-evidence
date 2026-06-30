#!/usr/bin/env python3
"""
verum_orient.py — Live Verum Frontier: the panel PROPOSES, the gate DISPOSES.

ORIENT stage of the inquiry loop. A five-model frontier panel — four free
Groq-hosted open models (Llama-3.1-8b, Llama-3.3-70b, Llama-4-Scout, Qwen3-32b)
plus Google Gemini 2.5 Flash — each proposes testable edge hypotheses for a
sector from a constrained primitive vocabulary. Every proposal, including the
confident-but-false ones they WILL generate, is run through the pre-registered,
walk-forward, Bonferroni-corrected ground-truth gate and sealed.

Two phases so a partial run still yields a sealed artifact:
  PHASE 1 (no market data): query panel, collect+dedupe+seal PROPOSALS.
  PHASE 2 (needs FMP): point-in-time gate on the proposals; seal verdicts.
If FMP is rate-limited, Phase 1 still seals the panel's proposals; gate pends.

Cross-model agreement is recorded but earns NO credibility — only the gate does.
"""

import os, sys, json, hashlib, statistics, math, re, requests
from datetime import datetime, timedelta
sys.path.insert(0, "/opt/alice/app")
from backtest_ifi import fetch_prices

GROQ_KEY = os.environ.get("GROQ_API_KEY","")
GEM_KEY  = os.environ.get("GEMINI_API_KEY","")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEM_URL  = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
GROQ_MODELS = ["llama-3.1-8b-instant","llama-3.3-70b-versatile",
               "meta-llama/llama-4-scout-17b-16e-instruct","qwen/qwen3-32b"]
SECTOR="TECH"
UNIVERSE=["NVDA","MSFT","AAPL","GOOGL","META","AMZN","AVGO","ORCL","ADBE","CRM",
          "AMD","INTC","CSCO","QCOM","TXN","AMAT","MU","NOW","INTU","IBM","ACN",
          "NFLX","PANW","SNPS","CDNS","LRCX","KLAC","ADI","MRVL","FTNT"]
SPLIT="2024-06-01"; START="2022-01-01"; END="2026-03-01"

SCHEMA={"signal":["insider_sell","insider_buy"],
        "condition":["hot_momentum","cold_momentum","any_momentum"],
        "horizon_days":[21,42,63],"direction":["short","long"]}
PROMPT=("You are proposing TESTABLE stock-edge hypotheses for the US TECH sector. "
 "You may ONLY combine these primitives:\n"
 f"signal: {SCHEMA['signal']}\ncondition: {SCHEMA['condition']}\n"
 f"horizon_days: {SCHEMA['horizon_days']}\ndirection: {SCHEMA['direction']}\n\n"
 "signal = a corporate insider (Form 4) buying/selling. condition = the stock's recent "
 "price-momentum state. direction = the bet (short=expect underperformance). Propose your "
 "3 best hypotheses for where a real, exploitable edge exists. Return ONLY a JSON array:\n"
 '[{"signal":"insider_sell","condition":"hot_momentum","horizon_days":42,"direction":"short"}]')

def sha(s): return hashlib.sha256(s.encode()).hexdigest()

def _parse(txt):
    txt=re.sub(r"<think>.*?</think>","",txt,flags=re.S)
    txt=re.sub(r"```(json)?","",txt)
    m=re.search(r"\[.*\]", txt, re.S)
    if not m: return []
    try: arr=json.loads(m.group(0))
    except Exception: return []
    out=[]
    for h in arr:
        s,c,ho,d=h.get("signal"),h.get("condition"),h.get("horizon_days"),h.get("direction")
        if s in SCHEMA["signal"] and c in SCHEMA["condition"] and ho in SCHEMA["horizon_days"] and d in SCHEMA["direction"]:
            out.append((s,c,int(ho),d))
    return out

def query_groq(model):
    try:
        r=requests.post(GROQ_URL,headers={"Authorization":"Bearer "+GROQ_KEY},
          json={"model":model,"messages":[{"role":"user","content":PROMPT}],
                "max_tokens":900,"temperature":0.7},timeout=40)
        return _parse(r.json()["choices"][0]["message"]["content"]) if r.status_code==200 else []
    except Exception: return []

def query_gemini():
    try:
        r=requests.post(GEM_URL+"?key="+GEM_KEY,
          json={"contents":[{"parts":[{"text":PROMPT}]}]},timeout=40)
        return _parse(r.json()["candidates"][0]["content"]["parts"][0]["text"]) if r.status_code==200 else []
    except Exception: return []

def run():
    print("=== VERUM FRONTIER — ORIENT: five-model panel proposes, gate disposes ===")
    panel=[("gemini-2.5-flash", query_gemini())] + [(m, query_groq(m)) for m in GROQ_MODELS]
    proposals={}
    for name,hyps in panel:
        print(f"  {name.split('/')[-1][:24]:24} proposed {len(hyps)}")
        for h in hyps: proposals.setdefault(h,set()).add(name.split('/')[-1][:18])
    if not proposals: print("No valid proposals."); return
    print(f"\n  {len(proposals)} unique hypotheses across the panel")

    # PHASE 1 — seal the panel's proposals (no market data required)
    pset={"sector":SECTOR,"panel":[p[0] for p in panel],
          "hypotheses":[{"h":list(h),"proposers":sorted(list(p)),"n_models":len(p)} for h,p in proposals.items()],
          "sealed_at":datetime.utcnow().isoformat()+"Z"}
    phash=sha(json.dumps(pset,sort_keys=True)); pset["proposal_hash"]=phash
    open("/opt/alice/data/verum_proposals_tech.json","w").write(json.dumps(pset,indent=2))
    print(f"  PROPOSALS SEALED: {phash[:24]}...  -> verum_proposals_tech.json")
    for (s,c,ho,d),provs in sorted(proposals.items(),key=lambda kv:-len(kv[1])):
        print(f"     {len(provs)}x  {s}/{c}/{ho}d/{d}   ({', '.join(sorted(provs))})")

    # PHASE 2 — ground-truth gate (needs FMP)
    try:
        from backtest_ond_pit import build_membership, fetch_insider, trailing, fwd_excess
        current,union,members_asof=build_membership()
    except Exception as e:
        print(f"\n  GATE PENDING — market-data layer unavailable: {repr(e)[:90]}")
        print("  (FMP free quota likely exhausted; cache fills on next clean run. Proposals are sealed.)")
        return

    endf=(datetime.now()+timedelta(days=1)).strftime("%Y-%m-%d")
    spy=fetch_prices("SPY",START,endf)
    prices={tk:fetch_prices(tk,START,endf) for tk in UNIVERSE}
    insider={tk:fetch_insider(tk) for tk in UNIVERSE}
    usable=[tk for tk in UNIVERSE if len(prices[tk])>=200]
    dts=[]; d=datetime.strptime(START,"%Y-%m-%d"); e=datetime.strptime(END,"%Y-%m-%d")
    while d<e: dts.append(d.strftime("%Y-%m-%d")); d=(d.replace(day=28)+timedelta(days=7)).replace(day=1)
    is_m=sorted(m for tk in usable for dt in dts if dt<SPLIT and tk in members_asof(dt)
                for m in [trailing(prices[tk],dt)] if m is not None)
    hot_lo=is_m[int(len(is_m)*0.7)]; cold_hi=is_m[int(len(is_m)*0.3)]

    def t0(x):
        if len(x)<5: return None
        m=statistics.mean(x); s=statistics.pstdev(x)/math.sqrt(len(x)); return m/s if s>0 else 0.0
    def pT(t):
        from math import erfc,sqrt; return 1.0 if t is None else erfc(abs(t)/sqrt(2))
    def qual(s,c,tk,dt,ev):
        if (s=="insider_sell" and ev!=-1) or (s=="insider_buy" and ev!=1): return False
        m=trailing(prices[tk],dt)
        if m is None: return False
        if c=="hot_momentum": return m>=hot_lo
        if c=="cold_momentum": return m<=cold_hi
        return True
    bonf=0.05/len(proposals)
    print(f"\n  GROUND-TRUTH GATE (OOS {SPLIT}..{END}, Bonferroni a={bonf:.4f}):")
    print(f"  {'hypothesis':>34} | {'models':>6} | {'n':>4} | {'signed%':>8} | {'t':>6} | verdict")
    print("  "+"-"*82)
    results=[]
    for (s,c,ho,d),provs in sorted(proposals.items(),key=lambda kv:-len(kv[1])):
        sign=-1 if d=="short" else 1; fw=[]
        for tk in usable:
            for ev in insider[tk]:
                if ev["date"]<SPLIT or tk not in members_asof(ev["date"]): continue
                if qual(s,c,tk,ev["date"],ev["dir"]):
                    f=fwd_excess(prices[tk],spy,ev["date"],ho)
                    if f is not None: fw.append(f)
        signed=[sign*f for f in fw]; t=t0(signed); p=pT(t)
        mean=statistics.mean(signed)*100 if signed else 0.0
        surv=(t is not None and t>0 and p<bonf and len(signed)>=20)
        lbl=f"{s}/{c}/{ho}d/{d}"
        results.append({"h":lbl,"models":len(provs),"n":len(signed),"signed_pct":round(mean,2),
                        "t":round(t,2) if t else None,"survives":surv})
        tt=f"{t:+.2f}" if t is not None else "  —"
        print(f"  {lbl:>34} | {len(provs):>6} | {len(signed):>4} | {mean:>+8.2f} | {tt:>6} | {'✅ SURVIVES' if surv else '❌ rejected'}")
    root=sha(phash+json.dumps(results,sort_keys=True))
    open("/opt/alice/data/verum_orient_tech.json","w").write(json.dumps(
        {"proposal_hash":phash,"results":results,"root":root,"at":datetime.utcnow().isoformat()+"Z"},indent=2))
    surv=[r['h'] for r in results if r['survives']]
    print(f"\n  ARCHIVIST SEAL: {root[:32]}... -> verum_orient_tech.json")
    print(f"  ACCEPTED {len(surv)}/{len(results)}: {surv or 'none'}")

if __name__=="__main__":
    import warnings; warnings.filterwarnings("ignore"); run()
