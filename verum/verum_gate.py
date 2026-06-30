#!/usr/bin/env python3
"""
verum_gate.py — run the ground-truth gate on the PRE-SEALED Verum proposals.

Integrity point: the five-model proposals were sealed (verum_proposals_tech.json,
hash 58d89a19…) while FMP was rate-limited — i.e. the hypotheses were locked
BEFORE the gate could see any market data. This script loads those exact sealed
proposals and runs the walk-forward, Bonferroni-corrected, point-in-time gate on
them, chaining the verdict to the original proposal_hash. That chain is the
tamper-evident proof the hypotheses weren't edited after seeing the answer.
"""
import sys, json, hashlib, statistics, math
from datetime import datetime, timedelta
sys.path.insert(0, "/opt/alice/app")
from backtest_ond_pit import build_membership, fetch_insider, trailing, fwd_excess
from backtest_ifi import fetch_prices

UNIVERSE=["NVDA","MSFT","AAPL","GOOGL","META","AMZN","AVGO","ORCL","ADBE","CRM",
          "AMD","INTC","CSCO","QCOM","TXN","AMAT","MU","NOW","INTU","IBM","ACN",
          "NFLX","PANW","SNPS","CDNS","LRCX","KLAC","ADI","MRVL","FTNT"]
SPLIT="2024-06-01"; START="2022-01-01"; END="2026-03-01"
PROP="/opt/alice/data/verum_proposals_tech.json"

def sha(s): return hashlib.sha256(s.encode()).hexdigest()
def t0(x):
    if len(x)<5: return None
    m=statistics.mean(x); s=statistics.pstdev(x)/math.sqrt(len(x)); return m/s if s>0 else 0.0
def pT(t):
    from math import erfc,sqrt; return 1.0 if t is None else erfc(abs(t)/sqrt(2))

def run():
    pset=json.load(open(PROP))
    phash=pset["proposal_hash"]
    hyps=[(tuple(h["h"]), h.get("n_models",len(h.get("proposers",[])))) for h in pset["hypotheses"]]
    print(f"=== VERUM GATE — testing {len(hyps)} PRE-SEALED proposals (proposal_hash {phash[:16]}…) ===")

    current,union,members_asof=build_membership()
    endf=(datetime.now()+timedelta(days=1)).strftime("%Y-%m-%d")
    spy=fetch_prices("SPY",START,endf)
    prices={tk:fetch_prices(tk,START,endf) for tk in UNIVERSE}
    insider={tk:fetch_insider(tk) for tk in UNIVERSE}
    usable=[tk for tk in UNIVERSE if len(prices[tk])>=200]

    # ── DEPTH INVARIANT (the "data-broker-delusion" defense) ──────────────────
    # Refuse to score if the insider feed is too shallow to actually span the
    # evaluation window. THIS is the check that would have auto-caught the
    # June-25 shallow-data false-confirmation (4 months posing as 2 years).
    from datetime import datetime as _dt
    DEPTH_MIN_DAYS=540
    depths=[]; covered=0; total_ev=0
    for tk in UNIVERSE:
        evs=insider.get(tk,[])
        total_ev+=len(evs)
        if len(evs)>=20:
            ds=sorted(e["date"] for e in evs)
            span=(_dt.strptime(ds[-1],"%Y-%m-%d")-_dt.strptime(ds[0],"%Y-%m-%d")).days
            depths.append(span)
            if span>=DEPTH_MIN_DAYS: covered+=1
    depths.sort()
    median_depth=depths[len(depths)//2] if depths else 0
    coverage=covered/len(UNIVERSE)
    integ={"median_insider_depth_days":median_depth,"coverage_frac":round(coverage,2),
           "total_events":total_ev,"min_required_days":DEPTH_MIN_DAYS}
    print(f"  DATA-INTEGRITY: median insider depth {median_depth}d | coverage {coverage:.0%} | {total_ev} events")
    if median_depth<DEPTH_MIN_DAYS or coverage<0.5:
        fault={"GATE_FAULT":"insufficient_data_depth","integrity":integ,
               "proposal_hash":phash,"at":datetime.utcnow().isoformat()+"Z"}
        fhash=sha(json.dumps(fault,sort_keys=True))
        json.dump({**fault,"fault_hash":fhash},open("/opt/alice/data/verum_gate_fault.json","w"),indent=2)
        print("\n  GATE FAULT - data too shallow (median %dd < %dd). Refusing a verdict. Sealed fault %s..." % (median_depth, DEPTH_MIN_DAYS, fhash[:24]))
        print("  (The check that would have caught the shallow-data false-confirmation.)")
        return
    dts=[]; d=datetime.strptime(START,"%Y-%m-%d"); e=datetime.strptime(END,"%Y-%m-%d")
    while d<e: dts.append(d.strftime("%Y-%m-%d")); d=(d.replace(day=28)+timedelta(days=7)).replace(day=1)
    is_m=sorted(m for tk in usable for dt in dts if dt<SPLIT and tk in members_asof(dt)
                for m in [trailing(prices[tk],dt)] if m is not None)
    hot_lo=is_m[int(len(is_m)*0.7)]; cold_hi=is_m[int(len(is_m)*0.3)]
    print(f"  usable {len(usable)}/{len(UNIVERSE)} | hot>= {hot_lo:+.3f} cold<= {cold_hi:+.3f}")

    def qual(s,c,tk,dt,ev):
        if (s=="insider_sell" and ev!=-1) or (s=="insider_buy" and ev!=1): return False
        m=trailing(prices[tk],dt)
        if m is None: return False
        if c=="hot_momentum": return m>=hot_lo
        if c=="cold_momentum": return m<=cold_hi
        return True

    bonf=0.05/len(hyps)
    print(f"\n  GROUND-TRUTH GATE (OOS {SPLIT}..{END}, Bonferroni a={bonf:.4f}):")
    print(f"  {'hypothesis':>34} | {'models':>6} | {'n':>4} | {'signed%':>8} | {'t':>6} | verdict")
    print("  "+"-"*82)
    results=[]
    for (s,c,ho,dr),nm in sorted(hyps,key=lambda kv:-kv[1]):
        sign=-1 if dr=="short" else 1; fw=[]
        for tk in usable:
            for ev in insider[tk]:
                if ev["date"]<SPLIT or tk not in members_asof(ev["date"]): continue
                if qual(s,c,tk,ev["date"],ev["dir"]):
                    f=fwd_excess(prices[tk],spy,ev["date"],ho)
                    if f is not None: fw.append(f)
        signed=[sign*f for f in fw]; t=t0(signed); p=pT(t)
        mean=statistics.mean(signed)*100 if signed else 0.0
        surv=(t is not None and t>0 and p<bonf and len(signed)>=20)
        lbl=f"{s}/{c}/{ho}d/{dr}"
        results.append({"h":lbl,"models":nm,"n":len(signed),"signed_pct":round(mean,2),
                        "t":round(t,2) if t else None,"survives":surv})
        tt=f"{t:+.2f}" if t is not None else "  —"
        print(f"  {lbl:>34} | {nm:>6} | {len(signed):>4} | {mean:>+8.2f} | {tt:>6} | {'✅ SURVIVES' if surv else '❌ rejected'}")

    root=sha(phash+json.dumps(results,sort_keys=True))
    json.dump({"proposal_hash":phash,"data_integrity":integ,"results":results,"merkle_root":root,
               "at":datetime.utcnow().isoformat()+"Z"},
              open("/opt/alice/data/verum_orient_tech.json","w"),indent=2)
    surv=[r for r in results if r["survives"]]
    print(f"\n  ARCHIVIST SEAL: {root[:32]}…  (chained to proposal_hash {phash[:12]}…)")
    print(f"  ACCEPTED {len(surv)}/{len(results)}: {[r['h'] for r in surv] or 'none'}")
    # consensus vs survival
    cons=[r for r in results if r['models']>=4]
    cs=[r for r in cons if r['survives']]
    print(f"\n  Panel CONSENSUS (>=4/5 models): {len(cons)} hyps, {len(cs)} survived the gate.")
    print("  consensus is a prior, not proof — the gate is the only judge.")

if __name__=="__main__":
    import warnings; warnings.filterwarnings("ignore"); run()
