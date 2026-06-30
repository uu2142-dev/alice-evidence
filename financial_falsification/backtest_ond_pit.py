#!/usr/bin/env python3
"""
backtest_ond_pit.py — OND backtest on a POINT-IN-TIME universe (Tier 0).

Fixes the two flaws that confounded backtest_ond.py:
  1. SURVIVORSHIP / hand-picking — universe is now S&P 500 membership reconstructed
     as-of each date from the FMP change-log (no cherry-picked winners, no look-ahead:
     a 2024 addition is excluded from 2022 tests).
  2. MOMENTUM confound — we now compare insider-SELL events to a BASELINE of all
     universe stocks in the SAME momentum decile. The OND question becomes:
     does insider selling add downside BEYOND just being a hot/cold stock?

Free-data caveat (Tier 0): prices for fully-delisted names aren't available free.
Missing the dead names UNDERSTATES a short edge (the crashers are the best shorts),
so any short-OND signal here is a CONSERVATIVE LOWER BOUND.

Object   : Form-4 insider direction (FMP).   Narrative : trailing-60d momentum.
Outcome  : 42d forward excess return vs SPY.
"""

import sys, statistics, requests, warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")
sys.path.insert(0, "/opt/alice/app")
from backtest_ifi import fetch_prices, price_on_or_after
import os as _os, json as _json, time as _time
_CACHE=_os.path.join(_os.getenv("DATA_DIR","/opt/alice/data"),"fmp_cache")
_os.makedirs(_CACHE,exist_ok=True)
def _cget(name,max_age_h=72):
    f=_os.path.join(_CACHE,name)
    if _os.path.exists(f) and (_time.time()-_os.path.getmtime(f))/3600 < max_age_h:
        try: return _json.load(open(f))
        except Exception: return None
    return None
def _cput(name,obj):
    try: _json.dump(obj,open(_os.path.join(_CACHE,name),"w"))
    except Exception: pass


KEY = _os.getenv("FMP_API_KEY", ""); BASE="https://financialmodelingprep.com/stable"
UA={'User-Agent':'Mozilla/5.0'}
N_TICKERS=140                 # insider API budget (<250/day FMP free)
PERIOD_START="2022-06-01"; PERIOD_END="2026-03-01"
FWD=42; LOOKBACK=60

def build_membership():
    cur=_cget("sp500_current.json")
    if cur is None:
        cur=requests.get(f"{BASE}/sp500-constituent?apikey={KEY}",timeout=30,headers=UA).json()
        if not isinstance(cur,list):
            raise RuntimeError("FMP sp500-constituent unavailable (rate-limited?) and no cache: "+str(cur)[:120])
        _cput("sp500_current.json",cur)
    current={x['symbol'] for x in cur}
    changes=_cget("sp500_changes.json")
    if changes is None:
        changes=requests.get(f"{BASE}/historical-sp500-constituent?apikey={KEY}",timeout=30,headers=UA).json()
        if not isinstance(changes,list):
            raise RuntimeError("FMP historical-sp500-constituent unavailable and no cache")
        _cput("sp500_changes.json",changes)
    changes=[c for c in changes if c.get('date')]
    changes.sort(key=lambda c:c['date'])
    union=set(current)
    for c in changes:
        if c.get('symbol'): union.add(c['symbol'])
        if c.get('removedTicker'): union.add(c['removedTicker'])
    def members_asof(date):
        m=set(current)
        for c in changes:
            if c['date']>date:                       # undo changes after `date`
                if c.get('symbol'): m.discard(c['symbol'])
                if c.get('removedTicker'): m.add(c['removedTicker'])
        return m
    return current, union, members_asof

def fetch_insider(sym, pages=15):
    # PREMIUM: paginate for DEEP history (page 0 alone = only ~last 4 months).
    _cached=_cget(f"insider_{sym}.json")
    if _cached is not None: return _cached
    out=[]
    try:
        for pg in range(pages):
            r=requests.get(f"{BASE}/insider-trading/search",
                           params={"symbol":sym,"page":pg,"apikey":KEY},timeout=20,headers=UA)
            if r.status_code!=200: break
            data=r.json()
            if not isinstance(data,list) or not data: break
            for x in data:
                ad=(x.get("acquisitionOrDisposition") or "").upper()
                d=(x.get("transactionDate") or "")[:10]
                if ad in ("A","D") and d: out.append({"date":d,"dir":1 if ad=="A" else -1})
            if len(data)<100: break
        if out: _cput(f"insider_{sym}.json",out)
        return out
    except Exception: return out

def trailing(series,date,lb=LOOKBACK):
    p0=price_on_or_after(series,(datetime.strptime(date,"%Y-%m-%d")-timedelta(days=lb)).strftime("%Y-%m-%d"))
    p1=price_on_or_after(series,date)
    return (p1/p0-1.0) if (p0 and p1 and p0>0) else None

def fwd_excess(series,spy,date,h=FWD):
    e0=price_on_or_after(series,date); s0=price_on_or_after(spy,date)
    if not(e0 and s0 and e0>0 and s0>0): return None
    fut=(datetime.strptime(date,"%Y-%m-%d")+timedelta(days=h)).strftime("%Y-%m-%d")
    e1=price_on_or_after(series,fut); s1=price_on_or_after(spy,fut)
    if not(e1 and s1 and e1>0 and s1>0): return None
    return (e1/e0-1.0)-(s1/s0-1.0)

def decile(boundaries,v):
    for i,b in enumerate(boundaries):
        if v<=b: return i
    return len(boundaries)

def run():
    print("Building point-in-time S&P 500 membership...")
    current,union,members_asof=build_membership()
    print(f"  current={len(current)} union(ever)={len(union)}")

    # neutral deterministic ticker sample from the full historical union
    sample=sorted(union)[:: max(1,len(union)//N_TICKERS)][:N_TICKERS]
    print(f"  sampled {len(sample)} tickers (neutral, alphabetical stride)")

    spy=fetch_prices("SPY",PERIOD_START,(datetime.now()+timedelta(days=1)).strftime("%Y-%m-%d"))
    prices={}; have=0; missing=0
    for tk in sample:
        px=fetch_prices(tk,PERIOD_START,(datetime.now()+timedelta(days=1)).strftime("%Y-%m-%d"))
        prices[tk]=px
        if len(px)>=120: have+=1
        else: missing+=1
    print(f"  free prices: {have}/{len(sample)} usable ({missing} missing/thin = survivorship gap proxy)")

    # month-end sample dates
    dates=[]
    d=datetime.strptime(PERIOD_START,"%Y-%m-%d")
    end=datetime.strptime(PERIOD_END,"%Y-%m-%d")
    while d<end:
        dates.append(d.strftime("%Y-%m-%d")); d=(d.replace(day=28)+timedelta(days=7)).replace(day=1)

    # BASELINE: all (ticker,month) momentum + fwd excess, member-as-of only
    base=[]
    for tk in sample:
        px=prices[tk]
        if len(px)<120: continue
        for dt in dates:
            if tk not in members_asof(dt): continue
            m=trailing(px,dt); f=fwd_excess(px,spy,dt)
            if m is not None and f is not None: base.append((m,f))
    base.sort(key=lambda x:x[0])
    moms=[m for m,_ in base]
    bounds=[moms[int(len(moms)*q/10)] for q in range(1,10)] if moms else []
    print(f"\nBaseline observations: {len(base)}  (momentum deciles built)")

    # baseline mean fwd-excess per decile
    bdec={i:[] for i in range(10)}
    for m,f in base: bdec[decile(bounds,m)].append(f)

    # EVENTS: insider sells/buys, member-as-of, bucket by same deciles
    sell={i:[] for i in range(10)}; buy={i:[] for i in range(10)}
    n_ev=0
    for tk in sample:
        px=prices[tk]
        if len(px)<120: continue
        for ev in fetch_insider(tk):
            if not(PERIOD_START<=ev["date"]<=PERIOD_END): continue
            if tk not in members_asof(ev["date"]): continue
            m=trailing(px,ev["date"]); f=fwd_excess(px,spy,ev["date"])
            if m is None or f is None: continue
            n_ev+=1
            (sell if ev["dir"]==-1 else buy)[decile(bounds,m)].append(f)

    print(f"Insider events scored (in-universe): {n_ev}")
    print(f"\n{'='*74}\n  42d fwd EXCESS vs SPY — insider SELL vs BASELINE, by momentum decile\n{'='*74}")
    print("  decile 0=coldest .. 9=hottest narrative.  'edge' = sell − baseline (neg = short works)")
    print(f"  {'decile':>6} | {'baseline%':>9} {'(n)':>6} | {'sell%':>7} {'(n)':>5} | {'edge%':>7}")
    print("  "+"-"*60)
    def mean(x): return statistics.mean(x) if x else None
    for i in range(10):
        b=mean(bdec[i]); s=mean(sell[i])
        if b is None: continue
        edge = (s-b) if s is not None else None
        sstr=f"{s*100:>7.2f} {len(sell[i]):>5}" if s is not None else f"{'—':>7} {len(sell[i]):>5}"
        estr=f"{edge*100:>+7.2f}" if edge is not None else f"{'—':>7}"
        print(f"  {i:>6} | {b*100:>9.2f} {len(bdec[i]):>6} | {sstr} | {estr}")

    # headline: hottest 3 deciles
    hot_sell=[f for i in (7,8,9) for f in sell[i]]
    hot_base=[f for i in (7,8,9) for f in bdec[i]]
    if hot_sell and hot_base:
        print(f"\n  HOT deciles (7-9): baseline {mean(hot_base)*100:+.2f}% (n={len(hot_base)}) | "
              f"insider-SELL {mean(hot_sell)*100:+.2f}% (n={len(hot_sell)}) | "
              f"OND edge {(mean(hot_sell)-mean(hot_base))*100:+.2f}%")
        # Welch's t-test: is hot-decile SELL mean < baseline mean?
        import math, random
        def var(x): m=mean(x); return sum((v-m)**2 for v in x)/(len(x)-1)
        ms,mb=mean(hot_sell),mean(hot_base); vs,vb=var(hot_sell),var(hot_base)
        se=math.sqrt(vs/len(hot_sell)+vb/len(hot_base))
        t=(ms-mb)/se if se>0 else 0
        print(f"  Welch t = {t:+.2f}  (|t|>1.96 ≈ 95% significant; negative = sell underperforms)")
        # Bootstrap: P(edge < 0) over 2000 resamples
        random.seed(42); neg=0; B=2000
        for _ in range(B):
            es=sum(random.choice(hot_sell) for _ in range(120))/120
            eb=sum(random.choice(hot_base) for _ in range(120))/120
            if es-eb<0: neg+=1
        print(f"  Bootstrap P(short-OND edge < 0) = {neg/B:.1%}  (n=2000 resamples)")
    print("\n  Short-OND validated only if 'edge' is NEGATIVE in HOT deciles (sell underperforms")
    print("  peers in the same momentum state). Free-data => conservative lower bound.")

if __name__=="__main__": run()
