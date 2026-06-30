#!/usr/bin/env python3
"""
backtest_cluster_deep.py — the clustering thesis, tested at depth.

Clustering is the one lead that survived the easy tests: congressional CONSENSUS
(multiple distinct members trading the same ticker+direction within a window)
showed a monotonic alpha lift where individual trades did not — but on only
24 members / 186 trades (cluster-3+ was n=3). Premium pagination now reaches
~2017, ~12.6k disclosures. Two questions the thin sample could not answer:
  1. Does the cluster-size -> alpha monotonic lift survive a LARGE n?
  2. Does it survive a MOMENTUM-DECILE control — is consensus real information,
     or just members piling into names already running? (The control that turned
     OND's "+16%" into a null.)
Disclosure-date basis (only actionable date). Survivorship-complete FMP prices.
Split BUY vs SELL (informed buying is the classic signal; selling is mostly noise).

Memory-light: streams one ticker at a time (1.8k tickers OOM'd if held resident).
"""
import sys, json, statistics, bisect
from collections import defaultdict
from datetime import datetime, timedelta
import requests
sys.path.insert(0, "/opt/alice/app")
from backtest_ond_pit import _cget, _cput, KEY, BASE, UA, LOOKBACK
from backtest_ond_definitive import fetch_prices_fmp, decile, welch

WINDOW = 30; FWD = 42; LB = LOOKBACK
START = "2017-01-01"; CALIB_END = "2019-06-01"; END = "2026-03-01"

# bisect-indexed lookups; only SPY + the current ticker held resident.
_PIDX = {}
def index_one(tk, series):
    _PIDX[tk] = ([d for d, _ in series], [c for _, c in series])
def _poa(tk, date):
    idx = _PIDX.get(tk)
    if not idx: return None
    ds, cs = idx
    i = bisect.bisect_left(ds, date)
    return cs[i] if i < len(ds) else None
def trailing(tk, date):
    p0 = _poa(tk, (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=LB)).strftime("%Y-%m-%d"))
    p1 = _poa(tk, date)
    return (p1/p0 - 1.0) if (p0 and p1 and p0 > 0) else None
def fwd_excess(tk, date, h=FWD):
    e0 = _poa(tk, date); s0 = _poa("SPY", date)
    if not (e0 and s0 and e0 > 0 and s0 > 0): return None
    fut = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=h)).strftime("%Y-%m-%d")
    e1 = _poa(tk, fut); s1 = _poa("SPY", fut)
    if not (e1 and s1 and e1 > 0 and s1 > 0): return None
    return (e1/e0 - 1.0) - (s1/s0 - 1.0)

def fetch_congress_deep(max_pages=130):
    c = _cget("congress_deep.json", max_age_h=168)
    if c is not None:
        return c
    out = []
    for ep in ("senate-latest", "house-latest"):
        for page in range(max_pages):
            try:
                r = requests.get(f"{BASE}/{ep}", params={"page": page, "limit": 100, "apikey": KEY},
                                 timeout=25, headers=UA)
                if r.status_code != 200: break
                j = r.json()
                if not j: break
                for it in j:
                    tk = (it.get("symbol") or "").upper().strip()
                    if not tk or not tk.isalpha(): continue
                    tt = (it.get("type") or "").lower()
                    d = 1 if ("purchase" in tt or "buy" in tt) else (-1 if ("sale" in tt or "sell" in tt) else 0)
                    if d == 0: continue
                    td = (it.get("transactionDate") or "")[:10]
                    dd = (it.get("disclosureDate") or td)[:10]
                    if not (td and dd): continue
                    mem = ((str(it.get("firstName") or "") + " " + str(it.get("lastName") or "")).strip()
                           or it.get("office") or "?")
                    out.append({"ticker": tk, "dir": d, "td": td, "dd": dd, "member": mem})
            except Exception:
                break
    seen, uniq = set(), []
    for t in out:
        k = (t["ticker"], t["dir"], t["td"], t["member"])
        if k not in seen:
            seen.add(k); uniq.append(t)
    _cput("congress_deep.json", uniq)
    return uniq

def annotate_clusters(trades):
    idx = defaultdict(list)
    for t in trades:
        idx[(t["ticker"], t["dir"])].append(t)
    for lst in idx.values():
        tds = [(t, datetime.strptime(t["td"], "%Y-%m-%d")) for t in lst]
        for t, t0 in tds:
            t["csize"] = len({o["member"] for o, od in tds if abs((od - t0).days) <= WINDOW})
    return trades

def run():
    print("=== CLUSTERING AT DEPTH — does congressional consensus survive n + momentum control? ===")
    trades = [t for t in fetch_congress_deep() if START <= t["dd"] <= END]
    print(f"  {len(trades)} trades, {len({t['member'] for t in trades})} distinct members, {START[:4]}-{END[:4]}")
    trades = annotate_clusters(trades)
    by_ticker = defaultdict(list)
    for t in trades:
        by_ticker[t["ticker"]].append(t)
    tickers = sorted(by_ticker)
    print(f"  {len(tickers)} distinct tickers; streaming survivorship-complete prices (cached)...")

    index_one("SPY", fetch_prices_fmp("SPY", START, END))     # resident
    dts = []; d = datetime.strptime(START, "%Y-%m-%d"); e = datetime.strptime(END, "%Y-%m-%d")
    while d < e:
        dts.append(d.strftime("%Y-%m-%d")); d = (d.replace(day=28) + timedelta(days=7)).replace(day=1)

    # PASS 1 — momentum deciles from in-sample window (one ticker resident at a time)
    is_m = []; plen = {}
    for tk in tickers:
        s = fetch_prices_fmp(tk, START, END); plen[tk] = len(s)
        if len(s) < 250: continue
        index_one(tk, s)
        for dt in dts:
            if dt >= CALIB_END: continue
            m = trailing(tk, dt)
            if m is not None: is_m.append(m)
        if tk != "SPY": _PIDX.pop(tk, None)
    have = sum(1 for tk in tickers if plen[tk] >= 250)
    print(f"  price coverage: {have}/{len(tickers)} = {have/len(tickers):.0%}")
    is_m.sort()
    bounds = [is_m[int(len(is_m)*q/10)] for q in range(1, 10)]

    # PASS 2 — baseline by decile (OOS) + cluster events (store decile, neutralize after)
    base = {i: [] for i in range(10)}
    events = []      # (dir, label, decile, fwd_excess)
    lab = lambda c: "1" if c == 1 else ("2" if c == 2 else "3+")
    for tk in tickers:
        if plen[tk] < 250: continue
        index_one(tk, fetch_prices_fmp(tk, START, END))
        for dt in dts:
            if dt < CALIB_END: continue
            m = trailing(tk, dt); f = fwd_excess(tk, dt)
            if m is None or f is None: continue
            base[decile(bounds, m)].append(f)
        for t in by_ticker[tk]:
            if t["dd"] < CALIB_END: continue
            m = trailing(tk, t["dd"]); f = fwd_excess(tk, t["dd"])
            if m is None or f is None: continue
            events.append((t["dir"], lab(t["csize"]), decile(bounds, m), f))
        if tk != "SPY": _PIDX.pop(tk, None)

    print(f"  events={len(events)}  baseline pts={sum(len(v) for v in base.values())}")
    base_mean = {i: (statistics.mean(base[i]) if base[i] else 0.0) for i in range(10)}
    edges = defaultdict(list); raws = defaultdict(list)
    for dv, L, dec, f in events:
        edges[(dv, L)].append(f - base_mean[dec])
        raws[(dv, L)].append(f)
    print(f"\n  momentum-NEUTRALIZED forward excess ({FWD}d, disclosure basis), OOS {CALIB_END[:7]}+")
    summary = {}
    for dname, dv in (("BUY", 1), ("SELL", -1)):
        print(f"\n  [{dname}]  edge = cluster alpha - same-momentum-decile baseline")
        singles = edges.get((dv, "1"), [])
        for L in ("1", "2", "3+"):
            e = edges.get((dv, L), []); r = raws.get((dv, L), [])
            if not e: continue
            t = welch(e, singles) if (L != "1" and len(singles) >= 5) else None
            ts = f"| t_vs_single={t:+.2f}" if t is not None else ""
            print(f"    cluster {L:>2}: n={len(e):>4} | raw alpha {statistics.mean(r)*100:+.2f}% "
                  f"| NEUTRALIZED edge {statistics.mean(e)*100:+.2f}% {ts}")
            summary[f"{dname}_{L}"] = {"n": len(e), "raw_pct": round(statistics.mean(r)*100, 3),
                                       "edge_pct": round(statistics.mean(e)*100, 3),
                                       "t_vs_single": round(t, 3) if t is not None else None}
    json.dump(summary, open("/opt/alice/data/cluster_deep.json", "w"), indent=2)
    print("\n  Thesis holds if BUY neutralized edge RISES with cluster size AND cluster-3+ t>2.")
    print("  If the lift vanishes after momentum control, the earlier signal was members")
    print("  chasing already-hot names. Whatever it shows is the answer.")

if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore"); run()
