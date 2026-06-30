#!/usr/bin/env python3
"""
backtest_ond_definitive.py — the OND short edge, tested properly for the first time.

Everything the earlier runs lacked, now that FMP premium is live:
  - SURVIVORSHIP-COMPLETE prices: FMP serves delisted names (SIVB, FRC, FTCH...)
    that yfinance drops. The crashers — the best shorts — are finally included.
  - DEEP insider history: paginated (years, not the page-0 ~4 months that fooled us).
  - BROAD universe: point-in-time S&P 500 membership, all sectors (not tech-only).
  - MULTI-REGIME: 2015-2026 spans the 2018 selloff, 2020 crash, 2022 bear, 2024-26 bull.
  - Momentum-decile control + held-out calibration + a per-regime / per-year breakdown,
    because tech-2024 showed the edge can be regime-dependent.

Question: in hot-momentum names (narrative running hot), do insiders SELLING
underperform their same-momentum peers — and does it depend on the regime?
Report whatever it shows: real edge, regime-conditional edge, or honest null.
"""
import sys, json, statistics, math, requests
from datetime import datetime, timedelta
sys.path.insert(0, "/opt/alice/app")
from backtest_ond_pit import (build_membership, fetch_insider, trailing, fwd_excess,
                              price_on_or_after, _cget, _cput, KEY, BASE, UA)

UNIVERSE_N = 150
START = "2015-01-01"; CALIB_END = "2018-01-01"; END = "2026-03-01"
FWD = 42; HOT_DECILES = (7, 8, 9)

def fetch_prices_fmp(sym, start, end):
    c = _cget(f"px_{sym}.json", max_age_h=336)
    if c is not None:
        return [(d, v) for d, v in c]
    try:
        r = requests.get(f"{BASE}/historical-price-eod/full",
                         params={"symbol": sym, "from": start, "to": end, "apikey": KEY},
                         timeout=30, headers=UA)
        if r.status_code != 200:
            return []
        data = r.json()
        rows = data if isinstance(data, list) else data.get("historical", [])
        out = []
        for x in rows:
            d = (x.get("date") or "")[:10]
            v = x.get("adjClose", x.get("close"))
            if d and v:
                out.append((d, float(v)))
        out.sort()
        if out:
            _cput(f"px_{sym}.json", out)
        return out
    except Exception:
        return []

def decile(bounds, v):
    for i, b in enumerate(bounds):
        if v <= b:
            return i
    return len(bounds)

def welch(a, b):
    if len(a) < 5 or len(b) < 5: return None
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.pvariance(a), statistics.pvariance(b)
    se = math.sqrt(va/len(a) + vb/len(b))
    return (ma-mb)/se if se > 0 else 0.0

def run():
    print("=== DEFINITIVE OND TEST — broad, deep, survivorship-complete, multi-regime ===")
    current, union, members_asof = build_membership()
    sample = sorted(union)[:: max(1, len(union)//UNIVERSE_N)][:UNIVERSE_N]
    print(f"  universe: {len(sample)} point-in-time S&P names (all sectors)")

    spy = fetch_prices_fmp("SPY", START, END)
    prices = {}; have = 0
    for tk in sample:
        px = fetch_prices_fmp(tk, START, END)
        prices[tk] = px
        if len(px) >= 250:
            have += 1
    print(f"  survivorship-complete prices (FMP incl. delisted): {have}/{len(sample)} "
          f"= {have/len(sample):.0%}  (yfinance gave ~56%)")
    usable = [tk for tk in sample if len(prices[tk]) >= 250]

    # depth invariant on insider feed
    insider = {tk: fetch_insider(tk) for tk in usable}
    depths = []
    for tk in usable:
        ev = insider[tk]
        if len(ev) >= 20:
            ds = sorted(e["date"] for e in ev)
            depths.append((datetime.strptime(ds[-1], "%Y-%m-%d") - datetime.strptime(ds[0], "%Y-%m-%d")).days)
    median_depth = sorted(depths)[len(depths)//2] if depths else 0
    print(f"  insider history median depth: {median_depth}d "
          f"({'OK' if median_depth >= 540 else 'GATE FAULT — too shallow'})")
    if median_depth < 540:
        print("  refusing verdict on shallow insider data."); return

    # month-end dates
    dts = []; d = datetime.strptime(START, "%Y-%m-%d"); e = datetime.strptime(END, "%Y-%m-%d")
    while d < e:
        dts.append(d.strftime("%Y-%m-%d")); d = (d.replace(day=28)+timedelta(days=7)).replace(day=1)

    # momentum deciles calibrated on IN-SAMPLE (2015-2018) only
    is_m = []
    for tk in usable:
        for dt in dts:
            if dt >= CALIB_END: continue
            if tk not in members_asof(dt): continue
            m = trailing(prices[tk], dt)
            if m is not None: is_m.append(m)
    is_m.sort()
    bounds = [is_m[int(len(is_m)*q/10)] for q in range(1, 10)]

    # baseline forward-excess by decile (OOS only), per regime/year
    def regime_at(dt):
        p_now = price_on_or_after(spy, dt)
        past = (datetime.strptime(dt, "%Y-%m-%d") - timedelta(days=180)).strftime("%Y-%m-%d")
        p_past = price_on_or_after(spy, past)
        if not (p_now and p_past and p_past > 0): return "?"
        return "bull" if (p_now/p_past - 1.0) > 0 else "bear"

    def collect():
        base = {i: [] for i in range(10)}
        base_reg = {"bull": [], "bear": []}; base_yr = {}
        sell = {i: [] for i in range(10)}
        sell_reg = {"bull": [], "bear": []}; sell_yr = {}
        for tk in usable:
            px = prices[tk]
            # baseline: all (ticker, month) OOS
            for dt in dts:
                if dt < CALIB_END or dt > END or tk not in members_asof(dt): continue
                m = trailing(px, dt); f = fwd_excess(px, spy, dt, FWD)
                if m is None or f is None: continue
                dec = decile(bounds, m)
                base[dec].append(f)
                if dec in HOT_DECILES:
                    reg = regime_at(dt); yr = dt[:4]
                    if reg in base_reg: base_reg[reg].append(f)
                    base_yr.setdefault(yr, []).append(f)
            # insider SELL events
            for ev in insider[tk]:
                if ev["dir"] != -1 or ev["date"] < CALIB_END or ev["date"] > END: continue
                if tk not in members_asof(ev["date"]): continue
                m = trailing(px, ev["date"]); f = fwd_excess(px, spy, ev["date"], FWD)
                if m is None or f is None: continue
                dec = decile(bounds, m)
                sell[dec].append(f)
                if dec in HOT_DECILES:
                    reg = regime_at(ev["date"]); yr = ev["date"][:4]
                    if reg in sell_reg: sell_reg[reg].append(f)
                    sell_yr.setdefault(yr, []).append(f)
        return base, sell, base_reg, sell_reg, base_yr, sell_yr

    base, sell, base_reg, sell_reg, base_yr, sell_yr = collect()
    mean = lambda x: statistics.mean(x) if x else None

    print(f"\n{'='*70}\n  HOT-decile (7-9) OND short edge — insider SELL vs same-decile BASELINE\n{'='*70}")
    print("  edge = sell - baseline (NEGATIVE = insider-sell underperforms peers = short works)")
    hb = [f for i in HOT_DECILES for f in base[i]]
    hs = [f for i in HOT_DECILES for f in sell[i]]
    t = welch(hs, hb)
    print(f"  OVERALL: baseline {mean(hb)*100:+.2f}% (n={len(hb)}) | sell {mean(hs)*100:+.2f}% (n={len(hs)}) "
          f"| edge {(mean(hs)-mean(hb))*100:+.2f}% | t={t:+.2f}")

    print(f"\n  BY REGIME (SPY 6-month trend at the event):")
    for reg in ("bull", "bear"):
        b, s = base_reg[reg], sell_reg[reg]
        if b and s:
            tr = welch(s, b)
            print(f"    {reg:>4}: baseline {mean(b)*100:+.2f}% (n={len(b)}) | sell {mean(s)*100:+.2f}% (n={len(s)}) "
                  f"| edge {(mean(s)-mean(b))*100:+.2f}% | t={tr:+.2f}")

    print(f"\n  BY YEAR:")
    for yr in sorted(set(base_yr) | set(sell_yr)):
        b, s = base_yr.get(yr, []), sell_yr.get(yr, [])
        if b and s and len(s) >= 20:
            print(f"    {yr}: baseline {mean(b)*100:+.2f}% | sell {mean(s)*100:+.2f}% "
                  f"| edge {(mean(s)-mean(b))*100:+.2f}% (n_sell={len(s)})")

    json.dump({"overall_edge_pct": round((mean(hs)-mean(hb))*100, 3), "t": round(t, 3),
               "n_sell": len(hs), "n_base": len(hb), "survivorship_coverage": round(have/len(sample), 3)},
              open("/opt/alice/data/ond_definitive.json", "w"), indent=2)
    print(f"\n  Negative edge + |t|>2 in the broad multi-regime test = a real OND short edge.")
    print(f"  A regime split (works in bear, fails in bull) would explain the tech-2024 failure.")
    print(f"  Honest null (edge ~0 everywhere) is also a valid, final answer.")

if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore"); run()
