#!/usr/bin/env python3
"""
backtest_cot.py — Does COT (managed-money positioning) lead price?

Unlike the congressional backtest (capped at ~200 trades), CFTC COT has ~10y of
weekly history per market — enough sample to QUANTIFY, not just direction.

Data (free):
  - COT     : CFTC Socrata disaggregated futures (resource 72hh-3qpy)
  - Prices  : yfinance daily closes of a liquid ETF tracking each market

Signals tested (managed money = speculators):
  1. net_level   : (mm_long - mm_short) / (mm_long + mm_short)         [level]
  2. net_chg_4w  : 4-week change in net_level                          [momentum]
  3. cot_index   : percentile of net position over trailing 156 weeks  [extremity]

For each weekly report we measure the tracking ETF's forward return at several
horizons and compute, pooled across markets (signal z-scored per market):
  - IC   : Pearson corr(signal, forward_return)  — the standard quant measure
  - hit% : sign(signal) == sign(forward_return)
Plus a contrarian check: forward return by cot_index quintile.

Honest: report n; |IC| ~0.0-0.05 = noise, 0.05-0.10 = weak-real, >0.10 = strong.
"""

import sys, statistics, time
from datetime import datetime, timedelta
import requests

sys.path.insert(0, "/opt/alice/app")
from backtest_ifi import fetch_prices, price_on_or_after

UA = {"User-Agent": "Mozilla/5.0"}
COT_URL = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
HORIZONS = [5, 10, 21, 42]
LOOKBACK_WEEKS = 156   # 3y for cot_index percentile

# market name (CFTC 'like' pattern) -> tracking ETF
MARKETS = {
    "GLD":  "GOLD - COMMODITY EXCHANGE INC.",
    "SLV":  "SILVER - COMMODITY EXCHANGE INC.",
    "USO":  "CRUDE OIL%NEW YORK MERCANTILE EXCHANGE",
    "UNG":  "NATURAL GAS%NEW YORK MERCANTILE EXCHANGE",
    "CORN": "CORN - CHICAGO BOARD OF TRADE",
    "SOYB": "SOYBEANS - CHICAGO BOARD OF TRADE",
    "WEAT": "WHEAT-SRW - CHICAGO BOARD OF TRADE",
    "CPER": "COPPER%COMMODITY EXCHANGE INC.",
}


def fetch_cot_series(like_pattern):
    """Weekly (date, {spec_net, comm_net}) for the dominant matching market.
    spec_net = managed-money net ratio; comm_net = (producer/merchant + swap
    dealer) net ratio — the 'commercial hedger / smart money' construct."""
    try:
        r = requests.get(COT_URL, params={
            "$where": f"market_and_exchange_names like '{like_pattern}'",
            "$select": ("report_date_as_yyyy_mm_dd,market_and_exchange_names,"
                        "m_money_positions_long_all,m_money_positions_short_all,"
                        "prod_merc_positions_long,prod_merc_positions_short,"
                        "swap_positions_long_all,swap__positions_short_all"),
            "$order": "report_date_as_yyyy_mm_dd ASC",
            "$limit": 2000,
        }, timeout=60, headers=UA)
        if r.status_code != 200:
            return []
        rows = r.json()
    except Exception:
        return []
    by_mkt = {}
    for x in rows:
        by_mkt.setdefault(x["market_and_exchange_names"], []).append(x)
    if not by_mkt:
        return []
    dominant = max(by_mkt.values(), key=len)

    def g(x, *keys):
        for k in keys:
            if x.get(k) not in (None, ""):
                try: return float(x[k])
                except (ValueError, TypeError): pass
        return 0.0

    series = []
    for x in dominant:
        try:
            sl = g(x, "m_money_positions_long_all"); ss = g(x, "m_money_positions_short_all")
            cl = g(x, "prod_merc_positions_long") + g(x, "swap_positions_long_all")
            cs = g(x, "prod_merc_positions_short") + g(x, "swap__positions_short_all")
            stot, ctot = sl + ss, cl + cs
            if stot < 100 or ctot < 100:
                continue
            series.append((x["report_date_as_yyyy_mm_dd"][:10],
                           {"spec_net": (sl - ss)/stot, "comm_net": (cl - cs)/ctot}))
        except (KeyError, ValueError, TypeError):
            continue
    return series


def pct_rank(window, value):
    if not window:
        return 0.5
    return sum(1 for w in window if w <= value) / len(window)


def pearson(xs, ys):
    n = len(xs)
    if n < 10:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    dx = sum((x-mx)**2 for x in xs) ** 0.5
    dy = sum((y-my)**2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return num / (dx*dy)


def zscore(vals):
    m = statistics.mean(vals); s = statistics.pstdev(vals) or 1.0
    return [(v-m)/s for v in vals]


SIGNALS = ("spec_net", "comm_net", "comm_index", "divergence")

def run():
    pooled = {sig: {h: {"sig": [], "ret": []} for h in HORIZONS} for sig in SIGNALS}
    quintile_ret = {q: [] for q in range(5)}   # comm_index quintile -> 21d returns
    per_market = {}

    for etf, pattern in MARKETS.items():
        cot = fetch_cot_series(pattern)
        if len(cot) < 60:
            print(f"  {etf}: only {len(cot)} COT weeks — skip")
            continue
        start = (datetime.strptime(cot[0][0], "%Y-%m-%d") - timedelta(days=10)).strftime("%Y-%m-%d")
        end   = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        px = fetch_prices(etf, start, end)
        if len(px) < 200:
            print(f"  {etf}: thin price history ({len(px)}) — skip")
            continue

        comm_hist = [v["comm_net"] for _, v in cot]
        recs = []
        for i, (d, v) in enumerate(cot):
            comm_idx = pct_rank(comm_hist[max(0, i-LOOKBACK_WEEKS):i], v["comm_net"]) if i >= 20 else None
            entry = price_on_or_after(px, d)
            if not entry or entry <= 0:
                continue
            fwd = {}
            for h in HORIZONS:
                exitd = (datetime.strptime(d, "%Y-%m-%d") + timedelta(days=h)).strftime("%Y-%m-%d")
                ex = price_on_or_after(px, exitd)
                fwd[h] = (ex/entry - 1.0) if ex and ex > 0 else None
            recs.append({
                "spec_net":   v["spec_net"],
                "comm_net":   v["comm_net"],
                "comm_index": comm_idx,
                "divergence": v["spec_net"] - v["comm_net"],
                "fwd": fwd,
            })

        # per-market IC at 21d on commercial net (smart-money construct)
        xs = [r["comm_net"] for r in recs if r["fwd"][21] is not None]
        ys = [r["fwd"][21] for r in recs if r["fwd"][21] is not None]
        per_market[etf] = (len(xs), pearson(xs, ys))

        for sig in SIGNALS:
            valid = [r for r in recs if r[sig] is not None]
            if len(valid) < 20:
                continue
            zs = zscore([r[sig] for r in valid])
            for j, r in enumerate(valid):
                for h in HORIZONS:
                    if r["fwd"][h] is not None:
                        pooled[sig][h]["sig"].append(zs[j])
                        pooled[sig][h]["ret"].append(r["fwd"][h])

        for r in recs:
            if r["comm_index"] is not None and r["fwd"][21] is not None:
                quintile_ret[min(4, int(r["comm_index"]*5))].append(r["fwd"][21])

        print(f"  {etf}: {len(recs)} weeks ({cot[0][0]}..{cot[-1][0]})")

    # ── RESULTS ──────────────────────────────────────────────────────────────
    print(f"\n{'='*68}\n  INFORMATION COEFFICIENT (pooled, signal z-scored per market)\n{'='*68}")
    print("  IC = corr(signal, forward return). |IC|<0.05 noise · 0.05-0.10 weak · >0.10 strong")
    desc = {"spec_net": "speculator net (managed money)",
            "comm_net": "COMMERCIAL net (producer+swap = 'smart money')",
            "comm_index": "commercial positioning percentile (156wk)",
            "divergence": "spec_net - comm_net (crowding)"}
    for sig in SIGNALS:
        print(f"\n  Signal: {sig}  — {desc[sig]}")
        print(f"    {'horizon':>8} | {'n':>5} | {'IC':>7} | {'hit %':>6}")
        print("    " + "-"*34)
        for h in HORIZONS:
            xs, ys = pooled[sig][h]["sig"], pooled[sig][h]["ret"]
            ic = pearson(xs, ys)
            if ic is None:
                continue
            hit = sum(1 for x, y in zip(xs, ys) if (x > 0) == (y > 0)) / len(xs) * 100
            print(f"    {h:>6}d | {len(xs):>5} | {ic:>+7.3f} | {hit:>6.1f}")

    print(f"\n{'='*68}\n  SMART-MONEY CHECK — 21d return by COMMERCIAL-index quintile\n{'='*68}")
    print(f"    {'quintile':>24} | {'n':>5} | {'mean 21d ret %':>14}")
    print("    " + "-"*50)
    labels = ["Q1 (comm most net-SHORT)", "Q2", "Q3", "Q4", "Q5 (comm most net-LONG)"]
    for q in range(5):
        rs = quintile_ret[q]
        if rs:
            print(f"    {labels[q]:>24} | {len(rs):>5} | {statistics.mean(rs)*100:>14.2f}")

    print(f"\n{'='*68}\n  PER-MARKET IC (commercial net, 21d)\n{'='*68}")
    for etf, (n, ic) in sorted(per_market.items(), key=lambda kv: -(kv[1][1] or 0)):
        if ic is not None:
            print(f"    {etf:>5} | n={n:>4} | IC={ic:+.3f}")

    print("\nNOTES:")
    print("  - comm_net +IC => commercials ('smart money') lead price -> a real edge.")
    print("  - smart-money check: if Q5(comm most long) OUTperforms Q1 => commercials predict.")
    print("  - divergence: large +IC contrarian read of crowded spec positioning.")
    print("  - COT released ~3 trading days after report date; this study uses report date.")


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    run()
