#!/usr/bin/env python3
"""
backtest_ifi.py — Honest backtest of the IFI thesis.

Central claim under test: congressional STOCK Act trades (the IFI signal)
contain information that LEADS the traded stock's price. If true, BUY trades
should precede positive forward returns and SELL trades negative ones, and
there should be a horizon at which that predictive power peaks (the real,
MEASURED lead time — replacing the assumed ~21 days in the brain).

Real data only:
  - Trades : FMP stable congressional feeds (senate-latest + house-latest)
  - Prices : yfinance daily closes (ground truth)

Two reference dates per trade:
  - transaction_date : when the member traded (tests whether they have an edge)
  - disclosure_date  : when the public could act (tests the TRADEABLE signal,
                       i.e. what ALICE/anyone following disclosures could capture)

Forward return for a trade at horizon h:
  entry = close on first trading day >= ref_date
  exit  = close on first trading day >= ref_date + h calendar days
  signed_return = (exit/entry - 1) * direction      (+ = trade was "right")

Honest reporting: sample sizes shown everywhere; no overclaiming on thin data.
"""

import os, sys, json, warnings, statistics
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import requests

warnings.filterwarnings("ignore")

FMP_KEY  = os.environ.get("FMP_API_KEY", "")
FMP_BASE = "https://financialmodelingprep.com/stable"
UA       = {"User-Agent": "Mozilla/5.0"}
HORIZONS = [5, 10, 21, 42, 63]   # calendar-day forward windows to test

# Optional: map tickers to ALICE sectors for a per-sector breakdown
try:
    sys.path.insert(0, "/opt/alice/app")
    from drift_indices.tech_sector import patch_existing_modules
    from drift_indices.new_sectors import patch_existing_modules as _pn
    patch_existing_modules(); _pn()
    import drift_indices.ifi as _ifi
    TICKER_SECTOR = dict(_ifi.TICKER_SECTOR)
except Exception:
    TICKER_SECTOR = {}


# ── DATA ──────────────────────────────────────────────────────────────────────

def fetch_congress_trades():
    """All available recent congressional trades from FMP, normalised."""
    out = []
    for ep in ("senate-latest", "house-latest"):
        try:
            r = requests.get(f"{FMP_BASE}/{ep}", params={"apikey": FMP_KEY},
                             timeout=20, headers=UA)
            if r.status_code != 200:
                print(f"  WARN {ep}: HTTP {r.status_code}")
                continue
            for it in r.json():
                tkr = (it.get("symbol") or "").upper().strip()
                if not tkr or not tkr.isalpha():
                    continue
                ttype = (it.get("type") or "").lower()
                if "purchase" in ttype or "buy" in ttype:
                    direction = 1
                elif "sale" in ttype or "sell" in ttype:
                    direction = -1
                else:
                    continue
                td = (it.get("transactionDate") or "")[:10]
                dd = (it.get("disclosureDate") or td)[:10]
                if not td:
                    continue
                member = ((str(it.get("firstName") or "") + " " +
                           str(it.get("lastName") or "")).strip()
                          or it.get("office") or "Unknown")
                out.append({"ticker": tkr, "direction": direction,
                            "transaction_date": td, "disclosure_date": dd,
                            "amount": it.get("amount", ""), "member": member})
        except Exception as e:
            print(f"  WARN {ep}: {e}")
    # dedupe identical rows
    seen, uniq = set(), []
    for t in out:
        k = (t["ticker"], t["direction"], t["transaction_date"], t["disclosure_date"])
        if k not in seen:
            seen.add(k); uniq.append(t)
    return uniq


def fetch_prices(ticker, start, end):
    """yfinance daily closes as a sorted list of (date, close)."""
    import yfinance as yf, math
    try:
        df = yf.download(ticker, start=start, end=end, interval="1d",
                         progress=False, auto_adjust=True)
        if df is None or len(df) == 0:
            return []
        closes = df["Close"].values.astype(float).flatten()
        dates  = [d.strftime("%Y-%m-%d") for d in df.index]
        return [(d, float(c)) for d, c in zip(dates, closes) if not math.isnan(c)]
    except Exception:
        return []


def price_on_or_after(series, target_date):
    """First (date, close) with date >= target_date."""
    for d, c in series:
        if d >= target_date:
            return c
    return None


# ── BACKTEST ──────────────────────────────────────────────────────────────────

def _raw_window_return(series, ref_date, horizon_days):
    """Unsigned (entry->exit) return of `series` over the window, or None."""
    entry = price_on_or_after(series, ref_date)
    if not entry or entry <= 0:
        return None
    exit_date = (datetime.strptime(ref_date, "%Y-%m-%d")
                 + timedelta(days=horizon_days)).strftime("%Y-%m-%d")
    exit_px = price_on_or_after(series, exit_date)
    if not exit_px or exit_px <= 0:
        return None
    return exit_px / entry - 1.0


def forward_return(series, ref_date, horizon_days, direction,
                   bench=None):
    """
    Signed forward return over `horizon_days` from ref_date.
    If `bench` (SPY series) is given, returns market-relative EXCESS return
    (stock return minus SPY return over the same window) * direction — i.e. alpha.
    """
    r = _raw_window_return(series, ref_date, horizon_days)
    if r is None:
        return None
    if bench is not None:
        b = _raw_window_return(bench, ref_date, horizon_days)
        if b is None:
            return None
        r = r - b
    return r * direction


def summarize(returns):
    """Mean signed return, hit rate, n for a list of signed returns."""
    rs = [r for r in returns if r is not None]
    if not rs:
        return None
    hit = sum(1 for r in rs if r > 0) / len(rs)
    return {"n": len(rs),
            "mean_signed_ret_pct": round(statistics.mean(rs) * 100, 3),
            "median_pct": round(statistics.median(rs) * 100, 3),
            "hit_rate_pct": round(hit * 100, 1)}


def run():
    if not FMP_KEY:
        print("FMP_API_KEY not set — aborting."); return

    print("Fetching congressional trades (FMP)...")
    trades = fetch_congress_trades()
    print(f"  {len(trades)} unique trades")
    if not trades:
        return

    dates = [t["transaction_date"] for t in trades]
    print(f"  date range: {min(dates)} -> {max(dates)}")

    # Fetch prices once per ticker over the full needed window (+ horizon buffer)
    tickers = sorted({t["ticker"] for t in trades})
    start = (datetime.strptime(min(dates), "%Y-%m-%d") - timedelta(days=10)).strftime("%Y-%m-%d")
    end   = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Fetching prices for {len(tickers)} tickers ({start}..{end})...")
    prices = {}
    for i, tk in enumerate(tickers):
        prices[tk] = fetch_prices(tk, start, end)
    have = sum(1 for tk in tickers if prices[tk])
    print(f"  price history for {have}/{len(tickers)} tickers")

    # Benchmark for market-relative (excess) returns
    print("Fetching SPY benchmark...")
    spy = fetch_prices("SPY", start, end)
    print(f"  SPY days: {len(spy)}")

    # Accumulate signed returns per (ref_basis, horizon), raw AND excess-vs-SPY
    for basis in ("transaction_date", "disclosure_date"):
        for mode, bench in (("RAW return", None), ("EXCESS vs SPY (alpha)", spy)):
            print(f"\n{'='*66}\n  {mode} by horizon — basis: {basis}\n{'='*66}")
            print(f"  {'horizon':>8} | {'n':>4} | {'mean signed %':>13} | {'median %':>9} | {'hit %':>6}")
            print("  " + "-"*54)
            for h in HORIZONS:
                rets = []
                for t in trades:
                    s = prices.get(t["ticker"])
                    if not s:
                        continue
                    rets.append(forward_return(s, t[basis], h, t["direction"], bench))
                summ = summarize(rets)
                if summ:
                    print(f"  {h:>6}d | {summ['n']:>4} | {summ['mean_signed_ret_pct']:>13} "
                          f"| {summ['median_pct']:>9} | {summ['hit_rate_pct']:>6}")

    # Per-sector breakdown at the canonical 21-day horizon (transaction basis)
    if TICKER_SECTOR:
        print(f"\n{'='*64}\n  PER-SECTOR (21d, transaction basis, ALICE watchlist only)\n{'='*64}")
        bysec = defaultdict(list)
        for t in trades:
            sec = TICKER_SECTOR.get(t["ticker"])
            s = prices.get(t["ticker"])
            if not sec or not s:
                continue
            bysec[sec].append(forward_return(s, t["transaction_date"], 21, t["direction"]))
        print(f"  {'sector':>12} | {'n':>3} | {'mean signed %':>13} | {'hit %':>6}")
        print("  " + "-"*44)
        for sec in sorted(bysec):
            summ = summarize(bysec[sec])
            if summ:
                print(f"  {sec:>12} | {summ['n']:>3} | {summ['mean_signed_ret_pct']:>13} | {summ['hit_rate_pct']:>6}")

    print("\nNOTES:")
    print("  - signed return = (exit/entry - 1) * direction; >0 means the trade was 'right'.")
    print("  - hit% > 50 and mean signed % > 0 => predictive edge at that horizon.")
    print("  - disclosure basis = the TRADEABLE signal (what a follower could capture).")
    print("  - small n => treat as directional, not conclusive.")


if __name__ == "__main__":
    run()
