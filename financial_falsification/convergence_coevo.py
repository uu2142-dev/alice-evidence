#!/usr/bin/env python3
"""
convergence_coevo.py — the one unfalsified claim: is the nonlinear COMBINATION of
individually-weak vectors, evolved under the anti-data gate, more than the sum of
its parts?

Every backtest tonight tested vectors IN ISOLATION; all were weak/null after
controls (OND t=0.09, COT noise, clustering +0.15% absolute). ALICE's actual
thesis was never single-vector — it's that the gated NEAT co-evolution finds a
nonlinear convergence the pieces can't show alone. This tests that directly.

Features (real, point-in-time, ticker-level — the subset we have honest data for):
  momentum, insider_net, insider_breadth, congress_net, congress_consensus
Label: forward 42d excess vs SPY.

Method:
  - Panel built point-in-time (no look-ahead), standardized on TRAIN stats only,
    split walk-forward (train < 2022-07, test >= 2022-07).
  - GATED NEAT: each generation scores on a FRESH train batch minus performance on
    label-shuffled data (the anti-data check). Real NEAT engine (neat_alice).
  - Baselines: (1) best SINGLE feature OOS IC; (2) NAIVE combination (fixed-batch
    overfit) OOS IC.  Convergence holds iff GATED OOS IC > best single, robustly.
This is still ticker-level statistical edge, not the personal/alignment layer.
"""
import sys, json, bisect, statistics, os
import numpy as np
sys.path.insert(0, "/opt/alice/app")
import neat_alice as NA
FEAT_NAMES = ["momentum", "insider_net", "insider_breadth", "congress_net", "congress_consensus"]
NFEAT = len(FEAT_NAMES); NA.INPUT_DIM = NFEAT
from neat_alice import NEATMutator, create_minimal_genome
from backtest_ond_pit import _cget, _cput, KEY, BASE, UA, LOOKBACK, fetch_insider
from backtest_ond_definitive import fetch_prices_fmp
from backtest_cluster_deep import fetch_congress_deep
from datetime import datetime, timedelta
from collections import defaultdict

START = "2017-01-01"; SPLIT = "2022-07-01"; END = "2026-03-01"; FWD = 42; LB = LOOKBACK
PANEL = "/opt/alice/data/conv_panel.npz"
POP = 50; GENS = 40; BATCH = 1500; ELITE = 5; NSEEDS = 6

_PIDX = {}
def _idx(tk, series): _PIDX[tk] = ([d for d, _ in series], [c for _, c in series])
def _poa(tk, date):
    idx = _PIDX.get(tk)
    if not idx: return None
    ds, cs = idx; i = bisect.bisect_left(ds, date)
    return cs[i] if i < len(ds) else None
def _ret(tk, d0, d1):
    p0 = _poa(tk, d0); p1 = _poa(tk, d1)
    return (p1/p0 - 1.0) if (p0 and p1 and p0 > 0) else None

def build_panel():
    if os.path.exists(PANEL):
        z = np.load(PANEL, allow_pickle=True)
        return z["X"], z["y"], z["dates"]
    print("  building point-in-time panel...")
    _idx("SPY", fetch_prices_fmp("SPY", START, END))
    cong = defaultdict(list)
    for t in fetch_congress_deep():
        if START <= t["td"] <= END: cong[t["ticker"]].append(t)
    tickers = sorted(tk for tk in cong if len(cong[tk]) >= 5)   # names with real congress activity
    print(f"  {len(tickers)} tickers with >=5 congress trades")
    dts = []; d = datetime.strptime(START, "%Y-%m-%d"); e = datetime.strptime(END, "%Y-%m-%d")
    while d < e:
        dts.append(d.strftime("%Y-%m-%d")); d = (d.replace(day=28) + timedelta(days=7)).replace(day=1)
    def shift(dt, days): return (datetime.strptime(dt, "%Y-%m-%d") + timedelta(days=days)).strftime("%Y-%m-%d")
    rows, labels, rdates = [], [], []
    for tk in tickers:
        s = fetch_prices_fmp(tk, START, END)
        if len(s) < 250: continue
        _idx(tk, s)
        ins = sorted(fetch_insider(tk), key=lambda x: x["date"])
        cg = sorted(cong[tk], key=lambda x: x["td"])
        for dt in dts:
            mom = _ret(tk, shift(dt, -LB), dt)
            e0 = _poa(tk, dt); s0 = _poa("SPY", dt)
            e1 = _poa(tk, shift(dt, FWD)); s1 = _poa("SPY", shift(dt, FWD))
            if mom is None or not (e0 and s0 and e1 and s1 and e0 > 0 and s0 > 0): continue
            fwd = (e1/e0 - 1.0) - (s1/s0 - 1.0)
            lo120 = shift(dt, -120); lo30 = shift(dt, -30)
            iw = [x for x in ins if lo120 <= x["date"] < dt]
            A = sum(1 for x in iw if x["dir"] == 1); D = sum(1 for x in iw if x["dir"] == -1)
            ins_net = (A - D) / (A + D) if (A + D) else 0.0
            ins_breadth = float(A + D)
            cw = [x for x in cg if lo120 <= x["td"] < dt]
            cong_net = float(sum(x["dir"] for x in cw))
            c30 = [x for x in cg if lo30 <= x["td"] < dt]
            mb = len({x["member"] for x in c30 if x["dir"] == 1})
            ms = len({x["member"] for x in c30 if x["dir"] == -1})
            cong_cons = float(mb - ms)
            rows.append([mom, ins_net, ins_breadth, cong_net, cong_cons])
            labels.append(fwd); rdates.append(dt)
        if tk != "SPY": _PIDX.pop(tk, None)
    X = np.array(rows, float); y = np.array(labels, float); dates = np.array(rdates)
    np.savez(PANEL, X=X, y=y, dates=dates)
    print(f"  panel: {len(X)} rows, {X.shape[1]} features")
    return X, y, dates

def ic(sig, lab):
    if sig.std() < 1e-12 or lab.std() < 1e-12: return 0.0
    return float(np.corrcoef(sig, lab)[0, 1])

def gsignal(genome, X):
    return np.array([genome.activate(X[i])[0] for i in range(len(X))])

def evolve(arm, rng, Xtr, ytr):
    mut = NEATMutator(rng); pop = [create_minimal_genome(i, rng) for i in range(POP)]
    fixed = rng.choice(len(Xtr), min(BATCH, len(Xtr)), replace=False)   # naive: same batch every gen
    for g in range(GENS):
        if arm == "naive":
            bi = fixed
        else:
            bi = rng.choice(len(Xtr), min(BATCH, len(Xtr)), replace=False)  # gated: fresh each gen
        Xb, yb = Xtr[bi], ytr[bi]
        ysh = yb.copy(); rng.shuffle(ysh)
        for gm in pop:
            sg = gsignal(gm, Xb)
            if arm == "naive":
                gm.fitness = ic(sg, yb)
            else:
                gm.fitness = ic(sg, yb) - max(0.0, abs(ic(sg, ysh)))   # anti-data penalty
        pop.sort(key=lambda x: x.fitness, reverse=True)
        nxt = pop[:ELITE]
        while len(nxt) < POP:
            a, b = rng.choice(pop[:25], 2, replace=False)
            p = a if a.fitness >= b.fitness else b
            c = mut.mutate(p, g); c.genome_id = len(nxt) + g * POP; nxt.append(c)
        pop = nxt
    return pop[0]

def run():
    print("=== CONVERGENCE CO-EVOLUTION — is the gated combination > the best single vector? ===")
    X, y, dates = build_panel()
    tr = dates < SPLIT; te = dates >= SPLIT
    mu = X[tr].mean(0); sd = X[tr].std(0); sd[sd < 1e-9] = 1.0
    Xs = (X - mu) / sd                                   # standardize on TRAIN stats only
    Xtr, ytr, Xte, yte = Xs[tr], y[tr], Xs[te], y[te]
    print(f"  panel {len(X)} rows | train {tr.sum()} (<{SPLIT}) | test {te.sum()} (>= {SPLIT})")

    single = {FEAT_NAMES[i]: ic(Xte[:, i], yte) for i in range(NFEAT)}
    best_name = max(single, key=lambda k: abs(single[k]))
    best_single = abs(single[best_name])
    print(f"  best SINGLE feature OOS |IC|: {best_name} = {single[best_name]:+.4f}")
    print(f"  all singles: " + ", ".join(f"{k}={v:+.3f}" for k, v in single.items()))

    gated_ics, naive_ics, wins = [], [], 0
    te_idx = np.random.RandomState(0).choice(len(Xte), min(4000, len(Xte)), replace=False)
    Xte_s, yte_s = Xte[te_idx], yte[te_idx]
    for seed in range(NSEEDS):
        gt = evolve("gated", np.random.RandomState(seed), Xtr, ytr)
        nv = evolve("naive", np.random.RandomState(seed), Xtr, ytr)
        gic = ic(gsignal(gt, Xte_s), yte_s); nic = ic(gsignal(nv, Xte_s), yte_s)
        gated_ics.append(gic); naive_ics.append(nic)
        wins += (abs(gic) > best_single)
        print(f"  seed {seed}: GATED OOS IC {gic:+.4f} | naive {nic:+.4f} | vs best-single {best_single:.4f} "
              f"{'WIN' if abs(gic) > best_single else '-'}")

    mg = statistics.mean(gated_ics); mn = statistics.mean(naive_ics)
    print(f"\n{'='*68}\n  VERDICT\n{'='*68}")
    print(f"  best single feature OOS |IC|     : {best_single:.4f} ({best_name})")
    print(f"  GATED combination OOS IC (mean)  : {mg:+.4f} +/- {statistics.pstdev(gated_ics):.4f}  "
          f"[beat best-single in {wins}/{NSEEDS}]")
    print(f"  NAIVE combination OOS IC (mean)  : {mn:+.4f}  (gate-vs-naive gap {mg-mn:+.4f})")
    json.dump({"best_single": best_single, "best_name": best_name, "gated_mean": mg,
               "naive_mean": mn, "wins": wins, "nseeds": NSEEDS,
               "gated_ics": gated_ics, "singles": single}, open("/opt/alice/data/convergence.json", "w"), indent=2)
    print("\n  CONVERGENCE PROVEN iff gated mean |IC| > best single AND wins majority of seeds")
    print("  AND gated > naive (the gate, not just nonlinearity, is doing the work).")
    print("  If gated ~ best single, the combination adds nothing — the vectors don't converge.")

if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore"); run()
