#!/usr/bin/env python3
"""
coevo_multiseed.py — multi-seed confirmation of the controlled co-evolution result.

One seed is suggestive, not proof (the lesson of the +12% edge). This runs the
anti-data-gated vs naive contrast across N independent seeds in the Goodhart-trap
regime (30 features, 1 weak real signal + 29 decoys, tiny sample) and reports how
often, and by how much, the gate generalizes better. Lean: no per-gen logging.
"""
import sys, statistics
import numpy as np
sys.path.insert(0, "/opt/alice/app")
import neat_alice as NA
N_FEAT = 30; NA.INPUT_DIM = N_FEAT
from neat_alice import NEATMutator, create_minimal_genome

BETA = 0.3; POP = 40; GENS = 30; TRAIN_N = 60; TEST_N = 1500; ELITE = 4; NSEEDS = 8

def make_data(rng, n):
    X = rng.randn(n, N_FEAT); y = BETA * X[:, 0] + rng.randn(n); return X, y

def perf(g, X, y):
    s = np.array([g.activate(X[i])[0] for i in range(len(X))])
    return 0.0 if (s.std() < 1e-9 or y.std() < 1e-9) else float(np.corrcoef(s, y)[0, 1])

def attr_f0(g, rng, samples=80, d=0.5):
    base = rng.randn(samples, N_FEAT)
    o0 = np.array([g.activate(base[i])[0] for i in range(samples)])
    sens = np.zeros(N_FEAT)
    for f in range(N_FEAT):
        Xp = base.copy(); Xp[:, f] += d
        op = np.array([g.activate(Xp[i])[0] for i in range(samples)])
        sens[f] = np.mean(np.abs(op - o0))
    s = sens.sum(); return float(sens[0] / s) if s > 1e-9 else 0.0

def lean_evolve(arm, rng, Xtr, ytr):
    mut = NEATMutator(rng); pop = [create_minimal_genome(i, rng) for i in range(POP)]
    for g in range(GENS):
        if arm == "naive":
            for gm in pop: gm.fitness = perf(gm, Xtr, ytr)
        else:
            Xf, yf = make_data(rng, TRAIN_N); ysh = yf.copy(); rng.shuffle(ysh)
            for gm in pop:
                gm.fitness = perf(gm, Xf, yf) - max(0.0, abs(perf(gm, Xf, ysh)))
        pop.sort(key=lambda x: x.fitness, reverse=True)
        nxt = pop[:ELITE]
        while len(nxt) < POP:
            a, b = rng.choice(pop[:20], 2, replace=False)
            p = a if a.fitness >= b.fitness else b
            c = mut.mutate(p, g); c.genome_id = len(nxt) + g * POP; nxt.append(c)
        pop = nxt
    return pop[0]

def run():
    Xte, yte = make_data(np.random.RandomState(999), TEST_N)
    gaps = []; cgaps = []
    print(f"=== MULTI-SEED CO-EVOLUTION CONFIRMATION ({NSEEDS} seeds, Goodhart-trap regime) ===")
    print(f"  {N_FEAT} feats, only f0 real (beta={BETA}), {TRAIN_N}-sample train. TEST=held-out {TEST_N}.")
    for s in range(NSEEDS):
        Xtr, ytr = make_data(np.random.RandomState(s), TRAIN_N)
        nv = lean_evolve("naive", np.random.RandomState(s), Xtr, ytr)
        gt = lean_evolve("gated", np.random.RandomState(s), Xtr, ytr)
        tn, tg = perf(nv, Xte, yte), perf(gt, Xte, yte)
        an, ag = attr_f0(nv, np.random.RandomState(7)), attr_f0(gt, np.random.RandomState(7))
        gaps.append(tg - tn); cgaps.append(ag - an)
        print(f"  seed {s}: naive TEST {tn:+.3f} (f0 {an:.2f}) | gated TEST {tg:+.3f} (f0 {ag:.2f}) | gap {tg-tn:+.3f}")
    wins = sum(1 for g in gaps if g > 0)
    cwins = sum(1 for c in cgaps if c > 0)
    print(f"\n  GENERALIZATION: gated beat naive in {wins}/{NSEEDS} seeds | "
          f"mean gap {statistics.mean(gaps):+.3f} +/- {statistics.pstdev(gaps):.3f}")
    print(f"  SIGNAL CONCENTRATION on true feature: gated higher in {cwins}/{NSEEDS} | "
          f"mean {statistics.mean(cgaps):+.3f}")
    print("\n  Robust iff gated wins the clear majority of seeds with a positive mean gap.")
    print("  Still a CONTROLLED synthetic claim about the MECHANISM, not a market edge.")

if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore"); run()
