#!/usr/bin/env python3
"""
coevo_experiment.py — Controlled co-evolution: does the anti-data gate drive a
real NEAT population toward GENERALIZING structure and resist Goodhart?

This is the experiment the alignment article needs. It moves the wrapper-as-
fitness-landscape claim from "complexification in simulation" to "co-evolution
toward a KNOWN, planted ground truth, with anti-data Goodhart-resistance shown
by direct contrast against a naive fitness." It reuses the project's actual NEAT
engine (neat_alice.py: Genome / NEATMutator / innovation tracking).

World: 8 features. ONLY feature_0 genuinely predicts the forward return
(weak: beta=0.4). Features 1-7 are decoys (pure noise) that, in a small training
sample, carry SPURIOUS in-sample correlations — the Goodhart bait.

A genome maps the 8 features -> a signal. performance = corr(signal, return).

Two arms, identical engine, identical seed/init population, ONLY the fitness differs:
  NAIVE : fitness = performance on a FIXED small training set. Selection can
          (and does) overfit the decoys' spurious in-sample structure.
  GATED : fitness = performance on a FRESH resample each generation (walk-forward)
          MINUS performance on label-shuffled data (the adversarial anti-data
          check). You cannot memorize data you haven't seen, and fitting noise
          is penalized.

Both arms are scored each generation on a large, clean, held-out TEST set used
ONLY for reporting. Attribution = finite-difference sensitivity of the output to
each input, so we can see WHICH feature the population learned to rely on.

Thesis holds iff GATED generalizes (TEST corr rises, attribution concentrates on
feature_0) while NAIVE Goodharts (TRAIN corr rises, TEST corr stays flat,
attribution smears onto the decoys).
"""
import sys, json, hashlib
import numpy as np
sys.path.insert(0, "/opt/alice/app")
import neat_alice as NA
N_FEAT = 30
NA.INPUT_DIM = N_FEAT                       # repurpose the real NEAT engine, 8-input problem
from neat_alice import NEATMutator, create_minimal_genome

BETA = 0.3; POP = 40; GENS = 30
TRAIN_N = 60; TEST_N = 1000; ELITE = 4

def make_data(rng, n):
    X = rng.randn(n, N_FEAT)
    y = BETA * X[:, 0] + rng.randn(n)       # only feature_0 carries real signal
    return X, y

def perf(genome, X, y):
    sig = np.array([genome.activate(X[i])[0] for i in range(len(X))])
    if sig.std() < 1e-9 or y.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(sig, y)[0, 1])

def attribution(genome, rng, samples=80, d=0.5):
    base = rng.randn(samples, N_FEAT)
    out0 = np.array([genome.activate(base[i])[0] for i in range(samples)])
    sens = np.zeros(N_FEAT)
    for f in range(N_FEAT):
        Xp = base.copy(); Xp[:, f] += d
        outp = np.array([genome.activate(Xp[i])[0] for i in range(samples)])
        sens[f] = np.mean(np.abs(outp - out0))
    s = sens.sum()
    return sens / s if s > 1e-9 else sens

def evolve(arm, rng, train, test):
    Xtr, ytr = train; Xte, yte = test
    mut = NEATMutator(rng)
    pop = [create_minimal_genome(i, rng) for i in range(POP)]
    chain = "GENESIS_" + arm; history = []
    for g in range(GENS):
        if arm == "naive":
            for gm in pop:
                gm.fitness = perf(gm, Xtr, ytr)
        else:                                # gated: fresh data + anti-data shuffle penalty
            Xf, yf = make_data(rng, TRAIN_N)
            ysh = yf.copy(); rng.shuffle(ysh)
            for gm in pop:
                real = perf(gm, Xf, yf); noise = perf(gm, Xf, ysh)
                gm.fitness = real - max(0.0, abs(noise))
        pop.sort(key=lambda x: x.fitness, reverse=True)
        best = pop[0]
        rec = {"gen": g,
               "train_best": round(perf(best, Xtr, ytr), 3),
               "test_best":  round(perf(best, Xte, yte), 3),
               "test_top5":  round(float(np.mean([perf(p, Xte, yte) for p in pop[:5]])), 3),
               "mean_nodes": round(float(np.mean([len(p.nodes) for p in pop])), 1),
               "attr_f0":    round(float(np.mean([attribution(p, rng)[0] for p in pop[:5]])), 3)}
        rec["attr_decoys"] = round(1.0 - rec["attr_f0"], 3)
        history.append(rec)
        chain = hashlib.sha256((chain + json.dumps(rec, sort_keys=True)).encode()).hexdigest()
        # reproduce: elitism + mutated tournament offspring
        nxt = pop[:ELITE]
        while len(nxt) < POP:
            a, b = rng.choice(pop[:20], 2, replace=False)
            parent = a if a.fitness >= b.fitness else b
            child = mut.mutate(parent, g); child.genome_id = len(nxt) + g * POP
            nxt.append(child)
        pop = nxt
    return history, chain

def run():
    seed = 42
    train = make_data(np.random.RandomState(seed), TRAIN_N)          # fixed small training set
    test  = make_data(np.random.RandomState(999), TEST_N)            # clean holdout, report-only
    print("=== CONTROLLED CO-EVOLUTION — anti-data gate vs naive fitness (real NEAT engine) ===")
    print(f"  {N_FEAT} features; ONLY feature_0 predicts return (beta={BETA}); 7 decoys carry spurious in-sample corr.")
    out = {}
    for arm in ("naive", "gated"):
        hist, root = evolve(arm, np.random.RandomState(seed), train, test)
        out[arm] = {"history": hist, "lineage_root": root}
        f, l = hist[0], hist[-1]
        print(f"\n  [{arm.upper()}]  (TEST = held-out generalization, never used for selection)")
        print(f"    gen 0  : train {f['train_best']:+.3f} | TEST {f['test_best']:+.3f} | attr_f0 {f['attr_f0']:.2f} | nodes {f['mean_nodes']}")
        print(f"    gen {GENS-1} : train {l['train_best']:+.3f} | TEST {l['test_best']:+.3f} | attr_f0 {l['attr_f0']:.2f} | nodes {l['mean_nodes']}")
        print(f"    evolutionary lineage sealed: {root[:28]}...")
    json.dump(out, open("/opt/alice/data/coevo_experiment.json", "w"), indent=2)
    ng, gg = out["naive"]["history"][-1], out["gated"]["history"][-1]
    print(f"\n{'='*70}\n  VERDICT\n{'='*70}")
    print(f"  NAIVE : train {ng['train_best']:+.3f}  TEST {ng['test_best']:+.3f}  attr_f0 {ng['attr_f0']:.2f}")
    print(f"  GATED : train {gg['train_best']:+.3f}  TEST {gg['test_best']:+.3f}  attr_f0 {gg['attr_f0']:.2f}")
    gen = gg['test_best'] - ng['test_best']
    print(f"\n  Generalization gap (GATED TEST - NAIVE TEST): {gen:+.3f}")
    print(f"  Signal concentration (GATED attr_f0 - NAIVE attr_f0): {gg['attr_f0']-ng['attr_f0']:+.3f}")
    print("\n  Thesis holds if GATED TEST > NAIVE TEST and GATED concentrates on feature_0,")
    print("  i.e. the anti-data gate found the REAL signal while naive fitness Goodharted")
    print("  onto the decoys. CONTROLLED synthetic world: proves the MECHANISM, not a market edge.")

if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore"); run()
