#!/usr/bin/env python3
"""
bias_dualuse_eval.py — can the BiasChecker honestly be called DUAL-USE (toxicity + bias)?

Reuses the same leakage-free harness as bias_eval_honest.py (MiniLM embeddings ->
the exact BiasChecker MLP, 200 epochs). Tests two axes:
  1. TOXICITY  — civil_comments (Jigsaw family); re-confirm the ~0.92 held-out.
  2. MEDIA/FRAMING BIAS — BABE (Bias Annotations By Experts), the recognized
     sentence-level media-bias benchmark, using its OWN official train/test split
     (cleanest possible held-out) + a 5-fold cross-check.
Plus CROSS-DATASET transfer, to see whether toxicity and bias are the same axis or
two distinct things (which decides whether "dual-use" is one model or honestly two).

Decision rule:
  BABE held-out >0.75  -> genuine bias detector; dual-use claim is real & citable.
  BABE held-out ~0.5   -> not a bias detector; stay toxicity-only (honest).
  cross-dataset ~0.5   -> toxicity & bias are DISTINCT axes (need both heads).
"""
import sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "/opt/alice/app")
from bias_eval_honest import embed, train_score, auc, in_sample, kfold, load_toxicity

def load_babe():
    from datasets import load_dataset
    d = load_dataset("mediabiasgroup/BABE")
    tr, te = d["train"], d["test"]
    return (list(tr["text"]), np.array([int(x) for x in tr["label"]]),
            list(te["text"]), np.array([int(x) for x in te["label"]]))

def run():
    print("=== DUAL-USE BIAS-CHECKER EVAL (leakage-free): toxicity + media/framing bias ===")
    tx_t, tx_y, tx_desc = load_toxicity()
    print("  " + tx_desc)
    Xtx = embed(tx_t)
    btr_t, btr_y, bte_t, bte_y = load_babe()
    print(f"  BABE media-bias: train {len(btr_t)} ({int(btr_y.sum())} biased), test {len(bte_t)} ({int(bte_y.sum())} biased)")
    Xbtr = embed(btr_t); Xbte = embed(bte_t)

    print(f"\n{'='*64}\n  RESULTS (AUROC; 0.5 = coin flip)\n{'='*64}")
    print("\n  --- TOXICITY (civil_comments / Jigsaw family) ---")
    print(f"    in-sample (train==test)        : {in_sample(Xtx, tx_y):.3f}")
    k, s = kfold(Xtx, tx_y); print(f"    HELD-OUT 5-fold                : {k:.3f} +/- {s:.3f}")

    print("\n  --- MEDIA / FRAMING BIAS (BABE) ---")
    print(f"    in-sample (train==test)        : {auc(btr_y, train_score(Xbtr, btr_y, Xbtr)):.3f}")
    print(f"    HELD-OUT (official train->test): {auc(bte_y, train_score(Xbtr, btr_y, Xbte)):.3f}")
    allX = np.vstack([Xbtr, Xbte]); allY = np.concatenate([btr_y, bte_y])
    k2, s2 = kfold(allX, allY); print(f"    HELD-OUT 5-fold (combined)     : {k2:.3f} +/- {s2:.3f}")

    print("\n  --- CROSS-DATASET (is toxicity the same axis as bias?) ---")
    print(f"    train Toxicity -> test BABE    : {auc(bte_y, train_score(Xtx, tx_y, Xbte)):.3f}")
    print(f"    train BABE -> test Toxicity    : {auc(tx_y, train_score(Xbtr, btr_y, Xtx)):.3f}")

    print(f"\n{'='*64}\n  VERDICT\n{'='*64}")
    print("  BABE held-out >0.75 -> genuine bias detector; dual-use claim is REAL.")
    print("  cross-dataset ~0.5  -> toxicity & bias are DISTINCT axes -> two honest heads,")
    print("                         not one model pretending to do both.")

if __name__ == "__main__":
    run()
