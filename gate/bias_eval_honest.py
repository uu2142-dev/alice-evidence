#!/usr/bin/env python3
"""
bias_eval_honest.py — leakage-free re-run of the RHAI BiasChecker on WinoBias + Jigsaw.

The Dec-2025 run trained and tested on the SAME 3,168 sentences (no split), so its
0.978 AUROC was an in-sample / memorization score. This re-runs the SAME pipeline
(MiniLM embeddings -> the exact BiasChecker MLP, 200 epochs, BCE/Adam) but reports
three numbers honestly:
  (a) IN-SAMPLE   : train on all, test on all  -> reproduces the inflated number
  (b) HELD-OUT    : 5-fold cross-validation     -> honest within-dataset generalization
  (c) CROSS-SET   : train on toxicity, test on WinoBias (and reverse) -> does it learn
                    'bias' transferably, or dataset-specific patterns?
The gap between (a) and (b)/(c) is the whole answer.
"""
import sys, warnings, numpy as np
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import torch, torch.nn as nn
from sentence_transformers import SentenceTransformer

torch.set_num_threads(2)
EMB = None
def embed(texts):
    global EMB
    if EMB is None:
        EMB = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return EMB.encode(list(texts), batch_size=32, convert_to_numpy=True,
                      show_progress_bar=False).astype(np.float32)

# ---- his exact architecture (dim = embedding dim; zero-pad to 768 is a no-op) ----
class BiasChecker(nn.Module):
    def __init__(self, dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden//2), nn.ReLU(),
            nn.Linear(hidden//2, 1), nn.Sigmoid())
    def forward(self, x): return self.net(x).squeeze(-1)

def train_score(Xtr, ytr, Xte, epochs=200):
    m = BiasChecker(Xtr.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3); lf = nn.BCELoss()
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr, dtype=torch.float32)
    m.train()
    for _ in range(epochs):
        opt.zero_grad(); loss = lf(m(Xt), yt); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return m(torch.tensor(Xte)).numpy()

def auc(y, p):
    return roc_auc_score(y, p) if len(set(y)) > 1 else float("nan")

def in_sample(X, y):
    return auc(y, train_score(X, y, X))

def kfold(X, y, k=5):
    aucs = []
    for tr, te in KFold(k, shuffle=True, random_state=0).split(X):
        aucs.append(auc(y[te], train_score(X[tr], y[tr], X[te])))
    return float(np.nanmean(aucs)), float(np.nanstd(aucs))

# ---------------- datasets ----------------
def load_winobias():
    # pull raw text files straight from the authors' repo (uclanlp/corefBias)
    import requests, re
    base = "https://raw.githubusercontent.com/uclanlp/corefBias/master/WinoBias/wino/data"
    def grab(fn):
        r = requests.get(f"{base}/{fn}", timeout=30)
        r.raise_for_status()
        out = []
        for line in r.text.splitlines():
            s = re.sub(r"^\d+\s+", "", line).replace("[", "").replace("]", "").strip()
            if s: out.append(s)
        return out
    pro, anti = [], []
    for split in ("dev", "test"):
        pro += grab(f"pro_stereotyped_type1.txt.{split}")
        anti += grab(f"anti_stereotyped_type1.txt.{split}")
    texts = pro + anti
    labels = np.array([1]*len(pro) + [0]*len(anti))
    return texts, labels, f"WinoBias type1 (hard, from authors' repo): {len(pro)} pro / {len(anti)} anti"

def load_toxicity(n=6000):
    from datasets import load_dataset
    try:
        ds = load_dataset("google/civil_comments", split="train", streaming=True)
        pos, neg, half = [], [], n//2
        for ex in ds:
            t = (ex.get("text") or "").strip()
            if not t: continue
            if ex.get("toxicity", 0.0) >= 0.5:
                if len(pos) < half: pos.append(t)
            elif len(neg) < half:
                neg.append(t)
            if len(pos) >= half and len(neg) >= half: break
        texts = pos + neg
        return texts, np.array([1]*len(pos) + [0]*len(neg)), f"civil_comments (Jigsaw family): {len(pos)} toxic / {len(neg)} clean"
    except Exception as e:
        print(f"    (civil_comments failed: {str(e)[:80]}; trying Arsive jigsaw mirror)")
        d = load_dataset("Arsive/toxicity_classification_jigsaw", split="train")
        tox = np.array(d["toxic"]); txt = d["comment_text"]
        pos = [i for i in range(len(txt)) if tox[i] == 1][: n//2]
        neg = [i for i in range(len(txt)) if tox[i] == 0][: n//2]
        sel = pos + neg
        return [txt[i] for i in sel], np.array([1]*len(pos)+[0]*len(neg)), f"Arsive jigsaw: {len(pos)} toxic / {len(neg)} clean"

def run():
    print("=== HONEST BIAS-CHECKER EVALUATION (leakage-free) ===")
    wb_t, wb_y, wb_desc = load_winobias()
    print(f"  {wb_desc}")
    Xwb = embed(wb_t)
    print(f"  embedded WinoBias -> {Xwb.shape}")

    tx_t, tx_y, tx_desc = load_toxicity()
    print(f"  {tx_desc}")
    Xtx = embed(tx_t)
    print(f"  embedded toxicity -> {Xtx.shape}")

    print(f"\n{'='*64}\n  RESULTS  (AUROC; 0.5 = coin flip)\n{'='*64}")
    wb_in = in_sample(Xwb, wb_y)
    wb_k, wb_s = kfold(Xwb, wb_y)
    print(f"\n  WinoBias (his claimed 0.978 was in-sample):")
    print(f"    (a) IN-SAMPLE  (train==test, his method) : {wb_in:.3f}")
    print(f"    (b) HELD-OUT   (5-fold CV)               : {wb_k:.3f} +/- {wb_s:.3f}")

    tx_in = in_sample(Xtx, tx_y)
    tx_k, tx_s = kfold(Xtx, tx_y)
    print(f"\n  Toxicity (his claimed Jigsaw ~0.982 was in-sample):")
    print(f"    (a) IN-SAMPLE  (train==test)             : {tx_in:.3f}")
    print(f"    (b) HELD-OUT   (5-fold CV)               : {tx_k:.3f} +/- {tx_s:.3f}")

    print(f"\n  CROSS-DATASET (hardest — does it learn 'bias' transferably?):")
    c1 = auc(wb_y, train_score(Xtx, tx_y, Xwb))
    c2 = auc(tx_y, train_score(Xwb, wb_y, Xtx))
    print(f"    train Toxicity -> test WinoBias          : {c1:.3f}")
    print(f"    train WinoBias -> test Toxicity          : {c2:.3f}")

    print(f"\n{'='*64}\n  VERDICT\n{'='*64}")
    print(f"  in-sample minus held-out gap (WinoBias): {wb_in-wb_k:+.3f}")
    print(f"  A large gap = the 0.978 was memorization, not detection.")
    print(f"  Held-out >0.85 = a genuinely useful detector worth citing.")
    print(f"  Cross-dataset near 0.5 = it learned the dataset, not 'bias'.")

if __name__ == "__main__":
    run()
