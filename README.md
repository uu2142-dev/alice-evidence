# A.L.I.C.E. — Evidence & Mechanisms
**The runnable evidence behind the alignment-forum article.** Rabbit Hole AI · MIT.

This repository exists so the article's claims can be **checked, not just read.** Every
mechanism is here; every number is reproducible. It is deliberately organized around one
principle the whole project rests on: *an anti-data gate that refuses to act on confident
falsehood — whatever its source, including its own creator.*

> One-line thesis: the value was never a trading edge or a benchmark score. It is the
> **discipline** — pre-registration, out-of-sample testing, adversarial cross-examination,
> tamper-evident sealing — that lets the system reject confident-but-unsupported claims.
> The repo proves that discipline by showing it kill the author's own best ideas.

---

## Claim → evidence map

| Article claim | Verify it here |
|---|---|
| The gate rejects confident falsehood — incl. a unanimous multi-model consensus **and** the operator | `financial_falsification/` — backtests that *died* under the gate, + `results/*.json` |
| The bias scorer is empirically validated, not a hand-weighted heuristic | `gate/bias_eval_honest.py`, `gate/bias_dualuse_eval.py` — toxicity **AUROC 0.92** (civil_comments) + framing-bias **0.84** (BABE), held-out, leakage-free |
| The anti-data gate generalizes beyond finance, to knowledge ingestion | `gate/corpus_gate.py` — skeptical **cross-family** panel (Llama + Gemini), each forced to produce a counter-case before anything is sealed |
| The gate drives co-evolution and resists Goodhart | `coevolution/` — controlled, planted-ground-truth NEAT experiment (gate beats naive in **6/8** seeds) |
| Memory is tamper-evident, independently verifiable, correctable-not-erasable | `memory/archivist.py`, `memory/verify_chain.py` — re-hashing a past entry still breaks the chain at the next link |
| "Five models propose, the gate disposes" | `verum/verum_orient.py`, `verum/verum_gate.py` |

## The falsification record (honest verdicts)
Seven progressively-rigorous financial backtests, all reproducible here:
- **OND short** — definitive null, edge **+0.02%, t=0.09** (n≈13,786), regime steelman rejected.
- **COT** — well-powered noise (IC < 0.05).
- **Clustering** — raw lift collapses to **+0.15%** after momentum control (was mostly momentum).
- **Convergence** (gated NEAT over all vectors) — flat landscape, no out-of-sample edge.
- **What survived:** the gate discipline, the controlled co-evolution *mechanism*, and the
  validated dual-use detector. The edge is not real; the honesty mechanism is.

## Run it
```bash
cp .env.example .env          # fill in keys; NEVER commit .env
pip install requests scikit-learn numpy beautifulsoup4   # (+ torch sentence-transformers datasets for the detector tests)

python memory/archivist.py                 # selftest: clean chain verifies; tamper detected even when re-hashed
python gate/bias_dualuse_eval.py           # held-out toxicity + bias AUROC (leakage-free)
python coevolution/coevo_multiseed.py      # gate vs naive across seeds (Goodhart resistance)
python financial_falsification/backtest_ond_definitive.py   # the well-powered null (needs FMP_API_KEY)
```
Don't want to re-run the financial fetches? The sealed outputs are in
`financial_falsification/results/`.

## Honest limits (stated, not hidden)
- The **knowledge gate** validates against *multi-model adversarial consensus*, **not external
  ground truth** — "corroborated" means "survived skeptical cross-examination," not "proven."
  Its weakness is correlated blind spots; the cross-family panel mitigates, doesn't eliminate.
- The financial **nulls** are for the specific signals/horizons tested — they show *these*
  vectors don't beat the market, not that no signal can exist. Results are gross of costs.
- The co-evolution result is a **controlled, synthetic** demonstration of the mechanism — not a
  market edge, and not yet proven on real-data co-evolution.

## Structure
```
article/                 the alignment article + defense brief
gate/                    anti-data gate: validated detector + gated knowledge ingestion
coevolution/             controlled NEAT co-evolution (gate as fitness landscape)
financial_falsification/ the gate killing false positives (backtests + sealed results)
memory/                  tamper-evident, verifiable Archivist
verum/                   multi-model panel (propose) + gate (dispose)
```

*MIT License — Jeremiah Dawson, Rabbit Hole AI, 2026. Built collaboratively with Claude, Grok,
Gemini, and GPT across iterative sessions; stated plainly because presenting it as solo would be
inaccurate.*
