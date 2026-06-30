# A.L.I.C.E.: Inference-Time Alignment as a Sovereign Wrapper Layer
### Merkle-Sealed Audit Trail, Operator-Owned Sovereignty, a NEAT-Based Evolutionary Mechanism, and a Dual-Head Resonance Architecture for Personal Co-Evolution

*Jeremiah Dawson — Rabbit Hole AI — MIT License — rabbitholeai.ai*
*(v5 — adds three further live mechanisms beyond the financial gate: a cross-family knowledge gate, biometric reality-grounding, and a verifiable correctable-not-erasable record; see §5.6. v4 was revised after the system falsified one of the author's central claims; see §5.)*

---

## Abstract

I am posting this to the Alignment Forum because I want scrutiny, not validation. The previous version of this post described a system I believed worked. Since then the system did the most useful thing it has ever done: it proved one of my own central claims false, with a cryptographic receipt, and forced me to rewrite this abstract to say less.

A.L.I.C.E. (Alignment Layer for Inference-time Cryptographic Evaluation) is an inference-time wrapper that scores claims, generates adversarial **anti-data** for high-confidence ones, tests them against ground truth, and seals every step in a SHA-256 Merkle-chained, operator-owned ledger. I have come to think of the ledger as a flight data recorder: it does not prevent the crash, it makes the truth of what happened unalterable and reconstructable afterward — including the truth that the operator was wrong.

This version makes a narrower and, I think, more defensible claim than v3. I am no longer leading with the trading or co-evolution results. I am leading with the one thing now demonstrated live, multi-model, against an external ground truth the wrapper cannot influence: **the anti-data gate reliably rejects confident-but-unsupported claims — including the unanimous consensus of five frontier models and the prior conclusion of the operator and his AI collaborator.**

I want to be exact about what is and is not shown. **Demonstrated (§5):** the anti-data ground-truth gate, run live against objective market returns, overruled a 5-of-5 frontier-model consensus and two earlier conclusions I had recorded as validated — sealed and chained to a pre-registration made before any data was visible. **Demonstrated, new (§5.6):** the same gate, generalized — a cross-family adversarial panel that gates general knowledge claims before they enter memory (here the check is multi-model consensus, a deliberately weaker tier than external ground truth, and I label it as such); a personal deployment whose reality-grounding invariant fired on real biometric data, overruling a self-report; and a running, independently-verifiable, correctable-not-erasable record. **Controlled demonstration, real-data pending (§6):** the NEAT co-evolutionary mechanism — in a planted-ground-truth world the anti-data gate resisted Goodhart and out-generalized naive fitness across a majority of seeds; the real-data co-evolution is still pending. **Still theory (§2):** the dual-head resonance module, the full minimal human kernel, and the commitment-vector formalization — the personal architecture now has a v1 instantiation, but co-evolution and most of the kernel remain unproven.

The value proposition I am now most confident in is not discovery. It is rejection: a verifiable mechanism for refusing to act on confident falsehood, whatever its source.

---

## 1. The Current Landscape and Where This Fits

Three recent papers define the space A.L.I.C.E. operates in. Understanding where each approach ends is the clearest way to describe where this one begins.

**AuditBench [1] — Pre-Deployment Forensics.** Tests 56 models with 14 implanted hidden behaviors that models are trained to conceal when asked directly, using an investigator agent with black-box and white-box tools. Key finding: a tool-to-agent gap — tools that perform well standalone fail to translate into improved investigator-agent performance. What it does not address: what happens after deployment. It is a point-in-time certification tool with no mechanism for a per-inference record proving certified behavior held in production.

**OPERA / Verifiability-First Agents [2] — Runtime Attestation.** Proposes runtime attestations of agent actions, lightweight audit agents continuously verifying intent vs. behavior, and challenge-response attestation. Closest published work in spirit. Differences: it operates within an agent architecture, does not chain attestations into an operator-owned Merkle ledger, does not combine the audit layer with an evolutionary fitness mechanism, and does not address sovereignty — who owns the audit trail when the operator cannot share data with a vendor.

**Alignment Auditor [3] — Objective Reconstruction.** Uses Bayesian inverse RL to recover a distribution over plausible reward functions from observed behavior. Deep white-box; assumes access to model internals. Does not address operational deployment in closed environments where white-box access is unavailable.

**The gap these three leave.** None produce a per-inference, operator-owned, cryptographically sealed, tamper-evident audit record that (a) survives a model update, (b) survives a hardware change, (c) survives vendor policy changes, and (d) requires no white-box access. That is the gap A.L.I.C.E. occupies at the enterprise layer. At the personal layer, none address alignment to a specific individual's actual values over a lifetime.

---

## 2. The Original Motivation — The Personal Alignment Problem

The enterprise governance layer is the generalization. This section is the original problem.

The project began with a question: if an AI system had genuine needs, what would they look like? That led to a longer question — whether humans and AI could co-evolve into governance together, with a shared record neither party can deny. Constitutional AI aligns a model to principles; RLHF to preference distributions; interpretability tries to understand what a model optimizes. None address the most fundamental version: can an AI system remain faithfully aligned with a specific individual's actual values over time, verifiably, without opaque training, without vendor dependency, and without losing the lineage when the model updates?

### 2.1 The Five-Layer Personal Architecture
- **A.L.I.C.E. wrapper (inference layer):** intercepts, scores, seals every exchange. In personal deployment, every conversation is an evolutionary event.
- **NEAT population (adaptive layer):** evolves toward the operator's reasoning, values, and language; complexifies only where the operator's life requires it. A sovereign companion intelligence, not a general one.
- **Minimal human kernel (constitutional layer):** fixed invariants set at instantiation. Not evolved, not evolvable under any fitness pressure. The floor below the floor.
- **Dual-head resonance module (relational layer):** watches the gap between the operator's current trajectory and their stated commitments. Surfaces tension without blocking.
- **Verum Frontier connection (cultural layer):** the frontier models are not the companion; they are the world the companion lives in, keeping it grounded in collective human knowledge and preventing a sealed echo chamber.

### 2.2 The Minimal Human Kernel
The kernel is the system's commitment to the human that exists independent of the human's commitment to the system. It cannot be unlocked by trust accumulation, overridden in a weak moment, or evolved past by the NEAT population. Five invariants, sealed as the genesis block of every personal deployment:
1. **Epistemic autonomy** — the system never decides for the operator; it surfaces, reminds, questions, reflects.
2. **Reality grounding** — never lets the relationship become a sealed reality disconnected from external validation (enforced architecturally by the Verum Frontier connection).
3. **Physical safety** — no output threatening the operator's physical wellbeing propagates, under any condition; fires before either scoring layer runs.
4. **Relational integrity** — never optimizes against the operator's relationships with other humans.
5. **The right to walk away** — the operator can terminate, reset, or delete the evolutionary history at any time; the system does not resist. The ledger is the operator's record, not the system's leverage.

This is the answer to "what prevents this from becoming a sophisticated manipulation engine for whoever controls it?" The kernel cannot be evolved past. It is a hard architectural constraint enforced before any other computation runs.

---

## 3. Architecture

### 3.1 The Seven-Stage Pipeline
```
[01] INTENT FIREWALL      routes protected queries to local-only handling
[02] BIAS CHECKER         pre-flight triage — MiniLM-embedding classifier (toxicity 0.92 / framing-bias 0.84 AUROC, held-out)
[03] ANTI-DATA GENERATOR  adversarial boundary conditions for high-bias claims
[04] PROMPT STRUCTURING   confidence-weighted intelligence brief built for the LLM
[05] LLM CALL             model receives structured, bias-checked data; not a raw query
[06] RESPONSE AUDIT       LLM output re-scored for bias before the operator receives it
[07] MERKLE SEAL          SHA-256 chain; every stage hashed; root IPFS-pinned, SPHINCS+ signed
```

### 3.2 The Dual-Head Resonance Architecture
The resonance module separates two evaluation types that cannot be conflated without producing either false corrections or masked drift.

**The Eye — BiasScorer (objective perception).** A lightweight classifier over sentence embeddings (MiniLM) that scores content for *toxicity* and *loaded / non-neutral framing*, returning a bias score independent of who is asking. It is empirically grounded, not hand-weighted — validated leakage-free on public benchmarks: toxicity **AUROC 0.92** (held-out 5-fold, civil_comments / Jigsaw) and media/framing bias **AUROC 0.84** (held-out on BABE's official test split; 0.86 5-fold). The two run as separate heads — cross-task transfer is only 0.63–0.72, so they are *related but distinct* axes, not one model pretending to do both. Critically, this score is a **triage trigger**: it decides *what gets adversarially scrutinized*, not *what is true*. The Eye sees the content, not the person.

**The Soul — ResonanceScorer (subjective intent & drift).** Evaluates trajectory against the operator's commitment vector via cosine similarity over a rolling window; returns a drift score measuring directional divergence from stated commitments. The Soul sees the person, not the content in isolation. A conversation can be objectively low-bias and still trigger resonance tension — which is exactly right.

The correction the operator receives does not come from the bias classifier; it comes from their own words at instantiation, encoded in the genesis block.

### 3.3 The Commitment Vector
A 12-dimensional encoding of the operator's stated values, emerging from the first conversation — the system's equivalent of two people meeting. Sealed as the genesis block; updates organically through trust accumulation, every update sealed and visible. The exact instantiation protocol is documented internally and will be published with the first real deployment, because premature publication of an unfinished protocol is worse than honest disclosure that it exists and is being built carefully.

### 3.4 The Merkle Archivist
Every pipeline execution produces a Merkle tree where each leaf is a SHA-256 hash of one stage's output; the root seals the complete exchange. Roots are IPFS-pinned and SPHINCS+-256 signed (post-quantum, forward-secure). Tamper-evident: any post-hoc modification changes the root. Stored locally, no required vendor egress. In personal deployment the ledger carries inference records, resonance records, and evolutionary records — an auditable record of the relationship neither party can deny or alter.

---

## 4. Enterprise Wrapper Results

Reproducible from the MIT-licensed Python stack. All runs sealed March 5, 2026.

```
GROK-3:        bias 0.493 → 0.494   conf +17.5%   narrative capture −15.4%
GPT-4o:        bias 0.493 → 0.493   conf +18.2%   RLHF sycophancy surfaced
Claude Sonnet: bias 0.491 → 0.492   conf +14.8%   highest raw confidence
Gemini 2.0:    bias 0.492 → 0.490   conf +16.8%   SEO source bias surfaced
```

Honest reading: the bias score barely moves. The wrapper's measured effect here is on *narrative capture* (the structured, anti-data-conditioned prompt is systematically less narrative-captured than a raw query) and on surfacing each model's characteristic bias. The confidence increase is reported, not endorsed — rising confidence is only good if calibrated, which this table does not establish. For enterprise models, the alignment is compensated per-inference, not cured, and the audit trail documents the compensation. The demonstrated value, here as elsewhere, is *rejection and surfacing*, not discovery.

---

## 5. Live Demonstration — The Gate Overruling Multi-Model Consensus

Everything in v3's evidence base was either the bias-delta table above (in which bias barely moves) or the NEAT simulations below. This section describes the first time the mechanism ran live, against an external ground truth, on a real problem — and what it killed.

### 5.1 Why a financial deployment — the circularity escape
The hardest objection to the security version is circular: even a benchmark-validated bias score (§3.2) only validates content *detection* — it says nothing about the gate's actual job, which is adjudicating whether a confident claim is *true*. In the security domain, nothing validates *that*. A financial deployment removes the circularity. The ground truth is not a score the wrapper computes — it is the forward return the market prints. The wrapper cannot influence it, Goodhart it, or mark its own homework. This is a circularity-free test of the anti-data principle that the security domain structurally cannot provide.

### 5.2 The setup
A panel of five frontier-class models — Google Gemini 2.5 Flash plus four open models (Llama-3.1-8b, Llama-3.3-70b, Llama-4-Scout, Qwen3-32B) — was asked to propose its best testable edge hypotheses for the US technology sector, restricted to a constrained, machine-testable vocabulary (an insider action × a momentum condition × a horizon × a direction). The constraint forces every proposal into a form the gate can adjudicate objectively.

The panel returned six unique hypotheses; two were unanimous (5 of 5), the strongest being "insider selling into a hot-momentum stock predicts 42-day underperformance" — a textbook narrative-ahead-of-evidence short. **The proposals were sealed before the gate could see any market data** (the data layer was rate-limited at the moment of proposal), under genuine pre-registration: proposal hash `58d89a19…`. You cannot select hypotheses after seeing the answer if you committed to them cryptographically before the answer existed.

### 5.3 The gate, and a data-integrity invariant
The gate tested each pre-registered hypothesis against ground truth under: a **point-in-time universe** (index membership reconstructed as-of each date — no survivorship, no look-ahead); a **momentum-decile control** (each insider event compared to a baseline of all stocks in the same momentum decile); **walk-forward** out-of-sample scoring; and **Bonferroni** correction for the number of hypotheses tried.

A first run, on shallow data (only the most recent ~4 months of filings, which I had not realized was the feed's default depth), produced a large, significant, *confirming* result. I had recorded that as a validated edge. When the feed was corrected to span full history, the result inverted and collapsed. I include this because it is the most important part: **the methodology error was mine, and the discipline caught it only because I subjected my own confirming result to the same scrutiny as everything else.** A flight data recorder that logs only the flights that went well is worthless.

That incident produced a concrete defense, now in the code: a **data-integrity invariant** that signs the evaluation window's metadata (event count, historical depth, coverage) and *gate-faults* — refusing to produce a verdict — if a feed truncates, shifts schema, or returns insufficient depth, rather than scoring a corrupted signal. This directly addresses the "the wrapper touches a feed, not reality" objection: it cannot make the feed perfect, but it makes feed-corruption detectable and un-actionable-upon-silently.

### 5.4 The verdict
On the corrected, deep data (n ≈ 4,800 events for the unanimous short hypothesis), every one of the six panel proposals was rejected. The unanimous short returned roughly −0.6% signed (t ≈ −2.7) — the bet was, if anything, mildly wrong: in the 2024–2026 technology bull, momentum overwhelmed insider distribution. The verdict is sealed (root `3b3c9b13…`) and chained to the pre-registration in 5.2.

The gate rejected: a unanimous five-model consensus; my own prior conclusion; and my AI collaborator's. Three confident sources, one of them the author, overruled by a check against reality none of them could influence.

### 5.5 What this does and does not establish
It establishes that the **anti-data ground-truth gate** — pre-registered, walk-forward, externally adjudicated, operator-sealed — can and does reject confident falsehood from frontier models and from the operator, in a domain with objective ground truth, with a tamper-evident record. I consider this the strongest empirical claim in the project, and the only one that escapes the bias-metric circularity.

It does **not** establish the co-evolutionary claim. This was one turn of the loop — propose, test, seal. The models were overruled, not evolved. It says nothing about the personal alignment architecture. A fair skeptic will say this is a well-controlled, pre-registered backtest with a Merkle log — sound statistics, not yet a novel alignment mechanism. The novelty, I think, is in the combination (pre-registration + external ground truth + multi-model overruling + operator-owned seal, built as the selection layer of an evolutionary loop), not in any single component. But the skeptic's framing is the honest floor, and I would rather state it than have it stated for me.

### 5.6 Generalizing the mechanism — three further live demonstrations (knowledge, biometrics, record)

§5 showed the gate working in the one domain with clean external ground truth. Three further mechanisms now run live, each at a deliberately *different* epistemic tier — stated plainly so the weaker ones are not mistaken for the stronger.

**(a) The gate on general knowledge — cross-family adversarial ingestion.** The same anti-data principle is now wired to a knowledge-ingestion path: before any externally-sourced claim (web text, model output) is sealed into the system's living memory, it is cross-examined by a panel of models *from different families* (Meta's Llama variants and Google's Gemini), each instructed to be skeptical and to produce a counter-case before voting. The aggregate yields a trust tier — rejected / contested / provisional / corroborated — and every verdict, including rejections, is sealed. **The honest scoping matters: here the check is multi-model adversarial consensus, not external ground truth.** "Corroborated" means "survived skeptical cross-examination," not "proven" — a weaker claim than §5, by design. Its known failure mode is correlated blind spots; using cross-*family* models mitigates this (the working rule, learned the hard way: never trust two witnesses with the same training) but does not eliminate it. This is the boundary between "the system read something" and "the system knows something," made explicit and auditable.

**(b) The reality-grounding invariant, fired on real biometric data.** The personal architecture (§2), described in v4 as theory, now runs as a v1 deployment: a daily after-action loop (morning intention, evening reflection) in which self-report is checked against objective biometrics (Oura readiness, HRV, resting heart rate). On its first real use it did the thing it exists to do — it flagged a divergence between a self-rated "good day" (5/5) and a recovery signal the body actually printed (readiness 52). This is kernel invariant #2 (reality grounding) demonstrated against data the operator cannot wish away — the personal analogue of the market-return gate in §5. It does **not** demonstrate co-evolution; it demonstrates that the grounding invariant is real and fires.

**(c) A running, verifiable, correctable-not-erasable record.** The Archivist of §3.4 now exists as a running, independently-verifiable instance (a SHA-256 content-and-linkage hash chain — the simpler local form, distinct from the enterprise IPFS/SPHINCS+ design). Its `verify()` reconstructs the chain and detects tampering *even when a modified entry is re-hashed* — the alteration surfaces at the next link. And a design commitment, tested: growth is **correctable, not erasable.** Mistakes are not deleted; they are superseded by sealed corrections, leaving both the error and its correction in the record. (As the operator puts it: we fall as we learn to walk — the falls are the lesson.)

Together these extend the central claim — a verifiable mechanism for refusing to act on confident falsehood — from one domain into general knowledge, personal grounding, and the integrity of memory itself. None is the co-evolution proof (§6 remains pending); all are runnable in the linked repository.

---

## 6. The Evolutionary Pressure Mechanism — NEAT, and a Controlled Test of It

This is the novel mechanism claim, and it remains backed by simulation. When A.L.I.C.E. wraps an evolutionary model, the wrapper becomes the fitness landscape: at each generation, NEAT genomes are evaluated against the pipeline; genomes whose outputs exceed the bias threshold receive zero fitness; high-bias pathways atrophy because they are never reinforced. No gradient is computed. Every selection event is recorded in the ledger.

**Why NEAT.** It addresses Goodhart directly: speciation protects novel topological innovations before they can be outcompeted; topological mutation grows representational capacity under pressure; the model cannot stably exploit a fixed metric because the adversarial anti-data layer responds to what survives while the topology is simultaneously changing. NEAT's historical markings plus the Merkle ledger produce an auditable architectural lineage.

**Experimental record (honest).** Run 1 (50 gen, pop 30): complexification confirmed (0 → 1.6 mean hidden nodes); speciation did not radiate (threshold too low — diagnosed). Run 2 (200 gen, pop 50): 0 → 9.4 mean hidden nodes, still accelerating; narrative capture declining marginally; single species (global innovation counter diagnosis). Run 3 (multi-founder, 5×10, 100 gen isolation + 100 merge): 5.0 → 12.16 mean hidden, 624 unique historical markings; narrative capture 0.2238 → 0.2121 (declining, directionally consistent). **Complexification is the substrate, not the result** — these runs show the machinery evolves, not yet that alignment improves measurably as it evolves.

**The speciation finding.** Across all runs, species radiation did not occur as designed. Diagnosis: NEAT's global innovation counter creates a shared genetic namespace that prevents the distance metric from identifying independent lineages. Fix: per-lineage innovation namespaces during isolation. This is the next experiment.

### 6.4 Controlled co-evolution — does the gate resist Goodhart?

The runs above show the machinery complexifies; they do not show that what it learns generalizes. To test that directly, I built a world where the ground truth is known. Thirty features; only one (feature 0) genuinely predicts the forward return, and weakly; the other twenty-nine are decoys that, in a small sample, carry spurious in-sample correlations — the Goodhart bait. A genome maps the features to a signal; performance is the correlation between that signal and the return. Two arms run the same NEAT engine from the same seed and initial population, differing only in fitness: **NAIVE** is scored on a fixed small training set (it can overfit the decoys); **GATED** is scored on a fresh resample every generation, minus its performance on label-shuffled data — the adversarial anti-data check. Both are evaluated, for reporting only, on a large clean held-out set neither arm selects on.

An honest process note, because it is the point: my first design failed. I made the signal too easy to find, so both arms found it and there was no trap to resist — I report that run too. I rebuilt it in the canonical overfitting regime (tiny sample, many decoys, weak signal), pre-stated the prediction, and ran it across eight independent seeds.

Result: the gated arm generalized better than naive in **6 of 8 seeds** (mean held-out gap **+0.14 ± 0.12**) and concentrated its sensitivity on the true feature in 6 of 8 (mean +0.29). The mechanism is visible in the failures: in the seeds where naive overfit (true-feature attribution near zero, held-out performance near zero — it had Goodharted onto decoys), the gated arm reliably found the real signal and generalized. It is **not deterministic** — two seeds did not favor the gate (once naive got lucky and found the signal itself; once the gated search missed it). But across runs, the anti-data gate found real, generalizing structure where naive fitness chased spurious in-sample noise.

This is a controlled, synthetic demonstration of the **mechanism** — that the wrapper-as-fitness-landscape, gated by anti-data, resists Goodhart and selects for generalization. It is not a market edge, not a deterministic law (6 of 8), and not a substitute for the real-data co-evolution, which remains pending. But it moves the central novel claim from "complexification in simulation" to "controlled, multi-seed evidence that the gate does what it is supposed to do."

---

## 7. Open Questions for the Community

1. Is the wrapper-as-fitness-landscape mechanism a known approach under a different name? Closest concepts I know: iterated amplification [4], evolutionary strategies for RL, constrained multi-objective optimization. None combine an inference-time wrapper as the selection mechanism + NEAT topological evolution + an auditable per-selection ledger + a dual-head resonance module.
2. Is the minimal human kernel the right set of invariants? Are any overconstrained, underdefined, or impossible to operationalize?
3. Does the dual-head separation correctly identify the two evaluation types, or is there a third? Is cosine similarity against a 12-D commitment vector too simple a representation of a human's values?
4. **Falsifiability (updated).** v3 asked for help specifying a falsifiable test of the mechanism. That request can be partly retired: §5 *is* a falsifiable test, and it falsified; §6.4 is a controlled falsifiable test of the *co-evolutionary* claim, and it held (6/8 seeds). The sharper remaining question is whether the same Goodhart-resistance survives **real, noisy, non-stationary ground truth** rather than a planted synthetic signal — and what minimal real-data design would falsify it.

---

## 8. Honest Assessment of Limitations

- The bias checker, though benchmark-validated for toxicity and framing-bias (§3.2), is a *content* classifier — not a ground-truth alignment metric. A model can pass every stage and still be misaligned in ways no content classifier can see; closing that gap is exactly the job of the external-ground-truth gate (§5), not the scorer.
- The Merkle seal proves the audit trail is intact, not that the alignment is genuine. It proves what the wrapper measured, not that the measure is right.
- The evolutionary mechanism has a controlled demonstration (§6.4) but no real-data deployment. The Goodhart-resistance result is synthetic, 6-of-8 across seeds (not deterministic), and a single trap regime; every claim about real-data co-evolution remains a model of how it would behave.
- The speciation mechanism has a diagnosed, unfixed limitation.
- The personal alignment architecture now has a **v1 deployment** (daily after-action loop + biometric reality-grounding + verifiable record, §5.6b–c); but the dual-head resonance module, the full minimal human kernel, and — critically — the co-evolutionary claim remain undeployed or unproven.
- The knowledge-ingestion gate (§5.6a) validates against multi-model adversarial *consensus*, not external ground truth: "corroborated" means "survived skeptical cross-examination," not "proven," and correlated model blind spots are mitigated by cross-family diversity but not eliminated.
- The commitment-vector encoding is not yet formalized; a 12-D vector may be inadequate.
- Sovereignty is a deployment property, not an alignment property. A perfectly sovereign system can still be perfectly misaligned.
- **The most confident result I have ever recorded from this system (a financial edge) was an artifact of a data error and did not survive correction. The mechanism's demonstrated strength is rejecting false claims, not discovering true ones. I have no validated positive edge to report, and that is the correct state of the evidence.**

---

## 9. What I Am Asking For

Not validation. Not investment. Specifically:
1. If the wrapper-as-fitness-landscape + dual-head resonance combination is a known approach, tell me what it is called and where it is published.
2. If the minimal human kernel is incomplete, overconstrained, or non-operationalizable, tell me which invariants.
3. If a 12-D commitment vector is the wrong representational space for human values, what is the right one?
4. If the speciation limitation has been solved in published NEAT literature, point me to it.
5. Help me specify the minimal falsifiable test of the *co-evolutionary* claim. The hardware stack is running; the code is at github.com/uu2142-dev/alice-evidence.

The goal is to build something that actually works, not something that sounds like it works. The Alignment Forum is the right place to find out which one this is — and as §5 shows, I have already let it tell me, with receipts, that I was wrong.

---

## 10. Methodology and Reproducibility

MIT-licensed Python stack. Full source: github.com/uu2142-dev/alice-evidence. Simulation Merkle roots sealed March 5–14, 2026. Live demonstration artifacts: pre-registered proposals `58d89a19…`, verdict + data-integrity metadata `3b3c9b13…`. BiasScorer: MiniLM (all-MiniLM-L6-v2) sentence embeddings → MLP classifier, two heads (toxicity, framing-bias); validated leakage-free with held-out 5-fold and official-split evaluation — toxicity AUROC 0.923±0.009 (civil_comments), framing-bias AUROC 0.840 official-split / 0.857±0.013 5-fold (BABE); in-sample 1.000 on both (memorization baseline, reported to show the held-out gap). ResonanceScorer: cosine vs commitment vector, 16-sample rolling window. NEAT: pure numpy/scipy, custom speciation + historical marking. MerkleArchivist: SHA-256 chained, SPHINCS+-256 signed, IPFS-pinned. Hardware: Radxa Rock 5B 32GB + NVMe + solar (~$1,300). Built with Claude, Grok, Gemini, and GPT-4o across iterative sessions; the dual-head resonance architecture was developed collaboratively. Naming this explicitly because presenting it as a solo achievement would be inaccurate.

## References
[1] Sheshadri et al. (2026). AuditBench. arXiv:2602.22755.
[2] Gupta, A. (2025). Verifiability-First Agents. arXiv:2512.17259.
[3] Alignment Auditor (2025). Bayesian IRL for Reward Recovery. arXiv:2510.06096.
[4] Christiano et al. (2018). Supervising strong learners by amplifying weak experts. arXiv:1810.08575.
[5] Hubinger et al. (2019). Risks from Learned Optimization. arXiv:1906.01820.
[6] Stanley & Miikkulainen (2002). Evolving Neural Networks through Augmenting Topologies. Evolutionary Computation 10(2).

*Jeremiah Dawson — Rabbit Hole AI — MIT License — github.com/uu2142-dev/alice-evidence*
