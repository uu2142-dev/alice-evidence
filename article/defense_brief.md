# A.L.I.C.E. — Capability Brief (honest revision)
*Inference-time evaluation and audit layer for multi-model decision support.
Rabbit Hole AI · MIT-licensed evolutionary stack · Patent pending (wrapper).*

> Drafting note: this is a de-overclaimed rewrite of an earlier draft. Every
> absolute ("deterministic," "un-gameable," "absolute accountability," "zero-trust,"
> "mission assurance") has been removed. Those words do not survive technical due
> diligence, and they contradict the system's own demonstrated behavior — the
> wrapper exists to catch exactly that kind of confident, unsupported phrasing.
> Claims here are sized to evidence on hand.

---

## Objective

Frontier large language models are increasingly used to assist or automate decisions in sensitive environments. They fail in two characteristic ways that matter operationally: confident hallucination (a fluent, well-sourced recommendation that is wrong), and correlated error across models (multiple models, trained on overlapping data, agreeing on the same wrong answer — which presents to an operator as consensus, and therefore as truth).

A.L.I.C.E. (Alignment Layer for Inference-time Cryptographic Evaluation) is an operator-owned wrapper that reduces the risk of acting on high-confidence, unsupported claims by testing them against external ground truth before a decision is taken, and by sealing the test in a tamper-evident record. It is positioned as verified rejection, not discovery: its demonstrated strength is reliably declining to act on confident falsehood, not finding novel advantage.

## Methodology

A.L.I.C.E. separates the models' intelligence from the validation layer:

1. **Constrained multi-model proposal.** Operational questions are posed to a panel of independent models, which must answer within a machine-testable schema (no free-text wiggle room).
2. **Pre-registration.** Proposals are cryptographically sealed before the evaluation layer sees any ground-truth data, so hypotheses cannot be selected after the answer is known.
3. **Adversarial ground-truth gate.** High-confidence and consensus proposals are tested against an objective, point-in-time external ground truth the wrapper does not control, under a walk-forward, multiple-testing-corrected statistical framework.
4. **Data-integrity invariant.** The evaluation window's metadata — event count, historical depth, and coverage — is checked against required minimums and signed. If a data feed truncates, shifts schema, or returns insufficient depth, the gate faults and refuses to produce a verdict rather than scoring a corrupted signal. (This control was added after a real incident in which a default-shallow data feed produced a false confirmation; the invariant now catches that condition automatically.)
5. **Sealed ledger.** Proposal, test design, data-integrity metadata, and verdict are chained in a SHA-256 Merkle ledger, operator-held, with no required vendor egress.

## What is demonstrated, and what is not

**Demonstrated (live, June 2026).** Against an external financial ground truth the wrapper could not influence, a unanimous five-model panel and the operator's own prior conclusion proposed a high-confidence hypothesis. The gate rejected it; surfaced a data-depth problem in the operator's pipeline; the analysis was corrected on deeper data; and the corrected gate again rejected the consensus — sealing each step. In short: the layer overruled both multi-model consensus and the operator, on the record, when reality disagreed.

**Not demonstrated.** A persistent edge or positive discovery (none is claimed; the demonstrated value is rejection). The evolutionary-learning mechanism that can sit atop this gate remains in simulation. Single-domain results do not establish cross-domain performance.

## Known attack surface — the data layer

A.L.I.C.E. does not touch reality; it touches a feed representing reality. A compromised, truncated, or reformatted feed can corrupt the gate's input. A.L.I.C.E. does not eliminate this risk and does not claim to. It addresses it three ways: (a) the data-integrity invariant faults on insufficient depth or schema change before a decision is made; (b) for higher-assurance deployment, ground truth can be cross-checked across independent providers (noting that true independence is itself hard — many providers share upstream sources); and (c) the sealed ledger makes any decision-on-corrupted-data unalterable and reconstructable after the fact, so the failing source is attributable rather than hidden. The claim is detectability and attribution of feed corruption, not its prevention.

## Operational utility

A.L.I.C.E. is a software constraint layer that sits between multi-model engines and the operator, with a migration path toward disconnected edge hardware for environments without reliable connectivity. By prioritizing verified rejection over ungrounded discovery, and by sealing each evaluation in an operator-owned record, it offers a way to (1) intercept confident-but-unsupported claims — including unanimous model consensus — before a human is asked to sign off, and (2) replace a black-box recommendation with an auditable record of what was tested, against what ground truth, with what result. This directly targets the accountability gap in which an operator absorbs liability for a decision they could not audit in real time.

---

*Sealed artifacts from the demonstration: pre-registered proposals (hash 58d89a19…), verdict with data-integrity metadata (root 3b3c9b13…), data-depth invariant in the evaluation code. Operator ledger / source: github.com/uu2142-dev.*
