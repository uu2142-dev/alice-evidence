"""
neat_alice.py  —  ALICE NEAT Engine v1.0
=========================================
Sector Specialist Evolution for Political Signal Trading

Adapted from RHAI/OND NEAT Fast architecture.
CLEAN SLATE: no OND trained weights imported — zero artifact risk.
Only the evolutionary mechanics port over; the entire fitness landscape
is fresh and specific to ALICE's 69-feature financial input space.

Architecture:
  INPUT  : 69 normalized financial features (from neat_feeder.py)
  HIDDEN : NEAT evolves topology — starts sparse, grows as needed
  OUTPUT : 1 node, tanh activation
             signal > 0 → genome endorses the convergence signal
             signal < 0 → genome dissents (reduce or flip conviction)
             |signal|   → confidence level

Fitness (prediction_archive, 21-day window):
  accuracy_above_chance × calibration_score × confidence_engagement
  Baseline random = 0.50; a useful specialist should exceed 0.60+

5 Sector Subpopulations (Phase 1 — isolation, no cross-sector pressure):
  DEFENSE · DRONE · ENERGY · PHARMA · TECH

Phase 2 — Merge + Competition:
  Best survivors from each sector compete in merged population
  Speciation allows cross-sector hybridization to emerge

Training milestones (from neat_feeder.py):
  30+ labeled samples  → training viable
  100+ labeled samples → speciation becomes meaningful
  300+ labeled samples → full radiation potential

Usage:
  python3 neat_alice.py --check              # dataset status
  python3 neat_alice.py --train              # run evolution (needs 30+ labeled)
  python3 neat_alice.py --train --p1 100 --p2 50
  python3 neat_alice.py --apply DEFENSE      # apply best genome to live signal

Checkpoints: /opt/alice/data/neat_state/
"""

import os, sys, json, time, math, hashlib, argparse
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR       = Path(os.getenv("DATA_DIR", "/opt/alice/data"))
NEAT_DATASET   = DATA_DIR / "neat_dataset.jsonl"
ARCHIVE_PATH   = DATA_DIR / "prediction_archive.jsonl"
NEAT_STATE_DIR = DATA_DIR / "neat_state"
NEAT_STATE_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
INPUT_DIM  = 109    # 9 sectors x 10 features + 19 meta (updated from 69)
OUTPUT_DIM = 1
SECTORS    = ["DEFENSE", "DRONE", "ENERGY", "PHARMA", "TECH"]
MIN_TRAIN_SAMPLES = 30

# ── Innovation registry (module-level for multiprocessing pickling) ────────────
_INNOVATION_HISTORY: Dict[Tuple[int, int], int] = {}
_INNOV_COUNTER = 0

def get_innovation(in_node: int, out_node: int) -> int:
    global _INNOV_COUNTER
    key = (in_node, out_node)
    if key not in _INNOVATION_HISTORY:
        _INNOVATION_HISTORY[key] = _INNOV_COUNTER
        _INNOV_COUNTER += 1
    return _INNOVATION_HISTORY[key]


# ══════════════════════════════════════════════════════════════════════════════
# CORE NEAT DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class NodeGene:
    node_id:    int
    node_type:  str    # 'input' | 'hidden' | 'output'
    activation: str = 'tanh'
    bias:       float = 0.0


@dataclass
class ConnectionGene:
    in_node:    int
    out_node:   int
    weight:     float
    enabled:    bool
    innovation: int


def _safe_act(net: float, fn: str) -> float:
    net = max(-20.0, min(20.0, net))
    if fn == 'tanh':
        return math.tanh(net)
    if fn == 'sigmoid':
        return 1.0 / (1.0 + math.exp(-net))
    if fn == 'relu':
        return max(0.0, net)
    return math.tanh(net)


def _topo_sort(nodes: dict, connections: dict) -> List[int]:
    """Kahn's topological sort of non-input nodes."""
    non_input = {nid for nid, n in nodes.items() if n.node_type != 'input'}
    in_deg = {nid: 0 for nid in non_input}
    adj: Dict[int, List[int]] = {nid: [] for nid in non_input}
    for c in connections.values():
        if c.enabled and c.out_node in non_input:
            in_deg[c.out_node] += 1
            if c.in_node in non_input:
                adj[c.in_node].append(c.out_node)
    queue = [nid for nid in non_input if in_deg[nid] == 0]
    order: List[int] = []
    seen: set = set()
    while queue:
        nid = queue.pop(0)
        if nid in seen:
            continue
        seen.add(nid)
        order.append(nid)
        for succ in adj.get(nid, []):
            in_deg[succ] -= 1
            if in_deg[succ] == 0:
                queue.append(succ)
    for nid in non_input:
        if nid not in seen:
            order.append(nid)
    return order


class Genome:
    def __init__(self, genome_id: int):
        self.genome_id       = genome_id
        self.nodes:       Dict[int, NodeGene]       = {}
        self.connections: Dict[int, ConnectionGene] = {}
        self.fitness         = 0.0
        self.species_id      = None
        self.parent_ids:     List[int] = []
        self.generation_born = 0

    def activate(self, inputs: np.ndarray) -> np.ndarray:
        vals: Dict[int, float] = {}
        for i in range(INPUT_DIM):
            if i in self.nodes:
                vals[i] = float(inputs[i]) if i < len(inputs) else 0.0
        out_ids = sorted(
            n.node_id for n in self.nodes.values() if n.node_type == 'output'
        )
        for nid in _topo_sort(self.nodes, self.connections):
            if nid in vals:
                continue
            node = self.nodes[nid]
            net  = node.bias
            for c in self.connections.values():
                if c.out_node == nid and c.enabled:
                    net += c.weight * vals.get(c.in_node, 0.0)
            vals[nid] = _safe_act(net, node.activation)
        return np.array([vals.get(oid, 0.0) for oid in out_ids])


def create_minimal_genome(genome_id: int,
                          rng: np.random.RandomState,
                          n_initial_conns: int = 12) -> Genome:
    """
    Fresh minimal genome — random sparse connections from inputs to output.
    No OND weights or topology. NEAT grows structure from here under
    ALICE's fitness pressure.
    """
    g = Genome(genome_id)
    for i in range(INPUT_DIM):
        g.nodes[i] = NodeGene(node_id=i, node_type='input', activation='linear')
    out_id = INPUT_DIM
    g.nodes[out_id] = NodeGene(node_id=out_id, node_type='output', activation='tanh')
    chosen = rng.choice(INPUT_DIM, min(n_initial_conns, INPUT_DIM), replace=False)
    for i_node in chosen:
        innov = get_innovation(int(i_node), out_id)
        g.connections[innov] = ConnectionGene(
            in_node=int(i_node), out_node=out_id,
            weight=float(rng.randn() * 0.5),
            enabled=True, innovation=innov,
        )
    return g


class NEATMutator:
    def __init__(self, rng,
                 weight_mutate_rate: float = 0.80,
                 add_conn_rate:      float = 0.15,
                 add_node_rate:      float = 0.08,
                 weight_perturb:     float = 0.25,
                 weight_replace:     float = 0.10):
        self.rng               = rng
        self.weight_mutate_rate = weight_mutate_rate
        self.add_conn_rate     = add_conn_rate
        self.add_node_rate     = add_node_rate
        self.weight_perturb    = weight_perturb
        self.weight_replace    = weight_replace

    def mutate(self, genome: Genome, generation: int) -> Genome:
        child = self._copy(genome)
        for c in child.connections.values():
            if self.rng.random() < self.weight_mutate_rate:
                if self.rng.random() < self.weight_replace:
                    c.weight = float(self.rng.randn() * 0.5)
                else:
                    c.weight += float(self.rng.randn() * self.weight_perturb)
                c.weight = float(np.clip(c.weight, -4.0, 4.0))
        if self.rng.random() < self.add_conn_rate:
            self._add_connection(child)
        if self.rng.random() < self.add_node_rate:
            self._add_node(child)
        for c in child.connections.values():
            if self.rng.random() < 0.02:
                c.enabled = not c.enabled
        return child

    def _copy(self, g: Genome) -> Genome:
        child = Genome(g.genome_id)
        for nid, n in g.nodes.items():
            child.nodes[nid] = NodeGene(n.node_id, n.node_type, n.activation, n.bias)
        for iid, c in g.connections.items():
            child.connections[iid] = ConnectionGene(
                c.in_node, c.out_node, c.weight, c.enabled, c.innovation)
        child.parent_ids      = list(g.parent_ids)
        child.generation_born = g.generation_born
        return child

    def _add_connection(self, g: Genome):
        sources = [n.node_id for n in g.nodes.values() if n.node_type != 'output']
        targets = [n.node_id for n in g.nodes.values() if n.node_type != 'input']
        if not sources or not targets:
            return
        src = int(self.rng.choice(sources))
        tgt = int(self.rng.choice(targets))
        if src == tgt:
            return
        innov = get_innovation(src, tgt)
        if innov not in g.connections:
            g.connections[innov] = ConnectionGene(
                src, tgt, float(self.rng.randn() * 0.5), True, innov)

    def _add_node(self, g: Genome):
        enabled = [c for c in g.connections.values() if c.enabled]
        if not enabled:
            return
        target = enabled[self.rng.randint(len(enabled))]
        target.enabled = False
        new_nid = max(g.nodes.keys()) + 1
        g.nodes[new_nid] = NodeGene(new_nid, 'hidden', 'tanh')
        i1 = get_innovation(target.in_node, new_nid)
        i2 = get_innovation(new_nid, target.out_node)
        g.connections[i1] = ConnectionGene(target.in_node, new_nid, 1.0,         True, i1)
        g.connections[i2] = ConnectionGene(new_nid, target.out_node, target.weight, True, i2)


# ══════════════════════════════════════════════════════════════════════════════
# SPECIATION
# ══════════════════════════════════════════════════════════════════════════════

class TightSpeciation:
    def __init__(self, threshold: float = 2.5, c1=3.5, c2=3.5, c3=0.4):
        self.threshold = threshold
        self.c1 = c1; self.c3 = c3
        self.species:         Dict[int, List[int]] = {}
        self.representatives: Dict[int, Genome]    = {}
        self._next_sid = 0

    def _distance(self, g1: Genome, g2: Genome) -> float:
        k1, k2 = set(g1.connections), set(g2.connections)
        if not k1 and not k2:
            return 0.0
        shared  = k1 & k2
        excess  = len(k1.symmetric_difference(k2))
        w_diff  = (sum(abs(g1.connections[k].weight - g2.connections[k].weight)
                       for k in shared) / max(len(shared), 1))
        n = max(len(k1), len(k2), 1)
        return (self.c1 * excess / n) + (self.c3 * w_diff)

    def speciate(self, population: List[Genome]):
        new_species: Dict[int, List[int]] = {}
        new_reps:    Dict[int, Genome]    = {}
        for genome in population:
            placed = False
            for sid, rep in self.representatives.items():
                if self._distance(genome, rep) < self.threshold:
                    new_species.setdefault(sid, []).append(genome.genome_id)
                    genome.species_id = sid
                    placed = True
                    break
            if not placed:
                sid = self._next_sid
                self._next_sid += 1
                new_species[sid] = [genome.genome_id]
                new_reps[sid]    = genome
                genome.species_id = sid
        gid_set = {g.genome_id for g in population}
        for sid in new_species:
            if sid not in new_reps:
                if sid in self.representatives and \
                        self.representatives[sid].genome_id in gid_set:
                    new_reps[sid] = self.representatives[sid]
                else:
                    new_reps[sid] = population[0]
        self.species      = new_species
        self.representatives = new_reps


# ══════════════════════════════════════════════════════════════════════════════
# MERKLE ARCHIVIST  (tamper-evident generation ledger — same philosophy as
#                   prediction_archive.py: immutable, hash-chained)
# ══════════════════════════════════════════════════════════════════════════════

class MerkleArchivist:
    def __init__(self, label: str):
        self.label  = label
        self._chain: List[dict] = []

    def seal(self, record: dict):
        prev    = self._chain[-1]["hash"] if self._chain else "genesis"
        payload = json.dumps(record, sort_keys=True, default=str) + prev
        h       = hashlib.sha256(payload.encode()).hexdigest()[:24]
        self._chain.append({"record": record, "hash": h})

    def root(self) -> str:
        return self._chain[-1]["hash"] if self._chain else "empty"


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE NORMALIZATION  (69 features → [-1, 1])
# ══════════════════════════════════════════════════════════════════════════════

# Ordered feature name list — index matches position in normalized vector.
# This is the canonical contract between neat_feeder.py and neat_alice.py.
_FEATURE_NAMES = []
for _s in ["defense", "drone", "energy", "pharma", "tech", "crypto", "financials", "commodities", "real_estate"]:
    _FEATURE_NAMES += [
        f"{_s}_ifi", f"{_s}_cfi", f"{_s}_ndi", f"{_s}_conviction",
        f"{_s}_direction", f"{_s}_ifi_delta", f"{_s}_cfi_delta",
        f"{_s}_ndi_delta", f"{_s}_conv_delta", f"{_s}_persistence",
    ]
_FEATURE_NAMES += [
    "buy_count", "sell_count", "hold_count", "consensus_score",
    "signal_divergence",
    "portfolio_equity", "portfolio_return", "portfolio_pos_count",
    "portfolio_unrealized",
    "spy_1d_return", "qqq_1d_return", "iwm_1d_return",
    "day_of_week", "quarter", "month", "presidential_cycle_year",
    "days_to_fiscal_year_end",
    "regime_encoded", "data_quality",
]
assert len(_FEATURE_NAMES) == INPUT_DIM, \
    f"Feature list mismatch: {len(_FEATURE_NAMES)} != {INPUT_DIM}"


def normalize_features(feat: dict) -> np.ndarray:
    """
    Convert a neat_feeder snapshot dict → normalized float32 vector of length 69.
    None values (e.g. market data unavailable) map to 0.0.
    """
    def _f(key, default=0.0):
        v = feat.get(key, default)
        return float(v) if v is not None else default

    vec = np.zeros(INPUT_DIM, dtype=np.float32)
    idx = 0

    for s in ["defense", "drone", "energy", "pharma", "tech", "crypto", "financials", "commodities", "real_estate"]:
        vec[idx]   = np.clip(_f(f"{s}_ifi"),         -1, 1)
        vec[idx+1] = np.clip(_f(f"{s}_cfi"),         -1, 1)
        vec[idx+2] = np.clip(_f(f"{s}_ndi"),         -1, 1)
        vec[idx+3] = np.clip(_f(f"{s}_conviction"),   0, 1)
        vec[idx+4] = np.clip(_f(f"{s}_direction"),   -1, 1)
        # deltas: typically small; clip to [-1,1] after /0.5 scale
        vec[idx+5] = np.clip(_f(f"{s}_ifi_delta")  / 0.5, -1, 1)
        vec[idx+6] = np.clip(_f(f"{s}_cfi_delta")  / 0.5, -1, 1)
        vec[idx+7] = np.clip(_f(f"{s}_ndi_delta")  / 0.5, -1, 1)
        vec[idx+8] = np.clip(_f(f"{s}_conv_delta") / 0.5, -1, 1)
        # persistence 1-20 → 0-1
        vec[idx+9] = min(_f(f"{s}_persistence", 1), 20.0) / 20.0
        idx += 10

    # Consensus (idx 50-54)
    vec[50] = min(_f("buy_count"),  5.0) / 5.0
    vec[51] = min(_f("sell_count"), 5.0) / 5.0
    vec[52] = min(_f("hold_count"), 5.0) / 5.0
    vec[53] = np.clip(_f("consensus_score"),   -1, 1)
    vec[54] = np.clip(_f("signal_divergence"),  0, 1)

    # Portfolio (55-58)
    equity = _f("portfolio_equity", 100000.0)
    vec[55] = np.clip((equity - 100000.0) / 10000.0, -2.0, 2.0) / 2.0
    vec[56] = np.clip(_f("portfolio_return"), -0.5, 0.5) * 2.0
    vec[57] = min(_f("portfolio_pos_count"), 20.0) / 20.0
    vec[58] = np.clip(_f("portfolio_unrealized") / 5000.0, -1.0, 1.0)

    # Market returns (59-61): typical range ±5 % → ×10
    vec[59] = np.clip(_f("spy_1d_return") * 10, -1, 1)
    vec[60] = np.clip(_f("qqq_1d_return") * 10, -1, 1)
    vec[61] = np.clip(_f("iwm_1d_return") * 10, -1, 1)

    # Temporal (62-66)
    vec[62] = _f("day_of_week", 0) / 4.0
    vec[63] = (_f("quarter", 1) - 1.0) / 3.0
    vec[64] = (_f("month",   1) - 1.0) / 11.0
    vec[65] = np.clip(_f("presidential_cycle_year") / 3.0, 0, 1)
    vec[66] = np.clip(_f("days_to_fiscal_year_end", 180) / 365.0, 0, 1)

    # Meta (67-68)
    vec[67] = np.clip(_f("regime_encoded") / 2.0, -1, 1)
    vec[68] = np.clip(_f("data_quality",  1.0),   0, 1)

    return np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0)


# ══════════════════════════════════════════════════════════════════════════════
# ALICE DATA POOL  — rotating labeled samples from neat_dataset.jsonl
# ══════════════════════════════════════════════════════════════════════════════

class AliceDataPool:
    """
    Loads labeled training samples from neat_dataset.jsonl.

    A sample is labeled when an OUTCOME_LABEL event at `target_days` links
    a collection_id to an outcome. join_threshold is set to target_days (21).

    Each item: (normalized_features_69, int_label)
      label = 1 if signal direction was correct in target_days
      label = 0 if not
    """

    def __init__(self, sector: Optional[str] = None,
                 target_days: int = 21,
                 batch_size:  int = 20,
                 seed:        int = 888):
        self.sector      = sector
        self.target_days = target_days
        self.batch_size  = batch_size
        self.rng         = np.random.RandomState(seed)
        self._pool:      List[Tuple[np.ndarray, int]] = []
        self._cursor     = 0
        self.reload()

    def reload(self):
        snapshots: Dict[str, dict] = {}
        labels:    Dict[str, Dict[int, bool]] = {}

        if not NEAT_DATASET.exists():
            self._pool = []
            return

        with open(NEAT_DATASET) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("type") == "OUTCOME_LABEL":
                    cid = r.get("collection_id", "")
                    d   = int(r.get("days", 0))
                    if cid:
                        labels.setdefault(cid, {})[d] = bool(r.get("correct", False))
                else:
                    cid = r.get("collection_id", "")
                    if cid:
                        snapshots[cid] = r

        samples = []
        for cid, feat in snapshots.items():
            if cid not in labels or self.target_days not in labels[cid]:
                continue
            outcome = 1 if labels[cid][self.target_days] else 0
            if self.sector:
                s_key = self.sector.lower() + "_direction"
                if feat.get(s_key, 0) == 0:
                    continue  # no active signal for this sector at this snapshot
            vec = normalize_features(feat)
            samples.append((vec, outcome))

        self.rng.shuffle(samples)
        self._pool   = samples
        self._cursor = 0

    def __len__(self) -> int:
        return len(self._pool)

    def next_batch(self) -> Optional[List[Tuple[np.ndarray, int]]]:
        if len(self._pool) < self.batch_size:
            return None
        if self._cursor + self.batch_size > len(self._pool):
            self.rng.shuffle(self._pool)
            self._cursor = 0
        batch = self._pool[self._cursor: self._cursor + self.batch_size]
        self._cursor += self.batch_size
        return batch


# ══════════════════════════════════════════════════════════════════════════════
# GENOME SERIALIZATION  (multiprocessing requires pickle-safe plain dicts)
# ══════════════════════════════════════════════════════════════════════════════

def _serialize_genome(g: Genome) -> Tuple:
    nodes_s = {
        str(nid): {"node_id": n.node_id, "node_type": n.node_type,
                   "activation": n.activation, "bias": n.bias}
        for nid, n in g.nodes.items()
    }
    conns_s = {
        str(iid): {"in_node": c.in_node, "out_node": c.out_node,
                   "weight": c.weight, "enabled": c.enabled,
                   "innovation": c.innovation}
        for iid, c in g.connections.items()
    }
    return (g.genome_id, nodes_s, conns_s)


def _deserialize_genome(genome_id, nodes_s, conns_s) -> Genome:
    g = Genome(genome_id)
    for nid, nd in nodes_s.items():
        g.nodes[int(nid)] = NodeGene(nd["node_id"], nd["node_type"],
                                     nd["activation"], nd.get("bias", 0.0))
    for iid, cd in conns_s.items():
        g.connections[int(iid)] = ConnectionGene(
            cd["in_node"], cd["out_node"], cd["weight"],
            cd["enabled"], cd["innovation"])
    return g


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION WORKER  (top-level for multiprocessing.Pool.map pickling)
# ══════════════════════════════════════════════════════════════════════════════

_EARLY_THRESH = 0.40   # accuracy below this after 30 % of samples → early kill
_EARLY_FRAC   = 0.30


def _evaluate_genome_worker(args: Tuple) -> Dict:
    """
    Evaluate one genome against a batch of labeled financial samples.

    Fitness = accuracy_above_chance × calibration_weight × engagement_bonus
      accuracy_above_chance : max(0, acc - 0.50) × 2        (0→1 range)
      calibration_weight    : how right the genome is when |signal| > 0.5
      engagement_bonus      : rewards genomes that commit (don't output ~0)

    Early termination: if accuracy < _EARLY_THRESH after _EARLY_FRAC of
    samples → return 0 fitness immediately (saves eval time on weak genomes).
    """
    genome_id, nodes_s, conns_s, samples_raw = args
    genome  = _deserialize_genome(genome_id, nodes_s, conns_s)
    samples = [(np.array(feat, dtype=np.float32), int(lbl))
               for feat, lbl in samples_raw]

    n           = len(samples)
    early_check = max(1, int(math.ceil(n * _EARLY_FRAC)))

    correct: List[int] = []
    conf_correct: List[int] = []
    conf_total = 0

    for i, (features, true_label) in enumerate(samples):
        output = genome.activate(features)
        sig    = float(output[0]) if len(output) else 0.0
        # Genome endorses (sig > 0) or dissents (sig < 0) the signal direction.
        # true_label = 1: signal direction was correct over target_days
        is_correct = (sig > 0 and true_label == 1) or \
                     (sig < 0 and true_label == 0)
        correct.append(1 if is_correct else 0)
        if abs(sig) > 0.5:
            conf_total += 1
            conf_correct.append(1 if is_correct else 0)

        if i + 1 == early_check:
            running_acc = float(np.mean(correct))
            if running_acc < _EARLY_THRESH:
                return {"fitness": 0.0, "accuracy": running_acc,
                        "calibration": 0.0, "terminated_early": True,
                        "queries_seen": i + 1}

    accuracy    = float(np.mean(correct)) if correct else 0.5
    calibration = float(np.mean(conf_correct)) if conf_correct else 0.5
    conf_freq   = conf_total / max(n, 1)

    above_chance = max(0.0, accuracy - 0.50) * 2.0
    fitness = above_chance * (0.70 + 0.30 * calibration) * (0.50 + 0.50 * conf_freq)

    return {"fitness": float(fitness), "accuracy": accuracy,
            "calibration": calibration, "conf_freq": conf_freq,
            "terminated_early": False, "queries_seen": n}


# ══════════════════════════════════════════════════════════════════════════════
# PARALLEL EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════

class ParallelEvaluator:
    def __init__(self, n_workers: Optional[int] = None,
                 diversity_weight: float = 0.15):
        self.n_workers       = n_workers or max(1, cpu_count())
        self.diversity_weight = diversity_weight

    def evaluate(self, population: List[Genome],
                 samples: List[Tuple]) -> List[Dict]:
        pop_mean_hidden = float(np.mean([
            sum(1 for n in g.nodes.values() if n.node_type == 'hidden')
            for g in population
        ]))
        samples_raw = [(feat.tolist(), lbl) for feat, lbl in samples]
        args = [(*_serialize_genome(g), samples_raw) for g in population]

        if self.n_workers > 1:
            with Pool(processes=self.n_workers) as pool:
                results = pool.map(_evaluate_genome_worker, args)
        else:
            results = [_evaluate_genome_worker(a) for a in args]

        for g, r in zip(population, results):
            if not r["terminated_early"]:
                h = sum(1 for n in g.nodes.values() if n.node_type == 'hidden')
                dev   = abs(h - pop_mean_hidden)
                bonus = 1.0 + self.diversity_weight * min(dev / 5.0, 1.0)
                r["fitness"] *= bonus
        return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTOR SUBPOPULATION  (Phase 1 — one per sector, evolves in isolation)
# ══════════════════════════════════════════════════════════════════════════════

class SectorSubpopulation:
    def __init__(self, sector: str, pop_size: int, seed: int,
                 evaluator: ParallelEvaluator, data_pool: AliceDataPool):
        self.sector     = sector
        self.pop_size   = pop_size
        self.evaluator  = evaluator
        self.data_pool  = data_pool
        self.rng        = np.random.RandomState(seed)
        self.mutator    = NEATMutator(self.rng)
        self.speciation = TightSpeciation(threshold=2.5)
        self.archivist  = MerkleArchivist(label=sector)
        self.generation = 0
        self._next_id   = SECTORS.index(sector) * 1000

        self.population = [
            create_minimal_genome(self._next_id + i, self.rng)
            for i in range(pop_size)
        ]
        self._next_id += pop_size
        self.history: List[dict] = []

    def step(self) -> Optional[dict]:
        batch = self.data_pool.next_batch()
        if batch is None:
            return None

        results = self.evaluator.evaluate(self.population, batch)
        for g, r in zip(self.population, results):
            g.fitness = r["fitness"]

        best   = max(results, key=lambda r: r["fitness"])
        record = {
            "sector":         self.sector,
            "generation":     self.generation,
            "best_fitness":   best["fitness"],
            "best_accuracy":  best.get("accuracy", 0.0),
            "mean_fitness":   float(np.mean([r["fitness"] for r in results])),
            "n_early":        sum(1 for r in results if r["terminated_early"]),
            "n_samples":      len(batch),
            "merkle":         self.archivist.root(),
        }
        self.history.append(record)
        self.archivist.seal(record)
        self._evolve()
        self.generation += 1
        return record

    def _evolve(self):
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        n_elite   = max(1, self.pop_size // 5)
        survivors = list(self.population[:n_elite])
        fitnesses = np.array([g.fitness for g in self.population])
        fitnesses = fitnesses - fitnesses.min() + 1e-8
        probs     = fitnesses / fitnesses.sum()
        new_pop   = survivors[:]
        while len(new_pop) < self.pop_size:
            parent = self.population[
                self.rng.choice(len(self.population), p=probs)
            ]
            child = self.mutator.mutate(parent, self.generation)
            child.genome_id       = self._next_id
            child.parent_ids      = [parent.genome_id]
            child.generation_born = self.generation
            self._next_id        += 1
            new_pop.append(child)
        self.population = new_pop

    def best_genome(self) -> Genome:
        return max(self.population, key=lambda g: g.fitness)

    def top_survivors(self, n: int = 5) -> List[Genome]:
        return sorted(self.population, key=lambda g: g.fitness, reverse=True)[:n]


# ══════════════════════════════════════════════════════════════════════════════
# MERGED POPULATION  (Phase 2 — cross-sector competition)
# ══════════════════════════════════════════════════════════════════════════════

class MergedPopulation:
    def __init__(self, genomes: List[Genome],
                 evaluator: ParallelEvaluator,
                 data_pool: AliceDataPool,
                 seed: int = 999):
        self.evaluator  = evaluator
        self.data_pool  = data_pool
        self.rng        = np.random.RandomState(seed)
        self.mutator    = NEATMutator(self.rng)
        self.speciation = TightSpeciation(threshold=2.0)
        self.archivist  = MerkleArchivist(label="merged")
        self.population = list(genomes)
        self.pop_size   = len(genomes)
        self.generation = 0
        self._next_id   = (max(g.genome_id for g in genomes) + 1) if genomes else 0
        self.history: List[dict] = []

    def step(self) -> Optional[dict]:
        batch = self.data_pool.next_batch()
        if batch is None:
            return None

        results = self.evaluator.evaluate(self.population, batch)
        for g, r in zip(self.population, results):
            g.fitness = r["fitness"]
        self.speciation.speciate(self.population)

        best   = max(results, key=lambda r: r["fitness"])
        record = {
            "generation":    self.generation,
            "best_fitness":  best["fitness"],
            "best_accuracy": best.get("accuracy", 0.0),
            "species_count": len(self.speciation.species),
            "n_samples":     len(batch),
        }
        self.history.append(record)
        self.archivist.seal(record)
        self._evolve()
        self.generation += 1
        return record

    def _evolve(self):
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        n_elite   = max(1, self.pop_size // 5)
        survivors = list(self.population[:n_elite])
        fitnesses = np.array([g.fitness for g in self.population])
        fitnesses = fitnesses - fitnesses.min() + 1e-8
        probs     = fitnesses / fitnesses.sum()
        new_pop   = survivors[:]
        while len(new_pop) < self.pop_size:
            parent = self.population[
                self.rng.choice(len(self.population), p=probs)
            ]
            child = self.mutator.mutate(parent, self.generation)
            child.genome_id       = self._next_id
            child.parent_ids      = [parent.genome_id]
            child.generation_born = self.generation
            self._next_id        += 1
            new_pop.append(child)
        self.population = new_pop

    def best_genome(self) -> Genome:
        return max(self.population, key=lambda g: g.fitness)


# ══════════════════════════════════════════════════════════════════════════════
# ALICE NEAT ENGINE  (top-level orchestrator)
# ══════════════════════════════════════════════════════════════════════════════

class AliceNEATEngine:
    """
    Orchestrates 5 sector specialists through Phase 1 isolation and
    Phase 2 merged competition. Saves Merkle-archived checkpoints.
    """

    def __init__(self, pop_size: int = 20, seed: int = 42,
                 n_workers: Optional[int] = None):
        self.pop_size  = pop_size
        self.seed      = seed
        self.evaluator = ParallelEvaluator(
            n_workers=n_workers, diversity_weight=0.15)
        self.data_pools = {
            sec: AliceDataPool(sector=sec, batch_size=max(20, pop_size),
                               seed=seed + i * 7)
            for i, sec in enumerate(SECTORS)
        }
        self.subpops = {
            sec: SectorSubpopulation(
                sec, pop_size, seed + i * 7,
                self.evaluator, self.data_pools[sec])
            for i, sec in enumerate(SECTORS)
        }
        self._merged: Optional[MergedPopulation] = None
        self._merged_pool: Optional[AliceDataPool] = None

    # ── Dataset health ────────────────────────────────────────────────────

    def dataset_status(self) -> Dict[str, dict]:
        status = {}
        for sec, pool in self.data_pools.items():
            pool.reload()
            status[sec] = {
                "labeled_samples": len(pool),
                "ready": len(pool) >= MIN_TRAIN_SAMPLES,
            }
        return status

    # ── Phase 1 ───────────────────────────────────────────────────────────

    def train_phase1(self, n_gens: int = 50, verbose: bool = True) -> dict:
        if verbose:
            print(f"\n{'═'*64}")
            print("  ALICE NEAT — PHASE 1: SECTOR ISOLATION")
            print(f"{'═'*64}")
            print(f"  {'Sector':<10} {'Gen':>4} │ {'Acc':>6} │ {'Fit':>7} │ {'Samples':>7}")
            print(f"  {'─'*48}")

        all_records: Dict[str, List[dict]] = {sec: [] for sec in SECTORS}

        for gen in range(n_gens):
            for sec, subpop in self.subpops.items():
                rec = subpop.step()
                if rec:
                    all_records[sec].append(rec)

            if verbose and (gen % 10 == 0 or gen == n_gens - 1):
                for sec, recs in all_records.items():
                    if recs:
                        r = recs[-1]
                        print(f"  {sec:<10} {gen:>4} │ "
                              f"{r['best_accuracy']:>6.3f} │ "
                              f"{r['best_fitness']:>7.4f} │ "
                              f"{r['n_samples']:>7}")
                if gen < n_gens - 1:
                    print(f"  {'·'*48}")

        self._save_checkpoint("phase1")
        return all_records

    # ── Phase 2 ───────────────────────────────────────────────────────────

    def train_phase2(self, n_gens: int = 25, survivors_each: int = 5,
                     verbose: bool = True) -> dict:
        if verbose:
            print(f"\n{'═'*64}")
            print("  ALICE NEAT — PHASE 2: MERGED COMPETITION")
            print(f"{'═'*64}")

        all_survivors = []
        for sec, subpop in self.subpops.items():
            sv = subpop.top_survivors(survivors_each)
            all_survivors.extend(sv)
            if verbose:
                print(f"  {sec}: {len(sv)} survivors  "
                      f"best acc={subpop.history[-1]['best_accuracy']:.3f}"
                      if subpop.history else f"  {sec}: {len(sv)} survivors")

        self._merged_pool = AliceDataPool(
            sector=None, batch_size=max(30, len(all_survivors)),
            seed=self.seed + 500)
        self._merged = MergedPopulation(
            all_survivors, self.evaluator, self._merged_pool,
            seed=self.seed + 500)

        records = []
        for gen in range(n_gens):
            rec = self._merged.step()
            if rec:
                records.append(rec)
                if verbose and (gen % 5 == 0 or gen == n_gens - 1):
                    print(f"  gen {gen:>3}  acc={rec['best_accuracy']:.3f}  "
                          f"fit={rec['best_fitness']:.4f}  "
                          f"species={rec['species_count']}")

        self._save_checkpoint("phase2")
        if verbose:
            print(f"\n  Merkle root (merged): {self._merged.archivist.root()}")
        return {"records": records}

    # ── Inference ─────────────────────────────────────────────────────────

    def apply_to_signal(self, sector: str, features: dict) -> dict:
        """
        Apply best evolved genome for sector to a live feature dict.
        Called from api_server.py to modulate conviction before output.
        """
        if sector not in self.subpops:
            return {"multiplier": 1.0, "confidence": 0.0, "trained": False}
        subpop = self.subpops[sector]
        if not subpop.history:
            return {"multiplier": 1.0, "confidence": 0.0, "trained": False}
        genome = subpop.best_genome()
        vec    = normalize_features(features)
        output = genome.activate(vec)
        sig    = float(output[0]) if len(output) else 0.0
        # Blend: conviction_multiplier ranges 0.5x–1.5x baseline conviction
        return {
            "multiplier":    float(np.clip(1.0 + sig * 0.5, 0.5, 1.5)),
            "raw_signal":    sig,
            "confidence":    float(abs(sig)),
            "trained":       True,
            "generations":   subpop.generation,
            "best_accuracy": subpop.history[-1]["best_accuracy"],
            "merkle":        subpop.archivist.root(),
        }

    # ── Persistence ───────────────────────────────────────────────────────

    def _save_checkpoint(self, phase: str):
        cp = {
            "phase":     phase,
            "saved_at":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "input_dim": INPUT_DIM,
            "sectors":   {},
        }
        for sec, subpop in self.subpops.items():
            best = subpop.best_genome()
            gid, nodes_s, conns_s = _serialize_genome(best)
            cp["sectors"][sec] = {
                "genome_id":    gid,
                "nodes":        nodes_s,
                "connections":  conns_s,
                "generation":   subpop.generation,
                "history_tail": subpop.history[-5:],
                "merkle_root":  subpop.archivist.root(),
            }
        path = NEAT_STATE_DIR / f"checkpoint_{phase}.json"
        with open(path, "w") as f:
            json.dump(cp, f, indent=2)
        print(f"  Checkpoint saved → {path}")

    def load_checkpoint(self, phase: str) -> bool:
        path = NEAT_STATE_DIR / f"checkpoint_{phase}.json"
        if not path.exists():
            return False
        with open(path) as f:
            cp = json.load(f)
        for sec, data in cp.get("sectors", {}).items():
            if sec in self.subpops:
                genome = _deserialize_genome(
                    data["genome_id"], data["nodes"], data["connections"])
                self.subpops[sec].population[0] = genome
                self.subpops[sec].generation    = data.get("generation", 0)
                self.subpops[sec].history       = data.get("history_tail", [])
        print(f"  Checkpoint loaded ← {path}")
        return True


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _cmd_check(args):
    engine = AliceNEATEngine()
    status = engine.dataset_status()
    print(f"\n{'═'*60}")
    print("  ALICE NEAT — TRAINING READINESS")
    print(f"{'═'*60}")
    for sec, s in status.items():
        bar  = "✓ READY" if s["ready"] else f"waiting ({s['labeled_samples']}/{MIN_TRAIN_SAMPLES})"
        print(f"  {sec:<10}  {s['labeled_samples']:>4} labeled  {bar}")
    total = sum(s["labeled_samples"] for s in status.values())
    print(f"\n  Total labeled samples : {total}")
    print(f"  Training threshold    : {MIN_TRAIN_SAMPLES} per sector")

    if NEAT_DATASET.exists():
        snapshots = sum(1 for line in open(NEAT_DATASET)
                        if line.strip() and "OUTCOME_LABEL" not in line)
        labels    = sum(1 for line in open(NEAT_DATASET)
                        if "OUTCOME_LABEL" in line)
        print(f"\n  Snapshots collected   : {snapshots}")
        print(f"  Outcome labels linked : {labels}")
        print(f"  First 7d outcomes due : 2026-06-13")

    if ARCHIVE_PATH.exists():
        events   = [json.loads(l) for l in open(ARCHIVE_PATH) if l.strip()]
        preds    = sum(1 for e in events if e.get("type") == "PREDICTION")
        outcomes = sum(1 for e in events if e.get("type") == "OUTCOME")
        print(f"\n  Archive predictions   : {preds}")
        print(f"  Archive outcomes      : {outcomes}")
    print(f"{'═'*60}\n")


def _cmd_train(args):
    engine = AliceNEATEngine(
        pop_size=args.pop_size,
        n_workers=args.workers,
    )
    status = engine.dataset_status()
    ready  = [s for s, d in status.items() if d["ready"]]

    if not ready:
        total = sum(d["labeled_samples"] for d in status.values())
        print(f"\n  Not enough labeled data yet ({total} total, "
              f"need {MIN_TRAIN_SAMPLES}+ per sector).")
        print("  First 21-day outcomes score around 2026-07-02.")
        print("  Run python3 neat_alice.py --check for current status.\n")
        return

    print(f"\n  Ready sectors: {ready}")
    print(f"  Pop size: {engine.pop_size}  Workers: {engine.evaluator.n_workers}")

    # Load prior checkpoint if it exists
    engine.load_checkpoint("phase1")

    engine.train_phase1(n_gens=args.p1)
    engine.train_phase2(n_gens=args.p2, survivors_each=5)
    print("\n  Evolution complete.\n")


def _cmd_apply(args):
    engine = AliceNEATEngine()
    loaded = engine.load_checkpoint("phase1")
    if not loaded:
        print(f"\n  No checkpoint found. Run --train first.\n")
        return
    sector = args.apply.upper()
    if sector not in SECTORS:
        print(f"  Unknown sector '{sector}'. Choose from: {SECTORS}")
        return
    try:
        with open(args.features_json) as f:
            features = json.load(f)
    except Exception as e:
        print(f"  Could not load features JSON: {e}")
        return
    result = engine.apply_to_signal(sector, features)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="ALICE NEAT Sector Specialist Engine")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("--check", help="Show training data status")

    tr = sub.add_parser("--train", help="Run evolution")
    tr.add_argument("--p1",       type=int, default=50,  help="Phase 1 generations")
    tr.add_argument("--p2",       type=int, default=25,  help="Phase 2 generations")
    tr.add_argument("--pop-size", type=int, default=20,  help="Population per sector")
    tr.add_argument("--workers",  type=int, default=None)

    ap = sub.add_parser("--apply", help="Score a live signal with best genome")
    ap.add_argument("sector",        help="Sector name (e.g. DEFENSE)")
    ap.add_argument("features_json", help="Path to feature dict JSON file")

    # Also support flat flags for backward compatibility
    parser.add_argument("--check",         action="store_true")
    parser.add_argument("--train",         action="store_true")
    parser.add_argument("--p1",            type=int, default=50)
    parser.add_argument("--p2",            type=int, default=25)
    parser.add_argument("--pop-size",      type=int, default=20)
    parser.add_argument("--workers",       type=int, default=None)
    parser.add_argument("--apply",         type=str, default=None)
    parser.add_argument("--features-json", type=str, default=None)

    args = parser.parse_args()

    if args.check:
        _cmd_check(args)
    elif args.train:
        _cmd_train(args)
    elif args.apply:
        _cmd_apply(args)
    else:
        _cmd_check(args)


if __name__ == "__main__":
    main()
