"""One pinned physical layout for every circuit of a run, chosen on the LIVE calibration by the same noise model the
engine predicts with -- graph-generic, so any protocol's circuits can use it.

Generalizes experiments/tw_teleport/tw2/layout.py (which enumerated 9-qubit paths for one fixed gate sequence):
  CANDIDATES  the level-3 transpiler's own choices over several seeds (Sabre + VF2 inside), plus every
              distinct initial layout they produce; any circuit family, any coupling graph.
  SCORE       the noise model's no-error probability, with readout, averaged over a representative set of the
              protocol's circuits AFTER routing on that layout:
                  proxy = mean_c [ ln P(no gate/idle error | c) + sum_measured ln(1 - (p01 + p10)/2) ]
              It is computed from noise.compile_ops -- the operations the engine simulates -- so a layout is ranked
              by the model that will predict it, including the routing SWAPs and every idle on the ALAP schedule.
  REFUSAL     a layout whose ROUTED circuits touch a qubit or an edge outside the policy is refused, with the
              reason (R138: one bad qubit in an otherwise good chain cost more than the whole idle budget).
The choice records the snapshot hash it was made on, so a frozen design names the calibration it was scored against.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .noise import NoiseParams, compile_ops
from .snapshot import DeviceSnapshot


class LayoutError(ValueError):
    """no candidate layout satisfies the policy"""


@dataclass(frozen=True)
class QubitPolicy:
    max_readout: float = 0.05           # (p01 + p10)/2 on a MEASURED qubit
    max_readout_any: float = 0.15       # p01 and p10 separately, on a measured qubit
    min_t1: float = 20e-6
    min_t2: float = 20e-6
    max_sx: float = 2e-3
    max_cz: float = 0.01                # vendor-reported error of any edge a routed circuit uses

    def describe(self) -> dict:
        return dict(self.__dict__)


@dataclass(frozen=True)
class LayoutScore:
    initial_layout: tuple
    proxy: float                        # mean over the representative circuits (higher is better)
    refusals: tuple                     # empty when the layout is admissible
    mean_cz: float
    physical_qubits: tuple              # every physical qubit any routed circuit touches


@dataclass(frozen=True)
class LayoutChoice:
    best: LayoutScore
    ranking: tuple = field(default=())
    n_candidates: int = 0
    snapshot_sha256: str = ""
    policy: dict = field(default_factory=dict)
    coupling_map: object = None             # the admissible piece every circuit of the run must be routed on

    @property
    def initial_layout(self) -> list:
        return list(self.best.initial_layout)


def refusals(snap: DeviceSnapshot, tqc, policy: QubitPolicy) -> list:
    """every reason this ROUTED circuit is outside the policy"""
    out = []
    measured = {tqc.find_bit(inst.qubits[0]).index for inst in tqc.data if inst.operation.name == "measure"}
    touched = {tqc.find_bit(x).index for inst in tqc.data for x in inst.qubits
               if inst.operation.name not in ("barrier", "delay")}
    for q in sorted(touched):
        if not snap.complete(q):
            out.append(f"q{q}: incomplete calibration")
            continue
        if snap.t1[q] < policy.min_t1:
            out.append(f"q{q}: T1 {snap.t1[q] * 1e6:.1f} us < {policy.min_t1 * 1e6:.0f} us")
        if snap.t2[q] < policy.min_t2:
            out.append(f"q{q}: T2 {snap.t2[q] * 1e6:.1f} us < {policy.min_t2 * 1e6:.0f} us")
        if snap.sx_error[q] > policy.max_sx:
            out.append(f"q{q}: sx error {snap.sx_error[q]:.2e} > {policy.max_sx:.0e}")
        if q in measured:
            if snap.readout_error(q) > policy.max_readout:
                out.append(f"q{q}: readout {snap.readout_error(q):.3f} > {policy.max_readout}")
            if max(snap.p01[q], snap.p10[q]) > policy.max_readout_any:
                out.append(f"q{q}: readout split ({snap.p01[q]:.3f}, {snap.p10[q]:.3f}) > {policy.max_readout_any}")
    for inst in tqc.data:
        if inst.operation.name == "cz":
            a, b = (tqc.find_bit(x).index for x in inst.qubits)
            err = snap.edge(a, b)[0]
            if err > policy.max_cz:
                out.append(f"edge {min(a, b)}-{max(a, b)}: CZ error {err:.4f} > {policy.max_cz}")
    return sorted(set(out))


def placement_refusals(snap: DeviceSnapshot, initial_layout, policy: QubitPolicy) -> list:
    """every physical qubit a logical qubit is PLACED on must pass the per-qubit policy, readout included: routing
    moves a logical qubit only some of the time, so a measured logical placed on a bad-readout qubit is measured
    there in some of a family's circuits. 2026-10-15: the representatives routed key0 off q72 (readout 0.063), the
    layout was admitted, and the HEALTHY guard then refused the frozen run -- some of its 208 circuits measured q72."""
    out = []
    for q in initial_layout:
        if not snap.complete(q):
            out.append(f"q{q}: incomplete calibration (placed)")
            continue
        if snap.readout_error(q) > policy.max_readout:
            out.append(f"q{q}: readout {snap.readout_error(q):.3f} > {policy.max_readout} (placed)")
        if max(snap.p01[q], snap.p10[q]) > policy.max_readout_any:
            out.append(f"q{q}: readout split ({snap.p01[q]:.3f}, {snap.p10[q]:.3f}) > {policy.max_readout_any} (placed)")
        if snap.t1[q] < policy.min_t1 or snap.t2[q] < policy.min_t2 or snap.sx_error[q] > policy.max_sx:
            out.append(f"q{q}: coherence or sx outside policy (placed)")
    return out


def proxy_log_fidelity(tqc, snap: DeviceSnapshot, params: NoiseParams) -> float:
    ops, active, meas = compile_ops(tqc, snap, params)
    lf = sum(math.log1p(-op.p_error) for op in ops if op.p_error > 0)
    if params.readout:
        lf += sum(math.log1p(-min(0.5, params.readout_factor * snap.readout_error(active[a]))) for a, _ in meas)
    return lf


def qubit_admissible(snap: DeviceSnapshot, q: int, policy: QubitPolicy) -> bool:
    """q passes every per-qubit criterion, readout included (as if it will be measured)"""
    return (snap.complete(q) and snap.t1[q] >= policy.min_t1 and snap.t2[q] >= policy.min_t2
            and snap.sx_error[q] <= policy.max_sx and snap.readout_error(q) <= policy.max_readout
            and max(snap.p01[q], snap.p10[q]) <= policy.max_readout_any)


def admissible_coupling(snap: DeviceSnapshot, policy: QubitPolicy):
    """the largest connected piece of the device whose every qubit and edge is inside the policy, as a CouplingMap.
    Routing on it cannot place, move or measure anything on a refused qubit or edge -- by construction, instead of
    hoping a transpiler that has never heard of the policy avoids them (2026-10-15: every one of 33 candidates on
    FakeKingston's calibration was refused once placements were checked)."""
    from qiskit.transpiler import CouplingMap
    ok = {q for q in range(snap.num_qubits) if qubit_admissible(snap, q, policy)}
    adj: dict = {q: set() for q in ok}
    for (a, b), (err, _) in snap.cz.items():
        if a in ok and b in ok and err <= policy.max_cz:
            adj[a].add(b)
            adj[b].add(a)
    best: set = set()
    seen: set = set()
    for q0 in sorted(ok):
        if q0 in seen:
            continue
        comp, stack = set(), [q0]
        while stack:
            q = stack.pop()
            if q in comp:
                continue
            comp.add(q)
            stack.extend(adj[q] - comp)
        seen |= comp
        if len(comp) > len(best):
            best = comp
    edges = [[a, b] for a in sorted(best) for b in sorted(adj[a]) if b in best]
    # CouplingMap keeps every index up to the largest as a node, so refused qubits survive as ISOLATED nodes the
    # router may park a non-interacting logical on (found 2026-10-15: an idle copy qubit landed on q146, incomplete
    # calibration). component_of reads the piece back; repair_placement moves such logicals inside it.
    return CouplingMap(edges)


def component_of(cm) -> frozenset:
    """the admissible piece of an admissible_coupling map: the qubits incident to an edge (every edge lies inside the
    piece, and every qubit of a piece of two or more has one). cm.size() is NOT it -- that is the largest index + 1."""
    return frozenset(q for e in cm.get_edges() for q in e)


def repair_placement(layout, cm, snap: DeviceSnapshot) -> tuple:
    """move every logical placed outside the admissible component onto the best free qubit inside it (lowest readout
    error). Only logicals with no two-qubit gate can be outside -- routing would have failed otherwise -- so routing is
    unaffected by the move."""
    comp = component_of(cm)
    used = {q for q in layout if q in comp}
    free = sorted((q for q in comp if q not in used), key=lambda q: (snap.readout_error(q), q))
    out = []
    for q in layout:
        if q in comp:
            out.append(q)
        else:
            out.append(free.pop(0))
    return tuple(out)


def place_and_route(qc, backend, snap: DeviceSnapshot, params: NoiseParams, policy: QubitPolicy, cm,
                    seeds: range = range(8)):
    """per-circuit compile: the level-3 transpiler on the admissible piece, each seed's placement repaired, the
    routed circuit kept only if it passes the policy, best of the seeds by the noise model's no-error proxy.
    Returns (isa circuit, proxy, seed) or raises LayoutError."""
    from qiskit import transpile
    n = qc.num_qubits
    best = None
    for s in seeds:
        t0 = transpile(qc, backend, optimization_level=3, seed_transpiler=s, coupling_map=cm)
        lay = repair_placement(tuple(t0.layout.initial_index_layout()[:n]), cm, snap)
        t = transpile(qc, backend, initial_layout=list(lay), optimization_level=3, seed_transpiler=s,
                      coupling_map=cm)
        if refusals(snap, t, policy):
            continue
        px = proxy_log_fidelity(t, snap, params)
        if best is None or px > best[1]:
            best = (t, px, s)
    if best is None:
        raise LayoutError(f"no seed of {len(seeds)} routes {qc.name} inside the policy")
    return best


def score_layout(circuits: list, backend, snap: DeviceSnapshot, params: NoiseParams, policy: QubitPolicy,
                 initial_layout, seed: int = 1, coupling_map=None) -> LayoutScore:
    from qiskit import transpile
    from qiskit.transpiler.exceptions import TranspilerError
    proxies: list[float] = []
    czs: list[int] = []
    reasons: set[str] = set(placement_refusals(snap, initial_layout, policy))
    phys: set[int] = set()
    for i, qc in enumerate(circuits):
        try:
            tqc = transpile(qc, backend, initial_layout=list(initial_layout), optimization_level=3,
                            seed_transpiler=seed + i, coupling_map=coupling_map)
        except TranspilerError as e:
            # a placement outside the admissible piece cannot be routed inside it: refused, by name, not crashed
            reasons.add(f"layout not routable inside the admissible coupling map ({str(e)[:80]})")
            proxies.append(-math.inf)
            czs.append(0)
            continue
        why = refusals(snap, tqc, policy)
        reasons.update(why)
        phys.update(tqc.find_bit(x).index for inst in tqc.data for x in inst.qubits
                    if inst.operation.name not in ("barrier", "delay"))
        czs.append(tqc.count_ops().get("cz", 0))
        # a refused layout is never scored: the noise model may not even be defined on it (a qubit with
        # incomplete calibration cannot be compiled), and its score could never make it admissible
        proxies.append(-math.inf if (why or reasons) else proxy_log_fidelity(tqc, snap, params))
    return LayoutScore(tuple(initial_layout), sum(proxies) / len(proxies), tuple(sorted(reasons)),
                       sum(czs) / len(czs), tuple(sorted(phys)))


def candidate_layouts(circuits: list, backend, seeds: range, coupling_map=None) -> list:
    """the distinct initial layouts the level-3 transpiler chooses for the representative circuits (on
    coupling_map when given: the admissible piece of the device)"""
    from qiskit import transpile
    seen, out = set(), []
    n = circuits[0].num_qubits
    for s in seeds:
        for qc in circuits:
            tqc = transpile(qc, backend, optimization_level=3, seed_transpiler=s, coupling_map=coupling_map)
            lay = tuple(tqc.layout.initial_index_layout()[:n])
            if lay not in seen:
                seen.add(lay)
                out.append(lay)
    return out


def choose_layout(circuits: list, backend, snap: DeviceSnapshot, params: NoiseParams = NoiseParams(),
                  policy: QubitPolicy = QubitPolicy(), seeds: range = range(1, 9), extra: list | None = None,
                  top_k: int = 5) -> LayoutChoice:
    """the admissible candidate with the highest proxy; LayoutError (with every candidate's reasons) if none"""
    if len({qc.num_qubits for qc in circuits}) != 1:
        raise LayoutError("every representative circuit must have the same number of logical qubits")
    cm = admissible_coupling(snap, policy)
    if len(component_of(cm)) < circuits[0].num_qubits:
        raise LayoutError(f"the admissible piece of {snap.name} has {len(component_of(cm))} qubits, fewer than the "
                          f"circuits' {circuits[0].num_qubits}")
    cands = [repair_placement(c, cm, snap) for c in candidate_layouts(circuits, backend, seeds, cm)]
    cands += [tuple(x) for x in (extra or [])]
    scored = [score_layout(circuits, backend, snap, params, policy, lay, coupling_map=cm)
              for lay in dict.fromkeys(cands)]
    ok = sorted((s for s in scored if not s.refusals), key=lambda s: (-s.proxy, s.initial_layout))
    if not ok:
        detail = "; ".join(f"{s.initial_layout}: {', '.join(s.refusals[:3])}" for s in scored[:5])
        raise LayoutError(f"all {len(scored)} candidate layouts are refused by the policy -- {detail}")
    return LayoutChoice(ok[0], tuple(ok[:top_k]), len(scored), snap.sha256(), policy.describe(), cm)
