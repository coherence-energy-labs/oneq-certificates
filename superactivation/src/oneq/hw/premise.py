"""The STRUCTURAL PREMISE of a preregistered run: what the frozen circuits must look like for the mechanism the
preregistration tests to be present at all -- checked on the frozen bytes, not remembered.

Why it exists (R183, 2026-10-19). R182 projected +0.030 to +0.047 F_e from removing a 2.57 us key wait at R147's
feed-forward, derived from R152's frozen circuits, which carry NO barrier. R153 then added a schedule barrier to the
builder, and R183 froze its compiled arm fresh, so the key sat behind `barrier(5 q)`. The server held that barrier
until the feed-forward was received, and the key waited in |0> instead. The run tested a different design. Its drift
control (the compiled arm within 2 sd of R152) passed and could not see it. The preregistration pinned the estimator
and the decision rules, but not the structure its mechanism needed.

A premise maps a circuit-name PREFIX ("" selects every circuit) to metric ranges, inclusive:

    {"asc__": {"barriers": [0, 0], "modelled_conditional_wait_s": [2.42e-6, 2.72e-6]}}

Metrics (`fingerprint`):
  barriers, if_else, measures, mid_circuit_measures (a read followed by more work on its qubit), two_qubit;
  modelled_conditional_wait_s -- the largest conditioned-qubit wait at any if_test on the twirled ASAP timeline
    (oneq.hw.dynamic_timing.conditional_waits; 0 with no conditional). It is a MODEL: the executed schedule is read
    after the run (dynamic_timing.executed_key_wait), but a premise about the design must be checkable before.
    It is the ASAP BOUND, and the server does not play it: R194's premise held this at 2.572 us while the executed
    key waited 0.11 us (2026-10-20). Kept so existing preregistrations still read their own metric.
  alap_conditional_wait_s -- the same wait under the server's MEASURED as-late-as-possible rule
    (conditional_waits(policy="alap")): only work the condition's read depends on can make a conditioned qubit wait.
    It reproduces R194's 130 executed key waits to <= 0.27 us where ASAP missed by up to 2.53 us. A premise about
    a wait the device will play should state this one.

Refusals (`check` returns them by name; `preflight` adds the PREMISE guard):
  - a prefix that selects no circuit (a premise about nothing passes everything);
  - an unknown metric or a range that is not [lo, hi] with lo <= hi;
  - a circuit the model cannot time (its metric cannot be read, so it does not pass);
  - every circuit outside a range, counted, with the first few named.
The preregistration must carry `declaration(premise)` VERBATIM (`declared_in`), so the premise in force is the one a
reader of the preregistration sees.
"""
from __future__ import annotations

import json

METRICS = ("barriers", "if_else", "measures", "mid_circuit_measures", "two_qubit", "modelled_conditional_wait_s",
           "alap_conditional_wait_s")
_TWO_Q = {"cz", "cx", "ecr", "swap", "rzz", "cp", "crz"}
_PREFIX = "STRUCTURAL PREMISE: "


class PremiseRefused(ValueError):
    """a premise that cannot be evaluated faithfully"""


def fingerprint(qc) -> dict:
    """the premise metrics of one circuit (top level; a conditional's body counts as its if_else)"""
    from . import dynamic_timing as DT
    names = [inst.operation.name for inst in qc.data]
    last_use: dict = {}
    for i, inst in enumerate(qc.data):
        for q in inst.qubits:
            last_use[qc.find_bit(q).index] = i
    mid = sum(1 for i, inst in enumerate(qc.data) if inst.operation.name == "measure"
              and last_use[qc.find_bit(inst.qubits[0]).index] > i)
    out: dict = dict(barriers=names.count("barrier"), if_else=names.count("if_else"), measures=names.count("measure"),
                     mid_circuit_measures=mid, two_qubit=sum(1 for n in names if n in _TWO_Q))
    try:
        waits = DT.conditional_waits(qc, twirled=True)
        late = DT.conditional_waits(qc, twirled=True, policy="alap")
    except ValueError as e:
        out["modelled_conditional_wait_s"] = out["alap_conditional_wait_s"] = None
        out["unmodelled"] = str(e)
        return out
    out["modelled_conditional_wait_s"] = max((max(w.values(), default=0.0) for w in waits), default=0.0)
    out["alap_conditional_wait_s"] = max((max(w.values(), default=0.0) for w in late), default=0.0)
    return out


def _validate(premise: dict) -> None:
    if not isinstance(premise, dict) or not premise:
        raise PremiseRefused("a premise is a non-empty {prefix: {metric: [lo, hi]}}")
    for prefix, ranges in premise.items():
        if not isinstance(prefix, str) or not isinstance(ranges, dict) or not ranges:
            raise PremiseRefused(f"premise entry {prefix!r}: not a prefix with metric ranges")
        for m, r in ranges.items():
            if m not in METRICS:
                raise PremiseRefused(f"premise entry {prefix!r}: unknown metric {m!r} (one of {METRICS})")
            if not (isinstance(r, (list, tuple)) and len(r) == 2 and all(isinstance(v, (int, float)) for v in r)
                    and r[0] <= r[1]):
                raise PremiseRefused(f"premise entry {prefix!r}: {m} range {r!r} is not [lo, hi] with lo <= hi")


def declaration(premise: dict) -> str:
    """the one line a preregistration must carry, canonical (sorted keys, no spaces) so it can be matched exactly"""
    _validate(premise)
    return _PREFIX + json.dumps(premise, sort_keys=True, separators=(",", ":"))


def declared_in(text: str, premise: dict) -> bool:
    """does a line of the text, stripped of whitespace, a list dash and code backticks, equal the declaration"""
    want = declaration(premise)
    return any(line.strip().lstrip("-").strip().strip("`") == want for line in text.splitlines())


def check(circuits: dict, premise: dict, max_named: int = 5) -> list:
    """every way the circuits (name -> QuantumCircuit) violate the premise, as refusal strings (empty = it holds)"""
    _validate(premise)
    bad = []
    prints: dict = {}
    for prefix, ranges in premise.items():
        chosen = sorted(n for n in circuits if n.startswith(prefix))
        if not chosen:
            bad.append(f"prefix {prefix!r} selects no circuit: a premise about nothing")
            continue
        for metric, (lo, hi) in ranges.items():
            off, unread = [], []
            for n in chosen:
                fp = prints.get(n) or prints.setdefault(n, fingerprint(circuits[n]))
                v = fp.get(metric)
                if v is None:
                    unread.append(n)
                elif not lo <= v <= hi:
                    off.append((n, v))
            if unread:
                bad.append(f"{prefix!r} {metric}: {len(unread)} of {len(chosen)} circuits cannot be modelled "
                           f"({prints[unread[0]].get('unmodelled')}), e.g. {unread[:max_named]}")
            if off:
                bad.append(f"{prefix!r} {metric} outside [{lo}, {hi}] on {len(off)} of {len(chosen)} circuits, e.g. "
                           + ", ".join(f"{n}={v:.4g}" for n, v in off[:max_named]))
    return bad


__all__ = ["fingerprint", "check", "declaration", "declared_in", "METRICS", "PremiseRefused"]
