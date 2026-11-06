"""R180 | Heisenberg measurement compiler for the channel-faithful two-use code.

The CZ1/2 physical Bob decoder is a per-pair measurement of O_beta:
 O_0 = (Z_a I_s + I_a Z_s + X_a X_s + Y_a Y_s)/2
 O_1 = (Z_a I_s + X_a X_s)/sqrt(2).

Instead of implementing an entangling measurement, sample a local-Pauli
term with independent, declared probability q>0; score (coefficient/q)
times its product of measurement eigenvalues. This is unbiased for every
density operator, not merely the six code inputs or some fitted noise model.

For m retained pairs, execute the original CNOT+CH measurement. For a
shadowed pair remove that two-qubit decoder block and measure its chosen
local Pauli factors. For the X-basis logical input the decoded fidelity's
scoring functional does not depend on ANY pair parity; remove all Bob
pair gates without additional sampling.

CRITICAL CLAIMS CEILING:
  * Preserves an observable EXPECTATION, not the resulting quantum state.
  * Does not implement a physical decoder channel or prove superactivation.
  * The signed score can be negative or >2. It is NOT a Bernoulli sample.
  * Full P_d on both uses is unchanged before the Bob decoder, and neither
    channel-unitary randomizer is supplied to the estimator as hidden advice.
  * A different measurement basis is allowed between the six cardinal
    inputs. The physical P_d channel itself must stay identical, enforced by
    barriers and a QPY-level gate-segment verifier when compiling.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Callable

TERMS = {
    0: {"ZI": 0.5, "IZ": 0.5, "XX": 0.5, "YY": 0.5},
    1: {"ZI": 1 / sqrt(2), "XX": 1 / sqrt(2)},
}
MEASURED_PAULIS = frozenset("XYZ")
# Commuting Pauli grouping needs only local Z measurements and uses BOTH
# single-qubit Z expectations from the same readout. This decreases the
# second moment without increasing the worst-case estimator range.
GROUPED_BETA0 = {"ZZ": 1.0, "XX": 0.5, "YY": 0.5}


class ShadowRefused(ValueError):
    """Observable decomposition, protected channel or score contract was violated."""


@dataclass(frozen=True)
class PairSetting:
    index: int
    term: str
    probability: float


@dataclass(frozen=True)
class Recipe:
    m: int
    branch: str
    beta: int
    input_basis: str
    settings: tuple[PairSetting, ...]
    measurement_tags: tuple[str, ...]
    skipped_irrelevant: bool
    estimator_scale: float
    range_min: float
    range_max: float
    channel_use_count: int = 2
    evidence_class: str = (
        "OFFLINE OBSERVABLE ESTIMATION; no decoded quantum-state certification"
    )

    @property
    def shadow_indices(self) -> tuple[int, ...]:
        return tuple(s.index for s in self.settings)


def _validate_terms(m, branch, beta, inp, choices):
    if type(m) is not int or not 1 <= m <= 12:
        raise ShadowRefused("m must be an integer in 1..12")
    if branch not in ("kept_kept", "kept_erased", "erased_erased"):
        raise ShadowRefused("unknown erasure branch")
    if type(beta) is not int or beta not in (0, 1):
        raise ShadowRefused("Bob beta must be exactly 0 or 1")
    if inp[0] not in ("X", "Y", "Z") or inp[1] not in (-1, 1):
        raise ShadowRefused("unknown logical input eigenstate")
    if branch == "erased_erased" and choices:
        raise ShadowRefused("both-erased branch has no shield decoder")
    if branch != "erased_erased" and inp[0] == "X" and choices:
        raise ShadowRefused("X input does not need any shield observable")
    seen = set()
    for s in choices:
        if (
            not isinstance(s, PairSetting)
            or type(s.index) is not int
            or not 0 <= s.index < m
            or s.index in seen
        ):
            raise ShadowRefused("duplicate or out-of-bounds shadow pair")
        seen.add(s.index)
        if s.term not in TERMS[beta] and not (beta == 0 and s.term == "ZZ"):
            raise ShadowRefused("Pauli term/group does not exist for Bob beta")
        if type(s.probability) is not float or not 0 < s.probability <= 1:
            raise ShadowRefused("probability must be positive rationalizable float")
    # A plan exposes one setting per pair. To be unbiased, EVERY alternative
    # term must have nonzero sampling probability in the ensemble schedule,
    # checked by validate_distribution, never by an individual shot.
    return tuple(sorted(choices, key=lambda s: s.index))


def validate_distribution(beta: int, sampling: dict[str, float]) -> None:
    if type(beta) is not int or beta not in TERMS:
        raise ShadowRefused("invalid beta for Pauli sampling distribution")
    valid = (set(TERMS[beta]),)
    if beta == 0:
        valid += (set(GROUPED_BETA0),)
    if type(sampling) is not dict or set(sampling) not in valid:
        raise ShadowRefused("missing local Pauli support makes the estimator biased")
    if any(type(q) is not float or not 0 < q <= 1 for q in sampling.values()):
        raise ShadowRefused("sampling probabilities must be positive floats")
    if abs(sum(sampling.values()) - 1) > 1e-12:
        raise ShadowRefused("sampling probabilities must sum to one")


def canonical_distribution(beta: int, *, grouped: bool = False) -> dict[str, float]:
    """Minimax Pauli importance; ZZ grouping is never worse in second moment."""
    if type(beta) is not int or beta not in TERMS:
        raise ShadowRefused("invalid beta")
    if grouped and beta == 0:
        sampling = {"ZZ": 0.5, "XX": 0.25, "YY": 0.25}
    else:
        n = len(TERMS[beta])
        sampling = {term: float(1 / n) for term in TERMS[beta]}
    validate_distribution(beta, sampling)
    return sampling


def score_range(
    beta: int, settings: tuple[PairSetting, ...]
) -> tuple[float, float, float]:
    amp = 1.0
    for s in settings:
        term_abs = (
            GROUPED_BETA0[s.term]
            if beta == 0 and s.term == "ZZ"
            else TERMS[beta][s.term]
        )
        amp *= term_abs / s.probability
    # The parity enters 1 + K * amp, with K in [-1,+1].
    return amp, 1 - amp, 1 + amp


def _tags(
    m: int, beta: int, selected: tuple[PairSetting, ...], irrelevant: bool
) -> tuple[str, ...]:
    if irrelevant:
        return ("ell", "b1", "out")
    by_id = {s.index: s.term for s in selected}
    tags = ["ell"]
    for t in range(m):
        if t not in by_id:
            tags.append(f"pair_a{t}")
            if beta == 0:
                tags.append(f"pair_s{t}")
        else:
            if by_id[t][0] in MEASURED_PAULIS:
                tags.append(f"a{t}")
            if by_id[t][1] in MEASURED_PAULIS:
                tags.append(f"s{t}")
    tags.extend(("b1", "out"))
    return tuple(tags)


def _mark_decoder_operations(base, m: int, selected: set[int], beta: int):
    """Only remove the canonical already-optimized ONE-CZ Bob pair decoder.

    The incumbent decoder is CZ1, *not* the older CNOT+controlled-H decoder.
    Anchor its exact local rotation sequence on CZ(a_t,s_t) and refuse an
    altered source pattern; upstream P_d CZ(key,s) operations are protected.
    """
    import math

    skipped = set()
    found = set()
    for i, ins in enumerate(base.data):
        if ins.operation.name != "cz":
            continue
        qs = [base.find_bit(q).index for q in ins.qubits]
        choices = [t for t in range(m) if sorted(qs) == sorted((4 + t, 4 + m + t))]
        if not choices:
            continue
        t = choices[0]
        if t in found:
            raise ShadowRefused("duplicate per-pair CZ decoder found")
        found.add(t)
        s, a = 4 + t, 4 + m + t
        if beta == 0:
            pattern = [
                ("ry", (a,), 3 * math.pi / 2),
                ("rz", (s,), math.pi),
                ("ry", (s,), 3 * math.pi / 2),
                ("cz", (a, s), None),
                ("ry", (a,), 7 * math.pi / 4),
                ("ry", (s,), 3 * math.pi / 4),
            ]
        else:
            pattern = [
                ("rz", (a,), math.pi),
                ("ry", (s,), 3 * math.pi / 2),
                ("rz", (s,), math.pi),
                ("cz", (a, s), None),
                ("ry", (a,), math.pi / 4),
            ]
        start = i - 3
        if start < 0 or start + len(pattern) > len(base.data):
            raise ShadowRefused("truncated CZ1 per-pair measurement gadget")
        for j, (gate, targets, theta) in enumerate(pattern):
            op = base.data[start + j]
            got = tuple(base.find_bit(q).index for q in op.qubits)
            if (
                op.operation.name != gate
                or got != targets
                or (
                    theta is not None
                    and (
                        len(op.operation.params) != 1
                        or abs(float(op.operation.params[0]) - theta) > 1e-10
                    )
                )
            ):
                raise ShadowRefused("CZ1 pair contract changed; refusing to drop gates")
        if t in selected:
            skipped.update(range(start, start + len(pattern)))
    if found != set(range(m)):
        raise ShadowRefused("missing one-CZ canonical Bob decoder pair")
    return skipped


def _measure_pauli(body, qubit: int, basis: str):
    if basis == "X":
        body.h(qubit)
    elif basis == "Y":
        body.sdg(qubit)
        body.h(qubit)
    elif basis not in ("Z", "I"):
        raise ShadowRefused("unknown local Pauli measurement")


def build(
    m,
    unit,
    choices: tuple[PairSetting, ...] = (),
    *,
    base_builder: Callable | None = None,
):
    """Build one channel-faithful circuit and its independent signed score recipe.

    unit: hw_faithful.FUnit, with two actual P_d unitary choices. Supports
    erase='none' only (discarding erased outputs by final partial trace).
    """
    if base_builder is None:
        raise ShadowRefused("explicit canonical two-P channel builder required")
    selected = _validate_terms(m, unit.branch, unit.beta, unit.inp, choices)
    if len(unit.choices) != 2:
        raise ShadowRefused("both P_d physical randomizer choices must remain present")
    original = base_builder(m, unit, decoder="cz1", erase="none")
    if unit.branch == "erased_erased":
        return original, Recipe(
            m, unit.branch, unit.beta, unit.inp[0], (), ("ell", "b1"), True, 1, 0, 1
        )
    irrelevant = unit.inp[0] == "X"
    shadow = {s.index for s in selected}
    removed = set(range(m)) if irrelevant else shadow
    indices = _mark_decoder_operations(original, m, removed, unit.beta)
    from qiskit import ClassicalRegister, QuantumCircuit

    tags = _tags(m, unit.beta, selected, irrelevant)
    body = QuantumCircuit(*original.qregs, name=original.name + "_heisenberg")
    for i, ins in enumerate(original.data):
        if ins.operation.name == "measure" or i in indices:
            continue
        if ins.clbits:
            raise ShadowRefused(
                "unsupported classical feed-forward before final measurements"
            )
        body.append(ins.operation, ins.qubits, [])
    for s in selected:
        a, shield = 4 + m + s.index, 4 + s.index
        _measure_pauli(body, a, s.term[0])
        _measure_pauli(body, shield, s.term[1])
    register = ClassicalRegister(len(tags), "c")
    qc = QuantumCircuit(*body.qregs, register, name=original.name + "_heisenberg")
    qc.compose(body, inplace=True)
    for j, tag in enumerate(tags):
        if tag == "ell":
            wire = 2
        elif tag == "b1":
            wire = 3 if unit.branch == "kept_kept" else 1
        elif tag == "out":
            wire = 0
        elif tag.startswith("pair_a"):
            wire = 4 + m + int(tag.removeprefix("pair_a"))
        elif tag.startswith("pair_s"):
            wire = 4 + int(tag.removeprefix("pair_s"))
        elif tag.startswith("a"):
            wire = 4 + m + int(tag[1:])
        elif tag.startswith("s"):
            wire = 4 + int(tag[1:])
        else:
            raise ShadowRefused("unmapped measurement")
        qc.measure(wire, j)
    amp, vmin, vmax = score_range(unit.beta, selected)
    return qc, Recipe(
        m,
        unit.branch,
        unit.beta,
        unit.inp[0],
        selected,
        tags,
        irrelevant,
        amp,
        vmin,
        vmax,
    )


def score(unit, recipe: Recipe):
    """Unbiased estimator of SAME decoded success expectation for any input rho.

    Some values are negative or >2, so a Bernoulli confidence interval is
    INVALID. Caller must budget Pauli setting randomness separately from
    physical shot noise and P_d-unitary sampling.
    """
    if recipe.branch == "erased_erased":

        def erased(bits):
            if len(bits) != 2:
                raise ShadowRefused("both-erased readout width changed")
            if unit.inp[0] != "X":
                return 0.5
            return float((1 - 2 * (bits[0] ^ bits[1])) == unit.inp[1])

        return erased
    tags = recipe.measurement_tags
    selected = {s.index: s for s in recipe.settings}
    b0, sign = {"X": ("Z", 1), "Z": ("X", 1), "Y": ("Y", -1)}[unit.inp[0]]
    z_flip = b0 in "XY"
    x_flip = b0 in "ZY"

    def estimate(bits):
        if len(bits) != len(tags) or any(v not in (0, 1) for v in bits):
            raise ShadowRefused("outcome bit width or value mismatch")
        data = dict(zip(tags, bits))
        ell = data["ell"]
        if ell != unit.beta:
            return 0.0
        b1, out = data["b1"], data["out"]
        target = (
            (1 - 2 * out) * sign * unit.inp[1] * (-1 if x_flip and (ell ^ b1) else 1)
        )
        if not z_flip:
            return 1.0 + target
        par = 1.0
        for t in range(recipe.m):
            s = selected.get(t)
            if s is None:
                par *= 1 - 2 * data[f"pair_a{t}"]
                if recipe.beta == 0:
                    par *= 1 - 2 * data[f"pair_s{t}"]
                continue
            eigen = 1
            if s.term[0] in MEASURED_PAULIS:
                eigen *= 1 - 2 * data[f"a{t}"]
            if s.term[1] in MEASURED_PAULIS:
                eigen *= 1 - 2 * data[f"s{t}"]
            if recipe.beta == 0 and s.term == "ZZ":
                par *= ((1 - 2 * data[f"a{t}"]) + (1 - 2 * data[f"s{t}"])) / (
                    2 * s.probability
                )
            else:
                par *= TERMS[recipe.beta][s.term] * eigen / s.probability
        return 1.0 + target * par

    return estimate
