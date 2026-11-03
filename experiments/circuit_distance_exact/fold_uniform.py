r"""fold_uniform.py (R198) -- the finite, T-independent hypotheses of FOLD_PROOF.md, checked mechanically, each with a
negative control that must FAIL it; then the template identification on chosen T and a cross-check against the
lane's recorded per-T fold rows.

    python fold_uniform.py                                        # morph72 + morph144, Z and X, hom x
    python fold_uniform.py --circuit morph144 --basis Z --template-T 1 2 3 4 --fresh-T 17

The theorem these hypotheses feed is FOLD_PROOF.md Theorem 1 (d_circ(T) = d_circ(1) for every T >= 1); the
archived runs are evidence/circuit_distance_exact/records/fold_uniform_hom_{x,y,xy}.jsonl (the JSON rows of a run).
One JSON object per (circuit, basis) on stdout, then a summary. Exit code 0 iff
every hypothesis holds, every negative control fails the condition it targets, every template check agrees and
every recorded fold row equals the prediction.

WHAT IS CHECKED (names as in FOLD_PROOF.md, section 2):
  H1   every CNOT layer of F_0, F_1 acts on pairwise distinct qubits (so each layer is an involution)
  H2   for each r, the right qubits are split exactly into M^X_r and M^Z_r (all measured, all reset)
  H3   every stabilizer generator (both types) and every memory logical, carried through F_r, has its right
       support inside the qubits measured in its own basis in round r (trackability)
  PAL  palindrome: for each r the mid-cycle images of the expansion noise of F_r^-1 equal those of the
       contraction noise of F_r, layer by layer, and the reset noise equals the measurement noise; with ROOT, the
       reason: every layer's noise set is invariant under that layer
  BND  boundary: prep images lie among the round-0 reset images, readout images among the measurement images of
       the same parity
  PURE every noise location that holds a Pauli A with nonzero memory part pi(A) also holds pi(A)
  COV, TWIN  the two consequences the theorem uses (implied by PAL + BND), checked directly on C_1, C_2
  INFO (not a hypothesis) every stabilizer contracts in exactly one round parity (detector windows <= 2 rounds)
The hypotheses are read off the schedule and the 1- and 2-round circuits (C_2 holds both round bodies).

TEMPLATE IDENTIFICATION (per chosen T; corroboration that bb_circuits.morphing_cycle builds the circuit the
theorem is about): the instruction skeleton equals the template's, the projected detectors and observables equal
the closed form, every other-type detector reads only other-basis records, every component's projected symptom
equals the theorem's exact formula (computed against the lane's independent backward-pass masks), the pure-part
lift holds on the actual masks, and the components' mid-cycle images lie in K(C_1).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

LANE = pathlib.Path(__file__).resolve().parent
RECORDS = LANE.parents[1] / "evidence" / "circuit_distance_exact" / "records" / "fold_lemma.jsonl"
sys.path.insert(0, str(LANE))

import numpy as np  # noqa: E402
import stim  # noqa: E402

import bb_circuits as bc  # noqa: E402
import dem_independent as di  # noqa: E402

BODY_PHASES = ("reset", "exp", "con", "meas")


# ----------------------------------------------------------------------------------------------- Pauli algebra
def bits(v: int):
    while v:
        b = v & -v
        yield b.bit_length() - 1
        v ^= b


def rowmask(row) -> int:
    return sum(1 << int(q) for q in np.nonzero(row)[0])


def apply_layer(layer, x: int, z: int) -> tuple[int, int]:
    """Conjugate X^x Z^z by a layer of CNOTs (control, target): X_c -> X_c X_t, Z_t -> Z_c Z_t."""
    for c, t in layer:
        if x >> c & 1:
            x ^= 1 << t
        if z >> t & 1:
            z ^= 1 << c
    return x, z


def apply_seq(seq, x: int, z: int) -> tuple[int, int]:
    for layer in seq:
        x, z = apply_layer(layer, x, z)
    return x, z


def to_xz(pauli: dict) -> tuple[int, int]:
    x = z = 0
    for q, p in pauli.items():
        if p in "XY":
            x |= 1 << q
        if p in "ZY":
            z |= 1 << q
    return x, z


# ----------------------------------------------------------------------------------------------- finite data D
def m2026_targets() -> list:
    """The standard-schedule targets of morph2026.py (2604.09797's Abelian-2BGA circuits): the theorem's template
    (data on the left block, every right qubit measured) applies to them as it is. The Def. 10 tree circuits put
    data on both halves, outside the template; they are not offered."""
    import morph2026 as M
    return sorted(k for k, t in M.TARGETS.items() if t["kind"] == "std")


def m2026_spec(kind: str, hom: str) -> dict:
    import morph2026 as M
    if hom != "x" or kind not in m2026_targets():
        raise ValueError(f"{kind}: not a standard-schedule morph2026 target with homomorphism x")
    return M.spec(kind)


class Data:
    """The finite data D of FOLD_PROOF.md section 1: schedule, measurement split, stabilizers, memory logicals."""

    def __init__(self, kind: str, basis: str, hom: str = "x", base: "Data | None" = None, gates=None, meas=None):
        self.kind, self.basis, self.hom = kind, basis, hom
        if kind in bc.MORPH_CODES:
            l, m, a_terms, b_terms = bc.MORPH_CODES[kind]
            g0, m0 = bc.morphing_schedule(l, m, a_terms, b_terms, hom)
            n = l * m
            code = None if base is not None else bc.BB(l, m, a_terms, b_terms)
        else:   # a standard-schedule morph2026 target (2604.09797): the same template, its own 2BGA code
            sp = m2026_spec(kind, hom)
            g0, m0, n = sp["gates"], sp["meas"], sp["code"].n2
            code = None if base is not None else sp["code"]
        self.gates = g0 if gates is None else gates
        self.meas = m0 if meas is None else meas
        self.n = n
        self.Lmask = (1 << n) - 1
        self.Rmask = self.Lmask << n
        if base is not None:
            self.stabs, self.logs = base.stabs, base.logs
        else:
            HX, HZ = code.check_matrices()
            self.stabs = {"X": [rowmask(r) for r in HX], "Z": [rowmask(r) for r in HZ]}
            LOG = code.logicals(HZ, HX) if basis == "Z" else code.logicals(HX, HZ)
            self.logs = [rowmask(v) for v in LOG]
        self.other = "X" if basis == "Z" else "Z"
        self.keep_type = 1 if basis == "Z" else 0
        self.mm = {t: [sum(1 << q for q in self.meas[t][r]) for r in (0, 1)] for t in ("X", "Z")}
        self._sig: dict = {}

    def carry(self, typ: str, supp: int, r: int) -> int:
        """Support of F_r(O) for the typ-type operator O on supp (contraction layers G_{r,0}, G_{r,1}, G_{r,2})."""
        if typ == "X":
            return apply_seq(self.gates[r], supp, 0)[0]
        return apply_seq(self.gates[r], 0, supp)[1]

    def split(self, typ: str, supp: int, r: int) -> tuple[int, int]:
        s = self.carry(typ, supp, r)
        return s & self.Rmask, s & self.Lmask

    def sigma(self, x: int, z: int) -> tuple[int, int]:
        """(syndrome bits over the memory-type generators, flip bits over the memory logicals) of X^x Z^z."""
        key = x if self.basis == "Z" else z
        hit = self._sig.get(key)
        if hit is None:
            sS = sum(1 << g for g, s in enumerate(self.stabs[self.basis]) if (key & s).bit_count() & 1)
            sL = sum(1 << k for k, s in enumerate(self.logs) if (key & s).bit_count() & 1)
            hit = self._sig[key] = (sS, sL)
        return hit

    def pure(self, x: int, z: int) -> tuple[int, int]:
        """pi(A): the part of A the memory-basis outcomes see (X part in memory-Z, Z part in memory-X)."""
        return (x, 0) if self.basis == "Z" else (0, z)


def image(D: Data, phase: str, rnd: int, layer, x: int, z: int, palindromic: bool = True) -> tuple[int, int]:
    """The mid-cycle image (FOLD_PROOF.md Lemma 3) of a noise Pauli at a tagged location of round `rnd`: forward through
    the rest of the expansion (prep, reset, exp), backward through the contraction so far (con, meas, readout).
    palindromic=False computes what a circuit whose expansion repeats F_prev's layers in forward order would give
    (negative control only)."""
    prev, cur = (rnd - 1) % 2, rnd % 2
    G = D.gates
    expansion = [G[prev][2], G[prev][1], G[prev][0]] if palindromic else [G[prev][0], G[prev][1], G[prev][2]]
    if phase in ("prep", "reset"):
        seq = expansion
    elif phase == "exp":
        seq = expansion[layer + 1:]
    elif phase == "con":
        seq = [G[cur][k] for k in range(layer, -1, -1)]
    elif phase in ("meas", "readout"):
        seq = [G[cur][2], G[cur][1], G[cur][0]]
    else:
        raise ValueError(phase)
    return apply_seq(seq, x, z)


# ----------------------------------------------------------------------------------------------- the circuits
def build(D: Data, T: int):
    tags: list = []
    if D.kind in bc.MORPH_CODES:
        c = bc.morphing_cycle(D.kind, D.basis, T, hom=D.hom, tags=tags)
    else:
        import morph2026 as M
        c = M.build(m2026_spec(D.kind, D.hom), D.basis, T, tags=tags)
    return c, tags


def components(D: Data, circuit, tags):
    """[(instruction index, phase, round, layer, x, z, mask)] with the lane's independent backward-pass masks."""
    comps, coords, nd, nobs = di.pauli_components(circuit)
    tag_of = {t[0]: t for t in tags}
    rows = []
    for idx, pauli, mask in comps:
        _, _, phase, rnd, layer = tag_of[idx]
        x, z = to_xz(pauli)
        rows.append((idx, phase, rnd, layer, x, z, mask))
    return rows, coords, nd, nobs


class Images:
    """Mid-cycle images of a component list, cached per (phase, round parity, layer, location Pauli)."""

    def __init__(self, D: Data, palindromic: bool = True):
        self.D, self.pal, self.cache = D, palindromic, {}

    def __call__(self, phase, rnd, layer, x, z):
        key = (phase, rnd % 2, layer, x, z)
        hit = self.cache.get(key)
        if hit is None:
            hit = self.cache[key] = image(self.D, phase, rnd, layer, x, z, self.pal)
        return hit


# ----------------------------------------------------------------------------------------------- hypotheses
def check_H1(D: Data) -> dict:
    bad = []
    for r in (0, 1):
        if len(D.gates[r]) != 3:
            bad.append((r, "layers", len(D.gates[r])))
        for k, layer in enumerate(D.gates[r]):
            used = [q for pair in layer for q in pair]
            if len(used) != len(set(used)) or any(not 0 <= q < 2 * D.n for q in used):
                bad.append((r, k))
    return dict(holds=not bad, violations=len(bad), first=bad[:3])


def check_H2(D: Data) -> dict:
    bad = []
    right = set(range(D.n, 2 * D.n))
    for r in (0, 1):
        X, Z = D.meas["X"][r], D.meas["Z"][r]
        if len(X) != len(set(X)) or len(Z) != len(set(Z)) or set(X) & set(Z) or set(X) | set(Z) != right:
            bad.append(r)
    return dict(holds=not bad, violations=len(bad), rounds=bad)


def check_H3(D: Data) -> dict:
    """Trackability of every operator the detectors and observables are built from, in both rounds."""
    bad = []
    ops = [(t, g, s) for t in ("X", "Z") for g, s in enumerate(D.stabs[t])]
    ops += [(D.basis, "L%d" % k, s) for k, s in enumerate(D.logs)]
    for typ, name, s in ops:
        for r in (0, 1):
            right, _ = D.split(typ, s, r)
            if right & ~D.mm[typ][r]:
                bad.append((typ, name, r))
    return dict(holds=not bad, violations=len(bad), operators=len(ops), first=bad[:3])


def contraction_info(D: Data) -> dict:
    """INFO, not a hypothesis: how many round parities each stabilizer contracts in (the builder asserts 1)."""
    counts = {}
    for typ in ("X", "Z"):
        hist = [0, 0, 0]
        for s in D.stabs[typ]:
            hist[sum(not D.split(typ, s, r)[1] for r in (0, 1))] += 1
        counts[typ] = hist
    return dict(holds=all(h[1] == D.n for h in counts.values()), contracting_in_0_1_2_parities=counts)


def body_sets(D: Data, rows, img: Images, rnd: int):
    """{(phase, layer): (set of location Paulis, set of images)} for the round `rnd` of a component list."""
    out: dict = {}
    for idx, phase, r, layer, x, z, mask in rows:
        if r != rnd:
            continue
        key = (phase, layer)
        loc, im = out.setdefault(key, (set(), set()))
        loc.add((x, z))
        im.add(img(phase, r, layer, x, z))
    return out


def check_PAL(D: Data, rows2, img: Images) -> dict:
    """On C_2 (round 0: prev 1 / cur 0; round 1: prev 0 / cur 1). For r: the round with prev = r supplies the
    expansion side, the round with cur = r the contraction side."""
    s = {t: body_sets(D, rows2, img, t) for t in (0, 1)}
    bad, root_bad = [], []
    for r in (0, 1):
        t_exp, t_con = (1 if r == 0 else 0), r
        for k in range(3):
            e = s[t_exp].get(("exp", 2 - k), (set(), set()))
            c = s[t_con].get(("con", k), (set(), set()))
            if e[1] != c[1]:
                bad.append(("layer", r, k))
            G = D.gates[r][k]
            for side, loc in (("con", c[0]), ("exp", e[0])):
                if {apply_layer(G, x, z) for x, z in loc} != loc:
                    root_bad.append((side, r, k))
        if s[t_exp].get(("reset", None), (set(), set()))[1] != s[t_con].get(("meas", None), (set(), set()))[1]:
            bad.append(("reset/meas", r))
    return dict(holds=not bad, violations=len(bad), first=bad[:3],
                root_layer_invariance=not root_bad, root_violations=len(root_bad))


def images_of(rows, img: Images, select=lambda phase, r: True) -> set:
    return {img(phase, r, layer, x, z) for idx, phase, r, layer, x, z, mask in rows if select(phase, r)}


def check_BND(D: Data, rows1, rows2, img: Images) -> dict:
    prep = images_of(rows2, img, lambda p, r: p == "prep")
    reset0 = images_of(rows2, img, lambda p, r: p == "reset" and r == 0)
    ro0 = images_of(rows1, img, lambda p, r: p == "readout")
    meas0 = images_of(rows1, img, lambda p, r: p == "meas" and r == 0)
    ro1 = images_of(rows2, img, lambda p, r: p == "readout")
    meas1 = images_of(rows2, img, lambda p, r: p == "meas" and r == 1)
    parts = dict(prep=prep <= reset0, readout_after_even=ro0 <= meas0, readout_after_odd=ro1 <= meas1)
    return dict(holds=all(parts.values()), **parts)


def check_PURE(D: Data, rows_list) -> dict:
    """Every location holding A with pi(A) != I also holds pi(A) (same instruction; within one noise instruction
    a qubit sits in one location, so (instruction, Pauli) names the location)."""
    bad = 0
    for rows in rows_list:
        present = {(idx, x, z) for idx, phase, r, layer, x, z, mask in rows}
        for idx, phase, r, layer, x, z, mask in rows:
            px, pz = D.pure(x, z)
            if (px or pz) and (idx, px, pz) not in present:
                bad += 1
    return dict(holds=bad == 0, violations=bad)


def check_COV(D: Data, rows_T, rows1, img_T: Images, img1: Images, level: str = "pauli") -> dict:
    """Every image of the T-round list lies among the images of C_1 (level 'pauli': as Paulis; 'sigma': as
    signatures, the weakest form the lower bound uses)."""
    k1 = images_of(rows1, img1)
    kT = images_of(rows_T, img_T)
    if level == "sigma":
        k1 = {D.sigma(*p) for p in k1}
        kT = {D.sigma(*p) for p in kT}
    miss = len(kT - k1)
    return dict(holds=miss == 0, outside=miss, size_T=len(kT), size_1=len(k1))


def check_TWIN(D: Data, rows1, rows2, img: Images, level: str = "pauli") -> dict:
    """Every image of C_1 occurs in round 0 of C_2 (prep or the round-0 body): the twins of the upper bound."""
    k1 = images_of(rows1, img)
    k0 = images_of(rows2, img, lambda p, r: r == 0 and p != "readout")
    if level == "sigma":
        k1 = {D.sigma(*p) for p in k1}
        k0 = {D.sigma(*p) for p in k0}
    miss = len(k1 - k0)
    return dict(holds=miss == 0, outside=miss)


# ----------------------------------------------------------------------------------------------- the template
def closed_form(D: Data, T: int):
    """The template's memory-type detectors {(g, coordinate round): records} and observables [records]; a record is
    (round, qubit), round T = the readout (FOLD_PROOF.md section 1.4)."""
    b = D.basis
    dets = {}
    for g, s in enumerate(D.stabs[b]):
        sp = [D.split(b, s, r) for r in (0, 1)]
        gamma = {r for r in (0, 1) if not sp[r][1]}
        cps = [t for t in range(T) if t % 2 in gamma]
        prev = -1
        for sj in cps:
            dets[(g, sj)] = frozenset((u, q) for u in range(prev + 1, sj + 1) for q in bits(sp[u % 2][0]))
            prev = sj
        if T - 1 not in cps:
            recs = {(u, q) for u in range(prev + 1, T) for q in bits(sp[u % 2][0])}
            recs |= {(T, q) for q in bits(sp[(T - 1) % 2][1])}
            dets[(g, T)] = frozenset(recs)
    obs = []
    for s in D.logs:
        sp = [D.split(b, s, r) for r in (0, 1)]
        recs = {(u, q) for u in range(T) for q in bits(sp[u % 2][0])} | {(T, q) for q in bits(sp[(T - 1) % 2][1])}
        obs.append(frozenset(recs))
    return dets, obs


def skeleton(D: Data, T: int) -> list:
    """The template's instruction stream (names and targets; noise probabilities and annotations ignored)."""
    n, b = D.n, D.basis
    left = list(range(n))
    flip = "X_ERROR" if b == "Z" else "Z_ERROR"
    out = [("R" if b == "Z" else "RX", left), (flip, left), ("TICK", [])]
    for t in range(T):
        prev, cur = (t - 1) % 2, t % 2
        out += [("RX", D.meas["X"][prev]), ("Z_ERROR", D.meas["X"][prev]), ("R", D.meas["Z"][prev]),
                ("X_ERROR", D.meas["Z"][prev]), ("DEPOLARIZE1", left), ("TICK", [])]
        for layer in [D.gates[prev][2], D.gates[prev][1], D.gates[prev][0]] + list(D.gates[cur]):
            pairs = [q for pair in layer for q in pair]
            out += [("CX", pairs), ("DEPOLARIZE2", pairs)]
            idle = sorted(set(range(2 * n)) - set(pairs))
            if idle:
                out.append(("DEPOLARIZE1", idle))
            out.append(("TICK", []))
        out += [("Z_ERROR", D.meas["X"][cur]), ("MX", D.meas["X"][cur]), ("X_ERROR", D.meas["Z"][cur]),
                ("M", D.meas["Z"][cur]), ("DEPOLARIZE1", left), ("TICK", [])]
    out += [(flip, left), ("M" if b == "Z" else "MX", left)]
    return out


def builder_view(D: Data, circuit, T: int):
    """(skeleton, projected detectors {(g, round): records}, other-type detectors' bases, observables)."""
    recs, skel, proj, other_bases, obs = [], [], {}, set(), {}
    ticks = 0
    for inst in circuit.flattened():
        name = inst.name
        tg = [t.value for t in inst.targets_copy()]
        if name == "DETECTOR":
            ids = [len(recs) + v for v in tg]
            co = tuple(int(round(v)) for v in inst.gate_args_copy())
            rs = frozenset((recs[i][0], recs[i][1]) for i in ids)
            if co[1] == D.keep_type:
                proj[(co[0], co[2])] = rs
            else:
                other_bases |= {recs[i][2] for i in ids}
            continue
        if name == "OBSERVABLE_INCLUDE":
            k = int(inst.gate_args_copy()[0])
            obs[k] = obs.get(k, frozenset()) ^ frozenset((recs[len(recs) + v][0], recs[len(recs) + v][1]) for v in tg)
            continue
        skel.append((name, tg))
        if name == "TICK":
            ticks += 1
        elif name in ("M", "MX"):
            rnd = (ticks - 1) // 8
            for q in tg:
                recs.append((rnd, q, "Z" if name == "M" else "X"))
    return skel, proj, other_bases, [obs.get(k, frozenset()) for k in range(len(D.logs))]


def template_check(D: Data, T: int, rows1, img1: Images, circuit=None, tags=None) -> dict:
    """Builder output for T rounds versus the template and the theorem's exact symptom formula."""
    t0 = time.time()
    if circuit is None:
        circuit, tags = build(D, T)
    skel, proj, other_bases, obs = builder_view(D, circuit, T)
    dets_cf, obs_cf = closed_form(D, T)
    skeleton_ok = skel == skeleton(D, T)
    detectors_ok = proj == dets_cf
    observables_ok = obs == obs_cf
    other_ok = other_bases <= {D.other}
    rows, coords, nd, nobs = components(D, circuit, tags)
    img = img1  # images depend on (phase, round parity, layer, Pauli) only, so one cache serves every T
    det_index = {(c[0], c[1], c[2]): d for d, c in coords.items()}
    pmask = sum(1 << d for d, c in coords.items() if c[1] == D.keep_type)
    obits = ((1 << nobs) - 1) << nd
    omask = ((1 << nd) - 1) & ~pmask
    gamma = []
    for s in D.stabs[D.basis]:
        sp = [D.split(D.basis, s, r) for r in (0, 1)]
        gamma.append({r for r in (0, 1) if not sp[r][1]})

    def window(g, t):
        if t % 2 in gamma[g]:
            return t
        if t + 1 <= T - 1 and (t + 1) % 2 in gamma[g]:
            return t + 1
        return T

    def iota(t, sS, sL):
        """The theorem's exact projected symptom of a fault with signature (sS, sL) at the mid-cycle point t."""
        m = sL << nd
        for g in bits(sS):
            m |= 1 << det_index[(g, D.keep_type, window(g, t))]
        return m

    mismatch = 0
    present = {}
    for idx, phase, r, layer, x, z, mask in rows:
        present[(idx, x, z)] = mask
    pure_bad = 0
    nz = 0
    k1 = images_of(rows1, img1)
    outside = 0
    actual_model = set()
    for idx, phase, r, layer, x, z, mask in rows:
        p = img(phase, r, layer, x, z)
        if p not in k1:
            outside += 1
        act = mask & (pmask | obits)
        if iota(r, *D.sigma(*p)) != act:
            mismatch += 1
        if act:
            nz += 1
            actual_model.add(act)
            px, pz = D.pure(x, z)
            m2 = present.get((idx, px, pz))
            if m2 is None or (m2 & (pmask | obits)) != act or m2 & omask:
                pure_bad += 1
    # Corollary 2 of FOLD_PROOF.md: the T-round projected model is the union over t of iota_t(the 1-round model)
    sig1 = {D.sigma(*p) for p in k1} - {(0, 0)}
    model = {iota(t, sS, sL) for t in range(T) for sS, sL in sig1}
    model_ok = model == actual_model
    return dict(T=T, skeleton=skeleton_ok, detectors_closed_form=detectors_ok, observables_closed_form=observables_ok,
                other_type_detectors_read_other_basis_only=other_ok, components=len(rows), nonzero=nz,
                symptom_formula_mismatches=mismatch, pure_lift_failures=pure_bad, images_outside_K1=outside,
                model_is_union_of_shifted_1round_models=model_ok, projected_mechanisms=len(actual_model),
                predicted_mechanisms=len(model), projected_detectors=len(dets_cf),
                agrees=bool(skeleton_ok and detectors_ok and observables_ok and other_ok and mismatch == 0
                            and pure_bad == 0 and outside == 0 and model_ok),
                seconds=round(time.time() - t0, 1))


# ----------------------------------------------------------------------------------------------- record cross-check
def count_model(D: Data, rows1, rows2, img: Images) -> dict:
    """Per-piece component counts and nonzero-signature counts (the theorem: a component has a nonzero projected
    symptom iff sigma(image) != 0), from C_1 and C_2."""
    def piece(rows, sel):
        n = nz = 0
        for idx, phase, r, layer, x, z, mask in rows:
            if sel(phase, r):
                n += 1
                nz += D.sigma(*img(phase, r, layer, x, z)) != (0, 0)
        return n, nz
    return dict(prep=piece(rows2, lambda p, r: p == "prep"),
                body={c: piece(rows2, lambda p, r, c=c: p in BODY_PHASES and r == c) for c in (0, 1)},
                body_C1=piece(rows1, lambda p, r: p in BODY_PHASES and r == 0),
                readout={0: piece(rows1, lambda p, r: p == "readout"), 1: piece(rows2, lambda p, r: p == "readout")})


def predict_row(cm: dict, T: int) -> dict:
    comps = cm["prep"][0] + sum(cm["body"][t % 2][0] for t in range(T)) + cm["readout"][(T - 1) % 2][0]
    mapped = cm["prep"][1] + sum(cm["body"][t % 2][1] for t in range(T)) + cm["readout"][(T - 1) % 2][1]
    return dict(components=comps, mapped=mapped, missing=0, conflicts=0, consistent=True, holds=True)


def model_size(D: Data, T: int, sig1: set) -> tuple[int, int]:
    """(projected mechanisms, projected detectors) of the T-round model predicted from the 1-round signatures alone
    (Corollary 2): mechanisms = |union over t of iota_t(M_1)|, detectors = checkpoints + finals per generator."""
    gamma = []
    for s in D.stabs[D.basis]:
        gamma.append({r for r in (0, 1) if not D.split(D.basis, s, r)[1]})

    def window(g, t):
        if t % 2 in gamma[g]:
            return t
        if t + 1 <= T - 1 and (t + 1) % 2 in gamma[g]:
            return t + 1
        return T
    model = {(frozenset((g, window(g, t)) for g in bits(sS)), sL) for t in range(T) for sS, sL in sig1}
    dets = sum(sum(t % 2 in gm for t in range(T)) + ((T - 1) % 2 not in gm) for gm in gamma)
    return len(model), dets


def model_records() -> list:
    """Every recorded projected morphing model's (circuit, basis, rounds, mechanisms, detectors), deduplicated."""
    seen = set()
    for f in sorted(RECORDS.parent.glob("morph*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.startswith("{"):
                continue
            r = json.loads(line)
            if r.get("projected") and "mechanisms" in r and r.get("circuit") in bc.MORPH_CODES:
                seen.add((r["circuit"], r["basis"], r["rounds"], r["mechanisms"], r["detectors"]))
    return sorted(seen)


def cross_check_models(D: Data, sig1: set, recs: list) -> dict:
    out, bad = [], 0
    for kind, basis, T, mech, dets in recs:
        if kind != D.kind or basis != D.basis:
            continue
        pm, pd = model_size(D, T, sig1)
        ok = (pm, pd) == (mech, dets)
        bad += not ok
        out.append(dict(T=T, recorded=[mech, dets], predicted=[pm, pd], agree=ok))
    return dict(rows=len(out), disagreements=bad, holds=bool(out) and bad == 0, detail=out)


def cross_check(kind: str, basis: str, cm: dict, rows: list) -> dict:
    out, bad = [], 0
    for row in rows:
        if row["circuit"] != kind or row["basis"] != basis:
            continue
        pred = predict_row(cm, row["T"])
        diff = {k: (row.get(k), v) for k, v in pred.items() if row.get(k) != v}
        bad += bool(diff)
        out.append(dict(T=row["T"], agree=not diff, diff=diff))
    return dict(rows=len(out), disagreements=bad, holds=bool(out) and bad == 0, detail=[o for o in out if not o["agree"]])


# ----------------------------------------------------------------------------------------------- controls
def doctored_circuit(circuit, edit):
    """A copy with instruction i replaced by edit(i, inst) (or kept); instruction boundaries are preserved."""
    flat = list(circuit.flattened())
    c2 = stim.Circuit()
    for i, inst in enumerate(flat):
        new = edit(i, inst)
        c2.append(new if new is not None else inst)
    flat2 = list(c2.flattened())
    assert len(flat2) == len(flat) and all(a.name == b.name for a, b in zip(flat, flat2)), "stim fused instructions"
    return c2


def controls(D: Data, rows1, rows2, img: Images, c2, tags2, rec_rows) -> dict:
    """Each entry: the condition evaluated on a doctored input; it must be False."""
    out = {}
    n = D.n
    # H1: a qubit used twice in one layer
    g = [[list(layer) for layer in D.gates[r]] for r in (0, 1)]
    g[0][0] = g[0][0] + [(g[0][0][0][0], g[0][0][1][1])]
    out["H1: a layer reusing a qubit"] = check_H1(Data(D.kind, D.basis, D.hom, base=D, gates=g))["holds"]
    # H2: one right qubit left unmeasured in round 0
    m2 = {t: [list(v) for v in D.meas[t]] for t in ("X", "Z")}
    m2["Z"][0] = m2["Z"][0][1:]
    out["H2: a right qubit never measured"] = check_H2(Data(D.kind, D.basis, D.hom, base=D, meas=m2))["holds"]
    # H3: move one qubit a memory-type stabilizer contracts onto into the other basis
    right, _ = D.split(D.basis, D.stabs[D.basis][0], 0)
    q = next(bits(right))
    m3 = {t: [list(v) for v in D.meas[t]] for t in ("X", "Z")}
    m3[D.basis][0].remove(q)
    m3[D.other][0].append(q)
    out["H3: a contracted qubit measured in the other basis"] = check_H3(Data(D.kind, D.basis, D.hom, base=D, meas=m3))["holds"]
    # PAL (images): a non-palindromic expansion (F_prev's layers in forward order)
    out["PAL: expansion not the reverse of the contraction"] = check_PAL(D, rows2, Images(D, palindromic=False))["holds"]
    # PAL root: the noise after one contraction layer keeps only X on each control (not invariant under the layer)
    controls_ = {c for c, t in D.gates[0][0]}

    def single_x_on_control(row):
        x, z = row[4], row[5]
        return z == 0 and x.bit_count() == 1 and x.bit_length() - 1 in controls_
    keep_x_on_control = [row for row in rows2
                         if not (row[1] == "con" and row[2] == 0 and row[3] == 0 and not single_x_on_control(row))]
    out["PAL root: one layer's noise not invariant under the layer"] = check_PAL(D, keep_x_on_control, img)["root_layer_invariance"]
    # BND / TWIN: the readout channel replaced by a correlated two-data-qubit flip
    corr = []
    for row in rows1:
        if row[1] == "readout":
            x, z = row[4], row[5]
            corr.append(row[:4] + (x | (x << 1 if x << 1 <= D.Lmask else 0), z | (z << 1 if z << 1 <= D.Lmask else 0)) + row[6:])
        else:
            corr.append(row)
    out["BND: correlated readout flips"] = check_BND(D, corr, rows2, img)["holds"]
    out["TWIN: correlated readout flips (Pauli level)"] = check_TWIN(D, corr, rows2, img)["holds"]
    out["TWIN: correlated readout flips (signature level)"] = check_TWIN(D, corr, rows2, img, "sigma")["holds"]
    # COV: the other homomorphism's 1-round table (Pauli level), and C_1 with its contraction noise removed (sigma)
    if D.kind in bc.MORPH_CODES:
        other_hom = "y" if D.hom != "y" else "x"
        Dy = Data(D.kind, D.basis, other_hom, base=D)
        c1y, t1y = build(Dy, 1)
        rows1y, *_ = components(Dy, c1y, t1y)
        out[f"COV: the hom-{other_hom} 1-round table (Pauli level)"] = check_COV(D, rows2, rows1y, img, Images(Dy))["holds"]
    else:
        # Z_N has one homomorphism onto Z_2, so the other schedule is the same code with its other CNOT order (the
        # non-special terms of A, else of B, reversed): its images differ. (Dropping C_1's reset noise is NOT a
        # control: PAL makes the measurement noise cover those images, so it cannot fail -- measured 2026-10-22.)
        import morph2026 as M
        code = m2026_spec(D.kind, D.hom)["code"]
        _, _, A, B = M.morphing_schedule(code, (1, 0))
        alt = (M.Code2BGA(code.l, code.m, A[:1] + A[1:][::-1], B) if len(A) > 2
               else M.Code2BGA(code.l, code.m, A, B[:1] + B[1:][::-1]))
        g_alt, m_alt, _, _ = M.morphing_schedule(alt, (1, 0))
        D_alt = Data(D.kind, D.basis, D.hom, base=D, gates=g_alt, meas=m_alt)
        t_alt: list = []
        c_alt = M.morphing_cycle_from(alt, g_alt, m_alt, D.basis, 1, tags=t_alt)
        rows_alt, *_ = components(D_alt, c_alt, t_alt)
        out["COV: the 1-round table of the code's other CNOT order (Pauli level)"] = check_COV(
            D, rows2, rows_alt, img, Images(D_alt))["holds"]
    no_con = [row for row in rows1 if row[1] != "con"]
    out["COV: 1-round table without its contraction noise (signature level)"] = check_COV(D, rows2, no_con, img, img, "sigma")["holds"]
    # PURE: one depolarizing location loses its X (memory-Z) / Z (memory-X) component
    target = next(row for row in rows2 if row[1] == "reset" and D.pure(row[4], row[5]) == (row[4], row[5])
                  and (row[4] or row[5]) and (row[4] | row[5]).bit_count() == 1 and (row[4] | row[5]) & D.Lmask)
    out["PURE: a location without its pure component"] = check_PURE(D, [[r for r in rows2 if r != target]])["holds"]
    # template: one projected detector with one record dropped; one CX pair removed from a contraction layer
    flat = list(c2.flattened())
    d_idx = next(i for i, inst in enumerate(flat) if inst.name == "DETECTOR"
                 and int(round(inst.gate_args_copy()[1])) == D.keep_type and len(inst.targets_copy()) > 1)

    def drop_record(i, inst):
        if i == d_idx:
            return stim.CircuitInstruction("DETECTOR", inst.targets_copy()[1:], inst.gate_args_copy())
        return None
    bad_det = doctored_circuit(c2, drop_record)
    tc = template_check(D, 2, rows1, img, bad_det, tags2)
    out["TEMPLATE: a detector missing one record (closed form)"] = tc["detectors_closed_form"]
    out["TEMPLATE: a detector missing one record (symptom formula)"] = tc["symptom_formula_mismatches"] == 0
    o_idx = next(i for i, inst in enumerate(flat) if inst.name == "OBSERVABLE_INCLUDE" and len(inst.targets_copy()) > 1)

    def drop_obs_record(i, inst):
        if i == o_idx:
            return stim.CircuitInstruction("OBSERVABLE_INCLUDE", inst.targets_copy()[1:], inst.gate_args_copy())
        return None
    out["TEMPLATE: an observable missing one record (closed form)"] = template_check(
        D, 2, rows1, img, doctored_circuit(c2, drop_obs_record), tags2)["observables_closed_form"]
    # a b'-type detector that also reads a b-basis record (the last readout record is in basis b)
    x_idx = next(i for i, inst in enumerate(flat) if inst.name == "DETECTOR"
                 and int(round(inst.gate_args_copy()[1])) != D.keep_type)
    meas_before = sum(len(inst.targets_copy()) for inst in flat[:x_idx] if inst.name in ("M", "MX"))
    b_rec = max(i for i, inst in enumerate(flat[:x_idx]) if inst.name == ("M" if D.basis == "Z" else "MX"))
    b_abs = sum(len(inst.targets_copy()) for inst in flat[:b_rec] if inst.name in ("M", "MX"))

    def add_b_record(i, inst):
        if i == x_idx:
            return stim.CircuitInstruction("DETECTOR", inst.targets_copy() + [stim.target_rec(b_abs - meas_before)],
                                           inst.gate_args_copy())
        return None
    out["TEMPLATE: an other-type detector reading a memory-basis record"] = template_check(
        D, 2, rows1, img, doctored_circuit(c2, add_b_record), tags2)["other_type_detectors_read_other_basis_only"]
    cx_seen = [i for i, inst in enumerate(flat) if inst.name == "CX"]
    cx_idx = cx_seen[3]  # round 0, first contraction layer

    def drop_pair(i, inst):
        if i == cx_idx:
            return stim.CircuitInstruction("CX", inst.targets_copy()[2:], inst.gate_args_copy())
        return None
    bad_cx = doctored_circuit(c2, drop_pair)
    tc = template_check(D, 2, rows1, img, bad_cx, tags2)
    out["TEMPLATE: one CNOT removed (symptom formula)"] = tc["symptom_formula_mismatches"] == 0
    out["TEMPLATE: one CNOT removed (skeleton)"] = tc["skeleton"]
    # INFO control: one CNOT removed from F_0's last layer changes which stabilizers contract
    g = [[list(layer) for layer in D.gates[r]] for r in (0, 1)]
    g[0][2] = g[0][2][1:]
    out["INFO: one CNOT removed from F_0"] = contraction_info(Data(D.kind, D.basis, D.hom, base=D, gates=g))["holds"]
    # records: a doctored row
    cm = count_model(D, rows1, rows2, img)
    mine = [r for r in rec_rows if r["circuit"] == D.kind and r["basis"] == D.basis]
    if mine:
        doctored = [dict(r) for r in mine]
        doctored[-1]["components"] += 1
        out["RECORDS: one recorded row with a wrong component count"] = cross_check(D.kind, D.basis, cm, doctored)["holds"]
    mrec = [r for r in model_records() if r[0] == D.kind and r[1] == D.basis]
    if mrec and D.hom == "x":
        sig1 = {D.sigma(*p) for p in images_of(rows1, img)} - {(0, 0)}
        doctored = mrec[:-1] + [mrec[-1][:3] + (mrec[-1][3] + 1, mrec[-1][4])]
        out["RECORDS: one recorded model size off by one"] = cross_check_models(D, sig1, doctored)["holds"]
    return out


# ----------------------------------------------------------------------------------------------- driver
def run(kind: str, basis: str, hom: str, template_T: list, fresh_T: list, rec_rows: list, do_controls: bool) -> dict:
    t0 = time.time()
    D = Data(kind, basis, hom)
    c1, t1 = build(D, 1)
    c2, t2 = build(D, 2)
    rows1, *_ = components(D, c1, t1)
    rows2, *_ = components(D, c2, t2)
    img = Images(D)
    hyp = dict(H1=check_H1(D), H2=check_H2(D), H3=check_H3(D), PAL=check_PAL(D, rows2, img),
               BND=check_BND(D, rows1, rows2, img), PURE=check_PURE(D, [rows1, rows2]))
    derived = dict(COV_pauli=check_COV(D, rows2, rows1, img, img), COV_sigma=check_COV(D, rows2, rows1, img, img, "sigma"),
                   TWIN_pauli=check_TWIN(D, rows1, rows2, img), TWIN_sigma=check_TWIN(D, rows1, rows2, img, "sigma"))
    info = contraction_info(D)
    tmpl = [template_check(D, T, rows1, img) for T in template_T]
    fresh = [template_check(D, T, rows1, img) for T in fresh_T]
    cm = count_model(D, rows1, rows2, img)
    lane = kind in bc.MORPH_CODES   # the archived fold rows and exported models exist for the lane's codes only
    rec = cross_check(kind, basis, cm, rec_rows) if hom == "x" and lane else dict(rows=0, holds=None)
    sig1 = {D.sigma(*p) for p in images_of(rows1, img)} - {(0, 0)}
    mrec = cross_check_models(D, sig1, model_records()) if hom == "x" and lane else dict(rows=0, holds=None)
    ctl = controls(D, rows1, rows2, img, c2, t2, rec_rows if hom == "x" else []) if do_controls else {}
    return dict(circuit=kind, basis=basis, hom=hom, n=D.n, hypotheses=hyp, derived=derived, info=info,
                counts=dict(prep=cm["prep"], body_even=cm["body"][0], body_odd=cm["body"][1],
                            body_even_in_C1=cm["body_C1"], readout_after_even=cm["readout"][0],
                            readout_after_odd=cm["readout"][1], one_round_signatures=len(sig1)),
                template=tmpl, fresh=fresh, records=rec, records_models=mrec, controls=ctl,
                seconds=round(time.time() - t0, 1))


def verdict(res: dict) -> bool:
    ok = all(v["holds"] for v in res["hypotheses"].values()) and res["hypotheses"]["PAL"]["root_layer_invariance"]
    ok &= all(v["holds"] for v in res["derived"].values())
    ok &= all(t["agrees"] for t in res["template"] + res["fresh"])
    ok &= res["counts"]["body_even"] == res["counts"]["body_even_in_C1"]
    for key in ("records", "records_models"):
        if res[key].get("holds") is not None:
            ok &= res[key]["holds"]
    ok &= not any(res["controls"].values())
    return bool(ok)


def summary(res: dict) -> str:
    lines = [f"== {res['circuit']} memory-{res['basis']} (hom {res['hom']}, n = {res['n']}) -- {res['seconds']} s"]
    for k, v in res["hypotheses"].items():
        extra = f" (root: layer invariance {'holds' if v['root_layer_invariance'] else 'FAILS'})" if k == "PAL" else ""
        lines.append(f"  {k:<5} {'PASS' if v['holds'] else 'FAIL'}{extra}")
    for k, v in res["derived"].items():
        lines.append(f"  {k:<10} {'PASS' if v['holds'] else 'FAIL'}")
    lines.append(f"  INFO  exactly one contraction parity per stabilizer: {res['info']['holds']} "
                 f"{res['info']['contracting_in_0_1_2_parities']}")
    for t in res["template"] + res["fresh"]:
        lines.append(f"  template T={t['T']:<3} {'AGREES' if t['agrees'] else 'DISAGREES'}: skeleton {t['skeleton']}, "
                     f"detectors {t['detectors_closed_form']}, observables {t['observables_closed_form']}, "
                     f"symptom mismatches {t['symptom_formula_mismatches']}/{t['components']}, pure-lift failures "
                     f"{t['pure_lift_failures']}, images outside K1 {t['images_outside_K1']}, model = union of "
                     f"shifted 1-round models {t['model_is_union_of_shifted_1round_models']} "
                     f"({t['projected_mechanisms']} mechanisms) ({t['seconds']} s)")
    r = res["records"]
    if r.get("holds") is not None:
        lines.append(f"  records: {r['rows']} fold rows, {r['disagreements']} disagree with the prediction")
    r = res["records_models"]
    if r.get("holds") is not None:
        sizes = ", ".join(f"T={d['T']}: {d['recorded'][0]}/{d['recorded'][1]}" for d in r["detail"])
        lines.append(f"  recorded projected models (mechanisms/detectors): {r['rows']} rows, {r['disagreements']} "
                     f"disagree with the 1-round prediction [{sizes}]")
    for k, v in res["controls"].items():
        lines.append(f"  control {'fails as required' if not v else 'DID NOT FAIL'}: {k}")
    lines.append(f"  VERDICT {'PASS' if verdict(res) else 'FAIL'}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", choices=sorted(bc.MORPH_CODES) + m2026_targets(), nargs="*",
                    default=sorted(bc.MORPH_CODES))
    ap.add_argument("--basis", choices=("Z", "X"), nargs="*", default=["Z", "X"])
    ap.add_argument("--hom", choices=sorted(bc.MORPH_HOMS), default="x")
    ap.add_argument("--template-T", type=int, nargs="*", default=[1, 2, 3, 4])
    ap.add_argument("--fresh-T", type=int, nargs="*", default=[])
    ap.add_argument("--no-controls", action="store_true")
    a = ap.parse_args()
    rec_rows = [json.loads(x) for x in RECORDS.read_text().splitlines() if x.strip()]
    results = []
    for kind in a.circuit:
        for basis in a.basis:
            res = run(kind, basis, a.hom, a.template_T, a.fresh_T, rec_rows, not a.no_controls)
            print(json.dumps(res), flush=True)
            results.append(res)
    print()
    for res in results:
        print(summary(res))
    ok = all(verdict(r) for r in results)
    print(f"\nOVERALL {'PASS' if ok else 'FAIL'}: {len(results)} (circuit, basis) runs")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
