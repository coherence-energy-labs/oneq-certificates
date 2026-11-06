r"""Cross-check against Pan's public circuits (external audit 2, finding P3-09): are our projected mechanism sets the
ones Pan's own builders and noise produce?

    python -I -B experiments/circuit_distance_exact/pan_crosscheck.py \
        --pan-zenodo DIR/zenodo --pan-morph DIR/morph.py --out evidence/circuit_distance_exact/records/pan_crosscheck.jsonl

INPUTS ARE UNTRUSTED DOWNLOADS, pinned by sha256 below and refused otherwise:
  * `--pan-zenodo`: the extracted `zenodo/` directory of the release asset `zenodo.tar.gz` of
    github.com/GavinNull/gross-code-distance-12, tag v2.0 (asset sha256 797747b7...). Used: `code/gross12/` (the
    qec-schedule/v1 template `schedule.memory_circuit`, SD6 `noise.add_sd6`, the code `code.load_code`) and
    `data/circuits/{ibm_gross_depth8,fig12}.json` (the circuits of Pan's SM S7 and Corollary 1).
  * `--pan-morph`: `search/experiments/e017/morph.py` of the same repository at commit 5ae848ab (git blob
    2fd16e2e...), Pan's morphing builder; `Morph(code, hom).memory_circuit(basis, T, first_j)`.
Every one of those files was read in full before it was first run (2026-11-03); nothing here downloads anything.

WHAT IS COMPARED, per (circuit, basis, T), T = 1..3, both bases:
  * OURS: the projected model the certificates are about -- `bb_circuits.build` with the lane's noise placement,
    `export_model.mechanisms` + `project_same_type` (memory-type detectors only, distinct nonzero symptoms).
  * PAN'S: Pan's noiseless memory circuit for the same schedule, with Pan's SD6 (`add_sd6`), the same projection.
Detectors are matched by (type, support of the check, round) -- never by index -- so the two builders' numbering
does not matter (Pan's morph rounds are 1-based; ours 0-based). Observables are different bases of the same
logical space, so the comparison fits the GF(2) map  L_ours = A L_pan + K D  on the matched mechanisms, requires
A invertible, and then requires the mapped mechanism SET to EQUAL ours. Equal sets with an invertible A mean the
same undetectable-logical sets, hence the same distance problem, in every weight.

Exit 0 iff every row is equal. The rows (JSON lines, LF) and a header with every pin go to --out.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bb_circuits as bc  # noqa: E402
import export_model as em  # noqa: E402

PAN_RELEASE = dict(repository="github.com/GavinNull/gross-code-distance-12", tag="v2.0",
                   tag_commit="5ae848ab86d25e5235f7db4fe3c8df17f071fe60", published_at_utc="2026-10-24T00:09:59Z",
                   zenodo_asset_sha256="797747b7ecdc639353c9b093de4bd77b532022cf64277346dde936718148e8e9")
PINS_ZENODO = {
    "code/gross12/__init__.py": "fe641fd80860e68108414b891216679737d1ddcad0fbd735eb8e187b752a7c26",
    "code/gross12/paths.py": "91acc9803bded73743e2d7d7eebb0f08f15c70fdbdf11f119dcf640d03a1f521",
    "code/gross12/code.py": "2c3f55014a15d49d2cf87e457502a9cbcd3f3fc9c944e710419aca9d68ae18a9",
    "code/gross12/noise.py": "884eee8add1852d84aeba84e88f2010c7be519ec5f7aaf337beb6e7c363aba32",
    "code/gross12/schedule.py": "8c5e51fe093110bec7584f2d5118d767088fa6fda0ad3bdc897c458310ccb6f4",
    "data/circuits/fig12.json": "2378a04d6ed569e369222ae15188155716d76e3a332999368ca68827f72ec175",
    "data/circuits/ibm_gross_depth8.json": "d4d402f9c9d20d08ceaccd6636fb20cf9ea8894107eca14419c40544cac7f578",
    "data/code/gross_code.json": "5bbbb55fdaa9f373d02cb45e24f1463667c807fbf060a0b5591b09468b0db664",
    # the controls of the three-colour rows: the conflict-free retimings of Fig. 12 that keep its CNOT orders (SM S9)
    "data/circuits/fig12_retiming_0.json": "43eab8b97bd26dfa787bf41cb6a0a081d29587c247c900eb1090f98f028403ac",
    "data/circuits/fig12_retiming_1.json": "51c2ff96c7a5b3f6750bd5b0649a9175f6e5a7da352aeca0bcdba157bbab7db1",
    "data/circuits/fig12_retiming_2.json": "949c7f507e0154624a1fff29c0a44a22a4d8632fcae0be6033eb2860c665a97f",
    "data/circuits/fig12_retiming_3.json": "32252b725a2f39c9be1604a9836c461e4a53ac37565059f6a425f77221a676d4",
}
PIN_MORPH = "b5bc3eb410434464aac07630ce911d8f408d9e992b5e4a8e415450ee0a5fdbcb"


def sha256(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_pan(zenodo: pathlib.Path, morph_file: pathlib.Path):
    bad = [rel for rel, h in PINS_ZENODO.items() if sha256(zenodo / rel) != h]
    if bad or sha256(morph_file) != PIN_MORPH:
        raise SystemExit(f"REFUSED: Pan's files are not the pinned v2.0 bytes: {bad or [str(morph_file)]}")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(zenodo / "code"))
    from gross12 import code as pcode, noise as pnoise, schedule as psched  # noqa: E402
    spec = importlib.util.spec_from_file_location("pan_e017_morph", morph_file)
    pmorph = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pmorph          # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(pmorph)
    return pcode, pnoise, psched, pmorph


def projected(circuit, key_of) -> set:
    """{(frozenset of detector keys, observable mask)}: distinct nonzero memory-type symptoms (key_of -> None drops)."""
    dem = circuit.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
    coords = dem.get_detector_coordinates()
    keys = {d: key_of(tuple(int(round(x)) for x in v)) for d, v in coords.items()}
    out = set()
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        dets, mask = set(), 0
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                k = keys[t.val]
                if k is not None:
                    dets ^= {k}
            elif t.is_logical_observable_id():
                mask ^= 1 << t.val
        if dets or mask:
            out.add((frozenset(dets), mask))
    return out, {k for k in keys.values() if k is not None}


def solve_gf2(rows: list[int], rhs: list[int], ncols: int):
    """One x with <row, x> = rhs for every row (rows as int bitmasks), or None."""
    piv = {}  # pivot column -> (row mask, rhs)
    for r, b in zip(rows, rhs):
        for c, (pr, pb) in piv.items():
            if r >> c & 1:
                r ^= pr
                b ^= pb
        if r == 0:
            if b:
                return None
            continue
        c = r.bit_length() - 1
        for c2 in list(piv):
            pr, pb = piv[c2]
            if pr >> c & 1:
                piv[c2] = (pr ^ r, pb ^ b)
        piv[c] = (r, b)
    x = 0
    for c, (r, b) in piv.items():
        if b:
            x |= 1 << c
    return x


def rank_gf2(vs: list[int]) -> int:
    piv = {}
    for v in vs:
        while v:
            h = v.bit_length() - 1
            if h in piv:
                v ^= piv[h]
            else:
                piv[h] = v
                break
    return len(piv)


def compare(ours: set, pan: set, ours_dets: set, pan_dets: set, k: int) -> dict:
    row = dict(ours_mechanisms=len(ours), pan_mechanisms=len(pan), detectors_ours=len(ours_dets),
               detectors_pan=len(pan_dets), detector_sets_equal=ours_dets == pan_dets)
    d_ours = {}
    for D, L in ours:
        d_ours.setdefault(D, []).append(L)
    d_pan = {}
    for D, L in pan:
        d_pan.setdefault(D, []).append(L)
    row["detector_parts_equal"] = set(d_ours) == set(d_pan)
    if not (row["detector_sets_equal"] and row["detector_parts_equal"]):
        row.update(equal=False)
        return row
    det_index = {d: i for i, d in enumerate(sorted(ours_dets, key=repr))}
    unique = [D for D in d_ours if len(d_ours[D]) == 1 and len(d_pan[D]) == 1]
    row["unique_detector_parts"] = len(unique)
    feats = []
    for D in unique:
        f = d_pan[D][0]
        for d in D:
            f |= 1 << (k + det_index[d])
        feats.append(f)
    A_cols = []
    sol = []
    for b in range(k):
        x = solve_gf2(feats, [d_ours[D][0] >> b & 1 for D in unique], k + len(det_index))
        if x is None:
            row.update(equal=False, observable_map="none fits")
            return row
        sol.append(x)
        A_cols.append(x & ((1 << k) - 1))
    row["observable_map_rank"] = rank_gf2(A_cols)

    def mapped(D, L):
        f = L
        for d in D:
            f |= 1 << (k + det_index[d])
        return sum(((bin(f & x).count("1") & 1) << b) for b, x in enumerate(sol))

    pan_mapped = {(D, mapped(D, L)) for D, L in pan}
    row["equal"] = row["observable_map_rank"] == k and pan_mapped == ours
    row["only_ours"], row["only_pan"] = len(ours - pan_mapped), len(pan_mapped - ours)
    return row


def swapped_layers(psched, spec, zcode, basis: str, T: int, pan_key, base: set):
    """Negative control for the spec circuits: the same schedule with two CNOT time steps exchanged for every check
    (a CNOT-layer reordering), the first such near-miss that the template accepts (legal, deterministic) and whose
    projected set differs from Pan's real circuit's. The comparison must then report UNEQUAL, or it cannot fail.
    (Dropping one DEPOLARIZE2 line is NOT a control here: measured 2026-11-03, no single removal of the 14 / 19 such
    lines changes the projected set of Pan's IBM / Fig. 12 circuit at T = 2, because other faults produce every
    symptom it removes.)"""
    P = spec.period
    for t1 in range(P):
        for t2 in range(t1 + 1, P):
            sw = lambda t: t2 if t == t1 else t1 if t == t2 else t  # noqa: E731
            ck = lambda c: psched.CheckTiming(c.prep, c.meas, tuple(sorted((a, sw(t)) for a, t in c.cnots)))  # noqa: E731
            s2 = psched.Schedule(P, tuple(ck(c) for c in spec.x_checks), tuple(ck(c) for c in spec.z_checks))
            if psched.validate_schedule(s2, zcode):
                continue
            try:
                circ = pnoise_add(psched.memory_circuit(s2, zcode, basis, T))
                cset, cdets = projected(circ, pan_key)
            except Exception:  # noqa: BLE001 -- a non-deterministic near-miss is not a usable control
                continue
            if cset != base:
                return cset, cdets, f"every check's CNOT time steps {t1} and {t2} exchanged (a legal layer reordering)"
    raise SystemExit("no legal near-miss schedule found for the control")


pnoise_add = None  # set in run(): Pan's add_sd6


def check_supports_ours(kind: str):
    code = bc.BB(12)
    if kind == "ibm":
        return {(t, k): frozenset((code.ibm_x_target(k, s) if t == "X" else code.ibm_z_target(k, s))
                                  for s in (bc.SX if t == "X" else bc.SZ) if s != "idle")
                for t in "XZ" for k in range(code.n2)}
    return {(t, k): frozenset(code.geo_neighbours(t, k).values()) for t in "XZ" for k in range(code.n2)}


def run(pan, T_list, kinds) -> list[dict]:
    global pnoise_add
    pcode, pnoise, psched, pmorph = pan
    pnoise_add = pnoise.add_sd6
    zcode = pcode.load_code()
    rows = []
    for kind in kinds:
        if kind in ("ibm", "3c"):
            spec = psched.load_schedule(pcode.CODE_JSON.parents[1] / "circuits" /
                                        ("ibm_gross_depth8.json" if kind == "ibm" else "fig12.json"))
            sup_o = check_supports_ours(kind)
            sup_p = {("X", i): frozenset(a for a, _ in c.cnots) for i, c in enumerate(spec.x_checks)}
            sup_p.update({("Z", i): frozenset(a for a, _ in c.cnots) for i, c in enumerate(spec.z_checks)})
            k = zcode.LX.shape[0]
            for basis in "ZX":
                tcode = 1 if basis == "Z" else 0
                for T in T_list:
                    _, mech, coords = em.mechanisms(bc.build(kind, basis, T))
                    ours_key = lambda c: (basis, sup_o[(basis, c[0])], c[2]) if c[1] == tcode else None  # noqa: E731
                    ours, odets = projected(bc.build(kind, basis, T), ours_key)
                    pc = pnoise.add_sd6(psched.memory_circuit(spec, zcode, basis, T))
                    pan_key = lambda c: (basis, sup_p[(basis, c[0])], c[1]) if c[2] == tcode else None  # noqa: E731
                    pset, pdets = projected(pc, pan_key)
                    em_proj, _ = em.project_same_type(mech, coords, basis)
                    row = dict(circuit=kind, basis=basis, T=T, export_model_mechanisms=len(em_proj),
                               **compare(ours, pset, odets, pdets, k))
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    if T == 2:
                        if kind == "ibm":
                            cset, cdets, what = swapped_layers(psched, spec, zcode, basis, T, pan_key, pset)
                        else:
                            # no legal deterministic layer swap exists for the pipelined Fig. 12 (measured); its
                            # near-misses are Pan's own retimings that keep the CNOT orders: the first that differs
                            for r_ in range(4):
                                s2 = psched.load_schedule(pcode.CODE_JSON.parents[1] / "circuits" /
                                                          f"fig12_retiming_{r_}.json")
                                cset, cdets = projected(pnoise.add_sd6(psched.memory_circuit(s2, zcode, basis, T)),
                                                        pan_key)
                                if cset != pset:
                                    what = f"Pan's fig12_retiming_{r_}.json (same CNOT orders, other timing)"
                                    break
                            else:
                                raise SystemExit("every retiming gives Fig. 12's projected set: no control")
                        ctl = dict(kind="control", circuit=kind, basis=basis, T=T, control=what,
                                   **compare(ours, cset, odets, cdets, k))
                        ctl["fails_as_required"] = not ctl["equal"]
                        rows.append(ctl)
                        print(json.dumps(ctl), flush=True)
        else:
            l, m, a_terms, b_terms = bc.MORPH_CODES[kind]
            el = lambda t: (t[1], 0) if t[0] == "x" else (0, t[1])  # noqa: E731
            pbb = pmorph.BB(l, m, A=[el(t) for t in a_terms], B=[el(t) for t in b_terms])
            M = pmorph.Morph(pbb, "x")
            ocode = bc.BB(l, m, a_terms, b_terms)
            HX, HZ = ocode.check_matrices()
            import numpy as np
            sup_o = {("X", g): frozenset(np.flatnonzero(HX[g]).tolist()) for g in range(l * m)}
            sup_o.update({("Z", g): frozenset(np.flatnonzero(HZ[g]).tolist()) for g in range(l * m)})
            PX, PZ = M.HX, M.HZ
            sup_p = {("X", r): frozenset(np.flatnonzero(PX[r]).tolist()) for r in range(l * m)}
            sup_p.update({("Z", r): frozenset(np.flatnonzero(PZ[r]).tolist()) for r in range(l * m)})
            for basis in "ZX":
                tcode = 1 if basis == "Z" else 0
                for T in T_list:
                    c = bc.build(kind, basis, T)
                    _, mech, coords = em.mechanisms(c)
                    ours_key = lambda c_: (basis, sup_o[(basis, c_[0])], c_[2]) if c_[1] == tcode else None  # noqa: E731
                    ours, odets = projected(c, ours_key)
                    pc = pnoise.add_sd6(M.memory_circuit(basis, T, first_j=0))
                    pan_key = lambda c_: (basis, sup_p[(basis, c_[0])], c_[1] - 1) if c_[2] == tcode else None  # noqa: E731
                    pset, pdets = projected(pc, pan_key)
                    em_proj, _ = em.project_same_type(mech, coords, basis)
                    row = dict(circuit=kind, basis=basis, T=T, export_model_mechanisms=len(em_proj),
                               **compare(ours, pset, odets, pdets, M.k))
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    if T == 2:
                        # near-miss: Pan's own other layer order (swap23, his paper-order variant) on the same code
                        Mc = pmorph.Morph(pbb, "x", swap23=True)
                        cset, cdets = projected(pnoise.add_sd6(Mc.memory_circuit(basis, T, first_j=0)), pan_key)
                        ctl = dict(kind="control", circuit=kind, basis=basis, T=T,
                                   control="Pan's builder with swap23=True (layer order of 2407.16336 Table 2/4)",
                                   **compare(ours, cset, odets, cdets, M.k))
                        ctl["fails_as_required"] = not ctl["equal"]
                        rows.append(ctl)
                        print(json.dumps(ctl), flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pan-zenodo", type=pathlib.Path, required=True)
    ap.add_argument("--pan-morph", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--rounds", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--circuits", nargs="*", default=["ibm", "3c", "morph72", "morph144"])
    a = ap.parse_args()
    pan = load_pan(a.pan_zenodo.resolve(), a.pan_morph.resolve())
    rows = run(pan, a.rounds, a.circuits)
    head = dict(kind="header", pan_release=PAN_RELEASE, pan_zenodo_files=PINS_ZENODO,
                pan_morph_file=dict(path="search/experiments/e017/morph.py", git_blob="2fd16e2eaa96bbfb63498b83cb58aa8dad11438c",
                                    sha256=PIN_MORPH),
                ours=dict(bb_circuits=sha256(HERE / "bb_circuits.py"), export_model=sha256(HERE / "export_model.py"),
                          this_script=sha256(pathlib.Path(__file__))),
                noise=dict(ours="the lane's placement (bb_circuits; paper Sec. II C)", pan="Pan's SD6, gross12/noise.add_sd6"),
                rounds=a.rounds,
                all_equal=all(r["equal"] for r in rows if r.get("kind") != "control"),
                controls_all_fail=all(r["fails_as_required"] for r in rows if r.get("kind") == "control"))
    if a.out:
        text = "".join(json.dumps(r) + "\n" for r in [head] + rows)
        a.out.write_bytes(text.encode("utf-8"))      # LF, UTF-8: the newline policy of SERIALIZATION.md
    cmp_rows = [r for r in rows if r.get("kind") != "control"]
    ctl_rows = [r for r in rows if r.get("kind") == "control"]
    print(f"{sum(r['equal'] for r in cmp_rows)}/{len(cmp_rows)} rows equal; "
          f"{sum(r['fails_as_required'] for r in ctl_rows)}/{len(ctl_rows)} controls fail as required")
    return 0 if head["all_equal"] and head["controls_all_fail"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
