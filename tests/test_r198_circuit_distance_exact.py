r"""R198: exact circuit distance with checkable certificates (experiments/circuit_distance_exact).

What is held here, each both ways:
  * the IBM cycle IS IBM's: term -> qubit maps re-derived from IBM's own matrix construction (np.roll shift
    matrices, rows of A / B for X checks, columns for Z checks), schedules sX / sZ verbatim;
  * the 3-colour cycle reproduces the six residual patterns of 2603.05481 Fig. 11, and stim accepts every
    detector in both bases (a mislabelled edge breaks commutation and stim refuses the DEM);
  * the landed exporter reproduces BYTE-IDENTICAL the models the recorded searches and proofs ran on;
  * the projection maps a real full-DEM solution to a projected solution (it is a lower bound);
  * the engine settles [[72,12,6]] (NONE <= 5, weight 6 found and lifted); an independent CDCL program agrees;
  * the prover's certificate VERIFIES, and every rejection path of the checker FIRES: a raised weight claim, a
    flipped bit, truncation, trailing bytes, a fake bound prune, a lying prover that ignores solutions, a model
    that is not invariant, orbit ids that disagree;
  * a real gross-code certificate (IBM cycle, T = 2, weight <= 7) verifies inside the suite;
  * the window lemma's preconditions hold on consecutive models and FAIL on a doctored one;
  * collect_results REFUSES missing records and reproduces the committed results.json;
  * the fault set is derived a second time WITHOUT stim's error analysis (dem_independent.py) and equals the
    exporter's; a mutated circuit copy (one noise line dropped, one CX reversed) breaks the equality, and an
    instruction the derivation does not handle is refused;
  * the morphing circuits of 2407.16336 have the paper's structure (degree-5 graph from its formula, weight-9
    end-cycle checks, k = 12) for both codes and every homomorphism; the 72-qubit one is settled (= 6);
  * the detectors span every deterministic parity but the 6 redundancy meta-checks, and a witness is tested
    against those too: a toy witness that flips one does NOT survive, the recorded ones do.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import struct
import sys

import numpy as np
import pytest
import stim

ROOT = pathlib.Path(__file__).resolve().parents[1]
LANE = ROOT / "experiments" / "circuit_distance_exact"
sys.path.insert(0, str(LANE))
import bb_circuits as bc  # noqa: E402
import collect_results as cr  # noqa: E402
import dem_independent as di  # noqa: E402
import export_model as em  # noqa: E402
import tools_cde as tc  # noqa: E402

RECORDS = ROOT / "evidence" / "circuit_distance_exact" / "records"
needs_gcc = pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not available")


@pytest.fixture(scope="module")
def exe(tmp_path_factory):
    if shutil.which("gcc") is None:
        pytest.skip("gcc not available")
    return tc.build_c(tmp_path_factory.mktemp("r198_bin"), liar=True)


@pytest.fixture(scope="module")
def bb72(tmp_path_factory):
    p = tmp_path_factory.mktemp("r198_m") / "bb72_Z2.txt"
    em.export("ibm", "Z", 2, p, l=6)
    return p


# ---------------------------------------------------------------- the circuits are the published ones

def test_ibm_term_maps_rederived_from_ibm_matrix_construction():
    code = bc.BB(12)
    l, m = 12, 6
    S_l, S_m = np.roll(np.eye(l, dtype=int), 1, axis=1), np.roll(np.eye(m, dtype=int), 1, axis=1)
    x = {i: np.kron(np.linalg.matrix_power(S_l, i), np.eye(m, dtype=int)) for i in range(l)}
    y = {i: np.kron(np.eye(l, dtype=int), np.linalg.matrix_power(S_m, i)) for i in range(m)}
    A = [x[3], y[1], y[2]]
    B = [y[3], x[1], x[2]]
    n2 = l * m
    for i in range(n2):
        for k in range(3):
            assert code.ibm_x_target(i, k) == np.nonzero(A[k][i, :])[0][0]
            assert code.ibm_x_target(i, k + 3) == n2 + np.nonzero(B[k][i, :])[0][0]
            assert code.ibm_z_target(i, k) == np.nonzero(B[k][:, i])[0][0]
            assert code.ibm_z_target(i, k + 3) == n2 + np.nonzero(A[k][:, i])[0][0]
    assert bc.SX == ("idle", 1, 4, 3, 5, 0, 2) and bc.SZ == (3, 5, 0, 1, 2, 4, "idle")


# Fig. 11 of 2603.05481, transcribed as direction names (a residual is equivalent to its complement in the check)
FIG11 = {
    ("Z", 2): [{"ll", "up"}, {"ur", "right"}, {"ll", "up", "down"}],
    ("Z", 0): [{"down", "ll"}, {"right", "left"}, {"down", "ll", "up"}],
    ("Z", 1): [{"down", "ll"}, {"ur", "right"}, {"down", "ll", "up"}],
    ("X", 2): [{"ll", "left"}, {"ur", "down"}, {"up", "ur", "down"}],
    ("X", 0): [{"ll", "left"}, {"up", "ur"}, {"down", "up", "ur"}],
    ("X", 1): [{"left", "right"}, {"up", "ur"}, {"down", "up", "ur"}],
}


@pytest.mark.parametrize("key", sorted(FIG11))
def test_three_colour_residuals_match_fig11(key):
    allx = set(bc.DIRECTIONS)
    got = {frozenset(min(r, frozenset(allx - r), key=lambda s: (len(s), sorted(s)))) for r in bc.residual_sets(*key)}
    want = {frozenset(min(frozenset(r), frozenset(allx - r), key=lambda s: (len(s), sorted(s)))) for r in FIG11[key]}
    assert got == want


def test_a_wrong_schedule_does_not_match_fig11():
    """The residual comparison can fail: swap two time steps of one check and it must."""
    saved = dict(bc.SCHEDULE_3C[("Z", 2)])
    try:
        bc.SCHEDULE_3C[("Z", 2)]["up"], bc.SCHEDULE_3C[("Z", 2)]["right"] = saved["right"], saved["up"]
        allx = set(bc.DIRECTIONS)
        got = {frozenset(min(r, frozenset(allx - r), key=lambda s: (len(s), sorted(s)))) for r in bc.residual_sets("Z", 2)}
        want = {frozenset(min(frozenset(r), frozenset(allx - r), key=lambda s: (len(s), sorted(s)))) for r in FIG11[("Z", 2)]}
        assert got != want
    finally:
        bc.SCHEDULE_3C[("Z", 2)].update(saved)


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_three_colour_cycle_measures_the_stabilizers(basis):
    dem = bc.three_colour_cycle(basis, 2).detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
    assert dem.num_detectors == 288 and dem.num_observables == 12


def test_landed_exporter_reproduces_the_recorded_models(tmp_path):
    proofs = [json.loads(x) for x in (RECORDS / "ibm_proofs_w9.jsonl").read_text().splitlines() if x.strip()]
    rec_ibm = next(r["dem_sha256"] for r in proofs if r["basis"] == "Z" and r["T"] == 2)
    em.export("ibm", "Z", 2, tmp_path / "ibm.txt", l=12)
    assert run_sha(tmp_path / "ibm.txt") == rec_ibm
    for r in (json.loads(x) for x in (RECORDS / "3c_models.jsonl").read_text().splitlines() if x.strip()):
        em.export("3c", r["basis"], 2, tmp_path / f"3c{r['basis']}.txt", l=12)
        assert run_sha(tmp_path / f"3c{r['basis']}.txt") == r["model_sha256"]


def run_sha(p):
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------- stim's error analysis out of the trusted base

@pytest.mark.parametrize("kind,l,basis,T", [("ibm", 6, "Z", 2), ("ibm", 6, "X", 2), ("3c", 12, "X", 2),
                                            ("morph72", 12, "Z", 3), ("morph144", 12, "X", 2)])
def test_independent_dem_equals_the_exporters_mechanisms(kind, l, basis, T):
    got = di.compare(bc.build(kind, basis, T, l))
    assert got["equal"] and got["mechanisms"] == got["stim_mechanisms"] > 0


def _mutate(circuit, prefix, change, at="first"):
    lines = str(circuit).splitlines()
    hits = [i for i, s in enumerate(lines) if s.startswith(prefix)]
    k = hits[0] if at == "first" else hits[len(hits) // 2]
    lines[k:k + 1] = change(lines[k])
    return stim.Circuit("\n".join(lines))


def _reverse_first_cx_pair(line):
    head, *tg = line.split()
    return [" ".join([head, tg[1], tg[0], *tg[2:]])]


@pytest.mark.parametrize("prefix,change,at", [("DEPOLARIZE2", lambda s: [], "first"),
                                              ("CX", _reverse_first_cx_pair, "middle")],
                         ids=["noise_line_dropped", "cx_reversed"])
def test_independent_dem_disagrees_with_a_mutated_circuit(prefix, change, at):
    """Run on a mutated copy, the derivation must DISAGREE with stim on the original: the equality can fail.

    The positions are measured, not arbitrary: the fault SET is redundant in most places ([[72,12,6]], memory-Z,
    T = 2). Dropping any DEPOLARIZE1 or X_ERROR line removes no mechanism (two-qubit noise already produces every
    symptom), and dropping a DEPOLARIZE2 line removes 36, 0, 0, 360, 360, 180, 0, 0, 180, 360, 360, 0, 180, 0
    mechanisms by layer; reversing the first CX layer moves none (it follows the resets), the middle layer's first
    pair moves 91."""
    c = bc.build("ibm", "Z", 2, 6)
    mine, _, _ = di.pauli_dem(_mutate(c, prefix, change, at))
    theirs, _, _ = di.stim_dem(c)
    assert mine != theirs


def test_single_qubit_noise_adds_no_mechanism_to_the_ibm_cycle():
    """Why the controls above sit where they do, held both ways: without any single-qubit noise line the fault set
    is unchanged; without the first two-qubit noise layer it is not."""
    c = bc.build("ibm", "Z", 2, 6)
    theirs, _, _ = di.stim_dem(c)
    kept = [s for s in str(c).splitlines() if not s.startswith(("DEPOLARIZE1", "X_ERROR", "Z_ERROR"))]
    assert di.pauli_dem(stim.Circuit("\n".join(kept)))[0] == theirs
    assert di.pauli_dem(_mutate(c, "DEPOLARIZE2", lambda s: []))[0] != theirs


@pytest.mark.parametrize("prefix,change", [("CX", lambda s: [s, "S 0"]), ("M ", lambda s: ["M(0.01)" + s[1:]])],
                         ids=["unhandled_gate", "noisy_measurement"])
def test_independent_dem_refuses_what_it_does_not_model(prefix, change):
    with pytest.raises(ValueError, match="not handled"):
        di.pauli_dem(_mutate(bc.build("ibm", "Z", 2, 6), prefix, change))


# ---------------------------------------------------------------- the projection and the witness

def test_projection_keeps_a_real_solution_a_solution():
    rec = json.loads((RECORDS / "bb72.jsonl").read_text().splitlines()[0])
    full = rec["lift"]["full"]
    assert tc.verify_full("ibm", "Z", 2, full, l=6)
    assert not tc.verify_full("ibm", "Z", 2, full[1:], l=6)
    _, mech, coords = em.mechanisms(bc.ibm_cycle("Z", 2, 6))
    pmech, _ = em.project_same_type(mech, coords, "Z")
    keep = sorted(d for d, v in coords.items() if int(v[1]) == 1)
    new_id = {d: i for i, d in enumerate(keep)}
    syn, obs = set(), set()
    for j in full:
        syn ^= {new_id[d] for d in mech[j][0] if d in new_id}
        obs ^= set(mech[j][1])
    assert not syn and obs


@needs_gcc
def test_engine_settles_bb72_and_the_witness_lifts(exe, bb72):
    out = tc.run(exe["cdist"], bb72, 1, 7, 18).stdout
    got = __import__("run_lane").parse_search(out)
    assert got["none_up_to"] == 5 and got["found_weight"] == 6
    full = tc.lift("ibm", "Z", 2, got["witness"], l=6)
    assert full and len(full) == 6 and tc.verify_full("ibm", "Z", 2, full, l=6)


def test_independent_cdcl_program_agrees_on_bb72(bb72):
    assert tc.sat_none(bb72, 4, seconds=120) is True       # no solution of weight <= 4
    assert tc.sat_none(bb72, 6, seconds=120) is False      # a solution of weight <= 6, re-verified


# ---------------------------------------------------------------- the certificate, both ways

def _parse(raw: bytes):
    """CDP3/CDP4/CDP5 -> (w, norb, maxround, [(code, [(nbits, bytearray)])] per anchor)"""
    w, norb, maxround, nanch = struct.unpack_from("<iiii", raw, 4)
    off, anchors = 20, []
    for _ in range(nanch):
        code = struct.unpack_from("<b", raw, off)[0]
        off += 1
        nstream = 0
        if code == 1:
            nstream = struct.unpack_from("<i", raw, off)[0]
            off += 4
        elif code == 2:
            nstream = 1
        streams = []
        for _ in range(nstream):
            n = struct.unpack_from("<q", raw, off)[0]
            nb = (n + 7) >> 3
            streams.append((n, bytearray(raw[off + 8: off + 8 + nb])))
            off += 8 + nb
        anchors.append((code, streams))
    return w, norb, maxround, anchors


def _write(path, w, norb, maxround, anchors, extra=b""):
    out = bytearray(b"CDP5" + struct.pack("<iiii", w, norb, maxround, len(anchors)))
    for code, streams in anchors:
        out += struct.pack("<b", code)
        if code == 1:
            out += struct.pack("<i", len(streams))
        for n, bits in streams:
            out += struct.pack("<q", n) + bytes(bits)
    path.write_bytes(bytes(out) + extra)


@needs_gcc
def test_a_mechanism_repeating_a_detector_is_refused_by_every_program(exe, tmp_path):
    """A detector listed twice cancels in the syndrome while the degree still counts it: the checker's size
    filters and empty-syndrome rule then lie -- the Lean pass (2026-10-21) built models where a checker printed
    VERIFIED for a false claim (F1, F5) and the prover crashed (F4). Every program now refuses such a model."""
    import struct
    import subprocess
    model = tmp_path / "rep.txt"   # the Lean pass's witness rep_z2: mechanism 2 lists detector 1 twice
    model.write_text("6 6 1 3 2 1\n2 0 1 1 0\n2 3 4 1 0\n4 1 1 1 2 0 1\n4 4 4 4 5 0 1\n6 0 0 0 0 0 2 0 2\n"
                     "6 3 3 3 3 3 5 0 2\n0 2 4\n0 0 0\n0 1 0\n0 0 1\n1 0 0\n1 1 0\n1 0 1\n")
    proof = tmp_path / "p.bin"     # a crafted CDP3 proof claiming weight <= 3 (root code 2, empty streams)
    proof.write_bytes(bytes([67, 68, 80, 51]) + struct.pack("<iiii", 3, 3, -1, 3) + (bytes([2]) + struct.pack("<q", 0)) * 3)
    run = lambda *a: subprocess.run([str(x) for x in a], capture_output=True, text=True, stdin=subprocess.DEVNULL)  # noqa: E731
    got = run(exe["ccheck"], model, proof, "full")
    assert got.returncode == 2 and "repeats detector" in got.stderr
    assert run(exe["cdist_prove"], model, 3, tmp_path / "q.bin").returncode == 2
    assert run(exe["cdist"], model, 1, 3, 18).returncode == 2


@needs_gcc
def test_checker_fast_and_sparse_leaf_paths_agree_on_mutated_proofs(exe, bb72, tmp_path):
    """The checker decides leaves by span masks + a degree-sorted scan (fast path) or by an exact sparse loop
    (CCHECK_SPARSE=1). On the valid proof and on 40 mutated copies (bit flips, truncations, an extra byte) the
    two must give the same verdict; the mutations must include rejections, so the agreement is not vacuous.
    (2026-10-21: the optimized checker also agreed with the original one on 423 such cases over three models.)"""
    import os
    import random
    import subprocess
    proof = tmp_path / "p.bin"
    assert tc.run(exe["cdist_prove"], bb72, 5, proof).returncode == 0
    raw, rng, verdicts = proof.read_bytes(), random.Random(198), []
    for t in range(41):
        b = bytearray(raw)
        if t % 3 == 1:
            i = rng.randrange(20, len(b)); b[i] ^= 1 << rng.randrange(8)
        elif t % 3 == 2:
            b = b[:rng.randrange(20, len(b))] if t % 2 else b + bytes([7])
        case = tmp_path / f"m{t}.bin"
        case.write_bytes(bytes(b))
        got = []
        for sparse in (False, True):
            env = dict(os.environ, OMP_NUM_THREADS="2", **({"CCHECK_SPARSE": "1"} if sparse else {}))
            got.append(subprocess.run([str(exe["ccheck"]), str(bb72), str(case), "full"], capture_output=True,
                                      text=True, env=env, stdin=subprocess.DEVNULL).returncode)
        assert got[0] == got[1], f"case {t}: fast {got[0]} vs sparse {got[1]}"
        verdicts.append(got[0])
    assert verdicts[0] == 0 and 1 in verdicts


# The CDP4/CDP5 rules (R205/R206) held against brute force: ccheck_oracle.c compiles ccheck.c ITSELF and answers
# with its bound_ok / leaf_exists3 / leaf_exists. Each mutant is a plausible one-token bug in one rule.
_CDP5_MUTANTS = {
    "terminal obstruction at three": ("if (++disjoint_count == 4) return 0;", "if (++disjoint_count == 3) return 0;"),
    "co-touch without the suffix union": ("for (size_t j = 0; j < (size_t)ND * W; j++) ct[j] |= next[j];",
                                          "for (size_t j = 0; j < 0; j++) ct[j] |= next[j];"),
    "dual raise ignores its own weight": ("if (bit(s, v)) used += weight[v];",
                                          "if (bit(s, v) && v != d) used += weight[v];"),
    "dual raise skips orbit o": ("int m = alist[t]; if (ORB[m] < o) continue;\n            int used = 0;",
                                 "int m = alist[t]; if (ORB[m] <= o) continue;\n            int used = 0;"),
    "terminal ignores a single": ("if (logical) return 1;", "if (0) return 1;"),
    "terminal skips orbit o": ("if (ORB[a] < o) continue;", "if (ORB[a] <= o) continue;"),
    "terminal drops pairs": ("leaf_exists(residual, hs ^ hz[a], pop, logical, 2)",
                             "leaf_exists(residual, hs ^ hz[a], pop, logical, 1)"),
    "cover bound at equality": ("    if (sum > 2520L * rem) return 1;\n    if (!modern_trace",
                                "    if (sum >= 2520L * rem) return 1;\n    if (!modern_trace"),
    "packing at equality": ("if (packed > rem) return 1;", "if (packed >= rem) return 1;"),
}


def _oracle_corpus(seed, nd=16, nm=22):
    """A random model (trivial group: every mechanism its own orbit, distinct syndromes, sparse so the co-touch
    graph has room for packings) and queries s = XOR of k random allowed mechanisms, each with the brute-force
    truth: min |Y| over allowed sets with S(Y) = s, and whether a logical completion of weight <= 3 (allowed) or
    <= 2 (any orbit) exists -- by breadth-first search over (syndrome, observable), sharing no code with ccheck."""
    import random
    rng = random.Random(seed)
    mech, seen = [], set()
    while len(mech) < nm:
        dets = tuple(sorted(rng.sample(range(nd), rng.choice([1, 2, 2, 2, 3, 3, 4]))))
        if dets not in seen:
            seen.add(dets)
            mech.append((dets, int(rng.random() < 0.35)))
    text = "\n".join([f"{nd} {nm} 1 {nm} 1 1"] + [" ".join(map(str, [len(d), *d, ob, m])) for m, (d, ob) in enumerate(mech)]
                     + [" ".join(map(str, range(nm)))] + [f"0 0 {d}" for d in range(nd)]) + "\n"
    enc = [sum(1 << d for d in dets) | ob << nd for dets, ob in mech]

    def bfs(gens, depth):
        dist = np.full(1 << (nd + 1), 127, np.int8)
        dist[0], frontier, g = 0, np.array([0], np.int64), np.array(gens, np.int64)
        for lev in range(1, depth + 1):
            nxt = np.unique((frontier[:, None] ^ g[None, :]).ravel())
            frontier = nxt[dist[nxt] == 127]
            dist[frontier] = lev
        return dist

    dist_all, queries, truth = bfs(enc, 2), [], []
    for o in sorted(rng.sample(range(nm), 6)):
        dist_o = bfs(enc[o:], 7)
        for _ in range(40):
            v = 0
            for m in rng.sample(range(o, nm), rng.randint(1, min(6, nm - o))):
                v ^= enc[m]
            s, l, rem = v & ((1 << nd) - 1), (v >> nd) ^ rng.randint(0, 1), rng.randint(3, 6)
            if s:
                flip = s | (l ^ 1) << nd                      # completions whose observable differs from l
                queries.append(f"{o} {rem} {l} {bin(s).count('1')} " + " ".join(str(d) for d in range(nd) if s >> d & 1))
                truth.append((rem, min(int(dist_o[s]), int(dist_o[s | 1 << nd])), dist_o[flip] <= 3, dist_all[flip] <= 2))
    return text, queries, truth


def _oracle_violations(oracle, corpora, mode, tmp_path):
    import subprocess
    bad, stats = [], dict(prune=0, leaf3_yes=0, leaf3_no=0)
    for i, (text, queries, truth) in enumerate(corpora):
        model = tmp_path / f"oracle_model_{i}.txt"
        model.write_text(text)
        r = subprocess.run([str(oracle), str(model), "full", str(mode)], input="\n".join(queries) + "\n",
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        answers = [tuple(map(int, ln.split())) for ln in r.stdout.splitlines() if ln.strip()]
        assert len(answers) == len(queries)
        for q, (bound, leaf3, leaf2), (rem, mincomp, ex3, ex2) in zip(queries, answers, truth):
            stats["prune"] += bound
            stats["leaf3_yes" if leaf3 else "leaf3_no"] += 1
            if bound and mincomp <= rem:
                bad.append(("bound prunes a completion within budget", i, q))
            if ex3 and not leaf3:
                bad.append(("weight-3 terminal misses a completion", i, q))
            if ex2 and not leaf2:
                bad.append(("leaf misses a completion", i, q))
    return bad, stats


@needs_gcc
def test_cdp5_rules_hold_against_brute_force_and_every_mutant_is_caught(tmp_path):
    """R205/R206 added three rules to the trusted checker: packing over the co-touch graph of ALLOWED mechanisms
    (orbit >= o, a suffix union), an exact-integer dual raise of the fractional cover, and a trace-free weight-3
    terminal (four pairwise non-co-touching firing detectors, else every allowed first member + the pair leaf).
    Held here against breadth-first brute force on 4,798 queries over 20 random models: a prune must never cut
    a completion within budget, a terminal must never miss one -- under the CDP3 rules and the CDP5 rules. Nine
    one-token bugs, one per rule, must each produce a violation, so the comparison is not dark."""
    import subprocess

    def build(name, src):
        d = tmp_path / name.replace(" ", "_")
        d.mkdir()
        (d / "ccheck.c").write_text(src)
        shutil.copy(LANE / "ccheck_oracle.c", d / "ccheck_oracle.c")
        subprocess.run(["gcc", "-O2", "-o", str(d / "oracle.exe"), str(d / "ccheck_oracle.c")], check=True,
                       capture_output=True, stdin=subprocess.DEVNULL)
        return d / "oracle.exe"

    src = (LANE / "ccheck.c").read_text()
    corpora = [_oracle_corpus(seed) for seed in range(20)]
    assert sum(len(q) for _, q, _ in corpora) == 4798
    oracle = build("original", src)
    stats = {}
    for mode in (0, 2):
        bad, stats[mode] = _oracle_violations(oracle, corpora, mode, tmp_path)
        assert bad == [], (mode, bad[:5])
    assert stats[0]["prune"] > 100 and stats[2]["leaf3_yes"] > 100 and stats[2]["leaf3_no"] > 100, stats
    assert stats[2]["prune"] > stats[0]["prune"], stats          # the CDP5 rules do prune more
    # lean/CertR198's ExampleDual: two copies of {0,2} {0,1,3} {0,4}, all ten detectors firing, budget 4 -- Lean
    # proves only the dual raise prunes it (plain cover exactly 2520 * 4), and this checker agrees
    gadget = tmp_path / "gadget.txt"
    gadget.write_text("10 6 1 6 1 1\n2 0 2 0 0\n3 0 1 3 0 1\n2 0 4 0 2\n2 5 7 0 3\n3 5 6 8 0 4\n2 5 9 0 5\n0 1 2 3 4 5\n"
                      + "".join(f"0 0 {d}\n" for d in range(10)))
    got = [subprocess.run([str(oracle), str(gadget), "full", str(mode)], input="0 4 0 10 0 1 2 3 4 5 6 7 8 9\n",
                          capture_output=True, text=True).stdout.split()[0] for mode in (0, 2)]
    assert got == ["0", "1"], got
    for name, (old, new) in _CDP5_MUTANTS.items():
        assert src.count(old) == 1, f"mutant {name!r} no longer applies: ccheck.c changed, update the mutant"
        bad, _ = _oracle_violations(build(name, src.replace(old, new)), corpora, 2, tmp_path)
        assert bad, f"mutant {name!r} survived: the oracle cannot see this rule"


def test_certificate_verifies_and_every_rejection_fires(exe, bb72, tmp_path):
    proof = tmp_path / "p.bin"
    assert tc.run(exe["cdist_prove"], bb72, 5, proof).returncode == 0
    ok = tc.run(exe["ccheck"], bb72, proof, "full")
    assert ok.returncode == 0 and ok.stdout.startswith("VERIFIED: no undetectable logical error of weight <= 5")
    raw = proof.read_bytes()
    w, norb, mr, anchors = _parse(raw)
    assert any(code == 1 and st for code, st in anchors), "the proof must cover an expanded anchor"
    forged = {}
    _write(tmp_path / "w6.bin", 6, norb, mr, anchors)
    forged["weight raised"] = tmp_path / "w6.bin"
    k = next(i for i, (code, st) in enumerate(anchors) if code == 1 and st)
    flip = [(c, [(n, bytearray(b)) for n, b in st]) for c, st in anchors]
    nonempty = next(((i, j) for i, (code, st) in enumerate(anchors)
                     for j, (n, _) in enumerate(st) if n), None)
    if nonempty is not None:
        i, j = nonempty
        flip[i][1][j][1][0] ^= 1
    else:
        # CDP5 resolves depth-three nodes without writing trace bits. A
        # bogus bit on an otherwise empty terminal stream is still rejected.
        flip[k][1][0] = (1, bytearray(b"\x00"))
    _write(tmp_path / "flip.bin", w, norb, mr, flip)
    forged["forged trace"] = tmp_path / "flip.bin"
    (tmp_path / "trunc.bin").write_bytes(raw[:-1])
    forged["truncated"] = tmp_path / "trunc.bin"
    _write(tmp_path / "trail.bin", w, norb, mr, anchors, extra=b"\x00")
    forged["trailing bytes"] = tmp_path / "trail.bin"
    fake = list(anchors)
    fake[0] = (0, [])                                       # claim the anchor-0 root is pruned by the bound
    _write(tmp_path / "fake.bin", w, norb, mr, fake)
    forged["fake root prune"] = tmp_path / "fake.bin"
    fewer = list(anchors)
    fewer[k] = (1, anchors[k][1][:-1])                      # one child stream missing
    _write(tmp_path / "fewer.bin", w, norb, mr, fewer)
    forged["child stream missing"] = tmp_path / "fewer.bin"
    _write(tmp_path / "scope.bin", w, norb, 1, anchors)     # claims time-shift scope on an unordered model
    forged["unearned time-shift scope"] = tmp_path / "scope.bin"
    assert tc.run(exe["liar"], bb72, 6, tmp_path / "liar.bin").returncode == 0      # the liar CLAIMS NONE <= 6
    forged["lying prover"] = tmp_path / "liar.bin"
    for name, path in forged.items():
        r = tc.run(exe["ccheck"], bb72, path, "full")
        assert r.returncode == 1 and r.stdout.startswith("REJECTED"), (name, r.stdout)
    assert "reason 2" in tc.run(exe["ccheck"], bb72, forged["lying prover"], "full").stdout
    assert "root bound does not hold" in tc.run(exe["ccheck"], bb72, forged["fake root prune"], "full").stdout


def _reverse_model(src: pathlib.Path, dst: pathlib.Path):
    """The same DEM listed backwards, orbits renumbered by first occurrence: a valid model whose orbit ids are
    NOT time-ordered."""
    lines = src.read_text().splitlines()
    nd, nm, no, norb, l, m = map(int, lines[0].split())
    mech = [ln.split() for ln in lines[1:1 + nm]][::-1]
    new_id, anchors = {}, []
    for i, p in enumerate(mech):
        o = p[-1]
        if o not in new_id:
            new_id[o] = len(new_id)
            anchors.append(i)
        p[-1] = str(new_id[o])
    dst.write_text("\n".join([lines[0]] + [" ".join(p) for p in mech] + [" ".join(map(str, anchors))]
                             + lines[2 + nm:]) + "\n")


@needs_gcc
def test_time_shift_anchoring_finds_what_it_must_and_refuses_unordered_models(exe, tmp_path):
    for T in (1, 2, 3, 4):
        model = tmp_path / f"t{T}.txt"
        em.export("ibm", "Z", T, model, l=6, time_order=True)
        got = __import__("run_lane").parse_search(tc.run(exe["cdist"], model, 5, 6, 18, 1).stdout)
        assert got["none_up_to"] == 5 and got["found_weight"] == 6, T     # a weight-6 logical starts in round <= 1
        proof = tmp_path / f"s{T}.bin"
        assert tc.run(exe["cdist_prove"], model, 5, proof, 1).returncode == 0
        ok = tc.run(exe["ccheck"], model, proof, "full")
        assert ok.returncode == 0 and "whose earliest round is <= 1" in ok.stdout
    rev = tmp_path / "rev.txt"
    _reverse_model(tmp_path / "t3.txt", rev)
    r = tc.run(exe["cdist"], rev, 5, 5, 18, 1)
    assert r.returncode == 2 and "not time-ordered" in r.stderr
    assert tc.run(exe["cdist_prove"], rev, 5, tmp_path / "x.bin", 1).returncode == 2
    full = tc.run(exe["cdist"], rev, 5, 6, 18)                        # without the time-shift claim it still runs
    assert __import__("run_lane").parse_search(full.stdout)["found_weight"] == 6


def test_observable_map_must_be_affine():
    _, mech, coords = em.mechanisms(bc.ibm_cycle("Z", 2, 6))
    pm, pc = em.project_same_type(mech, coords, "Z")
    shifts = bc.translations("full", 6, 6)
    assert em.observables_affine(pm, pc, shifts, 6, 6)
    bad = list(pm)
    dets, obs = bad[5]
    bad[5] = (dets, tuple(sorted(set(obs) ^ {0})))
    assert not em.observables_affine(bad, pc, shifts, 6, 6)


@needs_gcc
def test_checker_rejects_a_model_that_breaks_the_symmetry(exe, bb72, tmp_path):
    proof = tmp_path / "p.bin"
    assert tc.run(exe["cdist_prove"], bb72, 5, proof).returncode == 0
    lines = bb72.read_text().splitlines()
    head = lines[0].split()
    head[1] = str(int(head[1]) - 1)
    broken = tmp_path / "broken.txt"                       # drop one mechanism: no longer translation invariant
    broken.write_text("\n".join([" ".join(head)] + lines[2:]) + "\n")
    r = tc.run(exe["ccheck"], broken, proof, "full")
    assert r.returncode != 0 and "VERIFIED" not in r.stdout
    relabel = list(lines)                                  # orbit ids that disagree with the recomputed ones
    p = relabel[1].split()
    p[-1] = str(int(p[-1]) + 1)
    relabel[1] = " ".join(p)
    bad = tmp_path / "relabel.txt"
    bad.write_text("\n".join(relabel) + "\n")
    r = tc.run(exe["ccheck"], bad, proof, "full")
    assert r.returncode == 1 and "REJECTED" in r.stdout


@needs_gcc
def test_a_real_gross_code_certificate_verifies(exe, tmp_path):
    model = tmp_path / "ibm_Z2.txt"
    em.export("ibm", "Z", 2, model, l=12)
    proof = tmp_path / "g.bin"
    p = tc.run(exe["cdist_prove"], model, 7, proof, threads=8)
    assert p.returncode == 0 and "NONE at weight <= 7" in p.stdout
    c = tc.run(exe["ccheck"], model, proof, "full", threads=8)
    assert c.returncode == 0 and "VERIFIED: no undetectable logical error of weight <= 7" in c.stdout


# ---------------------------------------------------------------- the window lemma's preconditions

def test_window_preconditions_hold_and_can_fail(tmp_path):
    m3, m4 = tmp_path / "t3.txt", tmp_path / "t4.txt"
    em.export("ibm", "Z", 3, m3, l=6)
    em.export("ibm", "Z", 4, m4, l=6)
    assert tc.window_preconditions(m3, m4)["holds"]
    lines = m4.read_text().splitlines()
    nm = int(lines[0].split()[1])
    rnd = {d: int(lines[2 + nm + d].split()[2]) for d in range(int(lines[0].split()[0]))}
    for k in range(1, nm + 1):                             # move one interior-round mechanism's detector
        p = lines[k].split()
        dets = list(map(int, p[1:1 + int(p[0])]))
        if all(rnd[d] == 2 for d in dets):
            p[1] = str(next(d for d in rnd if rnd[d] == 2 and d not in dets))
            lines[k] = " ".join(p)
            break
    doctored = tmp_path / "t4_doctored.txt"
    doctored.write_text("\n".join(lines) + "\n")
    assert not tc.window_preconditions(m3, doctored)["holds"]
    with pytest.raises(ValueError):
        tc.window_preconditions(m3, m3)


def test_time_shift_needs_only_the_final_window_into_one_round_and_that_can_fail(tmp_path):
    m1, m2 = tmp_path / "t1.txt", tmp_path / "t2.txt"
    em.export("3c", "Z", 1, m1, time_order=True)
    em.export("3c", "Z", 2, m2, time_order=True)
    w = tc.window_preconditions(m1, m2)
    # a 1-round model's last round IS its start round, so its last-round window differs -- and is not needed
    assert w["P3_start"] and w["P3_final"] and not w["P3_last"] and w["time_shift_ok"]
    lines = m2.read_text().splitlines()
    nd, nm = map(int, lines[0].split()[:2])
    rnd = {d: int(lines[2 + nm + d].split()[2]) for d in range(nd)}
    for k in range(1, nm + 1):                                   # perturb one FINAL-layer mechanism
        p = lines[k].split()
        dets = list(map(int, p[1:1 + int(p[0])]))
        if all(rnd[d] == 2 for d in dets):
            p[1] = str(next(d for d in rnd if rnd[d] == 2 and d not in dets))
            lines[k] = " ".join(p)
            break
    bad = tmp_path / "t2_bad.txt"
    bad.write_text("\n".join(lines) + "\n")
    assert not tc.window_preconditions(m1, bad)["time_shift_ok"]


def test_space_time_window_holds_for_the_morphing_circuit_and_can_fail(tmp_path):
    """A morphing circuit repeats only up to the translation x per round: windows match after undoing r0 * x, and
    the observables (other logical representatives after a translation) must transform affinely. Holds on the 3-
    and 4-round models; FAILS without the drift, with the wrong drift, and with one interior observable flipped."""
    m3, m4 = tmp_path / "t3.txt", tmp_path / "t4.txt"
    em.export("morph72", "Z", 3, m3, time_order=True)
    em.export("morph72", "Z", 4, m4, time_order=True)
    w = tc.window_preconditions(m3, m4, bc.time_drift("morph72"))
    assert w["holds"] and w["time_shift_ok"] and w["obs_affine"] and w["drift"] == [1, 0]
    assert not tc.window_preconditions(m3, m4)["holds"]
    assert not tc.window_preconditions(m3, m4, (0, 1))["holds"]
    lines = m4.read_text().splitlines()
    nd, nm = map(int, lines[0].split()[:2])
    rnd = {d: int(lines[2 + nm + d].split()[2]) for d in range(nd)}
    for k in range(1, nm + 1):                                   # flip one observable of an interior mechanism
        p = lines[k].split()
        deg = int(p[0])
        if min(rnd[int(d)] for d in p[1:1 + deg]) == 2:
            p[1 + deg] = str(int(p[1 + deg]) ^ 1)
            lines[k] = " ".join(p)
            break
    bad = tmp_path / "t4_bad.txt"
    bad.write_text("\n".join(lines) + "\n")
    got = tc.window_preconditions(m3, bad, (1, 0))
    assert got["P2"] and not got["obs_affine"] and not got["holds"] and not got["time_shift_ok"]


# ---------------------------------------------------------------- the morphing circuits (Shaw & Terhal, 2407.16336)

def _reordered(terms, hom):
    f = bc.MORPH_HOMS[hom]
    c = [(t[1], 0) if t[0] == "x" else (0, t[1]) for t in terms]
    vals = [f(*e) for e in c]
    k = next(i for i, v in enumerate(vals) if vals.count(v) == 1)
    c[0], c[k] = c[k], c[0]
    return c


@pytest.mark.parametrize("kind", sorted(bc.MORPH_CODES))
@pytest.mark.parametrize("hom", sorted(bc.MORPH_HOMS))
def test_morphing_schedule_has_the_papers_structure(kind, hom):
    """Every CNOT layer is a perfect matching of all 2lm qubits; each round measures every right qubit once; every
    qubit has exactly FIVE partners, L(g) - R(a_i^-1 b_j g) for (i, j) in {(1,1), (1,2), (1,3), (2,1), (3,1)} with
    a_1, b_1 the elements the homomorphism singles out (the paper's degree-5 graph, written from its formula, not
    from the builder's loop); every end-cycle stabilizer has weight 9 and the end-cycle code keeps k = 12."""
    l, m, a_terms, b_terms = bc.MORPH_CODES[kind]
    gates, meas = bc.morphing_schedule(l, m, a_terms, b_terms, hom)
    n2 = l * m
    partners = {q: set() for q in range(2 * n2)}
    for rnd in gates:
        for layer in rnd:
            assert sorted(q for pair in layer for q in pair) == list(range(2 * n2))
            for c, t in layer:
                partners[c].add(t)
                partners[t].add(c)
    for r in (0, 1):
        assert sorted(meas["X"][r] + meas["Z"][r]) == list(range(n2, 2 * n2))
    A, B = _reordered(a_terms, hom), _reordered(b_terms, hom)
    for g in range(n2):
        i, j = divmod(g, m)
        want = {n2 + ((i - A[a][0] + B[b][0]) % l) * m + (j - A[a][1] + B[b][1]) % m
                for a, b in ((0, 0), (0, 1), (0, 2), (1, 0), (2, 0))}
        assert partners[g] == want and len(want) == 5
    assert {len(v) for v in partners.values()} == {5}
    code = bc.BB(l, m, a_terms, b_terms)
    HX, HZ = code.check_matrices()
    for r in (0, 1):
        ends = {}
        for p, H in (("X", HX), ("Z", HZ)):
            rows = [bc._contract(set(np.nonzero(H[g])[0].tolist()), p, gates[r]) - set(meas[p][r]) for g in range(n2)]
            rows = [x for x in rows if x]
            assert {len(x) for x in rows} == {9}
            ends[p] = np.array([[1 if q in x else 0 for q in range(n2)] for x in rows], dtype=np.uint8)
        assert n2 - bc.gf2_rank(ends["X"]) - bc.gf2_rank(ends["Z"]) == 12


@pytest.mark.parametrize("kind,basis", [("morph72", "Z"), ("morph72", "X"), ("morph144", "Z"), ("ibm", "Z")])
def test_detectors_span_every_deterministic_parity_but_the_redundancy_metachecks(kind, basis):
    """The derived detectors and observables span every deterministic parity of the noiseless circuit (stim's
    tableau counts the random outcomes exactly) except the other type's round-0 redundancy meta-checks: exactly
    n - rank(H_other) of them, the same family the IBM cycle omits."""
    c = bc.build(kind, basis, 2, 6 if kind == "ibm" else 12)
    gap = tc.deterministic_gap(c)
    l, m = bc.dims(kind, 6 if kind == "ibm" else 12)
    code = bc.BB(l, m, *bc.MORPH_CODES[kind][2:]) if kind in bc.MORPH_CODES else bc.BB(l)
    HX, HZ = code.check_matrices()
    other = HX if basis == "Z" else HZ
    assert len(gap["gap"]) == l * m - bc.gf2_rank(other) == 6
    assert gap["spanned"] + 6 == gap["deterministic"]


@pytest.mark.parametrize("kind,basis,l", [("morph72", "Z", 12), ("ibm", "X", 6), ("3c", "Z", 12)])
def test_projection_is_exact_here_every_projected_mechanism_has_a_pure_type_fault(kind, basis, l):
    """Why d(projected) = d_circ for these circuits: each projected mechanism is produced by a fault component
    that is pure X (memory-Z) / pure Z (memory-X), and such a component flips no detector of the other type."""
    c = bc.build(kind, basis, 2, l)
    comps, coords, nd, _ = di.pauli_components(c)
    keep = {d for d in range(nd) if int(coords[d][1]) == (1 if basis == "Z" else 0)}
    other = sum(1 << d for d in range(nd) if d not in keep)
    projected = {(m & ~other) for _, _, m in comps if m & ~other}
    pure = {m for _, p, m in comps if set(p.values()) == {"X" if basis == "Z" else "Z"} and not m & other}
    assert projected and projected <= pure


def test_a_witness_flipping_an_omitted_parity_does_not_survive_and_the_recorded_ones_do():
    toy = stim.Circuit("R 0 1 2\nX_ERROR(0.1) 1\nCX 1 2\nM 0 1 2\nDETECTOR rec[-3]\nOBSERVABLE_INCLUDE(0) rec[-2]")
    assert tc.deterministic_gap(toy)["gap"] == [[2]]
    assert tc.witness_survives_gap(toy, [0]) == dict(gap=1, survives=False)
    w = json.loads((RECORDS / "ibm_witnesses.jsonl").read_text().splitlines()[0])
    assert tc.witness_survives_gap(bc.build("ibm", w["basis"], w["rounds"]), w["full"]) == dict(gap=6, survives=True)


def test_fold_lemma_holds_on_the_morphing_circuit_and_each_control_fails(monkeypatch):
    """Every fault of the 3-round circuit, carried to its mid-cycle Pauli, has a single-fault twin in the 1-round
    circuit whose projected effect is a linear function of the original's (so d_proj(3) >= d_proj(1)). Each
    premise is load-bearing: without the propagation the twins conflict, with the other homomorphism's 1-round
    table they are missing, and one doctored observable breaks the linearity."""
    import fold_lemma as fl
    assert fl.fold_check("morph72", "Z", 3)["holds"]
    with monkeypatch.context() as mp:
        mp.setattr(fl, "_cx_layer", lambda layer, x, z: (x, z))
        got = fl.fold_check("morph72", "Z", 3)
        assert not got["holds"] and got["conflicts"] > 0
    table = fl.fault_table
    with monkeypatch.context() as mp:
        mp.setattr(fl, "fault_table", lambda kind, basis, T, hom="x": table(kind, basis, T, "y" if T == 1 else hom))
        got = fl.fold_check("morph72", "Z", 3)
        assert not got["holds"] and got["missing"] > 0

    def doctored(kind, basis, T, hom="x"):
        rows, nd = table(kind, basis, T, hom)
        if T == 1:
            k = next(i for i, r in enumerate(rows) if r[0])
            rows[k] = (rows[k][0], rows[k][1] ^ 1, *rows[k][2:])
        return rows, nd
    with monkeypatch.context() as mp:
        mp.setattr(fl, "fault_table", doctored)
        assert not fl.fold_check("morph72", "Z", 3)["holds"]


@needs_gcc
def test_engine_settles_the_72_qubit_morphing_circuit(exe, tmp_path):
    """Shaw & Terhal's bound d~circ <= 6 for the [[72,12,6]] end-cycle code is exact at T = 2 (memory-Z)."""
    p = tmp_path / "morph72.txt"
    em.export("morph72", "Z", 2, p)
    got = __import__("run_lane").parse_search(tc.run(exe["cdist"], p, 1, 7, 18).stdout)
    assert got["none_up_to"] == 5 and got["found_weight"] == 6
    full = tc.lift("morph72", "Z", 2, got["witness"])
    assert full and len(full) == 6 and tc.verify_full("morph72", "Z", 2, full)
    assert tc.witness_survives_gap(bc.build("morph72", "Z", 2), full)["survives"]


# ---------------------------------------------------------------- the evidence

def test_collect_results_refuses_missing_records(tmp_path, monkeypatch):
    monkeypatch.setattr(cr, "RECORDS", tmp_path)
    with pytest.raises(SystemExit, match="REFUSED"):
        cr.ibm_claim()
    with pytest.raises(SystemExit, match="REFUSED"):
        cr.three_colour_claim()
    with pytest.raises(SystemExit, match="REFUSED"):
        cr.three_colour_every_T_claim()
    with pytest.raises(SystemExit, match="REFUSED"):
        cr.dem_claim()
    with pytest.raises(SystemExit, match="REFUSED"):
        cr.morph72_claim()


def test_morph72_claim_needs_every_certificate_on_the_swept_model_and_every_window(tmp_path, monkeypatch):
    for f in RECORDS.iterdir():
        shutil.copy(f, tmp_path / f.name)
    monkeypatch.setattr(cr, "RECORDS", tmp_path)
    assert cr.morph72_claim()["evidence_class"].startswith("MACHINE-CHECKED")
    certs = [json.loads(x) for x in (tmp_path / "morph72_shift_certificates_w5.jsonl").read_text().splitlines() if x.strip()]
    certs[3]["model_sha256"] = "0" * 64                          # a certificate for another model certifies nothing
    (tmp_path / "morph72_shift_certificates_w5.jsonl").write_text("\n".join(json.dumps(c) for c in certs) + "\n")
    assert cr.morph72_claim()["evidence_class"].startswith("MEASURED")
    win = [json.loads(x) for x in (tmp_path / "morph72_window.jsonl").read_text().splitlines() if x.strip()]
    win[5]["obs_affine"] = False
    (tmp_path / "morph72_window.jsonl").write_text("\n".join(json.dumps(w) for w in win) + "\n")
    with pytest.raises(SystemExit, match="REFUSED"):
        cr.morph72_claim()


def test_morph144_equality_needs_our_own_witness_at_the_next_weight(tmp_path, monkeypatch):
    for f in RECORDS.iterdir():
        shutil.copy(f, tmp_path / f.name)
    monkeypatch.setattr(cr, "RECORDS", tmp_path)
    assert "memory-Z: d_circ = 12" in cr.morph144_claim()["claim"]
    ws = [json.loads(x) for x in (tmp_path / "morph144_witnesses.jsonl").read_text().splitlines() if x.strip()]
    for bad in (dict(survives_metachecks=False), dict(weight=13), dict(verified=False)):
        (tmp_path / "morph144_witnesses.jsonl").write_text(
            "".join(json.dumps({**w, **bad} if w["basis"] == "Z" else w) + "\n" for w in ws))
        got = cr.morph144_claim()
        assert "memory-Z: d_circ >= 12" in got["claim"] and "memory-X: d_circ = 12" in got["claim"]
        assert got["evidence_class"].startswith("MEASURED")


def test_morph144_every_T_needs_every_fold_and_every_embedding(tmp_path, monkeypatch):
    for f in RECORDS.iterdir():
        shutil.copy(f, tmp_path / f.name)
    monkeypatch.setattr(cr, "RECORDS", tmp_path)
    full = cr.morph144_claim()
    assert full["every_T"]["Z"]["tmax"] == 13 and "for every T >= 1" in full["claim"]     # the theorem's route
    folds = [json.loads(x) for x in (tmp_path / "fold_lemma.jsonl").read_text().splitlines() if x.strip()]
    (tmp_path / "fold_lemma.jsonl").write_text("\n".join(json.dumps(r) for r in folds if not (
        r["circuit"] == "morph144" and r["basis"] == "Z" and r["T"] == 5)) + "\n")
    assert cr.morph144_claim()["every_T"]["Z"]["tmax"] == 4        # a hole at T = 5 caps the run below 5
    (tmp_path / "fold_lemma.jsonl").write_text("\n".join(json.dumps(r) for r in folds) + "\n")
    emb = [json.loads(x) for x in (tmp_path / "morph144_witness_embedding.jsonl").read_text().splitlines() if x.strip()]
    (tmp_path / "morph144_witness_embedding.jsonl").write_text("\n".join(json.dumps(r) for r in emb if r["T"] != 9) + "\n")
    assert cr.morph144_claim()["every_T"]["Z"] is None


def test_morph144_is_machine_checked_only_with_every_certificate_on_the_searched_models(tmp_path, monkeypatch):
    for f in RECORDS.iterdir():
        shutil.copy(f, tmp_path / f.name)
    monkeypatch.setattr(cr, "RECORDS", tmp_path)
    assert cr.morph144_claim()["evidence_class"].startswith("PROVEN for every T >= 1")
    certs = [json.loads(x) for x in (tmp_path / "morph144_certificates_w11.jsonl").read_text().splitlines() if x.strip()]
    keep = [c for c in certs if not (c["basis"] == "X" and c["rounds"] == 1)]
    (tmp_path / "morph144_certificates_w11.jsonl").write_text("\n".join(json.dumps(c) for c in keep) + "\n")
    got = cr.morph144_claim()
    assert got["evidence_class"].startswith("MEASURED") and not got["every_T"]["X"]["certified"]
    bad = [dict(c, model_sha256="0" * 64) if (c["basis"], c["rounds"]) == ("Z", 1) else c for c in certs]
    (tmp_path / "morph144_certificates_w11.jsonl").write_text("\n".join(json.dumps(c) for c in bad) + "\n")
    with pytest.raises(SystemExit, match="REFUSED"):
        cr.morph144_claim()


# ---------------------------------------------------------------- the fold lemma for EVERY T (FOLD_PROOF.md)

def test_every_T_from_the_fold_theorem_needs_its_hypotheses_its_controls_and_corollary_2(tmp_path, monkeypatch):
    """Both morphing claims say "every T >= 1" (PROVEN from machine-checked premises) only while FOLD_PROOF.md's
    hypotheses hold in the archived fold_uniform run, every negative control there fails, and Corollary 2 matched
    stim at fresh T with both of its controls failing. Breaking any one of those drops each claim back to its
    per-T route (morph144: T = 1..13; morph72: T >= 2, MACHINE-CHECKED to 7)."""
    for f in RECORDS.iterdir():
        shutil.copy(f, tmp_path / f.name)
    monkeypatch.setattr(cr, "RECORDS", tmp_path)
    assert "= 12 for every T >= 1" in cr.morph144_claim()["claim"] and "every T >= 1" in cr.morph72_claim()["claim"]
    hom = (tmp_path / "fold_uniform_hom_x.jsonl").read_text()
    cor2 = (tmp_path / "fold_review_cor2.jsonl").read_text()

    def rows(text):
        return [json.loads(x) for x in text.splitlines() if x.strip()]

    def h3_fails(r):
        r["hypotheses"]["H3"]["holds"] = False

    def a_control_holds(r):
        r["controls"][next(iter(r["controls"]))] = True

    def a_template_disagrees(r):
        r["fresh"][-1]["agrees"] = False

    def stim_disagrees(c):
        c["equal"] = False

    def a_stim_control_agrees(c):
        if "control_flipped_window_equal" in c:
            c["control_flipped_window_equal"] = True

    for name, text, mutate in (("fold_uniform_hom_x.jsonl", hom, h3_fails), ("fold_uniform_hom_x.jsonl", hom, a_control_holds),
                               ("fold_uniform_hom_x.jsonl", hom, a_template_disagrees),
                               ("fold_review_cor2.jsonl", cor2, stim_disagrees),
                               ("fold_review_cor2.jsonl", cor2, a_stim_control_agrees)):
        rs = rows(text)
        for r in rs:
            mutate(r)
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in rs))
        m144, m72 = cr.morph144_claim(), cr.morph72_claim()
        assert "for every T = 1..13" in m144["claim"] and m144["evidence_class"].startswith("MACHINE-CHECKED"), mutate
        assert "every T >= 2" in m72["claim"] and "DERIVED beyond" in m72["evidence_class"], mutate
        (tmp_path / "fold_uniform_hom_x.jsonl").write_text(hom)
        (tmp_path / "fold_review_cor2.jsonl").write_text(cor2)
    (tmp_path / "fold_review_cor2.jsonl").unlink()
    assert "for every T = 1..13" in cr.morph144_claim()["claim"]


def test_fold_hypotheses_hold_on_the_lane_and_every_control_fails():
    """fold_uniform.py on the 72-qubit morphing circuit, both bases, template T = 1..3: H1-H3, PAL, BND, PURE hold,
    the template identification agrees, and all of its negative controls fail (its exit code requires both)."""
    import subprocess
    got = subprocess.run([sys.executable, "fold_uniform.py", "--circuit", "morph72", "--basis", "Z", "X",
                          "--template-T", "1", "2", "3"], cwd=LANE, capture_output=True, text=True,
                         stdin=subprocess.DEVNULL, timeout=600)
    assert got.returncode == 0 and "OVERALL PASS" in got.stdout, got.stdout[-2000:] + got.stderr[-2000:]
    rows = [json.loads(x) for x in got.stdout.splitlines() if x.startswith("{")]
    assert len(rows) == 2
    for r in rows:
        assert all(h["holds"] for h in r["hypotheses"].values()) and len(r["controls"]) >= 20
        assert not any(r["controls"].values())


def test_fold_twin_sends_the_1_round_witnesses_to_witnesses_at_T():
    """FOLD_PROOF.md Theorem (B)/(C) on the lane's weight-12 witnesses at T = 4: their twins are undetectable and
    logical in the T-round projected model, and their pure lifts are full-DEM witnesses of weight 12."""
    import subprocess
    got = subprocess.run([sys.executable, "fold_twin.py", "--T", "4"], cwd=LANE, capture_output=True, text=True,
                         stdin=subprocess.DEVNULL, timeout=600)
    rows = [json.loads(x) for x in got.stdout.splitlines() if x.startswith("{")]
    assert got.returncode == 0 and {r["basis"] for r in rows} == {"Z", "X"}, got.stdout + got.stderr
    for r in rows:
        assert r["weight"] == 12 and r["twin_images_in_T_model"] and r["projected_undetectable_logical_at_T"]
        assert r["full_dem_pure_lift_undetectable_logical_at_T"]


def _cor2_sets(circuit, keep_type):
    """(projected mechanism set {(frozenset (site, round)), frozenset observables)} of a circuit's DEM, by stim"""
    dem = circuit.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
    coords = {d: tuple(int(round(v)) for v in c) for d, c in dem.get_detector_coordinates().items()}
    out = set()
    for inst in dem.flattened():
        if inst.type == "error":
            dets = [t.val for t in inst.targets_copy() if t.is_relative_detector_id()]
            obs = frozenset(t.val for t in inst.targets_copy() if t.is_logical_observable_id())
            pd = frozenset((coords[d][0], coords[d][2]) for d in dets if coords[d][1] == keep_type)
            if pd or obs:
                out.add((pd, obs))
    return out


@pytest.mark.parametrize("kind,basis,T", [("morph72", "Z", 5), ("morph144", "X", 5)])
def test_fold_corollary2_matches_stim_at_fresh_T(kind, basis, T):
    """FOLD_PROOF.md Corollary 2 against stim ALONE (the 2026-10-22 audit's check; no image computation, no backward
    pass): the T-round projected model equals {(iota_t(s), l) : t < T, (s, l) in the 1-round model}, where iota_t puts
    generator g's bit on its detector whose window holds round t -- read off the 2-round circuit's detector
    coordinates. Two controls must disagree: the window parity flipped, and one round's expansion noise deleted."""
    keep = 1 if basis == "Z" else 0
    one = {(frozenset(g for g, _ in pd), obs) for pd, obs in _cor2_sets(bc.morphing_cycle(kind, basis, 1), keep)}
    c2 = bc.morphing_cycle(kind, basis, 2)
    present = {(int(round(c[0])), int(round(c[2]))) for c in c2.get_detector_coordinates().values()
               if int(round(c[1])) == keep}
    n = bc.MORPH_CODES[kind][0] * bc.MORPH_CODES[kind][1]
    gamma = [{r for r in (0, 1) if (g, r) in present} for g in range(n)]

    def window(g, t, flip):
        gm = {1 - r for r in gamma[g]} if flip else gamma[g]
        return next((s for s in range(t, T) if s % 2 in gm), T)

    def predict(flip=False):
        return {(frozenset((g, window(g, t, flip)) for g in gs), obs) for gs, obs in one for t in range(T)}

    tags: list = []
    cT = bc.morphing_cycle(kind, basis, T, tags=tags)
    actual = _cor2_sets(cT, keep)
    assert actual == predict() and len(actual) > len(one)
    assert actual != predict(flip=True)
    drop = {tg[0] for tg in tags if tg[2] == "exp" and tg[3] == T - 1}
    assert drop
    cut = stim.Circuit()
    for i, inst in enumerate(cT.flattened()):
        if i not in drop:
            cut.append(inst)
    assert _cor2_sets(cut, keep) != predict()


def test_trusted_base_claim_refuses_one_unequal_model(tmp_path, monkeypatch):
    for f in RECORDS.iterdir():
        shutil.copy(f, tmp_path / f.name)
    monkeypatch.setattr(cr, "RECORDS", tmp_path)
    assert cr.dem_claim()["models"] == 66
    rows = [json.loads(x) for x in (tmp_path / "dem_independent.jsonl").read_text().splitlines() if x.strip()]
    rows[-1]["only_stim"] = 1
    (tmp_path / "dem_independent.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(SystemExit, match="REFUSED"):
        cr.dem_claim()


def test_every_T_claim_refuses_a_hole_in_the_sweep(tmp_path, monkeypatch):
    for f in RECORDS.iterdir():
        shutil.copy(f, tmp_path / f.name)
    monkeypatch.setattr(cr, "RECORDS", tmp_path)
    assert cr.three_colour_every_T_claim()["claim"].startswith("3-colour")
    rows = [json.loads(x) for x in (tmp_path / "3c_sweep_shift_w10.jsonl").read_text().splitlines() if x.strip()]
    (tmp_path / "3c_sweep_shift_w10.jsonl").write_text("\n".join(json.dumps(r) for r in rows if r["rounds"] != 7) + "\n")
    with pytest.raises(SystemExit, match="REFUSED"):
        cr.three_colour_every_T_claim()


def test_every_T_certification_needs_all_24_verified_on_the_swept_models(tmp_path, monkeypatch):
    """The 3-colour every-T claim turns MACHINE-CHECKED only with a VERIFIED time-shift line for each of the 24
    (basis, T) cells, each on the model the sweep searched; one cell on another model, or missing, keeps it
    MEASURED. Synthetic certificate rows test the gate both ways before the real ones exist."""
    for f in RECORDS.iterdir():
        shutil.copy(f, tmp_path / f.name)
    monkeypatch.setattr(cr, "RECORDS", tmp_path)
    sweep = [json.loads(x) for x in (tmp_path / "3c_sweep_shift_w10.jsonl").read_text().splitlines() if x.strip()]
    rows = [dict(basis=r["basis"], rounds=r["rounds"], w=10, checker_exit=0, model_sha256=r["model_sha256"],
                 checker="VERIFIED: no undetectable logical error of weight <= 10 whose earliest round is <= 1 in this DEM")
            for r in sweep]
    certs = tmp_path / "3c_shift_certificates_w10.jsonl"
    certs.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert cr.three_colour_every_T_claim()["evidence_class"].startswith("PROVEN for every T >= 2 from MACHINE-CHECKED")
    rows[5] = dict(rows[5], model_sha256="0" * 64)
    certs.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert "premises MEASURED" in cr.three_colour_every_T_claim()["evidence_class"]
    certs.write_text("".join(json.dumps(r) + "\n" for r in rows[:5] + rows[6:]))
    assert "premises MEASURED" in cr.three_colour_every_T_claim()["evidence_class"]
    # without the window theorem's records the every-T step falls back to the window lemma's DERIVED form
    (tmp_path / "window_uniform_run.jsonl").unlink()
    certs.write_text("".join(json.dumps(r) + "\n" for r in rows[:5] + [dict(rows[5], model_sha256=sweep[5]["model_sha256"])]
                             + rows[6:]))
    got = cr.three_colour_every_T_claim()
    assert got["evidence_class"].startswith("MACHINE-CHECKED for T <= 12; DERIVED beyond") and "T = 2..12" in got["claim"]
    assert cr.ibm_claim()["evidence_class"].startswith("MACHINE-CHECKED for T <= 11, DERIVED beyond")


def test_the_committed_3c_certificates_prove_every_T_and_one_doctored_verdict_reverts_it(tmp_path, monkeypatch):
    """The REAL record of the 2026-10-23 campaign: one VERIFIED certificate per (basis, T) cell of the sweep, each at
    weight <= 10 with anchors in rounds 0-1 and on the exact model the sweep searched, workstation paths redacted --
    and the live 3-colour claim PROVEN. Flip one real verdict and the claim falls back: it reads these rows."""
    certs = [json.loads(x) for x in (RECORDS / "3c_shift_certificates_w10.jsonl").read_text().splitlines() if x.strip()]
    sweep = {(r["basis"], r["rounds"]): r["model_sha256"]
             for r in (json.loads(x) for x in (RECORDS / "3c_sweep_shift_w10.jsonl").read_text().splitlines() if x.strip())}
    assert len(sweep) == 24 and sorted((r["basis"], r["rounds"]) for r in certs) == sorted(sweep)
    for r in certs:
        assert r["checker_exit"] == 0 and r["checker"].startswith("VERIFIED") and (r["w"], r["maxround"]) == (10, 1)
        assert r["model_sha256"] == sweep[(r["basis"], r["rounds"])]
        assert "Users" not in json.dumps(r)
    assert cr.three_colour_every_T_claim()["evidence_class"].startswith("PROVEN for every T >= 2 from MACHINE-CHECKED")
    for f in RECORDS.iterdir():
        shutil.copy(f, tmp_path / f.name)
    monkeypatch.setattr(cr, "RECORDS", tmp_path)
    doctored = [dict(certs[7], checker=certs[7]["checker"].replace("VERIFIED", "REFUTED", 1))] + certs[:7] + certs[8:]
    (tmp_path / "3c_shift_certificates_w10.jsonl").write_text("".join(json.dumps(r) + "\n" for r in doctored))
    assert "premises MEASURED" in cr.three_colour_every_T_claim()["evidence_class"]


def test_window_theorem_needs_its_run_verdict_its_controls_and_the_audit(tmp_path, monkeypatch):
    """The IBM and 3-colour every-T claims are PROVEN by WINDOW_PROOF.md Theorem 1 only while the archived
    window_uniform.py run passes that script's own verdict with every control failing, and the audit's independent
    rebuild of the closed form equals stim with its controls failing; break any of that and the claims fall back."""
    for f in RECORDS.iterdir():
        shutil.copy(f, tmp_path / f.name)
    monkeypatch.setattr(cr, "RECORDS", tmp_path)
    assert cr.ibm_claim()["evidence_class"].startswith("PROVEN for every T >= 2")
    run = (tmp_path / "window_uniform_run.jsonl").read_text()
    rev = (tmp_path / "window_review_formula.jsonl").read_text()

    def fiddle(name, text, mutate):
        rs = [json.loads(x) for x in text.splitlines() if x.strip()]
        for r in rs:
            mutate(r)
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in rs))

    def a_control_holds(r):
        r["CONTROLS"][next(iter(r["CONTROLS"]))] = True

    def fir_has_no_minimum(r):
        r["FIR"]["mem"]["minimal"] = None

    def stim_disagrees(r):
        if r["step"] == "formula_vs_stim":
            r["all_equal"] = False

    def a_review_control_agrees(r):
        if r["step"] == "controls_must_be_false":
            r["all_fail_as_required"] = False

    for name, text, mutate in (("window_uniform_run.jsonl", run, a_control_holds),
                               ("window_uniform_run.jsonl", run, fir_has_no_minimum),
                               ("window_review_formula.jsonl", rev, stim_disagrees),
                               ("window_review_formula.jsonl", rev, a_review_control_agrees)):
        fiddle(name, text, mutate)
        assert cr.ibm_claim()["evidence_class"].startswith("MACHINE-CHECKED for T <= 11, DERIVED beyond"), mutate
        assert "T = 2..12" in cr.three_colour_every_T_claim()["claim"], mutate
        (tmp_path / "window_uniform_run.jsonl").write_text(run)
        (tmp_path / "window_review_formula.jsonl").write_text(rev)


def test_wu_bind_binds_the_window_theorem_to_the_recorded_models(tmp_path):
    """wu_bind (window_uniform's BIND step) re-exports a model the lane recorded and must reproduce its sha256 -- the
    link from WINDOW_PROOF.md's closed form to the certified models; a different T must not."""
    import wu_bind as wb
    recs = [r for r in wb.recorded_models() if r[0] == "ibm" and r[1] == "Z" and r[2] == 1 and not r[3]]
    assert recs, "no recorded IBM memory-Z T = 1 model to bind"
    kind, basis, T, to, sha = recs[0][:5]
    path, got = wb.export_and_hash(kind, basis, T, to, tmp_path)
    assert got == sha and len(wb.read_masks(path)) > 0
    _, other = wb.export_and_hash(kind, basis, T + 1, to, tmp_path)
    assert other != sha


def test_window_hypotheses_hold_on_the_ibm_cycle_and_every_control_fails():
    """window_uniform.py live on the IBM cycle, memory-Z, at small T (the archived run covers both circuits and both
    bases at T up to 32): every hypothesis holds, the closed form equals stim, and every control fails (its exit code
    requires all of it)."""
    import subprocess
    got = subprocess.run([sys.executable, "window_uniform.py", "--circuit", "ibm", "--basis", "Z", "--template-T", "1",
                          "2", "3", "4", "--fresh-T", "8", "--fresh-T-full", "6", "--bind-T", "1", "2"], cwd=LANE,
                         capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=1200)
    assert got.returncode == 0 and "OVERALL PASS" in got.stdout, got.stdout[-2000:] + got.stderr[-1000:]
    row = next(json.loads(x) for x in got.stdout.splitlines() if x.startswith("{"))
    assert len(row["CONTROLS"]) >= 14 and not any(row["CONTROLS"].values())


def test_a_certificate_for_another_model_does_not_certify(tmp_path, monkeypatch):
    """The 3-colour claim is certified only by a VERIFIED record on the SAME model the search ran on."""
    for f in RECORDS.iterdir():
        shutil.copy(f, tmp_path / f.name)
    monkeypatch.setattr(cr, "RECORDS", tmp_path)
    assert all(v["certified"] for v in cr.three_colour_claim()["bases"].values())
    certs = [json.loads(x) for x in (tmp_path / "3c_T2_certificates_w10.jsonl").read_text().splitlines() if x.strip()]
    certs[0]["model_sha256"] = "0" * 64
    (tmp_path / "3c_T2_certificates_w10.jsonl").write_text("\n".join(json.dumps(c) for c in certs) + "\n")
    got = cr.three_colour_claim()
    assert not got["bases"][certs[0]["basis"]]["certified"] and got["evidence_class"].startswith("MEASURED")


def test_committed_results_are_what_collect_results_computes():
    committed = json.loads((ROOT / "evidence" / "circuit_distance_exact" / "results.json").read_text())
    assert committed["ibm_gross"] == cr.ibm_claim()
    assert committed["three_colour"] == cr.three_colour_claim()
    assert committed["bb72"] == cr.bb72_claim()
    assert committed["three_colour_every_T"] == cr.three_colour_every_T_claim()
    assert committed["trusted_base"] == cr.dem_claim()
    assert committed["morph72"] == cr.morph72_claim()
    assert committed["morph144"] == cr.morph144_claim()
    assert committed["morph2026"] == cr.morph2026_claim()
    assert len(committed["ibm_gross"]["proofs"]) == 22


# ---------------------------------------------------------------- 2604.09797's Abelian-2BGA morphing circuits

def test_morph2026_builder_reproduces_the_lanes_morphing_circuits_byte_for_byte():
    """The generalized builder (any code, any number of layers, round-dependent data qubits) is the lane's own on the
    lane's codes -- so the exporter, engine, checker and lemmas apply to it unchanged."""
    import morph2026 as M
    assert all(M.same_as_lane(kind, basis, T) for kind in ("morph72", "morph144") for basis in "ZX" for T in (1, 2))


@needs_gcc
def test_morph2026_the_42_qubit_circuit_keeps_6_and_its_other_cnot_order_loses_2(tmp_path):
    """[[42,6,6]], degree 4 (Z_42, A = 1 + x + x^10, B = 1 + x^7): at T = 1 the engine finds no logical error of
    weight <= 5 and a weight-6 one that lifts to the full DEM, verifies, and loses its property under every single
    deletion; the checker VERIFIES the weight <= 5 certificate. The same code with the other order of A's two
    non-special terms (c42a2) has a verified weight-4 witness: the CNOT order alone costs two."""
    import morph2026 as M
    exe = tc.build_c(tmp_path)
    good = M.search(M.spec("c42a"), "Z", 1, 6, exe, tmp_path, 2)
    assert good["none_up_to"] == 5 and good["found_weight"] == 6 and good["verified"] and good["every_deletion_fails"]
    cert = M.certify(M.spec("c42a"), "Z", 1, 5, exe, tmp_path, 2)
    assert cert["checker_exit"] == 0 and cert["checker"].startswith("VERIFIED: no undetectable logical error of weight <= 5")
    assert cert["model_sha256"] == good["model_sha256"]
    bad = M.search(M.spec("c42a2"), "Z", 1, 6, exe, tmp_path, 2)
    assert bad["found_weight"] == 4 and bad["verified"] and bad["every_deletion_fails"]


def test_morph2026_fold_hypotheses_hold_on_the_42_qubit_circuit_and_every_control_fails():
    import subprocess
    got = subprocess.run([sys.executable, "fold_uniform.py", "--circuit", "c42a", "--basis", "Z", "--template-T", "1", "2"],
                         cwd=LANE, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=600)
    assert got.returncode == 0 and "OVERALL PASS" in got.stdout, got.stdout[-2000:] + got.stderr[-1000:]
    row = next(json.loads(x) for x in got.stdout.splitlines() if x.startswith("{"))
    assert len(row["controls"]) == 18 and not any(row["controls"].values())


def test_morph2026_claim_needs_every_certificate_window_and_the_theorem(tmp_path, monkeypatch):
    """Every T is PROVEN for a standard schedule only with the fold theorem's archived verdict; without it the claim
    falls back to MACHINE-CHECKED to w + 2. A certificate for another model, a missing time-shift certificate or a
    failed window removes even that."""
    for f in RECORDS.iterdir():
        shutil.copy(f, tmp_path / f.name)
    monkeypatch.setattr(cr, "RECORDS", tmp_path)
    got = cr.morph2026_claim()["targets"]["c42a"]["bases"]["Z"]
    assert got["d_circ"] == 6 and got["loses"] == 0 and got["every_T"].startswith("PROVEN")
    camp = (tmp_path / "morph2026_campaign.jsonl").read_text()
    folds = (tmp_path / "morph2026_fold_uniform.jsonl").read_text()
    rows = [json.loads(x) for x in folds.splitlines() if x.strip()]
    for r in rows:
        if r["circuit"] == "c42a" and r["basis"] == "Z":
            r["controls"][next(iter(r["controls"]))] = True
    (tmp_path / "morph2026_fold_uniform.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert cr.morph2026_claim()["targets"]["c42a"]["bases"]["Z"]["every_T"].startswith("MACHINE-CHECKED for T <= 7")
    (tmp_path / "morph2026_fold_uniform.jsonl").write_text(folds)

    def mutated(pred, change):
        out = []
        for x in camp.splitlines():
            r = json.loads(x)
            if pred(r):
                if change is None:
                    continue
                change(r)
            out.append(json.dumps(r))
        (tmp_path / "morph2026_campaign.jsonl").write_text("\n".join(out) + "\n")
        return cr.morph2026_claim()["targets"]["c42a"]["bases"]["Z"]

    is_c42a_z = lambda r: r.get("code") == "c42a" and r.get("basis") == "Z"  # noqa: E731
    got = mutated(lambda r: is_c42a_z(r) and r["kind"] == "cert" and r["rounds"] == 1,
                  lambda r: r.update(model_sha256="0" * 64))
    assert 1 not in got["certified_T"] and got["every_T"] is None
    got = mutated(lambda r: is_c42a_z(r) and r["kind"] == "shift_cert" and r["rounds"] == 4, None)
    assert got["every_T"] is None
    got = mutated(lambda r: is_c42a_z(r) and r["kind"] == "window" and r["T"] == 3, lambda r: r.update(time_shift_ok=False))
    assert got["every_T"] is None


# ---------------------------------------------------------------- the checker's soundness in Lean, and the campaign


def test_lean_receipt_certifies_the_current_sources_and_every_forgery_is_refused(tmp_path):
    """lean/CertR198 proves the certificate calculus sound; the receipt pins what Lean checked. The gate passes on
    the committed receipt and refuses: a source edited after the proof, a theorem resting on sorryAx, one resting
    on a non-standard axiom, and a theorem list that no longer matches AxiomCheck.lean."""
    import lean_receipt as lr
    assert lr.check() == [], lr.check()
    receipt = json.loads(lr.RECEIPT.read_text(encoding="utf-8"))
    assert len(receipt["theorems"]) >= 18 and receipt["build_exit"] == 0
    assert all(set(a) <= lr.STANDARD for a in receipt["theorems"].values())
    root = tmp_path / "CertR198"
    shutil.copytree(lr.LEAN, root, ignore=shutil.ignore_patterns(".lake"))

    def refused(mutate_receipt=None, mutate_tree=None):
        r = json.loads(json.dumps(receipt))
        if mutate_receipt:
            mutate_receipt(r)
        if mutate_tree:
            mutate_tree(root)
        p = tmp_path / "r.json"
        p.write_text(json.dumps(r))
        return lr.check(p, root)

    assert refused() == []
    model = root / "CertR198" / "Model.lean"   # a Windows checkout (core.autocrlf) is the same proved text
    model.write_bytes(model.read_bytes().replace(b"\n", b"\r\n"))
    assert refused() == []
    first = sorted(receipt["theorems"])[0]
    assert any("sorryAx" in b for b in refused(lambda r: r["theorems"][first].append("sorryAx")))
    assert any("non-standard" in b for b in refused(lambda r: r["theorems"][first].append("CertR198.trustMe")))
    assert any("theorem list" in b for b in refused(lambda r: r["theorems"].pop(first)))
    model = root / "CertR198" / "Model.lean"
    assert any("STALE" in b for b in refused(mutate_tree=lambda d: model.write_text(
        model.read_text(encoding="utf-8") + "\n-- edit\n", encoding="utf-8")))


def test_lean_kernel_modules_are_the_certified_models_and_a_doctored_one_is_refused(tmp_path, monkeypatch):
    """lean/CertR198/CertR198/Kernel/*.lean make four 1-round distances (2604.09797's [[42,6,6]] degree-4 circuit,
    6 in both bases; its [[30,8,5]] tree circuit, 5 in both) theorems checked by the Lean kernel. The gate regenerates
    each module from a fresh export bound to the campaign's certified model hash and minimal witness and refuses any
    difference (one number changed in a committed module is caught); the receipt lists every `distance` theorem on
    the standard axioms only."""
    import lean_kernel_proof as lk
    import lean_receipt as lr
    assert lk.check() == []
    theorems = json.loads(lr.RECEIPT.read_text(encoding="utf-8"))["theorems"]
    for name, *_ in lk.TARGETS:
        assert set(theorems[f"CertR198.Kernel.{name}.distance"]) <= lr.STANDARD
    got = cr.morph2026_claim()["targets"]
    assert got["c42a"]["bases"]["Z"]["kernel_checked_T1"] == "CertR198.Kernel.C42aZ1.distance"
    assert got["t30a"]["bases"]["X"]["kernel_checked_T1"] == "CertR198.Kernel.T30aX1.distance"
    assert got["c48a"]["bases"]["Z"]["kernel_checked_T1"] is None
    for f in lk.OUT.glob("*.lean"):
        shutil.copy(f, tmp_path / f.name)
    monkeypatch.setattr(lk, "OUT", tmp_path)
    assert lk.check() == []
    p = tmp_path / "C42aZ1.lean"
    p.write_text(p.read_text(encoding="utf-8").replace("(131075, 38)", "(131075, 39)", 1), encoding="utf-8")
    assert any("C42aZ1.lean" in b for b in lk.check())
    # a doctored module certifies nothing: the claim drops every kernel-checked theorem
    assert all(b is None or b["kernel_checked_T1"] is None
               for t in cr.morph2026_claim()["targets"].values() for b in t["bases"].values())


def test_campaign_finisher_refuses_a_certificate_for_another_model():
    sys.path.insert(0, str(LANE / "campaign"))
    import finish_3c_certs as fin
    sweep = {("Z", 1): "a" * 64, ("Z", 2): "b" * 64}
    ok = json.dumps(dict(basis="Z", rounds=1, checker_exit=0, model_sha256="a" * 64,
                         checker="VERIFIED: no irreducible undetectable logical error of weight <= 10 whose earliest round is <= 1 in this DEM"))
    old = json.dumps(dict(basis="Z", rounds=2, checker_exit=0, model_sha256="b" * 64,
                          checker="VERIFIED: no undetectable logical error of weight <= 10 whose earliest round is <= 1 in this DEM"))
    assert set(fin.verified([ok, old], sweep)) == {("Z", 1), ("Z", 2)}          # both wordings count
    other = json.dumps(dict(json.loads(ok), model_sha256="c" * 64))
    with pytest.raises(SystemExit, match="STOP"):
        fin.verified([other], sweep)
