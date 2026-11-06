r"""Two uses beat one: the first physical points of the superactivation channel's F(n) curve.

N~ (oneq.superactivation) is the flagged qubit channel one joint use of PPT (x) 50%-erasure
simulates. Its finite-blocklength fidelity F(n) crosses 3/4 -- superactivation -- at n = 17
(arXiv 2604.27042; certified, ledger R132). No device can reach n = 17 (ledger R133). This
measures the curve's first two points on hardware, with the encoder, the channel and the
decoders executed physically and every flag branch in the reported figure:

  n = 1: identity code. F_ideal = 1/2 + (1-p)/4 = 0.646447.
  n = 2: the optimal symmetric code (an isometry into Sym^2, frozen in code_n2.json) and its exact
         branch decoders. F_ideal = 0.687039.

THE CHANNEL IS REALISED EXACTLY, NOT APPROXIMATED. Flag 0 is the identity. Flag 1 is "measure in
Z, then flip the bit with probability p": the flagged qubit is used only as a control and then
discarded (deferred measurement: tracing a control after controlled operations IS measuring it
first), and the flip is an X gate in a separate circuit of weight p. The flags are known to Bob,
so each flag pattern is its own set of circuits, weighted 1/4.

DECODERS. (0,0): the encoder's inverse on (A1, A2); output A1. (1,0)/(0,1): the controlled
unitary U_b on the unflagged qubit, control = the flagged qubit; output = the unflagged qubit.
(1,1): measure-and-prepare -- it has no quantum part, so its fidelity is computed from the
measurement record of R (three bases) and the two flag bits; the bit flips are applied to the
record with their exact probabilities (an exact expectation, not a sampled one).

  python hw_coding_gain.py simulate     noise-free density-matrix check of every circuit (no device)
  python hw_coding_gain.py validate     + layout health / transpile / quota (submits nothing)
  python hw_coding_gain.py submit --i-understand-this-spends-credits     (ONEQ_IBM_SUBMIT=1)
  python hw_coding_gain.py fetch
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[2]
CODE = ROOT / "evidence" / "superactivation" / "code_n2.json"
PREREG = ROOT / "docs" / "PREREG_superactivation_coding_gain_2026-10-12.md"
PREREG_SHA256 = "e6f39ea301a9f18043db6d4343a3001a2904e9db369a5e3c0ab0e45b04a28b4e"
RECORDS = ROOT / "evidence" / "superactivation"
BACKEND = "ibm_kingston"
LAYOUT = [149, 148, 147]          # [R, A1, A2] -- pinned in the preregistration
RANDOMIZATIONS = 32
SHOTS = 4000
P = math.sqrt(2.0) - 1.0
F1_IDEAL = 0.5 + (1 - P) / 4
MIN_QUOTA_S = 90
BASES3 = ("X", "Y", "Z")
PAIR_BASES = ("XX", "YY", "ZZ")


# ------------------------------------------------------------------ the frozen code
def load_code():
    c = json.loads(CODE.read_text(encoding="utf-8"))
    psi = np.array(c["psi_re"]) + 1j * np.array(c["psi_im"])             # psi[r] in comp (A1 A2)
    U = {}
    for b in ("0", "1"):
        vec = np.array(c["k1"][b]["M_re"]) + 1j * np.array(c["k1"][b]["M_im"])   # vec[r, i]
        # U = conj(vec) (vec[r, :] = U^dag|r> for the adjoint-decoder Choi |v><v|), then projected
        # onto the NEAREST unitary (polar factor): the SDP gives unitarity to ~1e-9 and the
        # transpiler's Weyl decomposition refuses anything not unitary to machine precision.
        Wsvd, _, Vh = np.linalg.svd(np.conj(vec))
        U[int(b)] = Wsvd @ Vh
    G = {int(b): np.array(c["k2"][b]["G_re"]) + 1j * np.array(c["k2"][b]["G_im"]) for b in ("0", "1", "2")}
    return psi, U, G, c


def _u3(t, f, lam):
    return np.array([[math.cos(t / 2), -np.exp(1j * lam) * math.sin(t / 2)],
                     [np.exp(1j * f) * math.sin(t / 2), np.exp(1j * (f + lam)) * math.cos(t / 2)]])


def _ansatz(x):
    """[u3 x u3] CX(A1->A2) [u3 x u3] CX(A1->A2) [u3 x u3] on (A1, A2), index 2 a1 + a2."""
    CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    L = [np.kron(_u3(*x[6 * i:6 * i + 3]), _u3(*x[6 * i + 3:6 * i + 6])) for i in range(3)]
    return L[2] @ CX @ L[1] @ CX @ L[0]


def encoder_2cx(psi, seed: int = 7, tol: float = 1e-7):
    """A 2-CNOT circuit W with W|r,0> = psi_r (shared global phase), found by fitting.

    The generic unitary completion of this isometry transpiles to 3 CZ; a 1 -> 2 qubit isometry
    needs only 2 because the second input is fixed to |0>. The codewords are first made exactly
    orthonormal (polar factor; the SDP left them orthonormal to ~2e-9). The fit's loss is
    quadratic in the amplitude residual, so double precision floors it near 1e-8; the guard is
    1e-7 (a fidelity effect ~1e-14), and the circuit-level simulation re-checks F end to end."""
    from qiskit import QuantumCircuit
    from scipy.optimize import minimize
    Wsvd, _, Vh = np.linalg.svd(np.stack([psi[0], psi[1]], axis=1), full_matrices=False)
    target = Wsvd @ Vh                                               # 4 x 2, rows 2 a1 + a2

    from scipy.optimize import least_squares

    def resid_vec(y):
        """Amplitude residuals of W[:, inputs |0,0>, |1,0>] against e^{i theta} target --
        LINEAR in the error, so Levenberg-Marquardt reaches machine precision when an exact
        2-CNOT solution exists (a fidelity-style loss is quadratic and floors near 1e-8)."""
        W = _ansatz(y[:18])
        dlt = W[:, [0, 2]] - np.exp(1j * y[18]) * target
        return np.concatenate([dlt.real.ravel(), dlt.imag.ravel()])

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(60):
        r = least_squares(resid_vec, np.concatenate([rng.uniform(-math.pi, math.pi, 18), [0.0]]),
                          method="trf", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=20000)
        if best is None or r.cost < best.cost:
            best = r
        if np.max(np.abs(best.fun)) < tol * 1e-3:
            break
    del minimize
    W = _ansatz(best.x[:18])
    cols = W[:, [0, 2]]
    ph = np.trace(target.conj().T @ cols) / abs(np.trace(target.conj().T @ cols))
    resid = float(np.max(np.abs(cols / ph - target)))
    if resid > tol:
        raise RuntimeError(f"2-CNOT encoder fit failed: residual {resid:.2e}")
    x = best.x
    qc = QuantumCircuit(2, name="enc2cx")                            # qubit 0 = A1, qubit 1 = A2
    for i in range(3):
        qc.u(*x[6 * i:6 * i + 3], 0)
        qc.u(*x[6 * i + 3:6 * i + 6], 1)
        if i < 2:
            qc.cx(0, 1)
    qc.global_phase = -float(np.angle(ph))
    return qc.to_gate(label="enc2cx")


# ------------------------------------------------------------------ circuits
def circuits():
    """[(name, circuit, weight-tag)] for both n. Qubits: 0 = R, 1 = A1, 2 = A2."""
    from qiskit import QuantumCircuit
    psi, U, G, _ = load_code()
    out = []

    def meas(qc, qubits, bases):
        for q, b in zip(qubits, bases):
            if b == "X":
                qc.h(q)
            elif b == "Y":
                qc.sdg(q)
                qc.h(q)
        qc.measure(qubits, list(range(len(qubits))))

    # ---- n = 1 (qubits R, A1 only)
    for b in PAIR_BASES:
        qc = QuantumCircuit(3, 2, name=f"n1_f0_{b}")
        qc.h(0)
        qc.cx(0, 1)
        qc.barrier()
        meas(qc, [0, 1], b)
        out.append((f"n1_f0_{b}", qc))
    qc = QuantumCircuit(3, 2, name="n1_f1_ZZ")
    qc.h(0)
    qc.cx(0, 1)
    qc.barrier()
    meas(qc, [0, 1], "ZZ")
    out.append(("n1_f1_ZZ", qc))

    # ---- n = 2
    # THE ENCODER AS AN ISOMETRY, NOT A COMPLETED UNITARY: a 1 -> 2 qubit isometry needs at most
    # 2 CZ, while an arbitrary unitary completion of it transpiled to 3. Qubit list [A1, A2]: A1 is
    # the input (least significant), so the row index is a1 + 2 a2; psi is stored as 2 a1 + a2.
    enc_gate = encoder_2cx(psi)
    dec_gate = enc_gate.inverse()

    def encoded(name, ncl):
        qc = QuantumCircuit(3, ncl, name=name)
        qc.h(0)
        qc.cx(0, 1)                                   # Bell pair (R, A1); A1 carries the logical input
        qc.barrier()
        qc.append(enc_gate, [1, 2])
        qc.barrier()
        return qc

    for b in PAIR_BASES:                              # flags (0,0): invert the encoder, output A1
        qc = encoded(f"n2_f00_{b}", 2)
        qc.append(dec_gate, [1, 2])
        qc.barrier()
        meas(qc, [0, 1], b)
        out.append((f"n2_f00_{b}", qc))
    CU = np.zeros((4, 4), dtype=complex)              # control = more significant qubit
    CU[:2, :2], CU[2:, 2:] = U[0], U[1]
    for flagged, unflag in ((1, 2), (2, 1)):           # flags (1,0) and (0,1)
        tag = "f10" if flagged == 1 else "f01"
        for flip in (0, 1):
            for b in PAIR_BASES:
                qc = encoded(f"n2_{tag}_x{flip}_{b}", 2)
                if flip:
                    qc.x(flagged)
                qc.unitary(CU, [unflag, flagged], label=f"dec{tag}")
                qc.barrier()
                meas(qc, [0, unflag], b)
                out.append((f"n2_{tag}_x{flip}_{b}", qc))
    for rb in BASES3:                                  # flags (1,1): measure-and-prepare from the record
        qc = encoded(f"n2_f11_{rb}", 3)
        meas(qc, [0, 1, 2], rb + "ZZ")
        out.append((f"n2_f11_{rb}", qc))
    return out


# ------------------------------------------------------------------ analysis (probabilities in, numbers out)
def _norm(counts):
    n = sum(counts.values())
    return {k.replace(" ", ""): v / n for k, v in counts.items()}, n


def _corr(probs):
    """<P x P> parity of the two measured bits (keys 'c1c0')."""
    return sum(v * (1 if (int(k[-1]) ^ int(k[-2])) == 0 else -1) for k, v in probs.items())


def bell_F(c3):
    (pxx, nx), (pyy, ny), (pzz, nz) = (_norm(c3[b]) for b in PAIR_BASES)
    xx, yy, zz = _corr(pxx), _corr(pyy), _corr(pzz)
    F = (1 + xx - yy + zz) / 4
    se = math.sqrt((1 - xx ** 2) / nx + (1 - yy ** 2) / ny + (1 - zz ** 2) / nz) / 4
    return F, se


def analyse_n1(c):
    F0, s0 = bell_F({b: c[f"n1_f0_{b}"] for b in PAIR_BASES})
    probs, n = _norm(c["n1_f1_ZZ"])
    # flag 1: Bob reads b = a XOR flip; the optimal decoder outputs |b>; fidelity (1/2) Pr(R == b)
    agree = sum(v for k, v in probs.items() if int(k[-1]) == int(k[-2]))
    F1 = 0.5 * ((1 - P) * agree + P * (1 - agree))
    s1 = 0.5 * abs(1 - 2 * P) * math.sqrt(agree * (1 - agree) / n)
    return {"F": 0.5 * F0 + 0.5 * F1, "se": 0.5 * math.hypot(s0, s1), "F_flag0": F0, "F_flag1": F1}


def analyse_n2(c, G):
    """Branch fidelities and F. (1,1) branch: Bob's data is the observed weight b of the two
    flagged bits after independent flips; the measure-and-prepare decoder G_b (adjoint-decoder
    Choi on R, = phi_b^T) scores Tr(Gamma_b G_b)/4 with Gamma_b = 2 * (R's conditional state,
    unnormalized, summed over true strings z with Pr(b|z))."""
    F00, s00 = bell_F({b: c[f"n2_f00_{b}"] for b in PAIR_BASES})
    f1 = {}
    for tag in ("f10", "f01"):
        F, var = 0.0, 0.0
        for flip, w in ((0, 1 - P), (1, P)):
            Fx, sx = bell_F({b: c[f"n2_{tag}_x{flip}_{b}"] for b in PAIR_BASES})
            F += w * Fx
            var += (w * sx) ** 2
        f1[tag] = (F, math.sqrt(var))
    # (1,1): R's Bloch components conditioned on the true flag string z = (z1, z2)
    stats = {}
    for rb in BASES3:
        probs, n = _norm(c[f"n2_f11_{rb}"])
        stats[rb] = (probs, n)
    sigma = {}
    for z in itertools.product((0, 1), repeat=2):
        pz_by_basis = []
        comp = {}
        for rb in BASES3:
            probs, n = stats[rb]
            pz = sum(v for k, v in probs.items() if int(k[-2]) == z[0] and int(k[-3]) == z[1])
            pr = sum(v * (1 if int(k[-1]) == 0 else -1) for k, v in probs.items()
                     if int(k[-2]) == z[0] and int(k[-3]) == z[1])
            comp[rb] = pr
            pz_by_basis.append(pz)
        pzm = float(np.mean(pz_by_basis))
        I2 = np.eye(2)
        sx = np.array([[0, 1], [1, 0]])
        sy = np.array([[0, -1j], [1j, 0]])
        sz = np.diag([1.0, -1.0])
        sigma[z] = 0.5 * (pzm * I2 + comp["X"] * sx + comp["Y"] * sy + comp["Z"] * sz)   # unnormalized
    F11 = 0.0
    for b in range(3):
        Gam = np.zeros((2, 2), dtype=complex)
        for z, s in sigma.items():
            pb = 0.0
            for f in itertools.product((0, 1), repeat=2):
                if (z[0] ^ f[0]) + (z[1] ^ f[1]) == b:
                    pb += math.prod(P if fi else 1 - P for fi in f)
            Gam += 2 * pb * s
        F11 += float(np.real(np.trace(Gam @ G[b]))) / 4
    F = 0.25 * (F00 + f1["f10"][0] + f1["f01"][0] + F11)
    se = 0.25 * math.sqrt(s00 ** 2 + f1["f10"][1] ** 2 + f1["f01"][1] ** 2)   # F11 se below
    return {"F": F, "se_excluding_f11": se, "F00": F00, "F10": f1["f10"][0], "F01": f1["f01"][0],
            "F11": F11}


def analyse(counts):
    _, _, G, code = load_code()
    n1 = analyse_n1(counts)
    n2 = analyse_n2(counts, G)
    gain = n2["F"] - n1["F"]
    se_gain = math.hypot(n1["se"], n2["se_excluding_f11"])
    return {"n1": n1, "n2": n2, "gain": gain, "gain_se_approx": se_gain,
            "ideal": {"n1": F1_IDEAL, "n2": code["F_reduced"], "gain": code["F_reduced"] - F1_IDEAL}}


# ------------------------------------------------------------------ noise-free simulation (no device)
def simulate_counts(shots=None):
    """Exact outcome probabilities of every circuit (qiskit DensityMatrix), as pseudo-counts."""
    from qiskit.quantum_info import DensityMatrix
    out = {}
    for name, qc in circuits():
        q = qc.remove_final_measurements(inplace=False)
        measured = [qc.find_bit(ins.qubits[0]).index for ins in qc.data if ins.operation.name == "measure"]
        rho = DensityMatrix(q)
        probs = rho.probabilities_dict(qargs=measured)
        # probabilities_dict keys are little-endian over qargs: 'b_last...b_first' -> same as 'c_{k}..c0'
        out[name] = {k: v * (shots or 10 ** 9) for k, v in probs.items()}
    return out


def prereg_ok() -> bool:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest() == PREREG_SHA256


def _device():
    from qiskit_ibm_runtime import QiskitRuntimeService
    svc = QiskitRuntimeService()
    backend = svc.backend(BACKEND)
    t = backend.target
    bad = []
    for q in LAYOUT:
        e = t["measure"][(q,)].error
        if e is None or e > 0.05:
            bad.append(f"qubit {q} readout {e}")
    for a, b in ((LAYOUT[0], LAYOUT[1]), (LAYOUT[1], LAYOUT[2])):
        g = t["cz"].get((a, b)) or t["cz"].get((b, a))
        if g is None or g.error is None or g.error > 0.01:
            bad.append(f"edge {a}-{b} CZ {None if g is None else g.error}")
    u = svc.usage()
    remaining = (u.get("usage_limit_seconds") or 0) - (u.get("usage_consumed_seconds") or 0)
    return svc, backend, bad, remaining


def _isa(backend):
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend, initial_layout=LAYOUT)
    named = circuits()
    return [n for n, _ in named], [pm.run(qc) for _, qc in named]


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "simulate"
    if mode == "simulate":
        res = analyse(simulate_counts())
        print(json.dumps(res, indent=2))
        ok = (abs(res["n1"]["F"] - F1_IDEAL) < 1e-9 and abs(res["n2"]["F"] - res["ideal"]["n2"]) < 1e-7)
        print("SIMULATION REPRODUCES THE IDEAL FIDELITIES:", ok)
        return 0 if ok else 1
    if mode == "fetch":
        rec = json.loads(sorted(RECORDS.glob("coding_gain_*.job.json"))[-1].read_text(encoding="utf-8"))
        from qiskit_ibm_runtime import QiskitRuntimeService
        job = QiskitRuntimeService().job(rec["job_id"])
        if "DONE" not in str(job.status()):
            print(f"job {rec['job_id']}: {job.status()}")
            return 4
        result = job.result()
        counts = {nm: result[i].data.c.get_counts() for i, nm in enumerate(rec["circuit_names"])}
        spans = result.metadata.get("execution", {}).get("execution_spans")
        res = analyse(counts)
        out = {"job": rec, "counts": counts, "analysis": res,
               "execution_seconds": (spans.stop - spans.start).total_seconds() if spans else None}
        dest = RECORDS / f"coding_gain_{rec['job_id']}.result.json"
        dest.write_bytes((json.dumps(out, indent=2) + "\n").encode("utf-8"))
        print(json.dumps(res, indent=2))
        print(f"-> {dest}")
        return 0
    svc, backend, bad, remaining = _device()
    names, isa = _isa(backend)
    ncz = [sum(1 for ins in c.data if ins.operation.name == "cz") for c in isa]
    print(f"prereg hash matches: {prereg_ok()} | layout {LAYOUT}: {'OK' if not bad else bad} | "
          f"quota remaining {remaining} s")
    print(f"{len(isa)} circuits x {SHOTS} shots; CZ per circuit: {dict(zip(names, ncz))}")
    if mode == "validate":
        return 0 if (prereg_ok() and not bad) else 3
    if mode != "submit":
        print(__doc__)
        return 2
    if not ("--i-understand-this-spends-credits" in sys.argv and os.environ.get("ONEQ_IBM_SUBMIT") == "1"):
        print("REFUSED: needs --i-understand-this-spends-credits AND ONEQ_IBM_SUBMIT=1")
        return 3
    if not prereg_ok() or bad or remaining < MIN_QUOTA_S:
        print("REFUSED: prereg hash, layout health or quota")
        return 3
    from qiskit_ibm_runtime import SamplerV2
    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = SHOTS
    sampler.options.twirling.enable_gates = True
    sampler.options.twirling.enable_measure = False
    sampler.options.twirling.num_randomizations = RANDOMIZATIONS
    sampler.options.dynamical_decoupling.enable = False
    job = sampler.run(isa)
    rec = {"prereg": PREREG.name, "prereg_sha256": PREREG_SHA256, "backend": backend.name,
           "layout": LAYOUT, "shots": SHOTS, "twirling": {"gates": True, "num_randomizations": RANDOMIZATIONS},
           "circuit_names": names, "cz_per_circuit": ncz, "job_id": job.job_id(),
           "code_sha256": hashlib.sha256(CODE.read_bytes()).hexdigest(),
           "submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "quota_remaining_before_s": remaining}
    p = RECORDS / f"coding_gain_{job.job_id()}.job.json"
    p.write_bytes((json.dumps(rec, indent=2) + "\n").encode("utf-8"))
    print(f"submitted {len(isa)} circuits: job {job.job_id()} -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
