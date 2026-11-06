"""Per-shot IQ readout (SamplerV2 meas_type="kerneled") -> bits, in-circuit readout separation, and LEAKAGE.

Bits are what a classified job returns: IBM's discriminator has already thrown the IQ point away. Kept, the IQ point
says more (raw IQ is available on this account at no extra cost: evidence/iq_capability_probe.json):
  - each qubit's in-circuit assignment errors and the SEPARATION of its two clusters (d'), from the job's own shots;
  - readout drift across a job (the clusters move between the first and last circuits);
  - LEAKAGE out of the qubit subspace. |2> sits in its own cluster, which a two-state discriminator silently calls
    "1". No |2> reference can be prepared with the ISA gates, so a shot whose Mahalanobis distance from BOTH
    calibrated clusters exceeds D2_OUT is an OUTLIER, and a circuit's outlier fraction ABOVE the calibration circuits'
    own is the leakage-like excess. It is reported, never used to change a bit.

The discriminator per qubit, from shots prepared in |0> and in |1> (the job's readout circuits): linear discriminant
analysis on the (I, Q) plane -- w = S^-1 (mu1 - mu0), S the pooled covariance, threshold midway between the
projected means: the Bayes rule for two Gaussians of one covariance at equal priors. A |1> prepared and relaxed
during the read lands in the |0> cluster; that is assignment error, and it is counted as such (err10), as the
vendor's prob_meas0_prep1 counts it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

D2_OUT = 18.42           # chi2 quantile with 2 dof at p = 1e-4: a shot this far from BOTH clusters is an outlier


class IQRefused(ValueError):
    """IQ data or a calibration that cannot be read faithfully"""


@dataclass(frozen=True)
class Discriminator:
    mu0: complex
    mu1: complex
    cov: tuple                          # pooled 2x2 covariance, row-major (I, Q)
    w: tuple                            # LDA direction
    threshold: float                    # projection > threshold reads 1
    err01: float                        # P(read 1 | prepared 0), on the training shots
    err10: float                        # P(read 0 | prepared 1)
    d_prime: float                      # |projected mean difference| / projected pooled sd
    n0: int
    n1: int

    def to_json(self) -> dict:
        return dict(mu0=[self.mu0.real, self.mu0.imag], mu1=[self.mu1.real, self.mu1.imag], cov=list(self.cov),
                    w=list(self.w), threshold=self.threshold, err01=self.err01, err10=self.err10,
                    d_prime=self.d_prime, n0=self.n0, n1=self.n1)


def _xy(iq) -> np.ndarray:
    z = np.asarray(iq)
    if not np.iscomplexobj(z):
        raise IQRefused(f"IQ shots must be complex, got {z.dtype}")
    if not np.all(np.isfinite(z)):
        raise IQRefused("IQ shots carry a non-finite value")
    return np.stack([z.real, z.imag], axis=-1).astype(float)


def fit(iq0, iq1) -> Discriminator:
    """the discriminator from shots prepared in |0> (iq0) and |1> (iq1), each a 1-D complex array"""
    a, b = _xy(iq0).reshape(-1, 2), _xy(iq1).reshape(-1, 2)
    if len(a) < 10 or len(b) < 10:
        raise IQRefused(f"too few calibration shots ({len(a)}, {len(b)})")
    m0, m1 = a.mean(0), b.mean(0)
    S = (np.cov(a.T) * (len(a) - 1) + np.cov(b.T) * (len(b) - 1)) / (len(a) + len(b) - 2)
    if np.linalg.cond(S) > 1e12:
        raise IQRefused("degenerate IQ covariance: the clusters do not span the plane")
    w = np.linalg.solve(S, m1 - m0)
    p0, p1 = float(m0 @ w), float(m1 @ w)
    if not p1 > p0:
        raise IQRefused("the |1> cluster does not project above the |0> cluster")
    thr = (p0 + p1) / 2
    sd = float(np.sqrt(w @ S @ w))
    return Discriminator(mu0=complex(*m0), mu1=complex(*m1), cov=tuple(float(x) for x in S.ravel()),
                         w=(float(w[0]), float(w[1])), threshold=thr, err01=float(np.mean(a @ w > thr)),
                         err10=float(np.mean(b @ w <= thr)), d_prime=(p1 - p0) / sd, n0=len(a), n1=len(b))


def classify(d: Discriminator, iq) -> np.ndarray:
    """True where a shot reads 1"""
    return _xy(iq) @ np.array(d.w) > d.threshold


def outliers(d: Discriminator, iq) -> np.ndarray:
    """True where a shot is farther than D2_OUT (squared Mahalanobis) from BOTH calibrated clusters"""
    x = _xy(iq)
    Si = np.linalg.inv(np.array(d.cov).reshape(2, 2))
    out = np.ones(x.shape[:-1], dtype=bool)
    for mu in (d.mu0, d.mu1):
        r = x - np.array([mu.real, mu.imag])
        out &= np.einsum("...i,ij,...j->...", r, Si, r) > D2_OUT
    return out


def counts(bits: np.ndarray) -> dict:
    """(shots, n_clbits) booleans -> a qiskit-style histogram (clbit 0 the RIGHTMOST character)"""
    b = np.asarray(bits, dtype=bool)
    if b.ndim != 2:
        raise IQRefused(f"bits must be (shots, clbits), got shape {b.shape}")
    keys = ["".join("1" if v else "0" for v in row[::-1]) for row in b]
    out: dict = {}
    for k in keys:
        out[k] = out.get(k, 0) + 1
    return out


def measured_qubits(circuit) -> list:
    """the physical qubit each clbit records, in clbit order (refused if a clbit is written twice or never)"""
    owner: dict = {}
    for inst in circuit.data:
        if inst.operation.name == "measure":
            c = circuit.find_bit(inst.clbits[0]).index
            if c in owner:
                raise IQRefused(f"clbit {c} is written twice: a per-shot IQ column has one owner")
            owner[c] = circuit.find_bit(inst.qubits[0]).index
    if sorted(owner) != list(range(circuit.num_clbits)):
        raise IQRefused("a clbit is never measured")
    return [owner[c] for c in range(circuit.num_clbits)]


def calibrate(iq_by_name: dict, qubits_by_name: dict, prep0: str, prep1: str) -> dict:
    """physical qubit -> Discriminator, from the circuit `prep0` (every measured qubit in |0>) and `prep1` (|1>)"""
    q0, q1 = qubits_by_name[prep0], qubits_by_name[prep1]
    if q0 != q1:
        raise IQRefused("the two calibration circuits must read the same qubits in the same clbits")
    return {q: fit(iq_by_name[prep0][:, j], iq_by_name[prep1][:, j]) for j, q in enumerate(q0)}


def classify_all(iq_by_name: dict, qubits_by_name: dict, discs: dict) -> dict:
    """every circuit's histogram from its IQ, each column read by its own qubit's discriminator"""
    out = {}
    for name, iq in iq_by_name.items():
        qs = qubits_by_name[name]
        missing = [q for q in qs if q not in discs]
        if missing:
            raise IQRefused(f"{name}: no discriminator for qubits {missing}")
        bits = np.stack([classify(discs[q], iq[:, j]) for j, q in enumerate(qs)], axis=1)
        out[name] = counts(bits)
    return out


def frame_homogeneity(index: dict, iq_by_name: dict, qubits_by_name: dict, discs: dict) -> dict:
    """the frame-homogeneity map (oneq.hw.randomization.homogeneity_map, R178) of a KERNELED job: every shot
    classified by its own qubit's discriminator, tied to its frame by the execution spans, then the same exact chi2
    tests a bit job gets in its flight record. Each flagged clbit is also named by its PHYSICAL qubit."""
    from oneq.hw import randomization as RZ
    pub = {p["name"]: p["index"] for p in index["pubs"]}
    shots = {}
    for name, iq in iq_by_name.items():
        qs = qubits_by_name[name]
        missing = [q for q in qs if q not in discs]
        if missing:
            raise IQRefused(f"{name}: no discriminator for qubits {missing}")
        bits = np.stack([classify(discs[q], iq[:, j]) for j, q in enumerate(qs)], axis=1)
        R, n = RZ.shape(index["execution_spans"], pub[name], len(bits))
        shots[name] = RZ.from_bits(bits, R, n)
    out = RZ.homogeneity_map(shots)
    for s in out["strongest"]:
        if s["clbit"] != "hamming":
            s["qubit"] = int(qubits_by_name[s["circuit"]][s["clbit"]])
    return out


def outlier_fractions(iq_by_name: dict, qubits_by_name: dict, discs: dict) -> dict:
    """circuit -> {qubit: fraction of its shots that are outliers}"""
    return {name: {str(q): float(np.mean(outliers(discs[q], iq[:, j])))
                   for j, q in enumerate(qubits_by_name[name])}
            for name, iq in iq_by_name.items()}


def drift(iq_by_name: dict, qubits_by_name: dict, discs: dict, names_in_time_order: list) -> dict:
    """per qubit, the median projected position of each circuit's shots, in submission order, relative to the
    discriminator's threshold in units of the projected sd -- a readout that walks shows as a trend (the circuits'
    own states move it too, so compare circuits that prepare the same state)"""
    out: dict = {}
    for name in names_in_time_order:
        iq = iq_by_name[name]
        for j, q in enumerate(qubits_by_name[name]):
            d = discs[q]
            w = np.array(d.w)
            sd = float(np.sqrt(w @ np.array(d.cov).reshape(2, 2) @ w))
            proj = _xy(iq[:, j]) @ w
            out.setdefault(str(q), []).append(float((np.median(proj) - d.threshold) / sd))
    return out


def iq_columns(index: dict, arrays: dict, name: str) -> np.ndarray:
    """one circuit's per-shot IQ, (shots, clbits) complex, from a flight record written with meas_type kerneled"""
    pub = next((p for p in index["pubs"] if p["name"] == name), None)
    if pub is None:
        raise IQRefused(f"{name} is not in the flight record")
    regs = [r for r in pub["registers"].values() if r.get("kind") == "iq"]
    if len(regs) != 1:
        raise IQRefused(f"{name}: expected one IQ register, found {len(regs)}")
    a = arrays[regs[0]["key"]]
    if a.ndim != 3 or a.shape[-1] != 2:
        raise IQRefused(f"{name}: stored IQ must be (shots, clbits, 2), got {a.shape}")
    return a[..., 0].astype(float) + 1j * a[..., 1].astype(float)
