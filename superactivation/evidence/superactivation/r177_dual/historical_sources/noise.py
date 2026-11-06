"""The calibrated error model the engine samples from: a scheduled ISA circuit becomes a list of operations, each with
the Pauli errors that can follow it.

  CZ on edge (a, b)    a uniformly random non-identity two-qubit Pauli with probability
                       p = cz_factor x (5/4) x error(a, b). The vendor number is an AVERAGE gate infidelity r; a uniform
                       15-Pauli channel of probability p has r = 4p/5, so cz_factor = 1 reproduces the vendor exactly.
                       cz_factor is MEASURED, not assumed (CZ_FACTOR_R133 below): twirled CZs on ibm_kingston 148-149
                       decayed the Bell depolarizing parameter by eps = 0.6333% per CZ (job db6pne04qg6s73c561s0); a
                       uniform channel decays it by 16p/15, so p = 15 eps / 16 = 0.594%, against the calibration's
                       r = 0.207% -> p = 0.259%: factor 2.29. (Dividing eps by r directly gives 3.04, a ratio of two
                       different units.) Twirling every CZ server-side is what makes a stochastic Pauli model right:
                       the untwirled 4.6% was coherent accumulation.
                       STRUCTURE (R146): a snapshot may carry each edge's full Pauli-channel SHAPE -- 15 fractions,
                       DeviceSnapshot.cz_structure -- measured in-circuit by oneq.hw.cz_channel (the 9 learnable
                       quantities of a twirled CZ, symmetric gauge). Any Pauli channel of total p has r = 4p/5, so the
                       shape never changes the vendor-units relation. It does change what a Bell ladder reads: its
                       lambda is blind to XX, YY and ZZ errors (each commutes with every Bell stabilizer) and weighs the
                       rest unequally, so R133's 15 eps / 16 is not a uniform-channel rate.
  sx / x on qubit q    a random Pauli with probability sx_factor x 1.5 x error(q) (r = 2p/3 for a qubit)
  rz                   virtual: no error
  idle (delay t)       Pauli-twirled thermal relaxation: px = py = (1 - e^{-t/T1})/4, pz = (1 - e^{-t/T2})/2 - px,
                       with T2 clamped to 2 T1. This is the idle term whose omission once overstated a margin by ~11
                       sigma (R138). The schedule (NoiseParams.schedule; ALAP by default) uses the snapshot's own
                       durations, so every gap is counted -- except the wait BEFORE a qubit's first gate (it sits in
                       |0>, which neither decay nor dephasing moves, while the twirl would flip it with probability
                       a/2) and any wait AFTER its readout.
  quasi-static         (only where the snapshot carries DeviceSnapshot.sigma, R155) a detuning delta ~ N(0, sigma^2), fixed
                       within a shot, turns a qubit's phase by delta x (signed time) over each WINDOW it spends in
                       superposition. Z commutes with rz, cz and idles, so a window runs through them and closes only at
                       a gate that does not commute with Z: an sx. An x anticommutes with Z, so it flips the sign of what
                       follows -- an echo, exact. At each closing sx the window's phase becomes one Pauli-twirled Z error
                       of probability (1 - exp(-sigma^2 T^2 / 2)) / 2, T the window's signed time; a window still open at
                       the readout is dropped (a Z before a Z measurement changes nothing). The windows of one shot share
                       delta and are combined incoherently: exact for one window, a second-order approximation across an
                       sx. NoiseParams.quasistatic picks how a twirled CZ is read (QUASISTATIC). t2 is then the WHITE T2.
  coherent detuning    (only where the snapshot carries DeviceSnapshot.detuning, R161) a FIXED frequency offset Delta:
                       over each phase window (the same windows; phase_windows) the qubit turns by rz(Delta T), a
                       UNITARY placed before the sx that closes the window -- exact, not a Pauli: an x inverts what
                       came before it (an echo), a window open at the readout is dropped. q149 on ibm_kingston turned
                       at +49.7 kHz with contrast 0.984 (R161): what every earlier model read as a 3-5 us T2.
                       oneq.hw.frame cancels it with virtual rz.
  readout              asymmetric (p01, p10) per measured qubit, applied exactly by the engine, not sampled.
Everything is a function of the snapshot and NoiseParams, so a prediction names its model in full.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .snapshot import DeviceSnapshot

VIRTUAL = ("rz", "barrier", "id")
ISA = ("cz", "sx", "x", "rz", "id", "measure", "barrier", "delay")
PAULIS2 = tuple((a, b) for a in "IXYZ" for b in "IXYZ" if (a, b) != ("I", "I"))   # the engine's two-qubit order


def cz_pauli_probs(p: float, fractions: tuple | None) -> tuple:
    """the CZ error channel as 15 probabilities in PAULIS2 order: p x the edge's measured fractions. () when it has
    none: the engine's uniform fast path, so an unstructured snapshot predicts exactly what it did before structure
    existed."""
    if fractions is None:
        return ()
    return tuple(p * f for f in fractions)


class ScheduleError(ValueError):
    """the circuit cannot be scheduled or simulated faithfully (mid-circuit measurement, unknown gate, ...)"""


EPS_CZ_TWIRLED_R133 = 0.006332518678653698   # evidence/superactivation/twirl_db6pne04qg6s73c561s0.result.json
R_CZ_VENDOR_R133 = 2.07e-3                    # docs/PREREG_superactivation_hw_floor_2026-10-12.md, pair 148-149
CZ_FACTOR_R133 = (15 / 16 * EPS_CZ_TWIRLED_R133) / (5 / 4 * R_CZ_VENDOR_R133)


# How the DEVICE times a circuit given without delays -- which decides which waits exist, and so what the idles cost.
#   alap                  every instruction as late as possible; terminal readout aligned at the end (the default,
#                         and every prediction made before R146)
#   asap                  every instruction as soon as its qubits are free, readout included: a qubit is read the moment
#                         its last gate ends, and waits only before it
#   asap_aligned_readout  gates as soon as possible, terminal readout aligned at the end
# R146's timing circuits (shield_family/hw_diag_run.py) measure which one ibm_kingston runs.
SCHEDULES = ("alap", "asap", "asap_aligned_readout")
# How the snapshot's always-on crosstalk (DeviceSnapshot.zz) accumulates over a shared idle (R153):
#   coherent    the conditional phase adds UP while only diagonal operations (rz, idles) act on the pair -- they
#               commute with it -- and a stretch ends at any sx / x (which do not commute) and at any CZ touching
#               the pair (twirled, which randomizes the frame); one twirled CPhase error per stretch, placed
#               immediately before the gate that ends it. An approximation: the phases on either side of an sx
#               are combined incoherently
#   incoherent  a stretch ends at EVERY gate on either qubit, rz included (a lower bound on the coherent build-up)
#   off         ignored
CROSSTALK = ("coherent", "incoherent", "off")
# How a TWIRLED CZ treats a quasi-static window (DeviceSnapshot.sigma) on its qubits:
#   coherent  the window runs through the CZ. The twirl's frame P before a CZ and P' = CZ P CZ after it carry the same
#             X part on each qubit, so a diagonal phase accumulated on either side commutes through P' CZ P = CZ and
#             adds coherently -- exact when P and P' sit at the CZ
#   layer     the window closes at every CZ on the qubit: the reading in which the server merges P and P' into the
#             neighbouring single-qubit gates, so a physical X or Y lands at a merge point and the sign of the phase
#             beyond it is random (second moments add). Which one the device runs is an empirical question
#   off       sigma ignored
QUASISTATIC = ("coherent", "layer", "off")
# Which qubits the server's gate TWIRLING dresses at each layer of CZs (qiskit-ibm-runtime TwirlingOptions.strategy;
# the runtime's default is "active-accum"), for the DRESSING term: a qubit twirled while IDLE in a layer gets a random
# Pauli P before the layer and the same P after it -- two physical pulses whenever P is X or Y, so one pulse's error
# on average (a ladder cannot see this: every one of its qubits is in every layer). "off" models none (every
# prediction before 2026-10-17); "active" twirls only the layer's own qubits, so it dresses no idle qubit either.
TWIRL = ("off", "active", "active-accum", "active-circuit", "all")


@dataclass(frozen=True)
class NoiseParams:
    cz_factor: float = CZ_FACTOR_R133   # measured in-circuit twirled CZ Pauli rate / the vendor's, same units
    sx_factor: float = 1.0
    idle: bool = True
    readout: bool = True
    idle_factor: float = 1.0            # scenario knobs (a what-if device): every idle Pauli probability, and
    readout_factor: float = 1.0         # every p01 / p10, multiplied by this
    schedule: str = "alap"              # one of SCHEDULES
    crosstalk: str = "coherent"         # one of CROSSTALK; acts only where the snapshot carries zz
    spectator: float = 0.0              # R153: Z-error probability on every coupled NEIGHBOUR of a CZ's two qubits that
                                        # is out of |0> and not yet read while the CZ runs (a phase a CZ pulse leaves on
                                        # its spectators; a CZ ladder with its neighbours parked in |0> cannot see it)
    quasistatic: str = "coherent"       # one of QUASISTATIC; acts only where the snapshot carries sigma (R155)
    detuning: bool = True               # the snapshot's COHERENT fixed detunings (R161), as rotations; off = a what-if
    twirl: str = "off"                  # one of TWIRL: the server's twirl dressing of IDLE qubits, per CZ layer
    dressing_factor: float = 1.0        # the dressing's pulse error per idle layer in units of ONE pulse (1.5 x sx
                                        # error), scaling the dressing ALONE (sx_factor scales every sx). R165 measured
                                        # 1.5-1.75 once the engine's own idle T1 (exact under a random frame) is counted
    twirl_template: bool = False        # R166 (MEASURED from the executed schedule): a twirling server plays every
                                        # single-qubit run as ONE two-sx template, and gives every qubit of a real
                                        # twirled CZ a slot before it and after its last one (template_resynthesis)
    twirl_template_dress: bool = False  # with twirl_template, active-accum's dressing AS PULSES: every touched idle
                                        # qubit gets a template slot before each CZ layer and after the last, its own
                                        # neighbouring gates absorbed (R166: 9 slots over 8 layers for it AND ip)

    def __post_init__(self) -> None:
        if self.schedule not in SCHEDULES:
            raise ScheduleError(f"unknown schedule {self.schedule!r}; expected one of {SCHEDULES}")
        if self.crosstalk not in CROSSTALK:
            raise ScheduleError(f"unknown crosstalk mode {self.crosstalk!r}; expected one of {CROSSTALK}")
        if self.quasistatic not in QUASISTATIC:
            raise ScheduleError(f"unknown quasi-static mode {self.quasistatic!r}; expected one of {QUASISTATIC}")
        if self.twirl not in TWIRL:
            raise ScheduleError(f"unknown twirl strategy {self.twirl!r}; expected one of {TWIRL}")
        if self.twirl_template_dress and not self.twirl_template:
            raise ScheduleError("twirl_template_dress is the template's own dressing: it needs twirl_template")
        if not (0.0 <= self.spectator <= 0.5):
            raise ScheduleError(f"spectator Z probability must be in [0, 1/2]: {self.spectator}")
        if not (math.isfinite(self.dressing_factor) and self.dressing_factor >= 0.0):
            raise ScheduleError(f"dressing_factor must be finite and >= 0: {self.dressing_factor}")

    def describe(self) -> dict:
        d = dict(cz_factor=self.cz_factor, sx_factor=self.sx_factor, idle=self.idle, readout=self.readout,
                 idle_factor=self.idle_factor, readout_factor=self.readout_factor, schedule=self.schedule)
        if self.crosstalk != "coherent":                 # the defaults are left out: earlier records keep their bytes
            d["crosstalk"] = self.crosstalk
        if self.spectator:
            d["spectator"] = self.spectator
        if self.quasistatic != "coherent":
            d["quasistatic"] = self.quasistatic
        if not self.detuning:
            d["detuning"] = False
        if self.twirl != "off":
            d["twirl"] = self.twirl
        if self.dressing_factor != 1.0:
            d["dressing_factor"] = self.dressing_factor
        if self.twirl_template:
            d["twirl_template"] = True
        if self.twirl_template_dress:
            d["twirl_template_dress"] = True
        return d

    def scaled(self, lam: float) -> "NoiseParams":
        """every error source multiplied by lam: the device lam times as noisy (on the same schedule)"""
        return NoiseParams(cz_factor=self.cz_factor * lam, sx_factor=self.sx_factor * lam, idle=self.idle,
                           readout=self.readout, idle_factor=self.idle_factor * lam,
                           readout_factor=self.readout_factor * lam, schedule=self.schedule,
                           crosstalk=self.crosstalk, spectator=min(0.5, self.spectator * lam),
                           quasistatic=self.quasistatic, detuning=self.detuning, twirl=self.twirl,
                           dressing_factor=self.dressing_factor,     # a pulse COUNT: sx_factor carries the lam
                           twirl_template=self.twirl_template, twirl_template_dress=self.twirl_template_dress)


NOISELESS = NoiseParams(cz_factor=0.0, sx_factor=0.0, idle=False, readout=False)


@dataclass(frozen=True)
class Op:
    """one operation of the scheduled circuit on ACTIVE-qubit indices, with the error that may follow it"""
    name: str
    qubits: tuple                       # active indices
    matrix: np.ndarray | None           # unitary (None for an idle)
    p_error: float                      # total probability that a Pauli error follows
    pauli_probs: tuple                  # 1q: (px, py, pz) summing to p_error; 2q: () = uniform over the 15, or 15
                                        # probabilities in PAULIS2 order summing to p_error (cz_pauli_probs)


def rz_matrix(theta: float) -> np.ndarray:
    """qiskit's rz(theta): diag(e^{-i theta/2}, e^{+i theta/2})"""
    return np.array([[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]], dtype=complex)


def phase_windows(sched, snap: DeviceSnapshot, qubits: set, close_at_cz: bool) -> dict:
    """where each qubit's phase window closes: instruction index k -> [(qubit, T)], T the window's signed time in the
    frame of the instruction it closes before. A window opens when the qubit leaves |0> (its first sx or x), runs
    through every operation that commutes with Z (rz, cz, delays, id, barriers) and closes BEFORE an sx (and before a
    cz when close_at_cz). An x carries what came before it through into its own frame INVERTED (X rz(a) = rz(-a) X),
    so an x in the middle of a wait echoes it -- for a coherent rotation the sign matters, for a quasi-static spread
    only T^2 does. A window still open at the readout is dropped (a Z rotation before a Z measurement changes
    nothing). Times are the scheduled circuit's own, from the snapshot's durations, exactly as compile_ops clocks."""
    out: dict = {}
    clock: dict = {}
    win: dict = {}                                     # qubit -> [signed time, time of the last update]
    done: set = set()
    for k, inst in enumerate(sched.data):
        name = inst.operation.name
        phys = [sched.find_bit(x).index for x in inst.qubits]
        t0 = max((clock.get(q, 0.0) for q in phys), default=0.0)
        dur = _op_duration(inst, phys, snap)
        for q in phys:
            clock[q] = t0 + dur
        for q in phys:
            if q not in qubits or q in done:
                continue
            w = win.get(q)
            if w is None:
                if name in ("sx", "x"):
                    win[q] = [0.0, t0 + dur]
                continue
            w[0] += t0 - w[1]                          # an unpadded gap is time too
            if name == "measure":
                done.add(q)
            elif name == "sx" or (close_at_cz and name == "cz"):
                out.setdefault(k, []).append((q, w[0]))
                w[:] = [0.0, t0 + dur]
            elif name == "x":                          # its halves cancel; the past comes through inverted
                w[:] = [-w[0], t0 + dur]
            else:                                      # rz, cz, delay, id, barrier commute with Z: the window runs on
                w[0] += dur
                w[1] = t0 + dur
    return out


def quasistatic_z(sigma: float, window: float) -> float:
    """the Pauli-twirled Z probability a detuning delta ~ N(0, sigma^2) leaves after a window of signed time T:
    E[sin^2(delta T / 2)] = (1 - exp(-sigma^2 T^2 / 2)) / 2"""
    return (1.0 - math.exp(-0.5 * (sigma * window) ** 2)) / 2.0


def idle_pauli(t: float, t1: float, t2: float) -> tuple[float, float, float]:
    """Pauli-twirled amplitude damping + dephasing over an idle of t seconds"""
    if t <= 0:
        return (0.0, 0.0, 0.0)
    t2 = min(t2, 2 * t1)
    a = 1 - math.exp(-t / t1)
    px = py = a / 4
    pz = max(0.0, (1 - math.exp(-t / t2)) / 2 - a / 4)
    return (px, py, pz)


_UNIT = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12}


def _seconds(duration, unit: str, dt: float) -> float:
    if unit == "dt":
        return float(duration) * dt
    if unit not in _UNIT:
        raise ScheduleError(f"unknown duration unit {unit!r}")
    return float(duration) * _UNIT[unit]


def durations(snap: DeviceSnapshot):
    """qiskit InstructionDurations from the snapshot (seconds); rz is virtual (0)"""
    from qiskit.transpiler import InstructionDurations
    items: list[tuple[str, tuple[int, ...], float, str]] = []
    for (a, b), (_, dur) in snap.cz.items():
        items += [("cz", (a, b), dur, "s"), ("cz", (b, a), dur, "s")]
    for q in range(snap.num_qubits):
        sxd = snap.sx_duration[q] if math.isfinite(snap.sx_duration[q]) else 0.0
        md = snap.meas_duration[q] if math.isfinite(snap.meas_duration[q]) else 0.0
        items += [("sx", (q,), sxd, "s"), ("x", (q,), sxd, "s"), ("id", (q,), sxd, "s"), ("rz", (q,), 0.0, "s"),
                  ("measure", (q,), md, "s")]
    return InstructionDurations(items, dt=snap.dt)


def duration_s(qc, snap: DeviceSnapshot) -> float:
    """critical-path duration of an ISA circuit in seconds (what one shot occupies the device beyond reset and
    rep delay), from the snapshot's own durations: each instruction starts when all its qubits are free"""
    free: dict = {}
    for inst in qc.data:
        name = inst.operation.name
        qs = [qc.find_bit(x).index for x in inst.qubits]
        start = max((free.get(q, 0.0) for q in qs), default=0.0)
        if name in ("barrier", "rz"):
            dur = 0.0
        elif name == "cz":
            dur = snap.edge(*qs)[1]
        elif name in ("sx", "x", "id"):
            dur = snap.sx_duration[qs[0]]
        elif name == "measure":
            dur = snap.meas_duration[qs[0]]
        elif name == "delay":
            dur = _seconds(inst.operation.duration, inst.operation.unit, snap.dt)
        else:
            raise ScheduleError(f"gate {name} is not in the ISA basis")
        for q in qs:
            free[q] = start + dur
    return max(free.values(), default=0.0)


def _readout_aligned(qc):
    """the same circuit with one barrier over every qubit before its terminal readout (compile_ops has already refused
    anything after a measurement, so moving every measure behind the barrier changes no outcome)"""
    out = qc.copy_empty_like()
    for inst in qc.data:
        if inst.operation.name != "measure":
            out.append(inst)
    out.barrier()
    for inst in qc.data:
        if inst.operation.name == "measure":
            out.append(inst)
    return out


ONE_QUBIT = ("rz", "sx", "x")


def two_sx_angles(U: np.ndarray) -> tuple:
    """(l, t, p): rz(l), sx, rz(t + pi), sx, rz(p + pi) -- in circuit order -- equals U up to a global phase, with
    (t, p, l) U's U3 angles. ALWAYS two sx, also where fewer would do: the template a twirling server plays"""
    from qiskit.synthesis import OneQubitEulerDecomposer
    t, p, l = OneQubitEulerDecomposer("U3").angles(U)
    return float(l), float(t), float(p)


def cz_layers(qc, snap: DeviceSnapshot) -> dict:
    """{instruction index of a REAL CZ: its layer}, greedy in instruction order: a CZ joins the current layer unless
    one of its qubits is already in it or has run any other instruction since the layer began (a barrier on it
    counts). The server's own layering can differ (R166: one pair's third CZ in a layer of its own)"""
    layer, members, since = -1, set(), set()
    out = {}
    for k, inst in enumerate(qc.data):
        qs = [qc.find_bit(b).index for b in inst.qubits]
        if inst.operation.name == "cz" and snap.edge(*qs)[1] > 0:
            if layer < 0 or set(qs) & members or set(qs) & since:
                layer, members, since = layer + 1, set(), set()
            members.update(qs)
            out[k] = layer
        elif inst.operation.name != "delay":
            since.update(qs)
    return out


def template_resynthesis(qc, snap: DeviceSnapshot, dress_idle: bool = False):
    """the ISA circuit as a twirling server PLAYS it (R166, MEASURED from its executed schedule):
    - every maximal single-qubit run (rz, sx, x between any other instructions) becomes ONE template of exactly two
      sx carrying the run's unitary (two_sx_angles): an H compiled with one sx plays two;
    - every qubit of a REAL twirled CZ (calibrated duration > 0; a stand-in's zero-duration virtual edge is not a
      twirled layer) gets such a slot immediately before the CZ and one after its last CZ, even where nothing was
      compiled. A barrier or a delay ends a run but never forces a slot, so CZ-barrier-CZ has ONE slot between
      layers (R166: 18 sx over an 8-CZ ladder, with or without barriers);
    - a run of rz alone that no twirled CZ forces stays virtual;
    - dress_idle (active-accum): every TOUCHED qubit (one any gate or read has used) idle in a CZ layer gets a slot
      before the layer and after the last one, exactly as a CZ's own qubits do, and a barrier does not close its run,
      so its neighbouring gates are absorbed into those slots (R166, MEASURED: it and ip both play 9 slots over an
      8-CZ ladder, ip's two H's included).
    The unitary of the circuit is unchanged (tests/test_twirl_template.py checks it)."""
    import math as _m

    from qiskit.quantum_info import Operator
    out = qc.copy_empty_like()
    pending: dict = {}
    dirty: set = set()                                    # qubits a real CZ touched since their last slot
    layers = cz_layers(qc, snap) if dress_idle else {}
    starts: dict = {}
    for k, L in layers.items():
        starts.setdefault(L, k)
    members: dict = {}
    for k, L in layers.items():
        members.setdefault(L, set()).update(qc.find_bit(b).index for b in qc.data[k].qubits)
    first_of = {k: L for L, k in starts.items()}
    touched: set = set()
    read: set = set()

    def flush(q: int, force: bool) -> None:
        run = pending.pop(q, [])
        if not run and not force:
            return
        if not force and all(inst.operation.name == "rz" for inst in run):
            for inst in run:
                out.append(inst.operation, inst.qubits, inst.clbits)
            return
        U = np.eye(2, dtype=complex)
        for inst in run:
            U = Operator(inst.operation).data @ U
        lam, th, ph = two_sx_angles(U)
        qb = qc.qubits[q]
        out.rz(lam, qb)
        out.sx(qb)
        out.rz(th + _m.pi, qb)
        out.sx(qb)
        out.rz(ph + _m.pi, qb)
        dirty.discard(q)

    for k, inst in enumerate(qc.data):
        name = inst.operation.name
        qs = [qc.find_bit(b).index for b in inst.qubits]
        if name in ONE_QUBIT:
            pending.setdefault(qs[0], []).append(inst)
            touched.add(qs[0])
            continue
        if k in first_of:                                 # a layer begins: dress its touched idle qubits first
            L = first_of[k]
            for q in sorted(touched - members[L] - read):
                flush(q, force=True)
                dirty.add(q)
        real_cz = name == "cz" and snap.edge(*qs)[1] > 0
        passive = name in ("barrier", "delay", "id")
        for q in qs:
            if dress_idle and passive and q in touched:
                continue                                  # the server merges this run across the barrier
            flush(q, force=real_cz or (q in dirty and not passive))
        out.append(inst.operation, inst.qubits, inst.clbits)
        if real_cz:
            dirty.update(qs)
        if name == "measure":
            read.update(qs)
        if not passive:
            touched.update(qs)
    for q in sorted(set(pending) | dirty):
        flush(q, force=q in dirty)
    return out


def compile_ops(qc, snap: DeviceSnapshot, params: NoiseParams):
    """schedule an ISA circuit (terminal measurements only) and return (ops, active physical qubits, measured map)

    measured map: list of (active index, clbit index) in clbit order. Raises ScheduleError on anything the engine
    cannot represent faithfully."""
    from qiskit.circuit.library import CZGate
    from qiskit.quantum_info import Operator
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import ALAPScheduleAnalysis, ASAPScheduleAnalysis, PadDelay

    if params.twirl_template:                              # the circuit as the twirling server plays it (R166)
        qc = template_resynthesis(qc, snap, dress_idle=params.twirl_template_dress)

    seen_measure: set = set()
    clbits_written: list = []
    for inst in qc.data:
        q = [qc.find_bit(x).index for x in inst.qubits]
        if inst.operation.name not in ISA:
            raise ScheduleError(f"gate {inst.operation.name} is not in the ISA basis (transpile to cz/rz/sx/x first)")
        if inst.operation.name == "cz":
            snap.edge(*q)                              # SnapshotError names the uncalibrated edge
        if inst.operation.name == "measure":
            if q[0] in seen_measure:
                raise ScheduleError(f"qubit {q[0]} is measured twice")
            seen_measure.add(q[0])
            clbits_written.append(qc.find_bit(inst.clbits[0]).index)
        elif inst.operation.name not in ("barrier", "delay") and seen_measure & set(q):
            raise ScheduleError(f"{inst.operation.name} acts after a measurement on {q}: mid-circuit measurement "
                                "is not supported (feed it forward as a Pauli frame instead)")
    if sorted(clbits_written) != list(range(qc.num_clbits)):
        raise ScheduleError(f"every classical bit must be written by exactly one measurement; got {clbits_written} "
                            f"for {qc.num_clbits} clbits (a score reads bits in clbit order)")
    analysis = ALAPScheduleAnalysis if params.schedule == "alap" else ASAPScheduleAnalysis
    src = _readout_aligned(qc) if params.schedule == "asap_aligned_readout" else qc
    pm = PassManager([analysis(durations(snap)), PadDelay(durations=durations(snap))])
    sched = pm.run(src)
    active = sorted({sched.find_bit(x).index for inst in sched.data for x in inst.qubits
                     if inst.operation.name not in ("delay", "barrier")})
    idx = {p: i for i, p in enumerate(active)}
    started: set = set()                               # qubits that have left |0> (their first real gate ran)
    read: set = set()                                  # qubits whose readout has happened
    ops: list = []
    meas: list = []
    # TIMING, for the crosstalk only: each qubit's clock over the padded schedule; per appended op its physical
    # qubits and [start, end); per qubit the moment it left |0> and the moment its readout began
    clock: dict = {}
    op_time: list = []
    t_start: dict = {}
    t_read: dict = {}
    # PHASE WINDOWS (module docstring), each closing BEFORE instruction k: the quasi-static spread's (closing per
    # NoiseParams.quasistatic) and a coherent detuning's (closing only at sx)
    active_q = set(idx)
    qs_at = (phase_windows(sched, snap, active_q & set(snap.sigma), close_at_cz=params.quasistatic == "layer")
             if params.idle and params.quasistatic != "off" and snap.sigma else {})
    dt_at = (phase_windows(sched, snap, active_q & set(snap.detuning), close_at_cz=False)
             if params.idle and params.detuning and snap.detuning else {})
    for k, inst in enumerate(sched.data):
        name = inst.operation.name
        phys = [sched.find_bit(x).index for x in inst.qubits]
        t0 = max((clock.get(q, 0.0) for q in phys), default=0.0)
        dur = _op_duration(inst, phys, snap)
        for q in phys:
            clock[q] = t0 + dur
        for q, T in qs_at.get(k, ()):                  # placed BEFORE the gate that closes the window
            pz = params.idle_factor * quasistatic_z(snap.sigma[q], T)
            if pz > 0:
                ops.append(Op("quasistatic", (idx[q],), None, pz, (0.0, 0.0, pz)))
                op_time.append(((q,), t0, t0, "quasistatic"))
        for q, T in dt_at.get(k, ()):
            theta = snap.detuning[q] * T
            if theta:
                ops.append(Op("detuning", (idx[q],), rz_matrix(theta), 0.0, (0.0, 0.0, 0.0)))
                op_time.append(((q,), t0, t0, "detuning"))
        if name in ("sx", "x"):
            for q in phys:
                t_start.setdefault(q, t0 + dur)
        if name == "measure":
            meas.append((idx[phys[0]], sched.find_bit(inst.clbits[0]).index))
            read.add(phys[0])
            t_read.setdefault(phys[0], t0)
            continue
        if name == "barrier":
            continue
        n_before = len(ops)                            # every op appended below gets ONE timing row (idles too)
        if any(p not in idx for p in phys):
            continue                                   # an idle on a qubit the protocol never touches
        a = tuple(idx[p] for p in phys)
        if name in ("delay", "id"):
            # ALAP parks a late-used qubit in |0> at the start. |0> is a fixed point of amplitude damping and of
            # dephasing, so that wait costs nothing physically -- but its Pauli twirl would flip it with probability
            # a/2. Count only idles after the qubit's first NON-DIAGONAL gate (rz and cz leave a |0> qubit in |0>).
            # The rule is fixed at compile time, so a gate error that flips a parked qubit early leaves its later
            # wait uncharged: an O(p_gate x p_idle) omission.
            # A wait AFTER the qubit's readout (ASAP reads a qubit the moment its last gate ends, and pads the rest)
            # cannot touch the recorded bit: the engine applies readout to the final state, so charging it would.
            if not params.idle or phys[0] not in started or phys[0] in read:
                continue
            q = phys[0]
            t = (_seconds(inst.operation.duration, inst.operation.unit, snap.dt) if name == "delay"
                 else snap.sx_duration[q])
            if not snap.complete(q):
                raise ScheduleError(f"qubit {q} has incomplete calibration in {snap.name}")
            pp = tuple(params.idle_factor * x for x in idle_pauli(t, snap.t1[q], snap.t2[q]))
            if sum(pp) > 0:
                ops.append(Op("idle", a, None, sum(pp), pp))
                op_time.append((tuple(phys), t0, t0 + dur, "idle"))
            continue
        if name in ("sx", "x"):
            started.update(phys)
        U = Operator(inst.operation).data if name != "cz" else Operator(CZGate()).data
        if name == "cz":
            p = min(params.cz_factor * 1.25 * snap.edge(*phys)[0], 15 / 16)
            ops.append(Op(name, a, U, p, cz_pauli_probs(p, snap.edge_structure(*phys))))
        elif name == "rz":
            ops.append(Op(name, a, U, 0.0, (0.0, 0.0, 0.0)))
        elif name in ("sx", "x"):
            e = params.sx_factor * 1.5 * snap.sx_error[phys[0]]
            e = 0.0 if not math.isfinite(e) else min(e, 0.75)
            ops.append(Op(name, a, U, e, (e / 3, e / 3, e / 3)))
        else:
            raise ScheduleError(f"gate {name} is not in the ISA basis (transpile to cz/rz/sx/x first)")
        if len(ops) > n_before:
            op_time.append((tuple(phys), t0, t0 + dur, name))
    meas.sort(key=lambda t: t[1])
    if not meas:
        raise ScheduleError("the circuit measures nothing")
    if len(op_time) != len(ops):                       # the timing rows index the ops: a mismatch would misplace
        raise ScheduleError(f"internal: {len(ops)} ops but {len(op_time)} timing rows")   # every inserted error
    if params.spectator > 0:
        ops, op_time = _with_spectators(ops, op_time, idx, snap, params.spectator, t_start, t_read)
    if params.twirl not in ("off", "active"):
        ops, op_time = _with_twirl_dressing(ops, op_time, idx, snap, params.twirl,
                                            params.sx_factor * params.dressing_factor, t_read)
    if snap.zz and params.crosstalk != "off":
        ops = _with_crosstalk(ops, op_time, idx, snap, params.crosstalk, t_start, t_read, max(clock.values()))
    return ops, active, meas


GATE_ROWS = ("cz", "sx", "x", "rz")
# timing rows that are NOT gates (an idle or an inserted error): they neither twirl nor end a crosstalk stretch
NOT_GATES = ("delay", "id", "idle", "spectator", "quasistatic", "detuning", "dressing")


def twirl_layers(op_time: list) -> list:
    """the CZ LAYERS of a compiled op list, greedy as a twirling pass finds them (a CZ joins the layer after the last
    one either of its qubits was in): per layer its qubits, the index of its last CZ op, its start time, and the
    qubits any gate had touched before its first CZ (the 'up to the current layer' of active-accum)"""
    last: dict = {}
    layers: dict = {}
    touched: set = set()
    for k, (phys, s0, _s1, name) in enumerate(op_time):
        if name == "cz":
            L = max(last.get(q, -1) for q in phys) + 1
            for q in phys:
                last[q] = L
            info = layers.setdefault(L, dict(qubits=set(), end=k, start=s0, touched=set(touched)))
            info["qubits"].update(phys)
            info["end"] = max(info["end"], k)
            info["start"] = min(info["start"], s0)
        if name in GATE_ROWS:
            touched.update(phys)
    return [layers[L] for L in sorted(layers)]


def _with_twirl_dressing(ops, op_time, idx, snap, strategy, sx_factor, t_read):
    """after each CZ layer, on every qubit the strategy twirls that sits IDLE in that layer and has not been read, a
    depolarizing error of one physical pulse's probability (sx_factor x 1.5 x its sx error): the expected cost of the
    random Pauli the server puts before and after the layer (two pulses when it is X or Y). active-accum twirls the
    qubits a gate touched before the layer; active-circuit, every qubit a gate touches anywhere; all, every active
    qubit. A FIRST-ORDER model: the dressing merged into an active qubit's neighbouring gates is not charged, and the
    random echo the pulses give an idle qubit's dephasing is not credited."""
    layers = twirl_layers(op_time)
    circuit_q = {q for (phys, _s0, _s1, name) in op_time if name in GATE_ROWS for q in phys}
    after: dict = {}
    for info in layers:
        if strategy == "active-accum":
            twirled = info["touched"] | info["qubits"]
        elif strategy == "active-circuit":
            twirled = circuit_q
        else:                                          # "all"
            twirled = set(idx)
        for q in sorted(twirled - info["qubits"]):
            if q not in idx or t_read.get(q, math.inf) <= info["start"]:
                continue
            e = sx_factor * 1.5 * snap.sx_error[q]
            e = 0.0 if not math.isfinite(e) else min(e, 0.75)
            if e > 0:
                after.setdefault(info["end"], []).append((q, e, info["start"]))
    out, times = [], []
    for k, (op, tm) in enumerate(zip(ops, op_time)):
        out.append(op)
        times.append(tm)
        for q, e, s0 in after.get(k, ()):
            out.append(Op("dressing", (idx[q],), None, e, (e / 3, e / 3, e / 3)))
            times.append(((q,), s0, s0, "dressing"))
    return out, times


def _with_spectators(ops, op_time, idx, snap, p, t_start, t_read):
    """after every CZ on (a, b): a Z error of probability p on each coupled neighbour c of a or b (c not in {a, b})
    that the circuit uses, that has left |0> before the CZ starts and whose readout has not begun. Returns the ops and
    the matching timing list (the inserted errors carry the CZ's interval, and are not gates)."""
    nb: dict = {}
    for (u, v) in snap.cz:
        nb.setdefault(u, set()).add(v)
        nb.setdefault(v, set()).add(u)
    out, times = [], []
    for op, tm in zip(ops, op_time):
        out.append(op)
        times.append(tm)
        phys, s0, s1, name = tm
        if name != "cz":
            continue
        for c in sorted((nb.get(phys[0], set()) | nb.get(phys[1], set())) - set(phys)):
            if c in idx and c in t_start and t_start[c] <= s0 and t_read.get(c, math.inf) > s0:
                out.append(Op("spectator", (idx[c],), None, p, (0.0, 0.0, p)))
                times.append(((c,), s0, s1, "spectator"))
    return out, times


def _op_duration(inst, phys, snap: DeviceSnapshot) -> float:
    name = inst.operation.name
    if name in ("barrier", "rz"):
        return 0.0
    if name == "cz":
        return snap.edge(*phys)[1]
    if name in ("sx", "x", "id"):
        return snap.sx_duration[phys[0]]
    if name == "measure":
        return snap.meas_duration[phys[0]]
    if name == "delay":
        return _seconds(inst.operation.duration, inst.operation.unit, snap.dt)
    return 0.0


def cphase_pauli_probs(phi: float) -> tuple:
    """a Pauli-twirled CPhase(phi) as 15 probabilities in PAULIS2 order: ZI, IZ and ZZ each sin^2(phi/2)/4"""
    s = math.sin(phi / 2) ** 2 / 4
    return tuple(s if pair in (("Z", "I"), ("I", "Z"), ("Z", "Z")) else 0.0 for pair in PAULIS2)


def _with_crosstalk(ops, op_time, idx, snap, mode, t_start, t_read, t_end):
    """insert, for every coupled pair (a, b) both of which the circuit uses, a twirled CPhase(zeta x t) error for
    every stretch both qubits spend out of |0> and before their readout. A stretch ends at a twirled two-qubit gate
    touching a or b ('coherent') or at any gate on a or b ('incoherent'); time inside a CZ BETWEEN a and b is that
    gate's own (calibrated) interaction and is not counted. The error goes immediately before the op that ends the
    stretch, so it acts after everything that preceded it on a and b and before everything that follows. The last
    stretch, after the pair's last non-diagonal gate, is followed only by diagonal gates and the Z readout, which a
    CPhase error cannot change, so it is dropped."""
    inserts: dict = {}
    for (a, b), zeta in snap.zz.items():
        if a not in idx or b not in idx or zeta <= 0 or a not in t_start or b not in t_start:
            continue
        w0 = max(t_start[a], t_start[b])
        w1 = min(t_read.get(a, t_end), t_read.get(b, t_end))
        if w1 <= w0:
            continue
        last = w0
        for k, (phys, s0, s1, name) in enumerate(op_time):
            if (a not in phys and b not in phys) or name in NOT_GATES:
                continue                               # an idle is not a gate: it neither twirls nor ends a stretch
            mutual = set(phys) == {a, b}
            cut = mutual or name in ("cz", "sx", "x") or mode == "incoherent"
            if not cut:
                continue
            seg = max(0.0, min(s0, w1) - max(last, w0))
            if seg > 0:
                inserts.setdefault(k, []).append(((idx[a], idx[b]), zeta * seg))
            last = s1 if mutual else max(last, s0)
    if not inserts:
        return ops
    out = []
    for k, op in enumerate(ops + [None]):
        for (qa, qb), phi in inserts.get(k, ()):
            pp = cphase_pauli_probs(phi)
            out.append(Op("crosstalk", (qa, qb), None, sum(pp), pp))
        if op is not None:
            out.append(op)
    return out
