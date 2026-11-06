"""The only path from a frozen design to IBM hardware, and every guard on it. Nothing here runs unless asked.

A submission is REFUSED unless all of these hold, each checked and reported by `preflight` before any network call
that could spend anything:
  AUTHORIZED   the caller passed --i-understand-this-spends-credits AND the environment has ONEQ_IBM_SUBMIT=1
  PREREGISTERED the preregistration file hashes to the value pinned in the frozen record
  FROZEN       the QPY bytes and sources verify against the frozen record (freeze.load); circuits are never
               re-transpiled at submission
  ONE RUN      no job record for this preregistration exists yet (a second run is a replication with its own
               preregistration, never a silent retry)
  QUOTA        the account's remaining quota covers the conservative cost estimate with margin; read with
               service.usage(), which spends nothing
  HEALTHY      the live calibration still satisfies the layout policy on every physical qubit and edge the frozen
               circuits use; if not, postpone -- never re-site a preregistered run silently
  PREDICTED    the frozen prediction gives the decision a pass probability >= plan.min_pass_probability, OR the
               plan declares the run an exploratory MEASUREMENT with its reason (R133 ran runs predicted to fail,
               on purpose, to measure the noise): credits are never spent on a verdict the model already knows
The job record (job id, time, quota before, the frozen record's hashes) is written the moment the job exists.

QPU cost model (IBM does not publish one): seconds = shots x per_shot + instances x per_instance + per_job, where
an instance is one bound circuit or twirling randomization. Fitted to this repository's seven recorded jobs
(PAST_JOBS): 272-313 us per shot including the 250 us rep delay, and an instance overhead between 0.7 and 2.4 ms
that the data cannot pin down; the defaults take the conservative end of both.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import pathlib
from collections.abc import Mapping
from dataclasses import dataclass, field

from . import freeze as F

PER_SHOT_S = 320e-6
PER_INSTANCE_S = 2.5e-3
PER_JOB_S = 2.0
QUOTA_MARGIN = 1.25
MEAS_TYPES = ("classified", "kerneled")     # SamplerV2 execution.meas_type values a Plan may ask for

# (job id, circuits, shots per circuit, twirl randomizations per circuit or None, execution seconds reported)
PAST_JOBS = (
    ("db6pkeqmb58s738c6h80", 25, 4000, None, 27.227564),
    ("db6pne04qg6s73c561s0", 12, 4000, 32, 13.335266),
    ("db6pvrkvf2bc73d1k8ug", 22, 4000, 32, 27.00063),
    ("db6q46cvf2bc73d1kek0", 22, 4000, 32, 26.611637),
    ("db77t4mr6oqs7386v0b0", 20, 1000, None, 6.257611),
    ("db794cgr1c5c738k1a20", 44, 4000, None, 47.856065),
    # R152, 2026-10-17: one composite job of mixed circuits (static and dynamic, 1,000-3,672 shots each). Shots vary
    # per circuit, so the MEAN is recorded (circuits x shots = the 355,736 shots that ran); 106 s charged and summed
    # over its 12 execution spans alike. The model estimated 127.7 s: 1.20x, inside the asserted band.
    ("db9c50oc77vs73817rog", 594, 355736 / 594, 8, 106.0),
)


class SubmitRefused(RuntimeError):
    """a guard failed; the message names every failing guard"""


def qpu_seconds(total_shots: int, instances: int, per_shot: float = PER_SHOT_S,
                per_instance: float = PER_INSTANCE_S, per_job: float = PER_JOB_S) -> float:
    return total_shots * per_shot + instances * per_instance + per_job


def past_job_per_shot() -> list:
    return [(j, t / (c * s)) for j, c, s, _, t in PAST_JOBS]


def file_sha256(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


@dataclass
class Plan:
    """what one submission will do"""
    tag: str                            # file prefix for the job/result records
    backend: str
    shots: dict                         # circuit name -> shots
    prereg_path: str
    prereg_sha256: str
    twirl_randomizations: int | None = 32
    records_dir: str = "evidence/hw/jobs"
    predicted_pass_probability: float | None = None
    min_pass_probability: float = 0.5
    measurement_reason: str | None = None     # set to run a verdict the prediction expects to fail, and say why
    extra: dict = field(default_factory=dict)
    # "classified" (bits, the default) or "kerneled": every shot's complex IQ point per clbit, which oneq.hw.iq turns
    # into bits, in-circuit readout separation and leakage. The plan must then carry its own |0> and |1> readout
    # circuits, named in iq_calibration (prepared-0 circuit, prepared-1 circuit), or the IQ cannot be read
    meas_type: str = "classified"
    iq_calibration: tuple | None = None

    @property
    def total_shots(self) -> int:
        return sum(self.shots.values())

    @property
    def instances(self) -> int:
        return len(self.shots) * (self.twirl_randomizations or 1)

    def cost_s(self) -> float:
        return qpu_seconds(self.total_shots, self.instances)


def existing_jobs(plan: Plan) -> list:
    d = pathlib.Path(plan.records_dir)
    out = []
    for p in sorted(d.glob(f"{plan.tag}_*.job.json")) if d.exists() else []:
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("prereg_sha256") == plan.prereg_sha256:
            out.append(p)
    return out


def preflight(plan: Plan, frozen: dict, blob: bytes, *, authorized_flag: bool, remaining_s: float | None,
              health_refusals: list | None, env: dict | None = None) -> list:
    """every failing guard, as a list of reasons (empty = may submit). Pure: no network."""
    environ: Mapping[str, str] = os.environ if env is None else env
    bad = []
    if not (authorized_flag and environ.get("ONEQ_IBM_SUBMIT") == "1"):
        bad.append("AUTHORIZED: needs --i-understand-this-spends-credits AND ONEQ_IBM_SUBMIT=1")
    if not pathlib.Path(plan.prereg_path).exists() or file_sha256(plan.prereg_path) != plan.prereg_sha256:
        bad.append(f"PREREGISTERED: {plan.prereg_path} does not hash to {plan.prereg_sha256[:12]}...")
    if frozen.get("prereg_sha256") != plan.prereg_sha256:
        bad.append("PREREGISTERED: the frozen record pins a different preregistration")
    try:
        circuits = F.load(frozen, blob)
        if set(circuits) != set(plan.shots):
            bad.append("FROZEN: the plan's circuits are not exactly the frozen circuits")
    except F.FrozenMismatch as e:
        bad.append(f"FROZEN: {e}")
    if existing_jobs(plan):
        bad.append(f"ONE RUN: a job record for this preregistration already exists: {existing_jobs(plan)[0]}")
    need = plan.cost_s() * QUOTA_MARGIN
    if remaining_s is None:
        bad.append("QUOTA: remaining quota was not read")
    elif remaining_s < need:
        bad.append(f"QUOTA: {remaining_s:.0f} s remaining < {need:.0f} s (estimate {plan.cost_s():.1f} s x "
                   f"{QUOTA_MARGIN})")
    if plan.meas_type not in MEAS_TYPES:
        bad.append(f"MEASUREMENT: meas_type {plan.meas_type!r} is not one of {MEAS_TYPES}")
    elif plan.meas_type == "kerneled" and not (plan.iq_calibration and len(plan.iq_calibration) == 2
                                               and all(n in plan.shots for n in plan.iq_calibration)):
        bad.append("MEASUREMENT: a kerneled job must name its own |0> and |1> readout circuits (iq_calibration), "
                   "or its IQ cannot be classified")
    if health_refusals is None:
        bad.append("HEALTHY: the live calibration was not checked")
    elif health_refusals:
        bad.append(f"HEALTHY: postpone, do not re-site -- {health_refusals[:5]}")
    if not plan.measurement_reason:
        pp = plan.predicted_pass_probability
        if pp is None:
            bad.append("PREDICTED: no frozen prediction of the pass probability")
        elif pp < plan.min_pass_probability:
            bad.append(f"PREDICTED: pass probability {pp:.3g} < {plan.min_pass_probability} -- the model already "
                       "expects this verdict to fail (set measurement_reason to run it as a measurement)")
    return bad


def pass_probability(predicted: float, predicted_se: float, bar: float, halfwidth: float, run_sd: float) -> float:
    """P(lower bound > bar) when the run's estimate ~ Normal(predicted, run_sd^2 + predicted_se^2)"""
    s = math.hypot(run_sd, predicted_se)
    z = (predicted - halfwidth - bar) / s if s > 0 else (math.inf if predicted - halfwidth > bar else -math.inf)
    return 0.5 * math.erfc(-z / math.sqrt(2))


def remaining_quota_s(service) -> float:
    """read-only: QiskitRuntimeService.usage() spends nothing"""
    u = service.usage()
    return float((u.get("usage_limit_seconds") or 0) - (u.get("usage_consumed_seconds") or 0))


def submit(plan: Plan, frozen: dict, blob: bytes, backend, *, authorized_flag: bool, remaining_s: float,
           health_refusals: list) -> dict:
    """runs preflight; on success submits ONE SamplerV2 job of the frozen circuits and writes its record"""
    bad = preflight(plan, frozen, blob, authorized_flag=authorized_flag, remaining_s=remaining_s,
                    health_refusals=health_refusals)
    if bad:
        raise SubmitRefused("; ".join(bad))
    from qiskit_ibm_runtime import SamplerV2
    circuits = F.load(frozen, blob)
    order = list(frozen["circuit_order"])
    sampler = SamplerV2(mode=backend)
    sampler.options.dynamical_decoupling.enable = False          # the engine models free idles
    sampler.options.twirling.enable_measure = False              # the engine models the asymmetric readout split
    if plan.meas_type != "classified":                           # per-shot IQ (oneq.hw.iq); bits are the default
        sampler.options.execution.meas_type = plan.meas_type
    if plan.twirl_randomizations:
        sampler.options.twirling.enable_gates = True             # the measured CZ factor is for twirled CZs
        sampler.options.twirling.num_randomizations = plan.twirl_randomizations
    else:
        sampler.options.twirling.enable_gates = False
    pubs = [(circuits[n], None, plan.shots[n]) for n in order]
    # a LOWER bound on the submission, to the microsecond: stamped before the job exists. submitted_utc (after run
    # returns, whole seconds) is an upper bound to the second and cannot order the job against an event in the same
    # second -- a cross-QPU relay's causality check refused a receiver submitted 0.5 s after its source on it
    submit_called = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="microseconds")
    job = sampler.run(pubs)
    rec = dict(tag=plan.tag, job_id=job.job_id(), backend=getattr(backend, "name", plan.backend),
               submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               submit_called_utc=submit_called,
               quota_remaining_before_s=remaining_s, estimated_cost_s=plan.cost_s(), prereg=plan.prereg_path,
               prereg_sha256=plan.prereg_sha256, qpy_sha256=frozen["qpy_sha256"], circuit_order=order,
               shots=plan.shots, twirl_randomizations=plan.twirl_randomizations, extra=plan.extra)
    if plan.meas_type != "classified":                           # recorded only when it is not the default
        rec.update(meas_type=plan.meas_type, iq_calibration=list(plan.iq_calibration or ()))
    d = pathlib.Path(plan.records_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{plan.tag}_{rec['job_id']}.job.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    return rec


def counts_from_result(result, order: list) -> dict:
    """SamplerV2 PrimitiveResult -> {circuit name: counts}, by POSITION in the registered order (never by name
    lookup: the tw package's dry run read colliding names and silently copied one circuit's counts into another)"""
    if len(result) != len(order):
        raise SubmitRefused(f"{len(result)} results for {len(order)} registered circuits")
    counts = {}
    for i, name in enumerate(order):
        data = result[i].data
        regs = list(data.keys()) if hasattr(data, "keys") else [k for k in vars(data)]
        if len(regs) != 1:
            raise SubmitRefused(f"{name}: expected one classical register, found {regs}")
        reg = getattr(data, regs[0])
        if not hasattr(reg, "get_counts"):
            raise SubmitRefused(f"{name}: the result carries per-shot IQ (a kerneled job), not bits -- classify it "
                                "with oneq.hw.iq against the job's own readout circuits")
        counts[name] = reg.get_counts()
    return counts


def fetch(job_record: dict, service) -> dict:
    """counts by circuit name and the reported execution seconds, for a finished job (read-only)"""
    job = service.job(job_record["job_id"])
    status = str(job.status())
    if "DONE" not in status:
        return dict(status=status)
    result = job.result()
    spans = result.metadata.get("execution", {}).get("execution_spans")
    seconds = (spans.stop - spans.start).total_seconds() if spans else None
    return dict(status=status, counts=counts_from_result(result, job_record["circuit_order"]),
                execution_seconds=seconds)


def local_run(frozen: dict, blob: bytes, fake_backend, shots: dict, simulator=None, pad_alap: bool = False,
              method: str = "statevector") -> dict:
    """DRY RUN: the frozen circuits through SamplerV2 in LOCAL mode; exercises loading, PUB construction and counts
    extraction exactly as a real run would.

    simulator=None runs qiskit-aer's NoiseModel.from_backend(target) -- the model the engine is validated against
    (evidence/hw/engine_vs_aer.json), so a dry run is an end-to-end check of the whole design, engine against
    simulator. (Running the fake backend itself instead adds a relaxation error on every `measure` on top of the
    readout assignment error -- 3.6% process infidelity per measured qubit on FakeKingston q51 -- which the target
    model and the engine do not have: ~0.034 of F_avg on an 8-readout design. Pass it as `simulator` to get that.)

    *** THE SIMULATION METHOD IS PINNED to statevector, on any simulator. *** 2026-10-15 (R146): qiskit-aer 0.17.2's
    AUTOMATIC method, given a noisy batch of mixed circuits (R145's 184 fourteen-qubit kept circuits and 24 two-qubit
    both-erased ones), returned some two-qubit circuits as if their gates never acted -- P(00) 0.97-0.98 where the
    ideal is 0 or 0.25 -- on every noise model tried, with or without fusion or parallel experiments. The same
    circuits were right alone, right in the full job under ideal Aer, and right with method="statevector". Every
    earlier dry run read that branch at chance (F_e 0.25 against the device's own 0.368): an instrument defect the
    widths check could not see, since every key had the right length.

    pad_alap=True pads every circuit with explicit ALAP delays from the fake's own durations first. Aer relaxes a qubit
    only during an instruction, so an unpadded circuit idles for free there -- unlike the engine's ALAP model and,
    probably, the device. Padding makes the simulator a device whose idles are physical; a dry run that tests whether
    a model closes (R146) needs that, and one that only exercises the PUB path does not.

    method: always EXPLICIT (the automatic one is the defect above). "matrix_product_state" is for circuits too
    WIDE for a per-shot statevector whose entanglement stays local -- R152's calibration block reads 26 qubits at
    once, in product states and pairwise CZ ladders -- and must be checked against statevector where both fit."""
    from qiskit_ibm_runtime import SamplerV2
    if "fake" not in getattr(fake_backend, "name", "").lower():
        raise SubmitRefused("local_run is for fake backends only")
    if simulator is None:
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel
        simulator = AerSimulator(noise_model=NoiseModel.from_backend(fake_backend))
    if method not in ("statevector", "matrix_product_state"):
        raise SubmitRefused(f"simulation method {method!r}: statevector, or matrix_product_state for wide circuits")
    simulator.set_options(method=method)
    circuits = F.load(frozen, blob)
    order = list(frozen["circuit_order"])
    if pad_alap:
        from qiskit.transpiler import PassManager
        from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDelay

        from .noise import durations
        from .snapshot import from_backend
        dur = durations(from_backend(fake_backend))
        pm = PassManager([ALAPScheduleAnalysis(dur), PadDelay(durations=dur)])
        circuits = {n: pm.run(circuits[n]) for n in order}
    job = SamplerV2(mode=simulator).run([(circuits[n], None, shots[n]) for n in order])
    return counts_from_result(job.result(), order)


def instances_for(total_shots: int, budget_s: float) -> int:
    """the most instances a budget allows at a given total shot count (planning helper)"""
    return max(0, math.floor((budget_s - PER_JOB_S - total_shots * PER_SHOT_S) / PER_INSTANCE_S))
