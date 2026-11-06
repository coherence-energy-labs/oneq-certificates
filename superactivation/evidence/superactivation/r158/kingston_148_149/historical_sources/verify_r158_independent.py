#!/usr/bin/env python3
"""R158 producer-independent frozen IBM ISA/QPY verifier, NO provider import.

No oneq.hw.r158_native, no freeze() producer, no IBM service, no QPU job.
Recomputes all 60 circuit hashes and instruction/body semantics. The
frozen physical circuits, not an author's metadata, own acceptance.
Local integrity is NOT independent IBM provider execution attestation.
"""
from __future__ import annotations

import hashlib
import io
import itertools
import json
import math
import pathlib
import random
import re
import sys

ROOT=pathlib.Path(__file__).resolve().parents[1]
PRE=ROOT/"docs"/"PREREG_R158_NATIVE_SIGN_RAMSEY_V1.md"
SOURCE_PATHS={
    "src/oneq/hw/r158_native.py","src/oneq/hw/freeze.py",
    "src/oneq/hw/snapshot.py","src/oneq/hw/submit.py",
    "tools/r158_native_probe.py","tools/verify_r158_independent.py",
    "tools/analyze_r158_independent.py",
    "docs/PREREG_R158_NATIVE_SIGN_RAMSEY_V1.md",
}
FORMAT=re.compile(r"^r158_w(\d{5})_z([PM])_e([01])_([XYZ])$")
WAITS=(0,512,2048,4096,8192)
SHOTS=1024
N=60
SEED=15820261017


class IndependentRefused(ValueError):
    pass


def need(ok,why):
    if not ok:
        raise IndependentRefused(why)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _bytes(path):
    return pathlib.Path(path).read_bytes()


def expected():
    conditions=list(itertools.product(WAITS,(1,-1),(False,True),("X","Y","Z")))
    random.Random(SEED).shuffle(conditions)
    def name(c):
        t,z,e,axis=c
        return f"r158_w{t:05d}_z{'P' if z==1 else 'M'}_e{int(e)}_{axis}"
    return [(name(c),c) for c in conditions]


def _qpy_bytes(circuits):
    from qiskit import qpy
    out=io.BytesIO()
    qpy.dump(circuits,out)
    return out.getvalue()


def circuit_fingerprint(c):
    """Independent operation-level canonical encoder.

    Excludes the arbitrary ordering of QPY layout-register records, while
    preserving physical wire indices, exact float bit values, operation
    ordering, targets, params, classical bits, metadata, and global phase.
    The frozen whole-file QPY SHA still commits to every original byte.
    """
    operations=[]
    for inst in c.data:
        p=[]
        for value in inst.operation.params:
            if type(value) is int:
                p.append(["int",value])
            elif isinstance(value,float) and math.isfinite(value):
                p.append(["float64",float(value).hex()])
            else:
                raise IndependentRefused("unbound/native operation parameter")
        operations.append([
            inst.operation.name,
            [c.find_bit(q).index for q in inst.qubits],
            [c.find_bit(b).index for b in inst.clbits],
            p,
            inst.operation.unit if inst.operation.name=="delay" else None,
        ])
    phase=c.global_phase
    need(type(phase) in (int,float) and math.isfinite(phase),
         "non-numeric native global phase")
    canonical={"name":c.name,"num_qubits":c.num_qubits,
               "num_clbits":c.num_clbits,
               "global_phase":float(phase).hex(),
               "metadata":c.metadata,
               "instructions":operations}
    return digest(json.dumps(canonical,sort_keys=True,separators=(",",":"),
                             allow_nan=False).encode("utf-8"))


def inspect_circuit(qc,cond,physical):
    """Independent native timing, physical measurements and analysis-basis check."""
    import numpy as np
    from qiskit.quantum_info import Operator

    wait,z,echo,axis=cond
    target,neighbor=physical
    n=qc.num_qubits
    need(n==156 and qc.num_clbits==2,"device width/classical bit width changed")
    meta=qc.metadata or {}
    good={"schema":"oneq.r158.setting/1",
          "target_physical":target,"neighbor_physical":neighbor,
          "wait_dt":wait,"neighbor_z":z,"echo":echo,"target_axis":axis,
          "neighbor_prep":"matched_two_sx",
          "source":"LOGICAL_FOR_PHYSICAL_NATIVE_FREEZE_NOT_PROVIDER_RESULT"}
    need(meta==good,"circuit condition metadata is missing/altered")
    names=[]
    xcount={target:0,neighbor:0}
    sxcount={target:0,neighbor:0}
    delays={target:[],neighbor:[]}
    read=[]
    target_timeline=[]
    barriers=[]
    for i,ins in enumerate(qc.data):
        op=ins.operation.name
        qs=[qc.find_bit(q).index for q in ins.qubits]
        cs=[qc.find_bit(c).index for c in ins.clbits]
        need(op in {"rz","sx","x","measure","delay","barrier"},
             "non-native or two-qubit operation introduced")
        need(all(q in (target,neighbor) for q in qs),
             "instruction moved outside declared pair")
        names.append(op)
        if op=="barrier":
            barriers.append(i)
        if op=="measure":
            need(len(qs)==len(cs)==1,"malformed native measurement")
            read.append((i,qs[0],cs[0]))
        if op=="x":
            need(len(qs)==1, "X pulse must be one qubit")
            xcount[qs[0]]+=1
            if qs[0]==target:
                target_timeline.append(("x",0))
        if op=="sx":
            need(len(qs)==1, "SX pulse must be one qubit")
            sxcount[qs[0]]+=1
        if op=="delay":
            need(len(qs)==1 and ins.operation.unit=="dt",
                 "delay changed units or qubit count")
            duration=ins.operation.duration
            need(type(duration) is int and duration>0 and duration%16==0,
                 "physical dt granularity/positivity lost")
            delays[qs[0]].append(duration)
            if qs[0]==target:
                target_timeline.append(("delay",duration))
    need(len(read)==2 and [(r[1],r[2]) for r in read] ==
         [(target,0),(neighbor,1)],"readout no longer physical target/neighbor cbit 0/1")
    need([r[0] for r in read]==[len(qc.data)-2,len(qc.data)-1],
         "terminal readout has an operation after or among measures")
    need(xcount[target]==2*int(echo), "echo X pulses were optimized/changed")
    need(xcount[neighbor]==0 and sxcount[neighbor]==2,
         "sign control no longer has matched two-SX preparations")
    expected_delays=([wait//2,wait//2] if echo and wait else
                     [wait] if wait else [])
    need(delays[target]==delays[neighbor]==expected_delays,
         "native target/neighbor idle segments changed")
    wanted_timeline=([("delay",wait//2),("x",0),
                      ("delay",wait//2),("x",0)] if echo and wait else
                     [("x",0),("x",0)] if echo else
                     [("delay",wait)] if wait else [])
    need(target_timeline==wanted_timeline,
         "echo timing/order is not the frozen logical experiment")

    # The last global barrier separates the free evolution from analysis
    # gates. Determine what physical axis the following native 1q gates
    # REALLY measure, rather than trusting a circuit metadata string.
    need(len(barriers)>=3 and barriers[0]<barriers[-1]<read[0][0],
         "missing enforced preparation/idle/analysis barriers")
    # Before the first barrier, ONLY the neighbor can be driven.
    # Check its ACTUAL prepared eigenstate from native SU(2) gates.
    P=np.eye(2,dtype=complex)
    for ins in qc.data[:barriers[0]]:
        qs=[qc.find_bit(q).index for q in ins.qubits]
        need(qs==[neighbor] and ins.operation.name in ("rz","sx"),
             "unexpected drive during sign-matched neighbor initialization")
        P=Operator(ins.operation).data @ P
    prepared=P @ np.array([1.,0.],dtype=complex)
    expected_z=np.array([1.,0.]) if z==1 else np.array([0.,1.])
    need(abs(abs(np.vdot(expected_z,prepared))**2-1)<1e-10,
         "the native preparation no longer produces the declared neighbor Z state")

    R=np.eye(2,dtype=complex)
    for ins in qc.data[barriers[-1]+1:read[0][0]]:
        qs=[qc.find_bit(q).index for q in ins.qubits]
        if ins.operation.name=="barrier":
            continue
        need(qs==[target] and ins.operation.name in ("rz","sx","x"),
             "physical analysis frame contains unexpected operation")
        R=Operator(ins.operation).data @ R
    actual=R.conj().T @ np.diag([1.,-1.]) @ R
    axis_matrix=Operator.from_label(axis).data
    overlap=float(np.trace(actual @ axis_matrix).real/2)
    need(abs(overlap-1)<1e-10,
         f"reported {axis} analysis is not the actual native Pauli basis")
    return {"wait_segments_dt":expected_delays, "native_ops":len(names)}


def verify(folder):
    """Verify archived files against independent cartesian specification."""
    from qiskit import qpy
    directory=pathlib.Path(folder)
    raw=_bytes(directory/"circuits.qpy")
    rec=json.loads((directory/"frozen.json").read_text(encoding="utf-8"))
    snapshot=json.loads((directory/"snapshot.json").read_text(encoding="utf-8"))
    need(rec.get("schema")=="oneq.r158.native-frozen-sign-probe/1",
         "wrong freeze schema")
    need(rec.get("backend")=="ibm_kingston" and
         rec.get("physical_pair")==[148,149],"wrong frozen backend/physical pair")
    need(rec.get("owner_authorization") is False and
         rec.get("provider_submit_performed") is False,
         "a frozen design cannot assert an owner submission")
    need(rec.get("dt_seconds")==4e-9,
         "backend sample period differs from preregistered read-only calibration")
    need(rec.get("shuffle_seed")==SEED,
         "deterministic randomization seed changed")
    need(rec.get("n_circuits")==N,
         "not 60 circuit settings")
    need(rec.get("qpy_sha256")==digest(raw),"whole frozen QPY SHA mismatch")
    need(snapshot.get("name")=="ibm_kingston" and
         snapshot.get("num_qubits")==156,
         "not a Kingston device snapshot")
    canonical=json.dumps(snapshot,sort_keys=True,separators=(",",":"),
                         allow_nan=False).encode()
    need(digest(canonical)==rec.get("snapshot_sha256"),
         "frozen calibration snapshot hash is missing or altered")
    need(rec.get("calibrated_utc")==snapshot.get("calibrated_utc"),
         "provider calibration timestamp is not bound")
    need(rec.get("prereg_path")==str(PRE.relative_to(ROOT)).replace("\\","/"),
         "preregistration source name changed")
    need(digest(_bytes(PRE))==rec.get("prereg_sha256"),
         "the preregistration was changed after freezing")
    pins=rec.get("source_sha256",{})
    need(set(pins)==SOURCE_PATHS,"incomplete/stale producer source pins")
    for key,pin in pins.items():
        need(digest(_bytes(ROOT/key).replace(b"\r\n",b"\n"))==pin,
             "source bytes have changed since the native freeze: "+key)

    wanted=expected()
    need(rec.get("circuit_order")==[name for name,_ in wanted],
         "circuit order is not the preregistered randomized allocation")
    conditions=[dict(name=name,wait_dt=t,z=z,echo=e,axis=axis)
                for name,(t,z,e,axis) in wanted]
    need(rec.get("conditions")==conditions,
         "condition-to-register identities mismatch")
    need(rec.get("shots")=={name:SHOTS for name,_ in wanted},
         "wrong shot allocation: preregistration is 1024 per circuit")
    need(rec.get("circuit_sha256") is not None and
         set(rec["circuit_sha256"])==set(rec["circuit_order"]),
         "missing, extra or duplicate per-circuit QPY pins")
    expected_cost=61440*320e-6+N*2.5e-3+2.0
    need(math.isclose(rec.get("estimated_qpu_seconds_MODEL_ONLY",-1),expected_cost,
                      rel_tol=0,abs_tol=1e-12),
         "claimed QPU budget model differs from its declared arithmetic")
    need(rec.get("conservative_quota_margin")==1.25,
         "quota safety margin changed")

    decoded=qpy.load(io.BytesIO(raw))
    need(len(decoded)==N and
         [c.name for c in decoded]==rec["circuit_order"],
         "QPY does not decode into the registered circuit order")
    semantic=rec.get("semantic_circuit_sha256_v1")
    need(type(semantic) is dict and set(semantic)==set(rec["circuit_order"]),
         "missing independent operation-level structural fingerprints")
    for c,(name,cond) in zip(decoded,wanted):
        original=rec["circuit_sha256"][name]
        need(type(original) is str and len(original)==64
             and all(x in "0123456789abcdef" for x in original),
             "invalid original producer per-circuit QPY hash")
        # Standalone qpy.dump([c]) has nondeterministic ordering of the
        # transpiler's layout-register records on Qiskit 2.5.0; it is not
        # a sound roundtrip byte identity test. Whole QPY bytes ARE hashed.
        need(circuit_fingerprint(c)==semantic[name],
             "physical operation-level circuit fingerprint has changed")
        inspect_circuit(c,cond,(148,149))
    return {
        "schema":"oneq.r158.independent-native-check/1",
        "status":"OFFLINE_INTEGRITY_AND_PHYSICAL_ISA_CHECK_ONLY",
        "provider":"ibm_kingston","physical_pair":[148,149],
        "circuits_verified":N,"shots_registered":N*SHOTS,
        "qpy_sha256":rec["qpy_sha256"],
        "estimated_cost_seconds_MODEL_ONLY":expected_cost,
        "estimated_cost_with_margin_MODEL_ONLY":expected_cost*1.25,
        "producer_imported":False,"provider_job_submitted":False,
        "physical_ZZ_observed":False,
        "uncertainty":"Qiskit ISA/target calibration cannot authenticate actual pulse-level Ramsey timing",
    }


def main(argv):
    if len(argv)!=1:
        raise SystemExit("usage: python tools/verify_r158_independent.py evidence/superactivation/r158/kingston_148_149")
    print(json.dumps(verify(argv[0]),indent=2,sort_keys=True,allow_nan=False))


if __name__=="__main__":
    try:
        main(sys.argv[1:])
    except IndependentRefused as e:
        print("REFUSED:",e,file=sys.stderr)
        sys.exit(2)
