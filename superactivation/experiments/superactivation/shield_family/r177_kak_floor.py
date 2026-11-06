"""R177 -- the protocols' two-qubit cost against its floors, and what a fractional RZZ basis could save (OFFLINE, 2026-10-18).

OPEN_WORK's unstarted research item: "the protocol's own cost is the lever, not the compile -- the per-shield gate
count sets the loss slope". On the circuits that RAN (R152's frozen R146 and R147 ISA circuits; nothing recompiled),
per maximal two-qubit block (qiskit's Collect2qBlocks), its Weyl class (a, b, c) and four counts:

  cz_spent           the CZs it spends;
  cz_floor           its unitary's KAK floor (TwoQubitBasisDecomposer(CZ).num_basis_gates): CZs above it are waste;
  rzz_floor          its gate count in a fractional RZZ basis -- one per non-zero Weyl coordinate (RZZ(2a) covers
                     (a, 0, 0); IBM's fractional RZZ takes 0 < theta <= pi/2, the whole chamber). It saves a gate only
                     on class (a, 0, 0) with a off {0, pi/4}: 2 CZ -> 1 RZZ;
  cz_contract_floor  the CZs the block NEEDS given what the circuit asks of it (oneq.hw.contract_synth.plan: a fresh
                     input, outputs measured next) -- the same algebra the rewrite uses, so this census and the
                     rewrite cannot disagree; the INDEPENDENT check is the rewrite's exact outcome distributions.

plus the CZs outside every measurement's backward light cone (removable outright), and the CZs inside control-flow
bodies (counted, never rewritten: a body's wires start mid-circuit).

A COUNT, not a prediction (r177_price.py prices the rewrite on the device model).

  python r177_kak_floor.py -> evidence/superactivation/r177_kak_floor/r152_circuits.json
"""
from __future__ import annotations

import collections
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

from oneq.hw import contract_synth as CS  # noqa: E402

R152_DIR = ROOT / "evidence" / "superactivation" / "campaign_r152" / "20261017T003152Z"
OUT = ROOT / "evidence" / "superactivation" / "r177_kak_floor"
TOL = 1e-7
QUARTER = math.pi / 4


def rzz_count(coords: tuple, tol: float = TOL) -> int:
    """RZZ gates for a block of this class: one per non-zero Weyl coordinate"""
    return sum(1 for x in coords if abs(x) > tol)


def weyl_class(coords: tuple, tol: float = TOL) -> str:
    a, b, c = coords
    nz = [abs(x) > tol for x in (a, b, c)]
    if not any(nz):
        return "local"
    if nz == [True, False, False]:
        return "cz" if abs(a - QUARTER) < tol else "zz_partial"
    if nz == [True, True, False]:
        return "two_term"
    return "three_term"


def outside_light_cone(dag) -> list:
    """the op nodes that are NOT ancestors of any measurement or control-flow node: nothing they do can reach a
    classical bit, so deleting them leaves every outcome distribution exactly as it is"""
    readers = [n for n in dag.op_nodes() if n.op.name == "measure" or getattr(n.op, "blocks", None)]
    keep = set()
    for r in readers:
        keep.add(r)
        keep.update(a for a in dag.ancestors(r) if hasattr(a, "op"))
    return [n for n in dag.op_nodes() if n not in keep]


def body_blocks(dag) -> list:
    """(unitary, CZ spent) of every two-qubit block inside every control-flow body, recursively"""
    from qiskit.converters import circuit_to_dag
    out = []
    for n in dag.op_nodes():
        for body in getattr(n.op, "blocks", None) or ():
            bd = circuit_to_dag(body)
            out += [(b.unitary, b.cz) for b in CS.blocks(bd)]
            out += body_blocks(bd)
    return out


def circuit_floor(qc, seed: int = 177, unknown=None) -> dict:
    """one circuit's CZ spent and floors by Weyl class, top level and control flow apart. `unknown`: the input wires
    (default: the circuit's logical qubit 0, where both designs prepare the unit's input) -- the contract floor never
    leans on the input a unit happens to feed the channel (R177's correction)"""
    from qiskit.converters import circuit_to_dag
    dag = circuit_to_dag(qc)
    rng = np.random.default_rng(seed)
    rows = collections.defaultdict(collections.Counter)
    unknown = CS.input_wires(qc) if unknown is None else tuple(unknown)
    for b in CS.blocks(dag, unknown=unknown):
        coords = CS.weyl(b.unitary)
        r = rows["top|" + weyl_class(coords)]
        CS.plan(b, rng)
        r["blocks"] += 1
        r["cz_spent"] += b.cz
        r["cz_floor"] += b.floor
        r["rzz_floor"] += rzz_count(coords)
        r["cz_contract_floor"] += b.target
        r["fresh_inputs_" + str(sum(b.fresh))] += 1
        r["measured_outputs_" + str(sum(b.measured))] += 1
    for u, cz in body_blocks(dag):
        coords = CS.weyl(u)
        r = rows["flow|" + weyl_class(coords)]
        f = CS.cz_floor(u)
        r["blocks"] += 1
        r["cz_spent"] += cz
        r["cz_floor"] += f
        r["rzz_floor"] += rzz_count(coords)
        r["cz_contract_floor"] += f
    dead = outside_light_cone(dag)
    return dict(cz_total_top_level=sum(1 for i in qc.data if i.operation.name == "cz"),
                cz_in_control_flow=sum(v["cz_spent"] for k, v in rows.items() if k.startswith("flow|")),
                cz_outside_light_cone=sum(1 for n in dead if n.op.name == "cz"), ops_outside_light_cone=len(dead),
                by_class={k: dict(v) for k, v in sorted(rows.items())})


def summarize(per_circuit: dict) -> dict:
    agg = collections.defaultdict(collections.Counter)
    for c in per_circuit.values():
        for k, v in c["by_class"].items():
            agg[k].update(v)
    tot, top = collections.Counter(), collections.Counter()
    for k, v in agg.items():
        tot.update(v)
        if k.startswith("top|"):
            top.update(v)
    return dict(circuits=len(per_circuit), by_class={k: dict(v) for k, v in sorted(agg.items())}, total=dict(tot),
                cz_outside_light_cone=sum(c["cz_outside_light_cone"] for c in per_circuit.values()),
                cz_in_control_flow=sum(c["cz_in_control_flow"] for c in per_circuit.values()),
                compile_waste_cz=tot["cz_spent"] - tot["cz_floor"],
                contract_saving_cz=tot["cz_floor"] - tot["cz_contract_floor"],
                fractional_saving_gates=tot["cz_floor"] - tot["rzz_floor"],
                fractional_saving_top_level=top["cz_floor"] - top["rzz_floor"])


def main() -> int:
    from oneq.hw import freeze as F
    out = dict(schema="oneq.r177.kak_floor/1", label="EXPLORATORY -- a gate COUNT on circuits that ran; no prediction",
               source=R152_DIR.relative_to(ROOT).as_posix() if R152_DIR.is_relative_to(ROOT) else str(R152_DIR),
               protocols={})
    for tag in ("r146", "r147"):
        rec = json.loads((R152_DIR / f"{tag}.json").read_text(encoding="utf-8"))
        circs = F.load(rec, (R152_DIR / f"{tag}.qpy").read_bytes(), check_sources=False)
        per = {u["name"]: circuit_floor(circs[u["name"]]) for u in rec["units"]}
        s = summarize(per)
        out["protocols"][tag] = dict(summary=s, per_circuit=per)
        t = s["total"]
        print(f"{tag}: {s['circuits']} unit circuits  CZ spent {t.get('cz_spent', 0)}  floor {t.get('cz_floor', 0)}  "
              f"contract floor {t.get('cz_contract_floor', 0)}  RZZ floor {t.get('rzz_floor', 0)}  waste "
              f"{s['compile_waste_cz']}  fractional saving {s['fractional_saving_gates']}  outside light cone "
              f"{s['cz_outside_light_cone']}  in control flow {s['cz_in_control_flow']}", flush=True)
        for k, v in s["by_class"].items():
            print(f"   {k:20s} {dict(v)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "r152_circuits.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("wrote", OUT / "r152_circuits.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
