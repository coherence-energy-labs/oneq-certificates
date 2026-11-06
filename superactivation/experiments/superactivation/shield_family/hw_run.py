"""The two-use superactivation experiment through oneq.hw, end to end: plan -> freeze -> predict -> guards ->
(dry) run -> certified verdict. Submits NOTHING by itself: a real submission is `oneq.hw.submit.submit`, behind
every guard, and only on the owner's explicit decision.

DESIGN (sampled, stratified -- the statistically valid kind when the run's circuits are a SAMPLE of P_d's
1056-unitary mixture): the 13 strata of hw_two_use.strata() + both-erased, each with its exact weight; C circuits
drawn per stratum from the stratum's law; S shots each. The verdict is oneq.hw.stats' TWO-STAGE Hoeffding lower
bound on F_avg against 5/6 (F_e > 3/4): unit means are the bounded draws, so circuit-to-circuit spread is part of
the error, as it must be.

  python hw_run.py plan   M C S [--live B --snapshot J]   read-only go/no-go: layout, frozen design, prediction,
                                                           QPU cost, pass probability, the guards' verdict
  python hw_run.py dryrun M C S                            (needs qiskit-aer: tw-venv) the same on FakeKingston,
                                                           then SamplerV2 LOCAL -> counts -> the certified verdict
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
import hw_predict as HP  # noqa: E402
import hw_two_use as H  # noqa: E402
from oneq.hw import engine as E  # noqa: E402
from oneq.hw import freeze as F  # noqa: E402
from oneq.hw import noise as N  # noqa: E402
from oneq.hw import snapshot as S  # noqa: E402
from oneq.hw import stats as T  # noqa: E402
from oneq.hw import submit as U  # noqa: E402

OUT = ROOT / "evidence" / "superactivation" / "shield_family"
BAR_F_AVG = 5 / 6
ALPHA = 0.01
SOURCES = [HERE / "hw_two_use.py", HERE / "hw_predict.py", HERE / "hw_run.py"] + \
          sorted((ROOT / "src" / "oneq" / "hw").glob("*.py"))


def all_strata(m: int) -> list:
    return [(key, w) for key, w in H.strata(m)] + [(("erased_erased", None, 0, 0), H.BRANCH_WEIGHT["erased_erased"])]


def draw(m: int, key: tuple, rng) -> H.Unit:
    return H.sample_unit(m, "erased_erased", rng) if key[0] == "erased_erased" else H.sample_in_stratum(m, key, rng)


def build(m: int, C: int, S_: int, be, snap: S.DeviceSnapshot, seed: int = 11):
    """(units, named ISA circuits, stat units, scores, stratum weights, layout choice)"""
    choice = HP.choose_layout(be, m, snap, N.NoiseParams())
    rng = np.random.default_rng(seed)
    units, named, stat_units, scores, weights = [], [], [], {}, {}
    for key, W in all_strata(m):
        sname = "|".join(map(str, key))
        weights[sname] = W
        for c in range(C):
            u = draw(m, key, rng)
            name = f"{sname}#{c}"
            tqc = HP.isa(be, H.circuit(m, u), choice.initial_layout, seed=7000 + len(named),
                         coupling_map=choice.coupling_map)
            hi = 1.0 if u.branch == "erased_erased" else 2.0
            units.append(u)
            named.append((name, tqc))
            stat_units.append(T.Unit(name, 1.0 / C, S_, tqc.num_clbits, 0.0, hi, sname))
            scores[name] = H.score(m, u)
    return units, named, stat_units, scores, weights, choice


def predicted_halfwidth(stat_units: list, weights: dict) -> float:
    by: dict = {}
    for u in stat_units:
        by.setdefault(u.stratum, []).append(u)
    var = sum((weights[b] * max(u.hi for u in us)) ** 2 / len(us) for b, us in by.items())
    return math.sqrt(math.log(1 / ALPHA) * var / 2)


def plan(a, be, snap) -> dict:
    units, named, stat_units, scores, weights, choice = build(a.m, a.C, a.S, be, snap)
    pred = json.loads((OUT / f"hw_predict_m{a.m}_{snap.name}_cz{N.CZ_FACTOR_R133:.2f}.json").read_text())
    if pred["snapshot_sha256"] != snap.sha256():
        raise SystemExit(f"the prediction on file was made on another calibration ({pred['snapshot_sha256'][:12]}); "
                         "re-run hw_predict.py on this snapshot first")
    hw = predicted_halfwidth(stat_units, weights)
    # the run estimator's sd, from the prediction's OWN per-stratum spread of unit means (std of the per-unit losses
    # around the exactly known ideal) plus shot noise (variance <= range^2 / 4 per shot). Deriving it from the
    # Hoeffding halfwidth instead inflates the spread -- which makes a pass look MORE likely, the wrong direction
    # for a guard whose job is to refuse a doomed run.
    var = 0.0
    for name, W in weights.items():
        row = pred["strata"].get(name if name != "erased_erased|None|0|0" else "erased_erased")
        spread = row["se_loss"] * math.sqrt(row["units"]) if row else 0.0
        hi = 1.0 if name.startswith("erased") else 2.0
        var += W ** 2 * (spread ** 2 + hi ** 2 / 4 / a.S) / a.C
    run_sd = math.sqrt(var)
    pp = U.pass_probability(pred["F_avg_pred"], pred["F_avg_se"], BAR_F_AVG, hw, run_sd)
    meta = dict(protocol="superactivation two-use (hw_two_use.py)", m=a.m, design="sampled-stratified",
                circuits_per_stratum=a.C, shots_per_circuit=a.S, alpha=ALPHA, bar_F_avg=BAR_F_AVG,
                snapshot_sha256=snap.sha256(), calibrated_utc=snap.calibrated_utc, backend=snap.name,
                layout=choice.initial_layout, layout_proxy=choice.best.proxy, noise=N.NoiseParams().describe(),
                prediction=dict(F_e=pred["F_e_pred"], F_e_se=pred["F_e_se"], F_avg=pred["F_avg_pred"],
                                F_avg_se=pred["F_avg_se"], record=f"hw_predict_m{a.m}_{snap.name}_cz2.29.json"),
                predicted_halfwidth_F_avg=hw, predicted_run_sd_F_avg=run_sd, predicted_pass_probability=pp,
                prereg_sha256=None,
                sources=[str(p) for p in SOURCES])
    rec, blob = F.freeze(named, meta)
    shots = {n: a.S for n, _ in named}
    p = U.Plan(tag=f"superact_m{a.m}", backend=snap.name, shots=shots,
               prereg_path=str(ROOT / "docs" / f"PREREG_superactivation_two_use_m{a.m}.md"), prereg_sha256="",
               predicted_pass_probability=pp)
    # the guards' verdict on THIS plan, as an unauthorized caller would see it (no prereg exists: the run is a no-go)
    refusals = U.preflight(p, rec, blob, authorized_flag=False, remaining_s=None, health_refusals=None)
    out = dict(meta, n_circuits=len(named), total_shots=p.total_shots, qpu_seconds_estimate=p.cost_s(),
               qpy_sha256=rec["qpy_sha256"], guards=refusals,
               verdict_if_submitted="NO-GO: predicted to fail" if pp < 0.5 else "GO on prediction")
    return dict(out=out, rec=rec, blob=blob, stat_units=stat_units, scores=scores, weights=weights, shots=shots)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["plan", "dryrun"])
    ap.add_argument("m", type=int)
    ap.add_argument("C", type=int, help="circuits per stratum")
    ap.add_argument("S", type=int, help="shots per circuit")
    ap.add_argument("--live")
    ap.add_argument("--snapshot")
    a = ap.parse_args()
    if a.mode == "dryrun":
        be = HP.fake_kingston()
    elif a.live:
        from qiskit_ibm_runtime import QiskitRuntimeService
        be = QiskitRuntimeService().backend(a.live)
    else:
        be = HP.fake_kingston()
    snap = (S.from_backend(be) if not a.snapshot
            else S.DeviceSnapshot.from_json(json.loads(pathlib.Path(a.snapshot).read_text(encoding="utf-8"))))
    P = plan(a, be, snap)
    out = P["out"]
    tag = f"m{a.m}_{snap.name}_{a.mode}"
    if a.mode == "dryrun":
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel
        # the dry run's "device": the fake backend's OWN noise model (what SamplerV2 local mode runs by default)
        counts_own = U.local_run(P["rec"], P["blob"], be, P["shots"])
        tallies_own = T.tally_all(P["stat_units"], counts_own, P["scores"])
        b = T.lower_bound("sampled", P["stat_units"], tallies_own, ALPHA, P["weights"])
        d = T.decide("superactivation: F_avg > 5/6 (F_e > 3/4)", b, BAR_F_AVG)
        # THE PLUMBING CHECK runs the SAME sampler path on the TARGET-based Aer model -- the model the engine is
        # validated against (<= 0.0025, evidence/hw/engine_vs_aer.json) -- and the engine scores the SAME frozen
        # circuits under that noise (vendor-face gates, the symmetric measure error), through the same weights.
        # What is left is shot noise plus that model gap: a bit-order, register, lookup or scoring fault anywhere
        # in freeze -> sampler -> counts -> tally is gross. (First run, against the fake's OWN model, read z = -13:
        # that model adds relaxation during every readout, which the target model does not -- a model difference,
        # localized per gate, not a plumbing fault.)
        counts = U.local_run(P["rec"], P["blob"], be, P["shots"],
                             simulator=AerSimulator(noise_model=NoiseModel.from_backend(be)))
        tallies = T.tally_all(P["stat_units"], counts, P["scores"])
        import dataclasses
        sym = tuple(be.target["measure"][(q,)].error for q in range(be.num_qubits))
        snap_sym = dataclasses.replace(snap, p01=sym, p10=sym)
        circuits = F.load(P["rec"], P["blob"])
        rng = np.random.default_rng(3)
        er = {u.name: E.expectation(circuits[u.name], P["scores"][u.name], snap_sym, N.NoiseParams(cz_factor=1.0),
                                    40, rng) for u in P["stat_units"]}
        eng = {k: r.mean for k, r in er.items()}
        by: dict = {}
        for u in P["stat_units"]:
            by.setdefault(u.stratum, []).append(u)
        est_counts = sum(P["weights"][s] * sum(tallies[u.name].mean for u in us) / len(us) for s, us in by.items())
        est_engine = sum(P["weights"][s] * sum(eng[u.name] for u in us) / len(us) for s, us in by.items())
        shot_var = sum(P["weights"][s] ** 2 * sum((tallies[u.name].s2 / tallies[u.name].n - tallies[u.name].mean ** 2)
                                                  / tallies[u.name].n for u in us) / len(us) ** 2
                       for s, us in by.items())
        # the engine's own trajectory error is part of the comparison (an exact unit contributes 0)
        eng_var = sum(P["weights"][s] ** 2 * sum(er[u.name].se ** 2 for u in us) / len(us) ** 2 for s, us in by.items())
        z = (est_counts - est_engine) / math.sqrt(shot_var + eng_var)
        est_own = sum(P["weights"][s] * sum(tallies_own[u.name].mean for u in us) / len(us) for s, us in by.items())
        out["dryrun"] = dict(simulator="SamplerV2 local mode (qiskit-aer, the fake backend's own noise model)",
                             F_avg_hat=b.estimate, F_avg_lower=b.lower, halfwidth=b.halfwidth,
                             F_e_hat=(3 * b.estimate - 1) / 2, F_e_lower=(3 * b.lower - 1) / 2, passed=d.passed,
                             model_sensitivity=dict(F_avg_fake_own_model=est_own, F_avg_target_model=est_counts,
                                                    difference=est_own - est_counts,
                                                    note="the fake's own model adds relaxation during readout"),
                             plumbing=dict(F_avg_from_counts=est_counts, F_avg_engine_same_circuits=est_engine,
                                           shot_sd=math.sqrt(shot_var), engine_sd=math.sqrt(eng_var), z=z,
                                           note="SamplerV2 runs the circuits UNSCHEDULED: Aer applies no idle noise, the engine does (+0.003-0.005 expected)",
                                           per_unit={u.name: dict(counts=tallies[u.name].mean, engine=eng[u.name],
                                                                  cz=circuits[u.name].count_ops().get("cz", 0))
                                                     for u in P["stat_units"]}),
                             counts_sha256=hashlib.sha256(json.dumps(counts_own, sort_keys=True).encode()).hexdigest())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"hw_run_{tag}.json").write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"{tag}: {out['n_circuits']} circuits x {a.S} shots = {out['total_shots']} shots, ~"
          f"{out['qpu_seconds_estimate']:.0f} QPU-s; predicted F_e {out['prediction']['F_e']:.4f}, pass probability "
          f"{out['predicted_pass_probability']:.2e}; {out['verdict_if_submitted']}")
    for g in out["guards"]:
        print("  guard:", g)
    if "dryrun" in out:
        r = out["dryrun"]
        print(f"  dry run: F_e hat {r['F_e_hat']:.4f}, lower bound {r['F_e_lower']:.4f} -> "
              f"{'PASS' if r['passed'] else 'NOT ESTABLISHED'}")
        pl = r["plumbing"]
        print(f"  plumbing (target model): F_avg from counts {pl['F_avg_from_counts']:.4f} vs engine on the same "
              f"circuits {pl['F_avg_engine_same_circuits']:.4f} (shot sd {pl['shot_sd']:.4f}, z {pl['z']:+.2f})")
        ms = r["model_sensitivity"]
        print(f"  model sensitivity: the fake's own model reads F_avg {ms['F_avg_fake_own_model']:.4f}, "
              f"{ms['difference']:+.4f} vs the target model (relaxation during readout)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
