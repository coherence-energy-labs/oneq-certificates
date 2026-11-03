r"""R198: assemble evidence/circuit_distance_exact/results.json from the run records beside it.

    python collect_results.py            # reads evidence/circuit_distance_exact/records/, writes results.json

It computes nothing new: every number is copied from a record a run_lane.py subcommand (or the session's
sweep / certificate runs, archived verbatim under records/) printed, and each claim names the records it rests
on. It REFUSES a claim whose records are missing or disagree -- a summary with no producer is not evidence.

Claims and their classes:
  ibm_gross_all_T   d_circ = 10 for every T >= 2, both bases. MACHINE-CHECKED lower bound (22 checker
                    VERIFIED lines, memory-Z and memory-X, T = 1..11, weight <= 9) + window preconditions
                    (both bases) + verified full-DEM weight-10 witnesses (upper bound).
  three_colour_T2   d_circ = 11 at T = 2, both bases: search NONE <= 10 in both bases; certified iff the
                    checker records say VERIFIED; upper bound is the paper's (2603.05481), not ours.
  bb72_T2           d_circ = 6 at T = 2 (memory-Z): NONE <= 5, lifted weight-6 witness.
  morph72           the [[72,12,6]] morphing circuit of 2407.16336: d_circ = 6 for every T, both bases. MACHINE-
                    CHECKED for T <= 7 (14 time-shift certificates + space-time window preconditions); PROVEN for
                    every T >= 1 from the T = 1 certificates and FOLD_PROOF.md Theorem 1, whose hypotheses
                    fold_uniform.py checks (records/fold_uniform_hom_x.jsonl) -- else DERIVED beyond 7.
  morph144          the [[144,12,12]] morphing circuit: d_circ = 12 at T = 2 and for every T >= 1, both bases, by
                    the same theorem from the weight-11 1-round certificates (else the per-T fold check, T <= 13).
  trusted_base      every model's fault set is derived twice -- by stim's error analysis (what the exporter
                    projects from) and by dem_independent.py's backward Pauli pass -- and the two are equal on
                    every model searched or certified.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EVID = ROOT / "evidence" / "circuit_distance_exact"
RECORDS = EVID / "records"


def _shift_verified(text: str, w: int) -> bool:
    """A time-shift checker verdict for weight <= w and earliest round <= 1. The checker states it precisely as
    "no IRREDUCIBLE undetectable logical error ..." since 2026-10-21 (the Lean pass's finding F2: that is exactly
    what a scoped proof establishes, and all the time-shift lemma needs -- a minimum-weight solution is
    irreducible); certificates checked before then printed the same verified fact without the word."""
    return any(text.startswith(f"VERIFIED: no {q}undetectable logical error of weight <= {w} whose earliest round is <= 1")
               for q in ("", "irreducible "))


def jsonl(name: str) -> list[dict]:
    p = RECORDS / name
    if not p.exists():
        raise SystemExit(f"REFUSED: record {name} is missing")
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def text(name: str) -> str:
    p = RECORDS / name
    if not p.exists():
        raise SystemExit(f"REFUSED: record {name} is missing")
    return p.read_text()


def _window_theorem(kind: str):
    """WINDOW_PROOF.md Theorem 1 for `kind` ("ibm" or "3c"), per basis: the archived window_uniform.py run passes that
    script's own verdict (template, labelling, FIR/END, determinism, the closed form against stim at T = 16..32, the
    window structure, sizes, binding, witnesses, every control failing), and the 2026-10-22 audit's independent
    rebuild of the closed form equals stim's DEM at every T it tried with every one of its controls failing. A basis
    maps to None when any of that fails; the result is None when a record is missing."""
    if not all((RECORDS / f).exists() for f in ("window_uniform_run.jsonl", "window_review_formula.jsonl")):
        return None
    import window_uniform as WU
    run, rev = jsonl("window_uniform_run.jsonl"), jsonl("window_review_formula.jsonl")
    out = {}
    for b in "ZX":
        r = next((x for x in run if x["circuit"] == kind and x["basis"] == b), None)
        stim = [x for x in rev if x["step"] == "formula_vs_stim" and x["kind"] == kind and x["basis"] == b]
        ctl = [x for x in rev if x["step"] == "controls_must_be_false" and x["kind"] == kind and x["basis"] == b]
        ok = (r is not None and WU.verdict(r)[0] and len(r.get("CONTROLS", {})) >= 14
              and stim and all(x["all_equal"] and x["parts_stable_across_Tref_i0"] for x in stim)
              and ctl and all(x["all_fail_as_required"] for x in ctl))
        out[b] = dict(controls_failing=len(r["CONTROLS"]),
                      audit_stim_T=sorted({int(t) for x in stim for t in x["per_T"]})) if ok else None
    return out


def ibm_claim() -> dict:
    proofs = jsonl("ibm_proofs_w9.jsonl")
    cells = {(r["basis"], r["T"]) for r in proofs
             if r.get("checker_exit") == 0 and r.get("checker", "").startswith("VERIFIED") and r["W"] == 9}
    need = {(b, t) for b in "ZX" for t in range(1, 12)}
    if cells != need:
        raise SystemExit(f"REFUSED: IBM proofs missing or unverified for {sorted(need - cells)}")
    windows = jsonl("ibm_window.jsonl")
    if not all(w["holds"] for w in windows) or {w["basis"] for w in windows} != {"Z", "X"}:
        raise SystemExit("REFUSED: window preconditions not recorded as holding in both bases")
    wit = jsonl("ibm_witnesses.jsonl")
    if not all(w["verified"] and w["minus_one_fails"] and w["weight"] == 10 for w in wit) or len(wit) != 2:
        raise SystemExit("REFUSED: weight-10 witnesses not verified in both bases")
    thm = _window_theorem("ibm")
    proven = thm is not None and all(thm[b] for b in "ZX")
    base = ("the circuit's instruction list -- its fault set derived twice, by stim and by dem_independent.py, equal "
            "on every model (trusted_base) --, the projection, ccheck.c (its calculus PROVEN in Lean)")
    return dict(claim="IBM depth-8 gross cycle: d_circ = 10 for every T >= 2, memory-Z and memory-X",
                lower_bound=("MACHINE-CHECKED for T = 1..11: 22/22 proofs VERIFIED (weight <= 9 impossible); every T "
                             + ("by WINDOW_PROOF.md Theorem 1, which gives the window preconditions for EVERY "
                                "consecutive pair from finite hypotheses window_uniform.py machine-checks"
                                if proven else
                                "via the window lemma, whose preconditions are machine-checked on the computed models "
                                "and rest for T >= 12 on the builder's time-periodicity (DERIVED)")),
                upper_bound="verified full-DEM weight-10 witnesses (end-boundary type, identical window for T >= 2)",
                proofs=[{k: r[k] for k in ("basis", "T", "dem_sha256", "proof_sha256", "proof_bytes", "checker_s")}
                        for r in proofs],
                windows=windows, witnesses=wit, window_theorem=thm,
                evidence_class=(f"PROVEN for every T >= 2 from MACHINE-CHECKED premises: 22 VERIFIED certificates "
                                f"(T = 1..11) + WINDOW_PROOF.md Theorem 1 (a hand proof that survived an independent "
                                f"audit 2026-10-22; its hypotheses machine-checked) (trusted base: {base})" if proven else
                                f"MACHINE-CHECKED for T <= 11, DERIVED beyond (trusted base: {base}, the window lemma's "
                                f"argument, the builder's time-periodicity)"))


def three_colour_claim() -> dict:
    out = {}
    models = {r["basis"]: r["model_sha256"] for r in jsonl("3c_models.jsonl")}
    certs_path = RECORDS / "3c_T2_certificates_w10.jsonl"
    certs = {r["basis"]: r for r in (json.loads(x) for x in certs_path.read_text().splitlines() if x.strip())} \
        if certs_path.exists() else {}
    for b in "zx":
        m = re.search(r"w<=10: NONE\s+=> d_circ >= 11\s+\(([\d.]+)s wall, (\d+) nodes\)", text(f"3c_{b}_search_w10.log"))
        if not m:
            raise SystemExit(f"REFUSED: 3-colour {b.upper()} search record does not say NONE <= 10")
        c = certs.get(b.upper())
        # certified only by a VERIFIED checker line for weight <= 10 on the SAME model the search ran on
        verified = bool(c) and c.get("checker_exit") == 0 \
            and c.get("checker", "").startswith("VERIFIED: no undetectable logical error of weight <= 10 in this DEM") \
            and c.get("model_sha256") == models.get(b.upper())
        out[b.upper()] = dict(search_seconds=float(m.group(1)), search_nodes=int(m.group(2)), certified=verified,
                              certificate=({k: c[k] for k in ("model_sha256", "proof_sha256", "proof_bytes", "prover_s",
                                                             "checker_s")} if verified else None))
    certified = all(v["certified"] for v in out.values())
    wit = RECORDS / "3c_witnesses_w11.jsonl"
    witnesses = [json.loads(x) for x in wit.read_text().splitlines() if x.strip()] if wit.exists() else []
    upper = ("our own lifted full-DEM weight-11 witnesses: " + ", ".join(w["basis"] for w in witnesses if w["verified"])
             if witnesses else "the paper's extended-code bound (2603.05481); no weight-11 witness exhibited in this model yet")
    return dict(claim="3-colour cycle of 2603.05481: d_circ >= 11 at T = 2, both bases (= 11 with the upper bound below)",
                upper_bound=upper, witnesses=witnesses,
                bases=out, evidence_class="MACHINE-CHECKED" if certified else "MEASURED (engine search; certificate pending)",
                scope="T = 2 only; every-T needs weight 10 for T = 1..12 (see RESULTS.md)")


def three_colour_every_T_claim() -> dict:
    """d_circ = 11 for every T in 2..12 (and beyond by the window lemma, DERIVED): the time-shift sweep says no
    solution of weight <= 10 starts in round 0 or 1 for T = 1..12 (both bases); the time-shift preconditions hold
    on every consecutive pair; the weight-11 witnesses embed into every T = 2..12 and lift to verified full DEMs."""
    sweep = jsonl("3c_sweep_shift_w10.jsonl")
    need = {(b, t) for b in "ZX" for t in range(1, 13)}
    got = {(r["basis"], r["rounds"]) for r in sweep
           if r["none_up_to"] == 10 and r["found_weight"] is None and r["maxround"] == 1 and r["time_order"]}
    if got != need:
        raise SystemExit(f"REFUSED: time-shift sweep missing or not NONE <= 10 for {sorted(need - got)}")
    win = jsonl("3c_window.jsonl")
    if {(r["basis"], r["T"]) for r in win if r["time_shift_ok"]} != need:
        raise SystemExit("REFUSED: time-shift preconditions not recorded as holding on every pair (T, T+1), T = 1..12")
    emb = jsonl("3c_witness_embedding.jsonl")
    need_e = {(b, t) for b in "ZX" for t in range(2, 13)}
    if {(r["basis"], r["T"]) for r in emb if r.get("full_verified") and r["weight"] == 11} != need_e:
        raise SystemExit("REFUSED: the weight-11 witnesses are not verified in every T = 2..12")
    certs = RECORDS / "3c_shift_certificates_w10.jsonl"
    models = {(r["basis"], r["rounds"]): r["model_sha256"] for r in sweep}
    # certified only by VERIFIED time-shift checker lines for weight <= 10 on the very models the sweep searched
    certified = certs.exists() and {(r["basis"], r["rounds"]) for r in (json.loads(x) for x in certs.read_text().splitlines()
                                    if x.strip()) if r.get("checker_exit") == 0 and r.get("w") == 10
                                    and _shift_verified(r.get("checker", ""), 10)
                                    and r.get("model_sha256") == models.get((r["basis"], r["rounds"]))} == need
    thm = _window_theorem("3c")
    every = thm is not None and all(thm[b] for b in "ZX")
    return dict(claim="3-colour cycle of 2603.05481: d_circ = 11 for every T " + (">= 2" if every else "= 2..12")
                      + ", memory-Z and memory-X",
                lower_bound="no solution of weight <= 10 starting in round 0 or 1 for T = 1..12 (time-shift sweep, "
                            "both bases) + the time-shift lemma, whose preconditions "
                            + ("hold for EVERY consecutive pair by WINDOW_PROOF.md Theorem 1" if every else
                               "are machine-checked on every consecutive pair"),
                upper_bound="the T = 2 weight-11 witnesses embed into every T = 2..12 and lift to verified full DEMs"
                            + (" (and into every T, by the theorem's closed form)" if every else ""),
                sweep_seconds={b: round(sum(r["seconds"] for r in sweep if r["basis"] == b)) for b in "ZX"},
                window_theorem=thm,
                evidence_class=("PROVEN for every T >= 2 from MACHINE-CHECKED premises: 24 VERIFIED time-shift "
                                "certificates (T = 1..12) + WINDOW_PROOF.md Theorem 1 (audited 2026-10-22)"
                                if every and certified else
                                "every-T step PROVEN (WINDOW_PROOF.md Theorem 1, audited 2026-10-22); its T <= 12 "
                                "premises MEASURED (engine time-shift sweep; certificates pending)" if every else
                                ("MACHINE-CHECKED" if certified else "MEASURED (engine search; certificates pending)")
                                + " for T <= 12; DERIVED beyond (window lemma + the builder's time-periodicity)"))


def bb72_claim() -> dict:
    r = jsonl("bb72.jsonl")[0]
    if not (r["none_up_to"] == 5 and r["found_weight"] == 6 and r["lift"]["verified"]):
        raise SystemExit("REFUSED: [[72,12,6]] record does not show NONE <= 5 and a verified weight-6 lift")
    return dict(claim="[[72,12,6]] IBM-style cycle: d_circ = 6 at T = 2 (memory-Z)", record=r,
                evidence_class="MEASURED + MACHINE-CHECKED lower bound (tests/test_r198_circuit_distance_exact.py)")


def morph72_claim() -> dict:
    """Shaw & Terhal's morphing circuit with the [[72,12,6]] end-cycle code: d_circ = 6 for every T. The schedule is
    the authors' (gate-by-gate comparison); lower bound by 14 time-shift certificates (T = 1..7 = w + 2, both
    bases) on the swept models, with the space-time window preconditions (drift x, affine observables) on every
    consecutive pair; upper bound by weight-6 witnesses that lift, survive the redundancy meta-checks, and embed
    into every T = 2..7."""
    sched = jsonl("morph_schedule_vs_authors.jsonl")
    if len(sched) != 6 or not all(r["gates_identical"] and r["measurements_identical"]
                                  and r["contraction_rounds_identical"] for r in sched):
        raise SystemExit("REFUSED: the morphing schedule is not recorded identical to the authors' builder")
    need = {(b, t) for b in "ZX" for t in range(1, 8)}
    sweep = {(r["basis"], r["rounds"]): r["model_sha256"] for r in jsonl("morph72_sweep_shift_w5.jsonl")
             if r["circuit"] == "morph72" and r["none_up_to"] == 5 and r["found_weight"] is None
             and r["maxround"] == 1 and r["time_order"]}
    if set(sweep) != need:
        raise SystemExit(f"REFUSED: morph72 time-shift sweep missing or not NONE <= 5 for {sorted(need - set(sweep))}")
    if {(r["basis"], r["T"]) for r in jsonl("morph72_window.jsonl")
            if r["time_shift_ok"] and r.get("obs_affine") and r.get("drift") == [1, 0]} != need:
        raise SystemExit("REFUSED: space-time window preconditions not recorded on every pair (T, T+1), T = 1..7")
    certs = {(r["basis"], r["rounds"]) for r in jsonl("morph72_shift_certificates_w5.jsonl")
             if r.get("checker_exit") == 0 and r["w"] == 5 and _shift_verified(r.get("checker", ""), 5)
             and r["model_sha256"] == sweep.get((r["basis"], r["rounds"]))}
    wit = jsonl("morph72_witnesses.jsonl")
    if {w["basis"] for w in wit if w["verified"] and w["minus_one_fails"] and w["weight"] == 6
            and w["survives_metachecks"]} != {"Z", "X"}:
        raise SystemExit("REFUSED: weight-6 witnesses not verified (and meta-check-proof) in both bases")
    if {(r["basis"], r["T"]) for r in jsonl("morph72_witness_embedding.jsonl")
            if r.get("full_verified") and r["weight"] == 6} != {(b, t) for b in "ZX" for t in range(2, 8)}:
        raise SystemExit("REFUSED: the weight-6 witnesses are not verified in every T = 2..7")
    # every T >= 1 by FOLD_PROOF.md Theorem 1: d_circ(T) = d_circ(1), and the T = 1 certificates (bound to the swept
    # models) exclude every solution of weight <= 5 there; the T = 2 witnesses give the matching upper bound
    thm = _fold_theorem("morph72")
    proven = certs == need and thm is not None and all(thm[b] for b in "ZX")
    return dict(claim="morphing circuit of 2407.16336 with the [[72,12,6]] end-cycle code (mid-cycle [[144,12,12]], "
                      f"homomorphism x): d_circ = 6 for every T >= {1 if proven else 2}, memory-Z and memory-X -- the "
                      "authors' BP-OSD bound (<= 6) is exact",
                lower_bound="14/14 time-shift certificates VERIFIED (no solution of weight <= 5 starting in round 0 or "
                            "1, T = 1..7, both bases) + the space-time window lemma (drift x, affine observables), "
                            "machine-checked on every consecutive pair" if certs == need else "certificates incomplete",
                upper_bound="weight-6 witnesses (both bases) lift to verified full DEMs, survive the 6 redundancy "
                            "meta-checks, and embed unchanged into every T = 2..7",
                second_route=_morph72_fold_route(), fold_theorem=thm,
                evidence_class=("MACHINE-CHECKED for T <= 7 (time-shift certificates + window lemma); PROVEN for every "
                                "T >= 1 from MACHINE-CHECKED premises (the T = 1 certificates + FOLD_PROOF.md Theorem 1, "
                                "a hand proof audited 2026-10-22, its hypotheses checked by fold_uniform.py)" if proven else
                                ("MACHINE-CHECKED" if certs == need else "MEASURED (certificates incomplete)")
                                + " for T <= 7; DERIVED beyond (window lemma + the builder's space-time periodicity)"))


def _morph72_fold_route():
    """The same lower bound by a different lemma: the fold (d_proj(T) >= d_proj(1)) on T = 1..13 plus the 1-round
    certificates -- agreement of two independent routes, not a replacement for either."""
    fp = RECORDS / "fold_lemma.jsonl"
    if not fp.exists():
        return None
    certs = {r["basis"] for r in jsonl("morph72_shift_certificates_w5.jsonl") if r["rounds"] == 1
             and r.get("checker_exit") == 0 and r.get("checker", "").startswith("VERIFIED")}
    out = {}
    for b in "ZX":
        folds = {r["T"] for r in jsonl(fp.name) if r["circuit"] == "morph72" and r["basis"] == b and r["holds"]}
        tmax = max((t for t in range(1, 100) if set(range(1, t + 1)) <= folds), default=0)
        out[b] = f"fold lemma holds for T = 1..{tmax} + VERIFIED 1-round certificate: d_circ >= 6 for T <= {tmax}" \
            if b in certs and tmax >= 2 else None
    return out


def _morph144_certs() -> dict:
    """(basis, rounds) -> a VERIFIED weight <= 11 certificate record of the [[144,12,12]] morphing circuit."""
    p = RECORDS / "morph144_certificates_w11.jsonl"
    if not p.exists():
        return {}
    return {(r["basis"], r["rounds"]): r for r in jsonl(p.name) if r["circuit"] == "morph144" and r["w"] == 11
            and r.get("checker_exit") == 0
            and r.get("checker", "").startswith("VERIFIED: no undetectable logical error of weight <= 11 in this DEM")}


def morph144_claim() -> dict:
    """The [[144,12,12]] morphing circuit per basis: T = 2 from a certificate (else the searches), every T from the
    1-round model (certificate, else search) + the fold lemma + a witness verified in every T."""
    rows = [r for r in jsonl("morph144_search_T2.jsonl") if r["circuit"] == "morph144" and r["rounds"] == 2]
    certs = _morph144_certs()
    if not rows and not certs:
        raise SystemExit("REFUSED: no morph144 search or certificate record")
    out = {}
    for b in sorted({r["basis"] for r in rows} | {k[0] for k in certs}):
        rs = [r for r in rows if r["basis"] == b]
        c2 = certs.get((b, 2))
        shas = {r["model_sha256"] for r in rs} | ({c2["model_sha256"]} if c2 else set())
        if len(shas) != 1:
            raise SystemExit(f"REFUSED: morph144 {b} searches / certificate ran on different models")
        none = 11 if c2 else max(r["none_up_to"] for r in rs if r["found_weight"] is None or r["none_up_to"] > 0)
        out[b] = dict(none_up_to=none, certified=bool(c2), model_sha256=sorted(shas),
                      search_seconds=sum(r["seconds"] for r in rs),
                      certificate=({k: c2[k] for k in ("proof_sha256", "proof_bytes", "prover_s", "checker_s")}
                                   if c2 else None))
    wp = RECORDS / "morph144_witnesses.jsonl"
    wit = [json.loads(x) for x in wp.read_text().splitlines() if x.strip()] if wp.exists() else []
    for b, v in out.items():
        # equality only with OUR witness at exactly the next weight: verified on the full DEM, every proper subset
        # of one fewer fails, and it survives the redundancy meta-checks
        v["witness"] = next((w for w in wit if w["basis"] == b and w["rounds"] == 2 and w["verified"]
                             and w["minus_one_fails"] and w["survives_metachecks"]
                             and w["weight"] == v["none_up_to"] + 1), None)
    every = {b: _fold_every_T("morph144", b, 11, certs.get((b, 1))) for b in out}
    checked = all(v["certified"] and v["witness"] for v in out.values()) and set(out) == {"Z", "X"} \
        and all(e and e["certified"] for e in every.values())
    thm = _fold_theorem("morph144")
    proven = checked and thm is not None and all(thm[b] for b in "ZX")
    return dict(claim="morphing circuit of 2407.16336 with the [[144,12,12]] end-cycle code (mid-cycle [[288,12,18]]): "
                      + "; ".join(f"memory-{b}: d_circ {'=' if v['witness'] else '>='} {v['none_up_to'] + 1} at T = 2"
                                  + ((f", = {every[b]['d']} for every T >= 1" if proven
                                      else f", = {every[b]['d']} for every T = 1..{every[b]['tmax']}") if every[b] else "")
                                  for b, v in out.items())
                      + " -- equal to the end-cycle code's distance (12, the authors' integer program), so the circuit "
                        "loses none (their BP-OSD circuit bound: <= 12)",
                bases=out, every_T=every, fold_theorem=thm,
                evidence_class=("PROVEN for every T >= 1 from MACHINE-CHECKED premises: weight-11 certificates VERIFIED at "
                                "T = 1 (and T = 2) in both bases, weight-12 witnesses, and FOLD_PROOF.md Theorem 1 "
                                "(d_circ(T) = d_circ(1); a hand proof that survived an independent adversarial audit "
                                "2026-10-22) whose six T-independent hypotheses fold_uniform.py machine-checks with every "
                                "negative control failing; the per-T fold check (T = 1..13) is a second route" if proven else
                                "MACHINE-CHECKED: weight-11 certificates VERIFIED at T = 1 and T = 2 in both bases; every "
                                "T = 1..13 via the fold lemma (checked per T) from the certified 1-round model; DERIVED "
                                "beyond 13" if checked else
                                "MEASURED where a certificate is missing (uncertified cdist search; an independent CDCL "
                                "attempt could not decide weight 11, adversarial pass 2026-10-21); every T via the fold "
                                "lemma (checked per T) from the 1-round model, DERIVED beyond the checked T"))


def _fold_every_T(kind: str, basis: str, w: int, cert=None):
    """d_circ = w + 1 for every checked T, from the 1-round model: a certificate (else a search) NONE <= w there,
    the fold lemma d_proj(T) >= d_proj(1) holding for every T = 1..tmax, and a weight-(w + 1) witness verified in
    every T = 1..tmax. None if any piece is missing -- the claim then stays at what the T = 2 records say."""
    fp, sp, ep = RECORDS / "fold_lemma.jsonl", RECORDS / f"{kind}_search_T1.jsonl", RECORDS / f"{kind}_witness_embedding.jsonl"
    if not (fp.exists() and ep.exists()) or not (cert or sp.exists()):
        return None
    folds = {r["T"] for r in jsonl(fp.name) if r["circuit"] == kind and r["basis"] == basis and r["holds"]}
    tmax = max((t for t in range(1, 100) if set(range(1, t + 1)) <= folds), default=0)
    lower = [r for r in jsonl(sp.name) if r["basis"] == basis and r["rounds"] == 1 and r["none_up_to"] >= w] \
        if sp.exists() else []
    if cert and lower and cert["model_sha256"] != lower[0]["model_sha256"]:
        raise SystemExit(f"REFUSED: the {kind} {basis} 1-round certificate and search ran on different models")
    emb = {r["T"] for r in jsonl(ep.name) if r["basis"] == basis and r.get("full_verified") and r["weight"] == w + 1}
    if not (cert or lower) or tmax < 2 or not set(range(1, tmax + 1)) <= emb:
        return None
    how = ("1-round CERTIFICATE VERIFIED, NONE <= %d (prove %.0f s, check %.0f s)" % (w, cert["prover_s"], cert["checker_s"])
           if cert else "1-round search NONE <= %d (%.0f s)" % (w, lower[0]["seconds"]))
    return dict(d=w + 1, tmax=tmax, certified=bool(cert), lower=f"{how}; fold lemma holds for T = 1..{tmax}",
                upper="weight-%d witness verified in every T = 1..%d" % (w + 1, tmax),
                model_sha256_T1=(cert or lower[0])["model_sha256"])


def _fold_theorem(kind: str):
    """FOLD_PROOF.md Theorem 1 (d_circ(T) = d_circ(1) for EVERY T >= 1) applied to `kind`, per basis: its six
    T-independent hypotheses hold on the lane's schedule (homomorphism x) with every derived consequence, every template
    T and every recorded fold row agreeing and every one of its negative controls failing (`fold_uniform.py`,
    records/fold_uniform_hom_x.jsonl), and its Corollary 2 matched stim's DEM at fresh T with both controls failing
    (the 2026-10-22 adversarial audit, records/fold_review_cor2.jsonl). A basis maps to None when any of that fails;
    the whole result is None when a record is missing. The proof itself is a hand proof; the records are its premises."""
    if not ((RECORDS / "fold_uniform_hom_x.jsonl").exists() and (RECORDS / "fold_review_cor2.jsonl").exists()):
        return None
    rows, cor2 = jsonl("fold_uniform_hom_x.jsonl"), jsonl("fold_review_cor2.jsonl")
    out = {}
    for b in "ZX":
        r = next((r for r in rows if r["circuit"] == kind and r["basis"] == b and r["hom"] == "x"), None)
        mine = [c for c in cor2 if c["kind"] == kind and c["basis"] == b and c["hom"] == "x"]
        ctl = [c for c in mine if "control_flipped_window_equal" in c]
        ok = (r is not None and all(h["holds"] for h in r["hypotheses"].values())
              and all(v["holds"] for v in r["derived"].values()) and r["records"]["holds"]
              and len(r["controls"]) >= 20 and not any(r["controls"].values())
              and all(t["agrees"] for t in r["template"] + r["fresh"])
              and mine and all(c["equal"] for c in mine) and ctl
              and not any(c["control_flipped_window_equal"] or c["control_no_exp_noise_last_round_equal"] for c in ctl))
        out[b] = dict(hypotheses=sorted(r["hypotheses"]), controls_failing=len(r["controls"]),
                      template_T=[t["T"] for t in r["template"] + r["fresh"]],
                      stim_corollary2_T=sorted(c["T"] for c in mine)) if ok else None
    return out


def _m2026_theorem(row) -> bool:
    """fold_uniform.py's verdict on a morph2026 standard-schedule target: FOLD_PROOF.md's six hypotheses hold, every
    derived consequence and template T agrees, and every negative control (18: no archived per-T rows exist for
    these codes, so the two record controls do not apply) fails."""
    return bool(row and all(h["holds"] for h in row["hypotheses"].values())
                and row["hypotheses"]["PAL"]["root_layer_invariance"]
                and all(v["holds"] for v in row["derived"].values())
                and all(t["agrees"] for t in row["template"] + row["fresh"])
                and len(row["controls"]) >= 18 and not any(row["controls"].values()))


def _kernel_checked() -> dict:
    """(code, basis) -> the Lean theorem that makes its 1-round projected distance KERNEL-CHECKED: present in the
    Lean receipt on the standard axioms only, the receipt current for the sources (lean_receipt.check), and the
    module's model literal regenerated from the campaign's certified export (lean_kernel_proof.check)."""
    import lean_kernel_proof as LK
    import lean_receipt as LR
    if LR.check() or LK.check():
        return {}
    theorems = json.loads(LR.RECEIPT.read_text(encoding="utf-8"))["theorems"]
    out = {}
    for name, code, basis, _w in LK.TARGETS:
        t = f"CertR198.Kernel.{name}.distance"
        if t in theorems and set(theorems[t]) <= LR.STANDARD:
            out[(code, basis)] = t
    return out


def morph2026_claim() -> dict:
    """Shaw & Terhal's Abelian-2BGA morphing circuits (arXiv 2604.09797). The paper reports END-cycle code parameters
    and states that it did not check circuit-level distance (App. C.4); it names neither groups nor polynomials, so
    the codes are re-found by search (morph2026_codes.py, morph2026_trees.py). Per target and basis: the T = 1 search
    FOUND weight d with a lifted, full-DEM-verified, minimal witness; the same weight in every T = 2..w+2 (w = d - 1);
    unscoped certificates "no solution of weight <= w" at T = 1, 2 on exactly the searched models; time-shift
    certificates and the space-time window preconditions on every T = 1..w+2. Every T: PROVEN by FOLD_PROOF.md
    Theorem 1 where its hypotheses hold (the standard schedules: fold_uniform.py), else MACHINE-CHECKED to w + 2 and
    DERIVED beyond (the window lemma, as for morph72)."""
    rows = jsonl("morph2026_campaign.jsonl")
    authors = {r["target"]: r for r in jsonl("morph2026_vs_authors.jsonl")}
    if not any(r["target"].endswith("must FAIL)") and not r["gates_identical"] for r in authors.values()):
        raise SystemExit("REFUSED: the authors-builder comparison has no failing control")
    folds = {(r["circuit"], r["basis"]): r for r in jsonl("morph2026_fold_uniform.jsonl")}
    kernel = _kernel_checked()
    out = {}
    for inf in (r for r in rows if r["kind"] == "info"):
        code, per = inf["code"], {}
        for b in "ZX":
            mine = [r for r in rows if r.get("code") == code and r.get("basis") == b]
            srch = {r["rounds"]: r for r in mine if r["kind"] == "search"}
            s1 = srch.get(1)
            if not (s1 and s1["found_weight"] and s1.get("verified") and s1.get("every_deletion_fails")):
                per[b] = None
                continue
            d = s1["found_weight"]
            w = d - 1
            span = set(range(1, w + 3))
            exact = {T for T, r in srch.items() if r["found_weight"] == d and r["none_up_to"] >= w and r.get("verified")
                     and r.get("every_deletion_fails")}
            certs = {r["rounds"] for r in mine if r["kind"] == "cert" and r["w"] == w and r.get("checker_exit") == 0
                     and r.get("checker", "").startswith(f"VERIFIED: no undetectable logical error of weight <= {w} in")
                     and srch.get(r["rounds"]) and r["model_sha256"] == srch[r["rounds"]]["model_sha256"]}
            shift = {r["rounds"] for r in mine if r["kind"] == "shift_cert" and r["w"] == w and r.get("checker_exit") == 0
                     and _shift_verified(r.get("checker", ""), w)}
            windows = {r["T"] for r in mine if r["kind"] == "window" and r["time_shift_ok"] and r.get("obs_affine")
                       and r.get("drift") == [1, 0]}
            checked = span <= exact and {1, 2} <= certs and span <= shift and span <= windows
            proven = checked and _m2026_theorem(folds.get((code, b)))
            per[b] = dict(d_circ=d, loses=inf["d_end"] - d, exact_T=sorted(exact), certified_T=sorted(certs),
                          time_shift_certified_T=sorted(shift), window_T=sorted(windows),
                          every_T=("PROVEN for every T >= 1 (FOLD_PROOF.md Theorem 1, hypotheses checked by fold_uniform.py)"
                                   if proven else f"MACHINE-CHECKED for T <= {w + 2}, DERIVED beyond (window lemma)"
                                   if checked else None),
                          kernel_checked_T1=kernel.get((code, b)))
        kind = "tree" if code.startswith("t") else "standard"
        out[code] = dict(schedule=kind, n_end=inf["n_end"], k=inf["k"], end_cycle_d=inf["d_end"], degree=inf["degree"],
                         layers=inf["layers"], matches_authors_builder=(authors.get(code, {}).get("gates_identical")
                                                                        if kind == "standard" else None), bases=per)
    if not out:
        raise SystemExit("REFUSED: no morph2026 target in the campaign record")
    return dict(claim="circuit-level distance of 2604.09797's Abelian-2BGA morphing circuits (the paper reports end-cycle "
                      "parameters and did not check it): " + "; ".join(
                          f"{c} [[{t['n_end']},{t['k']},{t['end_cycle_d']}]] degree {t['degree']}: d_circ = "
                          + "/".join(str(t["bases"][b]["d_circ"]) if t["bases"][b] else "?" for b in "ZX") + " (Z/X)"
                          for c, t in sorted(out.items())),
                targets=out,
                evidence_class="per target and basis as `every_T` says: PROVEN for every T where FOLD_PROOF.md's "
                               "hypotheses hold (every standard schedule), MACHINE-CHECKED to w + 2 and DERIVED beyond "
                               "for the tree circuits; the distance itself is MACHINE-CHECKED (certificates + verified "
                               "minimal witnesses) at every T recorded; where `kernel_checked_T1` names a theorem, the "
                               "1-round distance is moreover a Lean theorem checked by the kernel (lean/CertR198 Exec + "
                               "Kernel: no C code and no compiler trusted)")


def dem_claim() -> dict:
    """Every model the claims rest on has its fault set derived twice and the two derivations agree exactly."""
    rows = jsonl("dem_independent.jsonl")
    need = {("ibm", 6, b, 2) for b in "ZX"} | {("ibm", 12, b, t) for b in "ZX" for t in range(1, 12)} \
        | {("3c", 12, b, t) for b in "ZX" for t in range(1, 13)} \
        | {("morph72", 12, b, t) for b in "ZX" for t in range(1, 8)} | {("morph144", 12, b, t) for b in "ZX" for t in (1, 2)}
    got = {(r["circuit"], r["l"], r["basis"], r["rounds"]) for r in rows if r["equal"] and r["only_independent"] == 0
           and r["only_stim"] == 0 and r["mechanisms"] == r["stim_mechanisms"]}
    if got != need:
        raise SystemExit(f"REFUSED: independent DEM derivation missing or unequal for {sorted(need - got)}")
    return dict(claim="every model's fault set derived twice -- stim's error analysis (export_model.mechanisms) and "
                      "dem_independent.py's backward Pauli pass -- is the same set, with the same detector coordinates "
                      "and observable count",
                models=len(need), mechanisms_min=min(r["mechanisms"] for r in rows),
                mechanisms_max=max(r["mechanisms"] for r in rows),
                scope="[[72,12,6]] T = 2, the IBM gross cycle T = 1..11, the 3-colour cycle T = 1..12, the morphing "
                      "circuits (72: T = 1..7, 144: T = 1..2); both bases",
                evidence_class="MEASURED (exact set equality on every model); stim still parses the instruction list")


def main():
    results = dict(lane="R198 circuit_distance_exact", ibm_gross=ibm_claim(), three_colour=three_colour_claim(),
                   three_colour_every_T=three_colour_every_T_claim(), bb72=bb72_claim(), morph72=morph72_claim(),
                   morph144=morph144_claim(), morph2026=morph2026_claim(), trusted_base=dem_claim())
    out = EVID / "results.json"
    out.write_text(json.dumps(results, indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
