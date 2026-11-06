"""The evidence writers added by the 2026-11-03 adjudication of paper 2, and the arithmetic each one scores with.

Each module below publishes into evidence/, so each is named here (tests/test_no_new_dark_evidence_writer.py), and
the part of each that decides a number -- the tier classifier, the paired exact McNemar, the set relation, the seed
rule, the per-dataset analysis -- is tested on inputs whose answer is known, apart from the run that produced the
committed artifact. The committed artifacts are then held to what the paper will say about them.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pathlib
import sys
from fractions import Fraction

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))
EV = ROOT / "evidence"

import qldpc_heldout_scaled_posthoc_all8 as ALL8  # noqa: E402
import qldpc_receipt_reverification as REVERIFY  # noqa: E402,F401  (named: it publishes evidence)
import qldpc_refsched_p3_equality as EQ  # noqa: E402
import qldpc_scale_probe_trees as SCALE  # noqa: E402
import qldpc_scaling_prereg_campaign as PRE  # noqa: E402
import qldpc_tier_by_receipt as TIER  # noqa: E402


def _fresh(doc: dict) -> bool:
    return all(hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == h for rel, h in doc["code_sha256"].items())


# ----------------------------------------------------------------------------- exact McNemar (two copies, one rule)
@pytest.mark.parametrize("n10, n01, p", [(0, 0, 1.0), (3, 0, 0.25), (0, 5, 0.0625), (2, 20, 0.000121)])
def test_exact_two_sided_mcnemar(n10, n01, p):
    for f in (ALL8.mcnemar, PRE.mcnemar):
        assert f(n10, n01) == pytest.approx(p, rel=2e-3)
        assert f(n10, n01) == f(n01, n10)


# ----------------------------------------------------------------------------- the receipt-defined tier classifier
class _Tiny(TIER.Instance):
    def __init__(self, checks, w):                    # an in-memory instance, no files
        self.checks = [tuple(c) for c in checks]
        self.supports = [frozenset(c) for c in self.checks]
        self.n = len(w)
        self.w = {i: Fraction(v) for i, v in enumerate(w)}


def test_classifier_reads_tier_by_what_the_receipt_proves():
    inst = _Tiny([(0, 1)], [1, 1])
    syn = [1]
    flat = [[0, [], 1, 1]]                            # L = 1 = U: attaining, and answer-free
    assert TIER.classify(inst, syn, [1], "flat", flat, [0]) == {"tier": "tier1", "dual": "attaining",
                                                                 "single_dual": True}
    inst2 = _Tiny([(0, 1), (1, 2)], [1, 3, 1])         # optimum {0, 2} weight 2; BP-OSD's {1} weighs 3
    flat2 = [[0, [], 1, 1], [1, [], 1, 1]]
    assert TIER.classify(inst2, [1, 1], [1], "flat", flat2, [0, 2])["tier"] == "tier2"
    tree = ["branch", 0, ["bound", [[0, [], 1, 1]]], ["bound", []]]
    assert TIER.classify(inst, syn, [1], "tree", tree, [0])["tier"] == "branching_bp_optimal"
    one_node = ["bound", [[0, [], 1, 1]]]
    assert TIER.classify(inst, syn, [1], "tree", one_node, [0])["tier"] == "tier1"


def test_classifier_refuses_an_answer_lighter_than_a_receipt():
    inst = _Tiny([(0, 1)], [1, 5])
    with pytest.raises(SystemExit):
        TIER.classify(inst, [1], [0], "flat", [[0, [], 1, 1]], [1])     # "proves" weight 5 while {0} weighs 1


def test_committed_receipt_tiers_partition_the_heldout_shots():
    d = json.loads((EV / "qldpc_tier_by_receipt.json").read_text(encoding="utf-8"))
    assert _fresh(d), "re-run experiments/m0_prior_art/qldpc_tier_by_receipt.py"
    t = d["heldout_totals"]
    assert t["N"] == 2019
    assert t["tier1"] + t["tier2"] + t["branching_bp_optimal"] + t["branching_repair"] == t["N"]
    assert t["single_dual_attaining"] + t["single_dual_lattice"] == t["tier1"] + t["tier2"]
    for lvl in d["strong_baseline"].values():
        for v in lvl.values():
            assert v["tier1"] >= v["recorded_tier1"]


# ----------------------------------------------------------------------------- the all-8 post-hoc re-decode
def test_committed_all8_redecode_is_held_by_its_controls_and_labelled_post_hoc():
    d = json.loads((EV / "qldpc_heldout_scaled_posthoc_all8.json").read_text(encoding="utf-8"))
    assert _fresh(d), "re-run experiments/m0_prior_art/qldpc_heldout_scaled_posthoc_all8.py"
    assert "NOT preregistered" in d["post_hoc"]
    assert list(d["rungs"]) == list(ALL8.RUNGS)
    for k, r in d["rungs"].items():
        assert all(r["controls"].values()), k
        assert r["alpha"]["1.0"]["reproduces_stored_candidate"] == r["shots"], k
        for a in ("1.0", "0.625"):
            assert r["alpha"][a]["lighter_than_receipt"] == 0, (k, a)
        h1, hs, pr = r["alpha"]["1.0"]["optimum_hit"], r["alpha"]["0.625"]["optimum_hit"], r["paired_optimum_hit"]
        assert h1 - pr["hit_at_1.0_only"] == hs - pr["hit_at_0.625_only"], k
        assert pr["exact_two_sided_mcnemar_p"] == ALL8.mcnemar(pr["hit_at_1.0_only"], pr["hit_at_0.625_only"])


# ----------------------------------------------------------------------------- the P2/P3 mechanism equality
def test_set_relation_names_every_case():
    a, b = {1, 2}, {1, 2, 3}
    assert EQ.relation(a, a)["relation"] == "equal"
    assert EQ.relation(a, b)["relation"] == "P2 subset of P3" and EQ.relation(a, b)["P3_only"] == 1
    assert EQ.relation(b, a)["relation"] == "P3 subset of P2"
    assert EQ.relation({1}, {2})["relation"] == "neither"


def test_committed_p3_equality_and_witness():
    d = json.loads((EV / "qldpc_refsched_p3_equality.json").read_text(encoding="utf-8"))
    assert _fresh(d), "re-run experiments/m0_prior_art/qldpc_refsched_p3_equality.py"
    for key in ("mechanisms_p2_observable", "mechanisms_13_observables"):
        assert d[key]["relation"] == "equal" and d[key]["P2"] == d[key]["P3"] == 2304
    assert d["witness"]["weight"] == 10 and d["witness"]["zero_syndrome_and_flips_p2_observable"]
    assert d["witness"]["no_9_column_subset_does_both"]
    assert d["stored_model"]["sha256"] == d["stored_model"]["manifest_sha256"]


# ----------------------------------------------------------------------------- the scale probe with trees
def test_committed_scale_probe_with_trees():
    p = EV / "qldpc_scale_robustness_trees.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    assert _fresh(d), "re-run experiments/m0_prior_art/qldpc_scale_probe_trees.py"
    assert list(d["scales"]) == [str(s) for s in SCALE.SCALES]
    for s, v in d["scales"].items():
        assert sum(v["by_route"].values()) == 300, s
        assert v["certified_exact"] == 300 - v["by_route"].get("unresolved", 0), s


# ----------------------------------------------------------------------------- the re-pin's re-verification record
# the bytes release v2026.11.03 pinned (its tree, read-only, hashed 2026-11-03): every history chain starts here, so
# an entry dropped from the front cannot leave a shorter chain that still closes
RELEASE_PIN = {"tools/external/tree_checker_b.py": "4f9b7022679be14ac09294513b757cd55b7269defa026350c2d6730d2939300b",
               "src/oneq/qldpc_check.py": "fa3ea1032535bfe021fb5e4a14050cadc736d934332fb54bd5b3a3586169ede2"}


def _pin_chain(current: str, hist: list, checker: str) -> dict:
    """A moved pin's history is one unbroken chain: entry 0 names the released bytes, entry k names the bytes record
    k replaced, record k names the bytes entry k+1 replaced, and the LAST record names the bytes pinned now. Returns
    the last record."""
    assert hist, f"{checker}: a moved pin with no history"
    assert hist[0]["sha256"] == RELEASE_PIN[checker], f"{checker}: the history does not start at the released pin"
    rec = None
    for k, h in enumerate(hist):
        p = ROOT / h["reverification_record"]
        assert p.is_file(), f"{checker}: history entry {k} points at a missing record {h['reverification_record']}"
        rec = json.loads(p.read_text(encoding="utf-8"))
        assert rec["replay"]["complete"] and rec["replay"]["receipts_passed"] == 2042
        nxt = hist[k + 1]["sha256"] if k + 1 < len(hist) else current
        assert rec["code_sha256"][checker] == nxt != h["sha256"], f"{checker}: the chain breaks at entry {k}"
    assert len({h["reverification_record"] for h in hist}) == len(hist), "two re-pins share one record"
    return rec


def test_every_moved_pin_rests_on_a_fresh_reverification_record():
    # qldpc_receipt_reverification.py writes ONE record per re-pin (--out), and an earlier record is never
    # overwritten: every pin that moved keeps each previous value beside it with a pointer to the record it moved on,
    # the chain of records accounts for every value the pin has held, and the newest record names the bytes now in
    # the tree -- a pin never moves silently, and an old pointer still shows what that move rested on
    add = json.loads((EV / "qldpc_heldout_addendum.json").read_text(encoding="utf-8"))
    corpus = json.loads((EV / "qldpc_tree_corpus.json").read_text(encoding="utf-8"))
    tb = "tools/external/tree_checker_b.py"
    last = {_pin_chain(add["tree_checker_b_sha256"], add["tree_checker_b_previous_pins"], tb)["date"],
            _pin_chain(corpus["checker_b_sha256"], corpus["checker_b_previous_pins"], tb)["date"]}
    assert len(last) == 1
    newest = add["tree_checker_b_previous_pins"][-1]["reverification_record"]
    assert corpus["checker_b_previous_pins"][-1]["reverification_record"] == newest
    rec = json.loads((ROOT / newest).read_text(encoding="utf-8"))
    assert _fresh(rec), f"re-run experiments/m0_prior_art/qldpc_receipt_reverification.py --out {newest}"
    ann = json.loads((ROOT / "docs" / "evidence_annotations.json").read_text(encoding="utf-8"))
    refs = [r for doc in ann.get("documents", ann if isinstance(ann, list) else []) for c in doc.get("claims", [])
            for r in c.get("references", []) if r.get("path") == "src/oneq/qldpc_check.py" and r.get("previous_pins")]
    assert len(refs) == 1, "the annotation that pins the production checker"
    pc = "src/oneq/qldpc_check.py"
    assert _pin_chain(refs[0]["sha256"], refs[0]["previous_pins"], pc)["code_sha256"] == rec["code_sha256"]
    assert refs[0]["previous_pins"][-1]["reverification_record"] == newest


# ----------------------------------------------------------------------------- the preregistered campaign's rules
def test_the_preregistration_states_every_seed_the_runner_derives():
    text = PRE.PREREG.read_text(encoding="utf-8")
    for name in PRE.SEED_NAMES:
        assert str(PRE.seed_for(PRE.SEED_DOMAIN, name)) in text, name
        assert PRE.seed_for(PRE.SMOKE_DOMAIN, name) != PRE.seed_for(PRE.SEED_DOMAIN, name)
    assert PRE.seed_for(PRE.SEED_DOMAIN, "bb_refsched") < 2 ** 31 - 1


def test_the_campaign_analysis_on_known_records():
    def rec(s1, s2, e1=False, e2=False):
        return {"receipt": {"kind": "flat"}, "answers": {"1.0": {"score": s1}, "0.625": {"score": s2}},
                "logical_error": {"1.0": e1, "0.625": e2, "receipt": False}}
    recs = [rec("tier1", "tier1"), rec("miss", "tier1", True, False), rec("miss", "optimum_branching", True, False),
            rec("tier1", "miss", False, True), rec("not_a_correction", "tier1", True, False)]
    out = PRE.analyse("x", recs, sampled=6, trivial=1, D={"seed": 7, "logical": object(), "obs_cols": None})
    assert out["per_configuration"]["1.0"]["optimum_hit"] == 2
    assert out["per_configuration"]["0.625"]["optimum_hit"] == 4
    assert out["per_configuration"]["0.625"]["optimum_on_branching_tree"] == 1
    assert out["paired_optimum_hit"] == {"hit_at_1.0_only": 1, "hit_at_0.625_only": 3,
                                         "exact_two_sided_mcnemar_p": PRE.mcnemar(1, 3)}
    assert out["paired_logical_error_secondary"]["error_at_1.0_only"] == 3
    assert "pooled" not in json.dumps(out)


def test_a_smoke_run_never_writes_under_evidence(monkeypatch):
    target = EV / "scratch_smoke_must_not_exist"
    monkeypatch.setattr(sys, "argv", ["x", "--smoke", "1", "--dataset", "uniform_p01", "--out-dir", str(target)])
    with pytest.raises(SystemExit, match="never writes under evidence"):
        PRE.main()
    assert not target.exists()
