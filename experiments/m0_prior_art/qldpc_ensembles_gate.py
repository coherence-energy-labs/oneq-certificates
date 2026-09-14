r"""The random-ensemble claims, as ARTIFACTS instead of scratch memories.

The paper audit found its own gap: five ensemble results (lazy-equals-
enumeration, capped-facet ordering and tightness, the three-tier ladder
against brute force, cut-lift, MILP agreement) were quoted from scratch
probes -- reproducible in spirit, backed by nothing the provenance gate
could sweep. This gate reruns all five with fixed seeds and writes ONE
artifact, so every ensemble number in the paper traces like the rest.
"""

from __future__ import annotations

import itertools
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from oneq.qldpc_cert import (QldpcCertificate, check_qldpc,  # noqa: E402
                             exact_milp, feldman_dual, feldman_dual_lazy)


def exact_min(n, checks, weights, syndrome):
    best = None
    for r in range(n + 1):
        for combo in itertools.combinations(range(n), r):
            e = set(combo)
            if all(len(e & set(c)) % 2 == syndrome[j]
                   for j, c in enumerate(checks)):
                w = sum(weights[i] for i in e)
                if best is None or w < best - 1e-12:
                    best = w
        if best is not None and r >= 2 \
                and best <= min(weights.values()) * (r - 1):
            break
    return best


def draw(rnd, max_deg=6):
    n = rnd.randint(6, 12)
    m = rnd.randint(3, max(4, n // 2))
    checks = [tuple(sorted(rnd.sample(range(n),
                                      rnd.randint(2, min(max_deg, n)))))
              for _ in range(m)]
    weights = {i: round(rnd.uniform(0.5, 3.0), 3) for i in range(n)}
    true_e = set(rnd.sample(range(n), rnd.randint(0, n // 2)))
    syndrome = [len(true_e & set(c)) % 2 for c in checks]
    return n, checks, weights, syndrome


def main() -> int:
    out: dict = {"schema": "oneq-qldpc-ensembles/1",
                 "seeds": {"lazy": 9, "capped": 5, "ladder": 20260801,
                           "milp": 99}}

    # 1) lazy == enumeration, 40 codes
    rnd = random.Random(9)
    worst = 0.0
    for _ in range(40):
        _n, checks, weights, syn = draw(rnd)
        _c, bf = feldman_dual(checks, weights, syn)
        _c2, bl = feldman_dual_lazy(checks, weights, syn)
        worst = max(worst, abs(bf - bl))
    out["lazy_vs_enum"] = {"codes": 40, "worst_abs_diff": worst}
    assert worst < 1e-9

    # 2) capped-facet ordering + tightness, 40 codes
    rnd = random.Random(5)
    viol = tf = tc = 0
    for _ in range(40):
        n, checks, weights, syn = draw(rnd)
        best = exact_min(n, checks, weights, syn)
        _c, bf = feldman_dual(checks, weights, syn)
        _c2, bc = feldman_dual_lazy(checks, weights, syn)  # warm not needed
        _c3, bcap = feldman_dual(checks, weights, syn, max_facet_size=3)
        if bcap > bf + 1e-9 or bf > best + 1e-6:
            viol += 1
        tf += abs(bf - best) < 1e-6
        tc += abs(bcap - best) < 1e-6
    out["capped_facets"] = {"codes": 40, "ordering_violations": viol,
                            "tight_full": tf, "tight_capped_F3": tc}
    assert viol == 0

    # 3) three-tier ladder vs brute force, 60 codes (base LP, no cuts)
    #    + cut lift on the same stream
    rnd = random.Random(20260801)
    proven = gaps = impossible = 0
    cut_tight = 0
    for t in range(60):
        n = rnd.randint(6, 14)
        m = rnd.randint(3, max(4, n // 2))
        checks = [tuple(sorted(rnd.sample(range(n),
                                          rnd.randint(2, min(5, n)))))
                  for _ in range(m)]
        weights = {i: round(rnd.uniform(0.5, 3.0), 3) for i in range(n)}
        true_e = set(rnd.sample(range(n), rnd.randint(0, n // 2)))
        syn = [len(true_e & set(c)) % 2 for c in checks]
        best = exact_min(n, checks, weights, syn)
        cert, bound = feldman_dual(checks, weights, syn)
        if bound > best + 1e-6:
            impossible += 1
        elif abs(bound - best) < 1e-6:
            proven += 1
        else:
            gaps += 1
        _c, bcut = feldman_dual(checks, weights, syn, rpc_rounds=8)
        cut_tight += abs(bcut - best) < 1e-6
    out["ladder_vs_bruteforce"] = {
        "codes": 60, "base_lp_tight": proven, "honest_gaps": gaps,
        "impossible_bucket": impossible, "with_cuts_tight": cut_tight}
    assert impossible == 0

    # 4) MILP tier vs brute force, 40 codes
    rnd = random.Random(99)
    agree = 0
    for _ in range(40):
        n, checks, weights, syn = draw(rnd, max_deg=5)
        best = exact_min(n, checks, weights, syn)
        _sup, mw = exact_milp(checks, weights, syn)
        agree += mw is not None and abs(mw - best) < 1e-6
    out["milp_vs_bruteforce"] = {"codes": 40, "exact_agreement": agree}
    assert agree == 40

    dest = ROOT / "evidence" / "qldpc_ensembles.json"
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"evidence -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
