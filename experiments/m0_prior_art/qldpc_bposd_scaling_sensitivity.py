r"""How the strong-baseline BP-OSD configurations depend on the DECLARED error rate and the min-sum scaling factor.

    python experiments/m0_prior_art/qldpc_bposd_scaling_sensitivity.py

WHY (paper 2 referee, 2026-11-02). Configuration B's failures at p = 0.05 (80 of 90 shots of true weight 7-10) against
none at p = 0.07 looked like a defect in the records. The records are exact -- re-decoding reproduces every stored
answer -- but the decoder is not well behaved: ldpc's min-sum at its default scaling factor 1 is sensitive to the
declared rate on this degenerate code. This measures, on the v3 shots (evidence/strong_baseline_v3), the number of
logical failures (judged against the sampled error, simulation truth) for:

  * A and B at their default scaling, told each rate in RATES, on the p = 0.02 and p = 0.05 syndromes;
  * B at the true rate, with each scaling factor in ALPHAS, on the p = 0.05 and p = 0.07 syndromes.

Post hoc; nothing here was preregistered.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))
OUT = ROOT / "evidence" / "qldpc_bposd_scaling_sensitivity.json"
V3 = ROOT / "evidence" / "strong_baseline_v3"
RATES = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.048, 0.05, 0.052, 0.055, 0.06, 0.07, 0.08, 0.1]
ALPHAS = [1.0, 0.9, 0.75, 0.625]


def main() -> int:
    from ldpc import BpOsdDecoder
    import ldpc
    from scipy.sparse import csr_matrix
    from qldpc_bb_gate import bb_gross_code, gf2_in_rowspace, gf2_row_reduce
    from qldpc_strong_baseline_gate import CONFIGS
    HX, HZ = bb_gross_code()
    n = HZ.shape[1]
    xb, xp = gf2_row_reduce(HX % 2)

    def failures(level: str, cfg: dict, rate: float, alpha: float | None = None) -> int:
        shots = [json.loads(x) for x in (V3 / f"p{level}_shots.jsonl").read_text(encoding="utf-8").splitlines()
                 if x.strip()]
        extra = {} if alpha is None else {"ms_scaling_factor": alpha}
        dec = BpOsdDecoder(csr_matrix(HZ), error_rate=rate, **cfg, **extra)
        bad = 0
        for s in shots:
            syn = np.zeros(HZ.shape[0], dtype=np.uint8)
            syn[s["syndrome_support"]] = 1
            e = np.asarray(dec.decode(syn)).astype(np.uint8)
            t = np.zeros(n, dtype=np.uint8)
            t[s["true_error_support"]] = 1
            f = (e ^ t) % 2
            bad += bool(((HZ @ f) % 2).any()) or not gf2_in_rowspace(xb, xp, f, n)
        return bad

    # WHERE 0.625 COMES FROM: the library's two defaults, read from the installed package, not quoted from memory --
    # the BpOsdDecoder class the campaign called, and the library's own Sinter wrapper around the same decoder
    import inspect
    from ldpc.sinter_decoders import SinterBpOsdDecoder
    probe = BpOsdDecoder(csr_matrix(np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)), error_rate=0.1)
    defaults = {"BpOsdDecoder": float(probe.ms_scaling_factor),
                "SinterBpOsdDecoder": float(inspect.signature(SinterBpOsdDecoder.__init__)
                                            .parameters["ms_scaling_factor"].default)}
    out = {"schema": "oneq-qldpc-bposd-scaling-sensitivity/2", "ldpc": ldpc.__version__,
           "post_hoc": "not preregistered", "shots_per_level": 200,
           "judged": "logical failure against the sampled error (simulation truth), v3 shots",
           "library_default_ms_scaling_factor": defaults,
           "declared_rate_scan": {}, "scaling_scan": {}}
    for level in ("0.02", "0.05"):
        for label in ("baseline_30_osdcs5", "strong_100_osdcs10"):
            row = {str(r): failures(level, CONFIGS[label], r) for r in RATES}
            out["declared_rate_scan"][f"p={level} {label}"] = row
            print(f"rates p={level} {label}: {row}", flush=True)
    # the matched control for "the unscaled decoder is erratic": the same declared-rate scan with scaling 0.625
    out["declared_rate_scan_scaled_0.625"] = {}
    for label in ("baseline_30_osdcs5", "strong_100_osdcs10"):
        row = {str(r): failures("0.05", CONFIGS[label], r, 0.625) for r in RATES}
        out["declared_rate_scan_scaled_0.625"][f"p=0.05 {label}"] = row
        print(f"rates scaled p=0.05 {label}: {row}", flush=True)
    for level in ("0.05", "0.07"):
        row = {str(a): failures(level, CONFIGS["strong_100_osdcs10"], float(level), a) for a in ALPHAS}
        out["scaling_scan"][f"p={level} strong_100_osdcs10"] = row
        print(f"alphas p={level}: {row}", flush=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
