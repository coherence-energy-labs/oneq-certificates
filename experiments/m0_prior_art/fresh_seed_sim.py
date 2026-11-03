r"""Paper 1's simulation rate on shots no run of this project has sampled: fresh seeds, the frozen base ladder.

    python experiments/m0_prior_art/fresh_seed_sim.py --workers 3      # certify, every receipt kept
    python experiments/m0_prior_art/fresh_seed_sim.py --merge          # summary

WHY (paper 1, second referee, 2026-11-02, minor 9). The simulation's headline rate is measured on the million-shot run's
seed plan (chunk seeds 1..4,000, plus supplements with other seed bases), and whether the producer was developed on
those shots is not recorded. This certifies 10,000 shots of the same configuration -- distance 5, 25 rounds,
p = 0.005 -- drawn with chunk seeds 20,261,102,001 .. 20,261,102,040, a date-coded range no run of this project used,
through the epsilon rerun's own unmodified machinery (sampler, producer call, checker, receipt format, forgery
battery) under the name `fresh_sim_d5_r25`. POST HOC: decided and run after the referee's report, not preregistered;
it is reported whatever it shows.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))
import epsilon_rerun as er  # noqa: E402

NAME = "fresh_sim_d5_r25"
SEED_LO = 20_261_102_001
er.CONFIGS[NAME] = dict(er.CONFIGS["sim_d5_r25"], seeds=(SEED_LO, SEED_LO + 39),
                        reproduces="post hoc, fresh seeds (paper 1 second referee, minor 9); not preregistered")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        er.merge(NAME)
    else:
        er.run(NAME, a.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
