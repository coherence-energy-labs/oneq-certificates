r"""Figures for paper 1, drawn from evidence artifacts and nothing else.

Every plotted value is read from a JSON artifact at run time; no number in
this file is data. Output PDFs carry no creation date, so a re-run on the
same artifacts is byte-stable enough to diff.

    python paper/tools/make_figures.py
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
EV = ROOT / "evidence"
OUT = ROOT / "paper" / "certifying-decoders" / "figures"

COLUMN_IN = 3.375                     # APS single column
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0, "pdf.fonttype": 42,
})
META = {"CreationDate": None, "Producer": None, "Creator": None}


def _load(name: str) -> dict:
    return json.loads((EV / name).read_text(encoding="utf-8"))


def _parse(config: str) -> tuple[int, str, int]:
    """'d7_at_q6_7_X_r30' -> (7, 'X', 30). The qubit label contains an
    underscore, so fields are read from the ends, never by position."""
    parts = config.split("_")
    return int(parts[0][1:]), parts[-2], int(parts[-1][1:])


def generation_ladder() -> pathlib.Path:
    """Uncertified fraction per Willow configuration, generation 1 -> 6.

    A configuration with NO uncertified shot is drawn as an open downward
    triangle at 1/n: the true value is below it, and a log axis cannot
    show zero."""
    h = _load("google_generation_history.json")["generations"]
    configs = sorted(h["1"]["per_config"], key=lambda c: (_parse(c)[0], _parse(c)[2], _parse(c)[1]))
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.7))
    colour = {(3, 13): "#9ecae1", (3, 30): "#3182bd", (5, 13): "#a1d99b",
              (5, 30): "#31a354", (7, 13): "#fdae6b", (7, 30): "#d62728"}
    seen = set()
    for c in configs:
        d, basis, rounds = _parse(c)
        gens, unc, zeros = [], [], []
        for g in sorted(h, key=int):
            pc = h[g]["per_config"].get(c)
            if not pc:
                continue
            gens.append(int(g))
            if pc["certified"] == pc["nontrivial"]:
                unc.append(1 / pc["nontrivial"])
                zeros.append((int(g), 1 / pc["nontrivial"]))
            else:
                unc.append(1 - pc["rate"])
        label = None if (d, rounds) in seen else f"d={d}, {rounds} rounds"
        seen.add((d, rounds))
        ax.plot(gens, unc, ls="-" if basis == "X" else "--", marker="o", ms=2.2,
                color=colour[(d, rounds)], label=label)
        for g, v in zeros:
            ax.plot([g], [v], marker="v", ms=4, mfc="white", mec=colour[(d, rounds)], ls="none")
    ax.set_yscale("log")
    ax.set_xlabel("producer generation")
    ax.set_ylabel("uncertified fraction")
    ax.set_xticks(range(1, 7))
    ax.set_ylim(1e-5, 1)
    ax.grid(True, which="major", lw=0.3, alpha=0.5)
    # above the axes: every region inside is crossed by some configuration
    ax.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              handlelength=1.6, columnspacing=0.9, borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    path = OUT / "generation_ladder.pdf"
    fig.savefig(path, metadata=META)
    plt.close(fig)
    return path


def gap_tails() -> pathlib.Path:
    """P(gap > g) on shots whose MWPM correction the exact checker proved.

    The curve starts at the fraction of shots with a nonzero gap; the rest of
    the mass sits at exactly zero, which a log axis cannot show, so it is
    stated in the legend."""
    doc = _load("gap_distribution_exact_rule.json")
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_IN, 2.0), sharey=True)
    colours = {3: "#3182bd", 5: "#31a354", 7: "#d62728"}
    for ax, dec, title in ((axes[0], "bp_osd", "BP-OSD"),
                           (axes[1], "belief_matching", "belief-matching")):
        for row in doc["rows"]:
            g = np.array(row["decoders"][dec]["per_shot_gap_vs_proven_optimum"])
            n = g.size
            pos = np.sort(g[g > 1e-9])
            k = pos.size
            xs = np.concatenate(([pos[0] / 3], pos))
            ys = np.concatenate(([k / n], (k - np.arange(1, k + 1)) / n))
            ax.step(xs[:-1], ys[:-1], where="post", color=colours[row["distance"]],
                    label=f"d={row['distance']}: {100 * (1 - k / n):.0f}% zero")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("excess weight $g$")
        ax.grid(True, which="major", lw=0.3, alpha=0.5)
        ax.legend(frameon=False, loc="lower left", handlelength=1.2)
    axes[0].set_ylabel("fraction with gap $> g$")
    fig.tight_layout(pad=0.3)
    path = OUT / "gap_tails.pdf"
    fig.savefig(path, metadata=META)
    plt.close(fig)
    return path


def checker_scaling() -> pathlib.Path:
    doc = _load("checker_scaling.json")
    rows = doc["rows"]
    x = np.array([r["nnz"] for r in rows], float)
    y = np.array([r["check_ms_median"] for r in rows], float)
    fit = doc["fit_time_vs_nnz"]
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.0))
    ax.loglog(x, y, "o", ms=3.5, color="#1f77b4", label="median check time")
    xx = np.geomspace(x.min(), x.max(), 50)
    ax.loglog(xx, np.exp(fit["intercept"]) * xx ** fit["exponent"], "-", color="#555555",
              label=f"fit: exponent {fit['exponent']:.3f}, $R^2$ {fit['r_squared']:.3f}")
    ax.set_xlabel("decoding-graph edges")
    ax.set_ylabel("check time per shot (ms)")
    ax.grid(True, which="major", lw=0.3, alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout(pad=0.3)
    path = OUT / "checker_scaling.pdf"
    fig.savefig(path, metadata=META)
    plt.close(fig)
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (generation_ladder, gap_tails, checker_scaling):
        print("wrote", f().relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
