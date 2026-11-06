"""Error bars that include the twirl's own sampling: resample a twirled job's RANDOMIZATIONS, not its shots.

A twirled PUB runs R randomizations of n shots each, flattened randomization-major; the execution span's slice says
so (`oneq.hw.campaign._spans`, kind TwirledSliceSpanV2: shape (R, n), flat). The n shots of one randomization share a
single Pauli frame, so they are NOT independent draws of the twirled channel. A coherent error that the twirl turns
into a Pauli channel only ON AVERAGE over frames shows as a spread between the R frame means -- a spread that counts
cannot show and a shot-level error bar cannot see. R169 (2026-10-17) measured it on 148-149: the shot-level sd of
R146's p was 1.4-2.7x too small there, and R161's preregistered S3 contrasts fell from z -4.4 to -2.0 once it was
counted.

The operator here is estimator-agnostic. Any statistic computed from per-circuit COUNTS is re-run on counts rebuilt
from resampled blocks:
  mode "randomization" -- each circuit's R randomization blocks drawn with replacement, independently per circuit
                          (every PUB draws its own frames), each block keeping its own n shots;
  mode "shot"          -- each circuit's R*n shots drawn with replacement: the error bar counts leave out the frames,
                          computed by the SAME route, so the ratio of the two is apples to apples (`inflation`).
A bootstrap over B blocks underestimates the variance of a mean of B iid blocks by (B - 1) / B; the sd is scaled by
sqrt(B / (B - 1)) with B the SMALLEST block count among the resampled circuits (exact when every circuit has the same
B, conservative otherwise). With R = 8 the scale is 1.069, which is not negligible.

A TRUNCATED last frame. A TwirledSliceSpanV2 also records the shots REQUESTED (pub_shots). When fewer than R * n were
requested, the runtime ran R frames of n and returned the first pub_shots of the randomization-major flattening
(qiskit_ibm_runtime's TwirledSliceSpanV2.mask: the merged axis cut to pub_shots), so the LAST frame is short. R183
asked for 250 shots and ran 8 x 32; until 2026-10-18 `shape` refused that, and the job's frame-counting bars and
frame-homogeneity map both came back empty. A short last frame is kept as one smaller block (`Shots.last`); the
bootstrap draws whole frames with replacement, each with its own shots (a cluster bootstrap with unequal clusters).
The truncation is accepted only for kind TwirledSliceSpanV2, and only when the span's own count of this PUB's shots
(`pubs`, the mask's sum) or its recorded pub_shots says exactly n_shots.

Refusals (RandomizationRefused, before any number is computed): a PUB whose shots are not covered by exactly one span
slice of a known kind; a slice that does not cover its shots; mode "randomization" on a circuit that ran ONE frame
(untwirled: there is no frame spread to sample; use mode "shot"); an estimator that returns a non-finite value on any
resample (a statistic undefined on a plausible resample is a fragility the error bar must not hide); fewer than two
resamples.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

MODES = ("randomization", "shot")
TRUNCATING_SPAN = "TwirledSliceSpanV2"


class RandomizationRefused(ValueError):
    """a job's shots cannot be tied to their randomizations, or a bootstrap cannot be computed faithfully"""


# ================================================================== tying shots to frames
def _per_pub(table, pub_index: int):
    """a span's per-PUB entry: a flight record fresh from the runtime keys it by int, one read back from JSON by str"""
    table = table or {}
    return table.get(str(pub_index), table.get(int(pub_index)))


def shape(spans: list, pub_index: int, n_shots: int) -> tuple:
    """(randomizations, shots per randomization) of one PUB from the execution spans' slices. Exactly one span must
    carry the PUB's slice, covering every shot:
      twirled   -- [[R, n], flat=True, [0, R, None], [0, n, None]]  (TwirledSliceSpanV2, randomization-major)
      untwirled -- [[n], [0, 1, None], [0, n, None]]                (DoubleSliceSpan: one frame, returned as (1, n))
    A TwirledSliceSpanV2 whose n_shots < R * n is a TRUNCATED job (module docstring): (ceil(n_shots / n), n) is
    returned, the last frame short, provided the span's own shot count for the PUB (or its recorded pub_shots) is
    exactly n_shots. Pass the n to `from_bitstrings` / `from_bits`."""
    found = []
    for s in spans:
        sl = _per_pub(s.get("slices"), pub_index)
        if sl is None:
            continue
        if len(sl) == 4:
            shp, flat, rs, ss = sl
            if not flat or len(shp) != 2 or list(rs) != [0, shp[0], None] or list(ss) != [0, shp[1], None]:
                raise RandomizationRefused(f"PUB {pub_index}: twirled slice {sl} is not one randomization-major block")
            found.append((int(shp[0]), int(shp[1]), s))
        elif len(sl) == 3:
            shp, a_s, ss = sl
            if len(shp) != 1 or list(a_s) != [0, 1, None] or list(ss) != [0, shp[0], None]:
                raise RandomizationRefused(f"PUB {pub_index}: untwirled slice {sl} is not one whole binding")
            found.append((1, int(shp[0]), s))
        else:
            raise RandomizationRefused(f"PUB {pub_index}: slice {sl} is of no known kind")
    if len(found) != 1:
        raise RandomizationRefused(f"PUB {pub_index}: {len(found)} spans carry its slice; exactly one is required")
    R, n, span = found[0]
    if R * n == n_shots:
        return R, n
    if R > 1 and span.get("kind") == TRUNCATING_SPAN and 0 < n_shots < R * n:
        requested = _per_pub(span.get("pub_shots"), pub_index)
        masked = _per_pub(span.get("pubs"), pub_index)
        if requested is None and masked is None:
            raise RandomizationRefused(f"PUB {pub_index}: slice {R} x {n} holds more than its {n_shots} shots and the "
                                       "span records neither its pub_shots nor its own count of them")
        for what, v in (("pub_shots", requested), ("shot count", masked)):
            if v is not None and int(v) != n_shots:
                raise RandomizationRefused(f"PUB {pub_index}: the span's {what} is {v}, not the {n_shots} returned")
        return -(-n_shots // n), n
    raise RandomizationRefused(f"PUB {pub_index}: slice {R} x {n} does not cover its {n_shots} shots")


@dataclass(frozen=True)
class Shots:
    """one circuit's shots in shot order as codes into `keys` (qiskit bitstrings, clbit 0 rightmost), R frames of n,
    the LAST one holding only `last` shots when the runtime truncated it (`shape`)"""
    keys: tuple
    code: np.ndarray
    R: int
    n: int
    last: int | None = None

    def __post_init__(self):
        if self.last is not None and not (self.R >= 2 and 1 <= self.last < self.n):
            raise RandomizationRefused(f"a short last frame of {self.last} shots in {self.R} frames of {self.n}")
        if self.R < 1 or self.n < 1 or self.code.shape != (self.total,):
            raise RandomizationRefused(f"{self.code.shape[0]} shot codes for {self.R} frames of {self.n}"
                                       + ("" if self.last is None else f" (the last of {self.last})"))
        if self.code.size and (self.code.min() < 0 or self.code.max() >= len(self.keys)):
            raise RandomizationRefused("a shot code names no key")

    @property
    def total(self) -> int:
        return self.R * self.n if self.last is None else (self.R - 1) * self.n + self.last

    @property
    def sizes(self) -> np.ndarray:
        """shots per frame, in frame order"""
        s = np.full(self.R, self.n, dtype=np.int64)
        if self.last is not None:
            s[-1] = self.last
        return s

    def counts(self, idx: np.ndarray | None = None) -> dict:
        """the histogram of the shots at `idx` (every shot when None), keys with zero count left out"""
        c = np.bincount(self.code if idx is None else self.code[idx], minlength=len(self.keys))
        return {self.keys[k]: int(v) for k, v in enumerate(c) if v}

    def frame_counts(self) -> list:
        """one histogram per randomization, in frame order"""
        return [self.counts(np.arange(r * self.n, r * self.n + int(sz))) for r, sz in enumerate(self.sizes)]


def _frames(total: int, R: int, n: int | None) -> tuple:
    """(R, n, last) for `total` shots in R frames: equal frames when n is None, else frames of n with a short last"""
    if n is None:
        if total % R:
            raise RandomizationRefused(f"{total} shots do not split into {R} frames")
        return int(R), total // int(R), None
    last = total - (int(R) - 1) * int(n)
    if not 1 <= last <= n:
        raise RandomizationRefused(f"{total} shots are not {R} frames of {n} with the last one short")
    return int(R), int(n), (None if last == n else last)


def from_bitstrings(bitstrings, R: int, n: int | None = None) -> Shots:
    """Shots from bitstrings in shot order (randomization-major) and the number of frames they ran in (with `n`,
    frames of n shots, the last one possibly short: `shape`)"""
    strs = np.asarray(list(bitstrings))
    if strs.size == 0:
        raise RandomizationRefused("no shots")
    R_, n_, last = _frames(strs.size, R, n)
    keys, code = np.unique(strs, return_inverse=True)
    return Shots(tuple(str(k) for k in keys), code.astype(np.int64).reshape(-1), R_, n_, last)


def from_bits(bits, R: int, n: int | None = None) -> Shots:
    """Shots from (shots, clbits) booleans in shot order, clbit 0 FIRST (an IQ-classified job: `oneq.hw.iq`); keys are
    qiskit bitstrings, clbit 0 rightmost, exactly as `oneq.hw.iq.counts` writes them"""
    b = np.asarray(bits, dtype=bool)
    if b.ndim != 2 or b.shape[0] == 0 or b.shape[1] == 0:
        raise RandomizationRefused(f"bits must be a non-empty (shots, clbits) array, got shape {b.shape}")
    R_, n_, last = _frames(b.shape[0], R, n)
    rows, code = np.unique(b, axis=0, return_inverse=True)
    keys = tuple("".join("1" if v else "0" for v in row[::-1]) for row in rows)
    return Shots(keys, code.astype(np.int64).reshape(-1), R_, n_, last)


def from_flight(index: dict, arrays: dict, names=None, register: str | None = None) -> dict:
    """name -> Shots for every (or every named) PUB of a flight record (`oneq.hw.campaign.flight_record`)"""
    from oneq.hw import campaign as C
    pubs = {p["name"]: p for p in index["pubs"]}
    out = {}
    for name in (list(pubs) if names is None else names):
        if name not in pubs:
            raise RandomizationRefused(f"{name}: not a PUB of this flight record")
        p = pubs[name]
        reg = p["registers"][register or next(iter(p["registers"]))]
        if reg.get("kind") == "iq":
            raise RandomizationRefused(f"{name}: a kerneled register holds IQ, not bits -- classify it (oneq.hw.iq) "
                                       "and use from_bits")
        R, n = shape(index["execution_spans"], p["index"], reg["num_shots"])
        out[name] = from_bitstrings(C.bitstrings(index, arrays, name, register), R, n)
    return out


# ================================================================== do the frames agree? (exact, no resampling)
FLAG_ALPHA = 0.01         # family-wise: a test is flagged below FLAG_ALPHA / (number of tests in the report)


def _key_bits(s: Shots) -> np.ndarray:
    return np.array([[int(ch) for ch in k[::-1]] for k in s.keys], dtype=float)


def homogeneity(s: Shots) -> dict:
    """are a circuit's R frames the same distribution? Pearson statistics of the frame means against the pooled mean,
    chi2 with R - 1 dof under the null (shots iid across frames; large n):
      hamming -- the per-shot Hamming fraction, against its pooled per-shot variance;
      per clbit -- P(1), against the binomial (a clbit that never or always reads 1 is skipped).
    A flag says WHERE frames disagree: a coherent error a twirl samples (R169), or a frame POSITION effect in a circuit
    whose frames are the same circuit (R176's q147). Untwirled circuits (R = 1) have nothing to test."""
    from scipy.stats import chi2
    if s.R < 2:
        return dict(frames=s.R, tested=False)
    K = _key_bits(s)
    # equal frames keep the exact arithmetic they always had; a short last frame weights each frame by its shots
    # (Pearson's statistic with unequal cells: sum_r n_r (p_r - p)^2 / p(1 - p), p pooled over shots, chi2 R - 1)
    sizes_f = s.sizes.astype(float)
    size = s.n if s.last is None else sizes_f                          # each frame's weight: its shots
    starts = np.arange(s.R) * s.n
    per = np.stack([np.bincount(s.code[a:a + int(m)], minlength=len(s.keys)) for a, m in zip(starts, s.sizes)])
    p_frame = per @ K / s.n if s.last is None else per @ K / sizes_f[:, None]    # (frames, clbits)
    pooled = p_frame.mean(axis=0) if s.last is None else (per @ K).sum(axis=0) / s.total
    dof = s.R - 1
    clbits = []
    for j, p in enumerate(pooled):
        if p <= 0.0 or p >= 1.0:
            continue
        x2 = float((size * (p_frame[:, j] - p) ** 2).sum() / (p * (1 - p)))
        clbits.append(dict(clbit=j, chi2=x2, p_value=float(chi2.sf(x2, dof))))
    h_shot = (K.mean(axis=1))[s.code]                                  # per-shot Hamming fraction
    var = float(h_shot.var(ddof=1))
    ham = None
    if var > 0:
        if s.last is None:
            frame_means = h_shot.reshape(s.R, s.n).mean(axis=1)
        else:
            frame_means = np.array([h_shot[a:a + int(m)].mean() for a, m in zip(starts, s.sizes)])
        x2 = float((size * (frame_means - h_shot.mean()) ** 2).sum() / var)
        ham = dict(chi2=x2, p_value=float(chi2.sf(x2, dof)))
    out = dict(frames=s.R, shots_per_frame=s.n, tested=True, dof=dof, hamming=ham, clbits=clbits)
    if s.last is not None:
        out["last_frame_shots"] = s.last
    return out


def homogeneity_map(shots_by_name: dict, alpha: float = FLAG_ALPHA, top: int = 20) -> dict:
    """homogeneity() of every circuit, with the family-wise flag: Bonferroni over every test in the map (each tested
    clbit and each Hamming statistic), so a clean job of thousands of tests flags ~nothing by chance"""
    per = {n: homogeneity(s) for n, s in shots_by_name.items()}
    tests = []
    for n, h in per.items():
        if not h.get("tested"):
            continue
        if h["hamming"] is not None:
            tests.append((h["hamming"]["p_value"], n, "hamming", h["hamming"]["chi2"]))
        tests += [(c["p_value"], n, c["clbit"], c["chi2"]) for c in h["clbits"]]
    cut = alpha / max(1, len(tests))
    flagged = sorted(t for t in tests if t[0] < cut)
    return dict(alpha=alpha, tests=len(tests), threshold=cut, flagged=len(flagged),
                strongest=[dict(circuit=n, clbit=c, chi2=x2, p_value=p) for p, n, c, x2 in sorted(tests)[:top]],
                flagged_circuits=sorted({n for _, n, _, _ in flagged}),
                untwirled=sorted(n for n, h in per.items() if not h.get("tested")))


# ================================================================== the bootstrap
def draw(shots: Shots, rng: np.random.Generator, mode: str) -> np.ndarray:
    """shot indices of one resample of a circuit"""
    if mode == "randomization":
        if shots.R < 2:
            raise RandomizationRefused("one frame: an untwirled circuit has no frame spread to resample (mode 'shot')")
        frames = rng.integers(shots.R, size=shots.R)
        if shots.last is None:
            return (frames[:, None] * shots.n + np.arange(shots.n)[None, :]).reshape(-1)
        sizes = shots.sizes                                # each drawn frame keeps its own shots, the short one too
        return np.concatenate([f * shots.n + np.arange(sizes[f]) for f in frames])
    if mode == "shot":
        return rng.integers(shots.total, size=shots.total)
    raise RandomizationRefused(f"mode {mode!r} is not one of {MODES}")


@dataclass(frozen=True)
class Bootstrap:
    estimate: dict              # the statistic on the full data, per key
    sd: dict                    # its bootstrap sd, scaled by sqrt(B / (B - 1))
    mode: str
    n_boot: int
    blocks: int                 # B: the smallest number of resampled blocks per circuit
    scale: float                # sqrt(B / (B - 1))

    def to_json(self) -> dict:
        return dict(estimate=self.estimate, sd=self.sd, mode=self.mode, n_boot=self.n_boot, blocks=self.blocks,
                    scale=self.scale)


def _as_mapping(value) -> dict:
    if isinstance(value, Mapping):
        return {str(k): float(v) for k, v in value.items()}
    return {"value": float(value)}


def bootstrap(estimator: Callable, shots_by_name: dict, n_boot: int = 300, seed: int = 0,
              mode: str = "randomization") -> Bootstrap:
    """the statistic estimator(counts_by_name) (a float, or a mapping of floats) on the full data and its sd over
    `n_boot` resamples of every circuit in `shots_by_name`. Counts the estimator also needs but that are not to be
    resampled (a readout calibration held fixed, say) belong in its closure."""
    if mode not in MODES:
        raise RandomizationRefused(f"mode {mode!r} is not one of {MODES}")
    if n_boot < 2:
        raise RandomizationRefused(f"{n_boot} resamples cannot give an sd")
    if not shots_by_name:
        raise RandomizationRefused("no circuits to resample")
    names = list(shots_by_name)
    blocks = min((s.R if mode == "randomization" else s.total) for s in shots_by_name.values())
    if blocks < 2:
        raise RandomizationRefused("one frame: an untwirled circuit has no frame spread to resample (mode 'shot')")
    estimate = _as_mapping(estimator({n: shots_by_name[n].counts() for n in names}))
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {k: [] for k in estimate}
    for b in range(n_boot):
        got = _as_mapping(estimator({n: shots_by_name[n].counts(draw(shots_by_name[n], rng, mode)) for n in names}))
        if set(got) != set(estimate):
            raise RandomizationRefused(f"resample {b}: the estimator returned keys {sorted(got)} not {sorted(estimate)}")
        for k, v in got.items():
            if not math.isfinite(v):
                raise RandomizationRefused(f"resample {b}: the estimator's {k!r} is {v}")
            draws[k].append(v)
    scale = math.sqrt(blocks / (blocks - 1))
    sd = {k: float(np.std(v, ddof=1) * scale) for k, v in draws.items()}
    return Bootstrap(estimate=estimate, sd=sd, mode=mode, n_boot=n_boot, blocks=int(blocks), scale=scale)


def inflation(frames: Bootstrap, shots: Bootstrap) -> dict:
    """per key, the randomization sd over the shot sd of the same statistic: how much the frames add (1 when they
    add nothing; R169's 148-149 read 1.4-2.7)"""
    if frames.mode != "randomization" or shots.mode != "shot":
        raise RandomizationRefused("inflation is a randomization bootstrap over a shot bootstrap")
    if set(frames.sd) != set(shots.sd):
        raise RandomizationRefused("the two bootstraps are of different statistics")
    return {k: (frames.sd[k] / shots.sd[k] if shots.sd[k] > 0 else float("inf")) for k in frames.sd}


def contrast(a_est: float, a_sd: float, b_est: float, b_sd: float) -> dict:
    """a - b with independent errors (different circuits draw their frames independently) and its z"""
    diff = a_est - b_est
    sd = math.hypot(a_sd, b_sd)
    return dict(diff=diff, sd=sd, z=diff / sd if sd > 0 else float("nan"))
