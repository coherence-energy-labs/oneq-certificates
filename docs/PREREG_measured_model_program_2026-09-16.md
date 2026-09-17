# Preregistration — the measured model: certified, extended in span, and asked the spatial question

**Status: WRITTEN 2026-09-16 (UTC) after the r=40 arms and the far-range readings (ledger R59, R62,
R63) and BEFORE any of the three analyses below is run. No shots are spent: every analysis reads
committed hardware records. The sha256 of this document up to the Receipts section is minted before
the first command runs; receipts are appended, never edited.**

## What is known, and what is not

`measured_model_lane` (ledger R23) estimates a decoding graph from the detector statistics alone and
judges it held out: on four records across two devices and two round counts the calibration DEM's
held-out logical errors fall 80–200x under it (kingston r=40: 403 -> 2; fez r=40: 759 -> 9). That is
already a decoding result. Three things are NOT known:

1. **Whether the certified pipeline can honour it.** Phase 7 certified minimum weight against the
   calibrated model through `rep_code_cert.certify_shots`. The measured model has never been passed
   through that accept path, so no shot has ever been certified minimum-weight with respect to a
   model measured from the device's own syndrome stream, and no certified logical-error count exists
   under it.
2. **How long the memory is.** The far-range excess is beyond the measured model at span 6 on all four
   records, and the model's share of it FALLS with rounds (15% at r=13, 12% and 4% at r=40). Whether a
   longer pairwise span carries the rest, or the correlation is not pairwise at all, is open.
3. **Whether any of it is spatial.** On a d=3 chain the two detector positions are one hop apart, so
   every far pair is far in time. `evidence/rep_d5_r13_fez.json` (R25, 40,000 shots, four positions)
   has pairs three positions apart at the same round. Nobody has split the statistic by family.

## A. Certified under the measured model (held out)

Tool: `experiments/dual_observable/certified_measured_lane.py`. For each record the model is estimated
on the first half of the shots (`measured_graph`, span 6, exactly as the lane does) and the SECOND half
is passed to `certify_shots(..., hard=True, edges=model, observed=truth)` three times: the record's
calibration DEM, the uniform graph, and the measured graph. Records: `rep_d3_r13_kingston`,
`rep_d3_r40_kingston`, `hardware_noise_floor_fez_d3_20k` (R22 fez r=13), `rep_d3_r40_fez`,
`rep_d5_r13_fez`, `rep_d3_r13_fez_nested`.

| # | prediction | met if | floor |
|---|---|---|---|
| A1 | the checker certifies under measured weights | certified >= 0.99 of nontrivial held-out shots, every record | >= 100 nontrivial |
| A2 | the certified logical-error count under the measured model beats the DEM's | measured <= DEM / 1.5 on every record | >= 30 DEM errors held out |
| A3 | the accept path and the lane's scorer agree on the measured model | certified-lane count within 3 shots of `score()`'s count on the same half, every record | -- |
| A4 | the parity rule reads the logical class | opposite-cut errors >= 10x declared-cut errors under the measured model, every record with >= 1 declared error | -- |

Decision table: A1–A4 met — the assurance layer can carry a model read from the shots it certifies,
and that is the model to declare in the next hardware preregistration. A1 fails — the float weight
range (probabilities 1e-5 to 0.49) breaks the exact checker; record which shots and why. A3 fails by
more than 3 — the two decoders disagree on tie-breaking or on the observable convention; the
difference is a finding about the lane, not the model, and is bounded before anything is claimed.

## B. The span sweep (r=40 records)

B-decode: `measured_model_lane --max-span S --runs <record> --no-in-sample` for S in 6, 10, 20, 40, on
`rep_d3_r40_kingston` and `rep_d3_r40_fez`. B-far: the far-range statistic against the measured null
built at the same spans, 200 draws each (`far_range_families.py --span S --null measured`).

| # | prediction | met if | floor |
|---|---|---|---|
| B1 | longer span does not destroy decoding | held-out measured errors at S=10 and S=20 <= 1.5x the S=6 count + 3, both records | -- |
| B2 | the model's share of the far-range excess grows with span | null_mean / observed is non-decreasing in S on both records | -- |
| B3 | the memory is pairwise and long | at S=40 the measured null carries >= 50% of the observed excess on both records | -- |

Decision table: B3 met — the excess is long pairwise memory; the instrument is a longer span, no new
model. B2 met and B3 not — pairwise memory to span 40 is real but is not the whole excess; what is
left is not pairwise (a three-detector term is next). B2 fails — the two-point estimator at long
spans is noise; the span floor is where the share stops rising. B1 fails — the same, read from the
decoder: variance at long spans is buying edges the data cannot price.

## C. The spatial question (d=5 fez, with the d=3 nested control)

Tool: `experiments/ibm_live/far_range_families.py`, which computes the SAME statistic as
`far_range_null.py` (pairwise plug-in mutual information in bits, excess over the fitted independent
base, mean over pairs at detector-graph distance >= 3) but reports it by family, with node
(position q, round t): **time-far** Δq = 0 and Δt >= 3; **space-far** Δt = 0 and Δq >= 3; **mixed**
Δq >= 1 and Δt >= 1 with distance >= 3. The union must equal the original far-pair set (a check).
Both nulls, 200 draws each, on `rep_d5_r13_fez` (40,000 shots) and, as the control that has no
space-far family, `rep_d3_r13_fez_nested`.

| # | prediction | met if | floor |
|---|---|---|---|
| C1 | the temporal reading replicates at d=5 | time-far excess beyond both nulls (p < 0.05) | >= 30 pairs |
| C2 | there is no purely spatial far-range correlation | space-far excess INSIDE the measured null: p >= 0.05 or \|z\| < 3 | >= 30 pairs |
| C3 | the whole is the sum of its families | all-far excess equals the pair-weighted mean of the families to 1e-9 | -- |

Decision table: C1 and C2 — the far-range result is temporal memory and nothing else on this device;
the scoreboard row is narrowed once more and the d>=5 hardware question is closed for this chain
family. C2 fails beyond BOTH nulls with z >= 3 — a spatial far-range correlation on one device, one
record; nothing is claimed until `ibm_kingston` d=5 r=13 at 20,000 shots is preregistered and run
(the quota allows it). C1 fails — the d=3 reading does not transfer to d=5 and is bounded to d=3.

## Commands

```
python experiments/ibm_live/far_range_fast.py --self-test
python experiments/dual_observable/certified_measured_lane.py --json evidence/certified_measured_lane.json
python experiments/dual_observable/measured_model_lane.py --runs evidence/rep_d3_r40_kingston.json,evidence/rep_d3_r40_fez.json --no-in-sample --max-span S --json evidence/span_sweep/decode_span_S.json   # S = 6, 10, 20, 40
python experiments/ibm_live/far_range_families.py evidence/rep_d3_r40_kingston.json kingston --span S --draws 200     # S = 6, 10, 20, 40; and fez
python experiments/ibm_live/far_range_families.py evidence/rep_d5_r13_fez.json fez --draws 200
python experiments/ibm_live/far_range_families.py evidence/rep_d3_r13_fez_nested.json fez_nested --draws 200
```

## Receipts (append-only from here)

### 2026-09-16 10:12 UTC — minted BEFORE the first run

sha256 of this document up to and excluding this Receipts section:
`f092b57b1037ecb122b7932152026d519d62a5fe64b6998d97152d8586b8f9ca`. `far_range_fast.py --self-test`
passed first (108 pairs, max difference 2.9e-16). One smoke run of each tool on `rep_d3_r13_kingston`
with 2 null draws preceded the mint only to catch crashes; its artifact was deleted and its numbers
are not receipts. The four program runs (A; B-decode; B-far; C) were launched at 10:13 UTC.

### 2026-09-16 12:30 UTC — A, B and C READ against the tables above

**A** (`evidence/certified_measured_lane.json`, 2,670 s; the r=40 fez certification alone 1,904 s):

| record | held out | certified under measured | DEM errors | uniform | measured | score() | opposite cut | A1 | A2 | A3 | A4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kingston r=13 | 10,000 | 2,777/2,777 | 139 | 141 | **2** | 2 | 40 | MET | MET (70x) | MET | MET |
| kingston r=40 | 10,000 | 5,801/5,802 (0.9998) | 403 | 413 | **2** | 2 | 44 | MET | MET (202x) | MET | MET |
| fez r=13 (R22) | 10,000 | 8,026/8,026 | 123 | 247 | **2** | 2 | 216 | MET | MET (62x) | MET | MET |
| fez r=40 | 10,000 | 9,778/9,793 (0.9985) | 759 | 751 | **9** | 9 | 271 | MET | MET (84x) | MET | MET |
| fez d=5 r=13 | 20,000 | 11,397/11,397 | 52 | 43 | **1** | 1 | 121 | MET | MET (52x) | MET | MET |
| fez d=3 nested | 10,000 | 3,184/3,184 | 41 | 53 | **1** | 1 | 74 | MET | MET (41x) | MET | MET |

Decision row **A1–A4 met on every record**: the certified accept path honours a model measured
from the device's own syndrome stream; the 16 refusals on the two r=40 records (of 15,595
nontrivial shots) are the exact checker declining a float-weight instance, not a wrong
correction, and are counted as UNKNOWN there as the pipeline always does. Beside the table: the
uniform model is WORSE than the calibration DEM on fez r=13 (247 vs 123) and better on fez d=5
(43 vs 52), so "calibrated beats uniform" (Phase 7, 54 vs 74) is a fact about a device and a day.

**B-decode** (`evidence/span_sweep/decode_span_{6,10,20,40}.json`): held-out measured errors at
spans 6/10/20/40 — kingston r=40 **2/0/0/0**, fez r=40 **9/7/7/6**. B1 MET on both.

**B-far** (`evidence/phase7/far_range_families_{kingston,fez}_rep_d3_r40_span{6,10,20,40}.json`,
200 draws each; the all-far number equals the earlier artifacts' to 1e-6 on both records):

| record | family | observed bits | measured-null share at span 6 / 10 / 20 / 40 |
|---|---|---|---|
| kingston r=40 | time-far (1,482 pairs) | +0.001213 | 18.1% / 16.5% / 14.5% / 13.7% |
| kingston r=40 | mixed (1,560) | +0.000603 | 0% at every span (the model has no such edge) |
| fez r=40 | time-far (1,482) | +0.003092 | 3.9% / 2.4% / 1.1% / — |
| fez r=40 | mixed (1,560) | +0.000074 | 0% at every span |

B2 **FAILS** on both (the share is monotonically DEcreasing in span); B3 **FAILS** (13.7% at span
40, not >= 50%). Decision row "B2 fails": the two-point estimator at long spans adds edges that
carry less of the excess, not more; no pairwise same-position model at any span is this excess.
And the decoder says the opposite of the null: longer spans decode slightly better. Both are
consistent with a mechanism whose two-point projection is a poor generative model but a good
prior, which is R23a's finding again from the other side.

**C** (`evidence/phase7/far_range_families_fez_rep_d5_r13_span6.json`, 40,000 shots, 200 draws;
control `..._fez_nested_rep_d3_r13_span6.json`):

| family | pairs | observed bits | crosstalk null (z, p) | measured null (z, p) | verdict |
|---|---|---|---|---|---|
| time-far | 264 | +0.004122 | +2,850, 0.005 | +132, 0.005 (share 26.5%) | C1 **MET** |
| space-far | **14** | −0.000014 | −1.2, 0.906 | −1.2, 0.915 | C2 **UNDECIDABLE** (floor 30 pairs); reads INSIDE |
| mixed | 1,014 | +0.000142 | +153, 0.005 | +159, 0.005 (share 0.8%) | recorded |
| all-far | 1,292 | +0.000953 | +1,206 | +152 | C3 MET (families reconstruct the mean to 1e-9) |

The nested d=3 control on the same device reads as every other d=3 record (time-far +0.003593,
beyond both nulls; measured share 27.4%; no space-far family). A d=7 arm was preregistered to reach
the floor (`docs/PREREG_spatial_d7_2026-09-16.md`, submitted on both devices 11:52 UTC).
