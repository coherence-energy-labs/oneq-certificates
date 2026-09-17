# Prior-art register — every distinctiveness claim, with its nearest neighbour

Status: **B4, fourth sweep, 2026-09-08** (rows 14-23 added for the estimator campaign, ledger R25a and R29). This is not a novelty claim. It
is a register of what was searched, what was found, what the nearest
known work actually says, and — the part that matters most — **what was
not searched**. `docs/OISC_PRIOR_ART.md` covers the OISC lower-bound
question and is not repeated here; this covers the rest of the estate and
records one finding that **narrows a claim we were making**.

> **Not-found ≠ absent.** Every row below is a statement about a search,
> not about the literature. A row marked `NEAREST FOUND` means a search
> ran and returned that; a row marked `NOT SEARCHED` means exactly what
> it says and must never be read as "nothing exists".

> **This file is the literature appendix an external audit already asked
> for, and it is late.** `docs/reviews/PAPER2_EXTERNAL_AUDIT.md` §10
> ends the defensible novelty statement with: *"Before using 'first,'
> add a searchable literature appendix listing databases, dates,
> queries, and inclusion criteria."* That appendix did not exist. It is
> §"The searches, verbatim" below.
>
> The same audit had already narrowed the claim correctly (C17: *"Too
> broad against prior LP certificates, LP post-processing, quantum LP,
> and recent certifying decoders"*), and the manuscript was rewritten as
> **four deltas against the cited literature**. So the HyperBlossom
> finding below **corroborates a narrowing that already happened** — it
> does not overturn an unnarrowed claim, and presenting it as a
> discovery would misrepresent the record.

---

## ⚠️ The finding, and it corroborates rather than overturns

**A primal-dual LP model for QEC decoding with certifiable bounds is NOT
novel, and one 2025 paper says so in our own vocabulary.** The external
audit reached the same conclusion from the other direction, without
naming this paper.

[EXTERNAL] **HyperBlossom**, *Minimum-Weight Parity Factor Decoder for
Quantum Error Correction*, [arXiv:2508.04969](https://arxiv.org/abs/2508.04969)
(August 2025), formulates most-likely-error decoding as a Minimum-Weight
Parity Factor problem and *"generalizes the blossom algorithm to
hypergraphs via a similar primal-dual linear programming model with
**certifiable proximity bounds**"*. Its own framing is that it
*"bridges the gap between heuristic and certifying decoders"*.

That is the same neighbourhood as this repository's core, in the same
words, and anyone reviewing ONE-Q will find it immediately. **Leading
with "we built a certifying decoder using LP duality" would be
discredited on contact.**

**What the read established, and the distinction is narrow:** the
abstract claims **proximity bounds** — a computable distance from
optimality — not a per-shot *proof* that the correction is
minimum-weight. It does not describe exact-arithmetic verification, nor a
checker sharing no code with the solver, nor refusal as the default
outcome.

So the honest position, stated as three separate claims at three
different strengths:

| claim | class | status |
|---|---|---|
| "primal-dual LP certification for QEC decoding" | — | **NOT NOVEL.** Prior art, and it must never be presented otherwise |
| "a computable bound on the decode's distance from optimal" | — | **NOT NOVEL.** HyperBlossom's certifiable proximity bounds |
| "a per-shot certificate re-checked in **exact rational arithmetic** by an independent checker that is the **only** accept path, refusing by default" | HYPOTHESIS | **not found in the sweeps run.** That is a statement about three searches, not about the field |

The third row is the only one that may be argued, it is graded
HYPOTHESIS rather than established, and the venues below that were not
searched are where it would most likely be overturned.

---

## The register

| # | Mechanism | Nearest found | Class |
|---|---|---|---|
| 1 | Certifying decoder, LP-dual optimality witness | HyperBlossom, [2508.04969](https://arxiv.org/abs/2508.04969) — primal-dual LP, certifiable **proximity** bounds | **NEAREST FOUND** |
| 2 | OISC: integer-programming aggregation over overlapping information sets | Classical **Brouwer–Zimmermann** already uses overlapping sets; the literature aggregates **additively** (`Σ max(0, t+1−r+k_i)`), not as an IP. See `docs/OISC_PRIOR_ART.md` | **NEAREST FOUND** — overlap is *not* distinguishing |
| 3 | IP-based bounds from set-cover-like aggregation | *An Integer Programming Based Bound for Locally Repairable Codes*, [1409.0952](https://arxiv.org/pdf/1409.0952) — converts parameter estimation to a **set cover** problem | **NEAREST FOUND** — adjacent formulation, different object |
| 4 | Covering arguments for code bounds | *Linear Programming Bounds for Codes via a Covering Argument* (Discrete & Comput. Geom.) — recovers MRRW via covering | **NEAREST FOUND** |
| 5 | Exact distance by a **solver portfolio** (MILP / SAT / PB / enumeration) | *Distance-Finding Algorithms for Quantum Codes and Circuits*, [2603.22532](https://arxiv.org/pdf/2603.22532) benchmarks BZ, connected-cluster, SAT and MIP as **genuinely distinct families** | **NEAREST FOUND — and it CONFIRMS the premise.** The portfolio is standard practice, not an invention |
| 6 | Hardness of the distance decision | [2509.21469](https://arxiv.org/abs/2509.21469), and NP-completeness known earlier | **NEAREST FOUND** — bounds what any engine can promise |
| 7 | Logical-failure decomposition `p_L = p_search + p_ambiguity + p_model + p_residual` | Searched; the nearest is a *two*-term waterfall/floor decomposition in neural-decoder work. No four-way **attribution by certificate class** found | **NOT FOUND in the sweep run** |
| 8 | Per-shot capability manifest with `sdk_expressible` vs `hardware_executable` provenance | **NOT SEARCHED** | **NOT SEARCHED** |
| 9 | Minimum-ratio-cycle formulation of the decoder's robustness radius | **NOT SEARCHED.** The min-ratio-cycle problem itself is classical (Lawler); its use as an exact calibration radius here is what would need checking | **NOT SEARCHED** |
| 10 | Two-sided **anytime** bounds on a partition function by truncated search with the unexplored part PRICED by an admissible heuristic (`coset_enclosure` E1+E2) | E. Dechter & R. Dechter, *A Note on Bounding the Partition Function by Search*; Lou, Dechter & Ihler, *Anytime Anyspace AND/OR Search for Bounding the Partition Function*; the admissible upper bounds come from mini-bucket elimination (Dechter & Rish) and Liu & Ihler, *Bounding the Partition Function using Hölder's Inequality* | **NOT NOVEL.** This is the same shape and must never be presented otherwise |
| 11 | The pricing being an **exact odd-parity factorisation** over a mechanism-disjoint family of unsatisfied GF(2) checks (`coset_enclosure` E6), rather than a variational relaxation | Searched; the graphical-models bounds found are variational (Hölder / weighted mini-bucket / gauge). The identity itself — `Σ_{odd} = (f(1) − f(−1))/2` — is elementary and certainly not new **as an identity**; what was not found is its use as the admissible heuristic that prices a discarded search node | **NOT FOUND in the sweep run** — and the identity is elementary, so this row claims a USE, not a discovery |
| 12 | A **per-shot machine-checkable certificate of the maximum-likelihood logical class** for a quantum code: a document an independent checker replays in exact integer arithmetic, in a second language, refusing by default | ML decoding as a coset partition-function comparison is standard and #P-hard (Iyer & Poulin); approximate ML decoders are many (tensor networks — Bravyi–Suchara–Vargo, Chubb; BP with cluster corrections; coset ensembles). None found emits a per-instance checkable object. Certified **model counting** exists (CPOG / knowledge compilation, Bryant et al.; formally certified *approximate* model counting) but compiles an exact form and is not applied here | **NOT FOUND in the sweep run** |
| 13 | A certified **two-sided interval on the per-shot posterior** `P(class given syndrome)` — a rigorous bound on the quantity a *complementary gap* estimates | Soft-output decoding and the complementary gap are established (Gidney; soft-output methods for boundary-less codes; neural-network confidence as a *learned* proxy for the logical gap). Every one found is a proxy or a calibrated estimate; none bounds the posterior | **NOT FOUND in the sweep run** |
| 14 | Two-point estimation of edge probabilities from detector correlations (`oneq.dem_measured.pair_probability`, R23) | Spitz, Tarasinski, Beenakker, O'Brien, *Adaptive weight estimator for QEC*, [1712.02360](https://arxiv.org/abs/1712.02360) (2018): weights "determined from the measured data in the absence of an error model"; Google's pij analysis (Korotkov) in [2102.06132](https://arxiv.org/abs/2102.06132); Takou, Benito, Vezvaee, Lidar, Brown, [2606.11496](https://arxiv.org/abs/2606.11496) Eq. 17 and hyperedge moment inversion; Blume-Kohout & Young, [2504.14643](https://arxiv.org/abs/2504.14643) (theory) | **NOT NOVEL — prior art, credited.** The formula is theirs; R23's "no calibration, no logical outcomes" is the 2018 abstract's own claim |
| 15 | An estimated DEM beats the device-informed DEM on IBM hardware (R22/R23/R25: 474 -> 47 held out, six runs, two Heron devices, d=3/5) | Takou et al. [2606.11496](https://arxiv.org/abs/2606.11496): d=3 surface code on `ibm_miami`, 50k shots, "typically at the 5-10% level and with larger gains in some IBM cases" (largest -37%, r=1), fit and judged on the same shots; Stein et al. [2601.16123](https://arxiv.org/abs/2601.16123) (PRR 2026): calibration-weighted MWPM on Fez/Kingston/Pittsburgh repetition codes beaten **4.5x at d=7, 11.1x at d=11** by a calibration-conditioned neural decoder, 2.7M shots, held out on unseen chains; Sivak, Newman, Klimov [2406.02700](https://arxiv.org/abs/2406.02700) +16% (Sycamore rep., RL priors); Cao, Feng, Ye, Pan [2602.19722](https://arxiv.org/abs/2602.19722) -30.6% (Google rep., differentiable MLE) | **NEAREST FOUND — adjacent-occupied.** The delta is MAGNITUDE with a PLAIN GRAPH MODEL: 10.1x held out from the two-point estimator alone, MWPM-decodable and certifiable, at d=3/5 where Stein et al. report their gain "particularly in d>=7" |
| 16 | The vendor DEM's error MODE (89.8% spurious flips) and the mechanism it mis-prices (space-like family overstated 7-38x on 7 of 7 runs; time-like 1.4-1.7x) — R22, R23, R25 | Takou et al. identify anomalous detector families but "do not quantify differential pricing by error category"; Stein et al. say only that "bias, inhomogeneity and correlated errors" degrade MWPM; PAEMS ([2603.29439](https://arxiv.org/abs/2603.29439)) compares correlation SECTORS of simulated models to experiment, not a vendor DEM's pricing by family | **NOT FOUND in the search** (queries 2026-09-08 #3, #7). A per-family overstatement of a vendor-derived DEM on IBM was not seen. Not-found is not absent. **NARROWED 2026-09-08 (R29, row 23):** Google's model-derived SI1000 prior on Google's Willow shots reads space-like 0.31-0.37 on 12 of 12 configurations, so the overstatement is IBM's calibration-derived pipeline as this instrument reads it, not a property of model-derived priors |
| 17 | Held-out estimation (fit half, judge half), decoder-independence (BP overlap 74.7x chance), and the structure-vs-pricing decomposition — R23, R23b, R23d | Takou et al.: "on the same 50,000 publicly released experimental shots", no train/test split; Stein et al. hold out chains and a later snapshot for a NEURAL decoder; no estimated-DEM paper found decomposes its gain | **NOT FOUND** — a METHODOLOGICAL delta, recorded as such, not a scientific one |
| 18 | The multi-round correlation tail and the fixed-price burst model (R20, R21, R22: 0.5 transfers across devices) | McEwen et al. [2102.06131](https://arxiv.org/abs/2102.06131) (Nat. Commun. 2021): leakage "opens a path to errors that are correlated in space and time"; PAEMS [2603.29439](https://arxiv.org/abs/2603.29439) (Apr 2026): on IBM Brisbane/Sherbrooke/Torino 21-qubit 30-round repetition codes, distinct timelike / spacelike / LEAKAGE correlation sectors, an ancilla's timelike correlation climbing 0.096 -> 0.465 over 29 rounds | **NOT NOVEL as a mechanism.** The tail is leakage accumulation, already measured on IBM. The burst model's transfer at a FIXED price is NEAREST FOUND (PAEMS refits per platform) |
| 19 | "A fast-reset device shrinks the multi-round tail" (the R25 phoenix prediction) | McEwen et al. 2021 measured exactly this: with reset applied the slow decay of detection events "is no longer visible" | **REPLICATION, not a prediction.** Recorded so it is never presented as one; a phoenix run is preregistered as a McEwen-2021 replication |
| 20 | A hidden-state (leaked / healthy) per-ancilla model as a decoder prior (R25 proposal, `oneq.leakage_hmm`) | Varbanov et al. [2002.07119](https://arxiv.org/abs/2002.07119) (npj QI 2020): leakage detection from syndromes via a hidden Markov model; PAEMS models leakage + seepage explicitly | **PORT of prior art.** Credited as Varbanov's HMM applied as a per-shot matching prior; the decode-gain measurement is what would be ours |
| 21 | The raw, marginal-inconsistent estimate decodes BEST; every consistency-restoring damping costs the gain (two-sided optimum at alpha=1) — R23a, R23b | Takou et al. floor low-SNR correlators and ZERO estimates outside [0,1], using the zeroing as a diagnostic of an inadequate model; PAEMS reads negative pij as the Markovian assumption failing | **NEAREST FOUND.** The neighbours REPAIR the inconsistency; none found reports that repairing it worsens the decode. NOT SEARCHED beyond the estimated-DEM papers |
| 22 | A syndrome-side, structurally-controlled bound on site-to-site noise correlation against a decoherence-free-subspace crossover (R24, R25: 0.0068 at d=5) | *Revealing correlated noise with single-qubit operations*, [2502.09706](https://arxiv.org/abs/2502.09706) (PRR Dec 2025): detects correlated dephasing via DFS / superdecoherence with single-qubit ops; pulse-based noise spectroscopy [2603.19373](https://arxiv.org/abs/2603.19373) | **NEAREST FOUND.** Different instrument (syndromes of a running code, no extra circuits); a negative result on Heron. Not seen elsewhere |
| 23 | A vendor's own decoder prior read AGAINST its own public shots by error family, with a hyperedge-aware pair prediction verified on DEM-sampled shots (R29: Google SI1000 prior on Willow, 12/12: time-like 0.59-0.77, space-like 0.31-0.37, space-time 0.22-0.26, detection rate 0.43-0.53x; multi-round tail 10-50x below every IBM run) | Google's own pij analysis ([2102.06132](https://arxiv.org/abs/2102.06132)) and pij-refined priors are the published comparison on their data; Takou et al. [2606.11496](https://arxiv.org/abs/2606.11496) estimate DEMs on Willow open data and improve decoding 5-10%; none read the SHIPPED prior by family | **NEAREST FOUND.** That a nominal SI1000 prior under-prices a real device is presumably known to its authors; the per-family reading, the dark check with hyperedges, and the IBM-vs-Google contrast on one instrument were not seen |

| 24 | The RESIDUE OUTER CODE at weight <= 4, decided exactly by sorting every signed weight-<= 2 pattern (R31): 33,968 colliding pairs survive the parity ancilla at n = 1,344, all of combined weight 4, in two families -- the integer identity 2^i + 2^(i+1) = 2^(i+2) - 2^i at every width, and a modular one at a - b = 899 | Residue number systems and AN codes are classical (Watson-Hastings 1966; Massey-Garcia 1972); the arithmetic-code literature treats additive errors, not the SIGNED, branch-dependent shifts an X error produces on a coherently-read register. FTQC_CONCEPTS' own arXiv sweep found the intersection empty three ways | **NOT NOVEL as an enumeration method** -- it is a sort. The kill is ours, and so is the observation that the design rule 'ord_m(2) >= register width' is necessary and not sufficient: the binding constraint is on PRODUCTS of the orders across moduli |
| 25 | The PLAN-CONDITIONED saturation wall (R36): strip an engineering programme's noise from the CIRCUIT, re-derive the detector error model, and ask whether any graphlike logical error survives. Perfect two-qubit gates buy 1.5x-2.8x and then stop; every noise source removed CUTS | Min-cut on a decoding graph and `shortest_graphlike_error` are standard (stim; Higgott-Gidney). Ablation ranking of noise channels is routine practice. FTQC_COHERENCE_NATIVE's sweep found no published object that answers CUT-or-WALL for a named engineering plan before the money is spent | **NOT NOVEL as graph algorithms.** What is claimed is the framing -- a plan is a circuit rewrite, and the a-priori asymptote is computable with zero Monte Carlo -- plus the three refusals (no observables is not a cut; hyperedges are refused not dropped; a gauge cut is not a cut) |
| 26 | The exact CX-schedule distance audit (R37): 20 of 24 layer orders are not schedules at all, and of the 4 valid ones half give d_eff = ceil((d+1)/2) -- confirmed out of sample at d = 7 and d = 9 | **THE PHYSICS IS TEXTBOOK.** Hook errors and the distance cost of a bad CNOT ordering: Dennis-Kitaev-Landahl-Preskill 2002 ([quant-ph/0110143](https://arxiv.org/abs/quant-ph/0110143)); Tomita-Svore 2014 ([1404.3747](https://arxiv.org/abs/1404.3747)) for the rotated surface code's orderings | **NOT NOVEL.** The closed form and the three-way classification -- 'not a schedule' separated from 'a bad schedule' -- are what this repository adds, and the ledger says the physics is not ours |
| 27 | Read-coverage blindness in configuration checking (R38): a per-read integrity check cannot see a corruption that makes something UNREAD. A dets corruption emptying a mechanism's detector set hides it from every syndrome-driven read, and one such mechanism carried an observable | ABFT is Huang-Abraham 1984; CRAM scrubbing at 10-100 ms is the industrial baseline `abft_l0` quotes; memory-integrity tagging is standard. No located treatment of the case where the corrupted field is itself what determines coverage | **NEAREST FOUND is the general ABFT literature.** The claim is deliberately narrow: it is about per-read checks over tables whose own fields decide what is read, and the remedy (full coverage for the structural fields, per-read for the rest) follows from it |
| 29 | **A per-shot optimality certificate for a model ESTIMATED FROM THE SHOTS BEING DECODED** (R64): 40,963 of 40,979 held-out shots proven minimum-weight under a graph fitted on the run's own first half, on two devices | HyperBlossom [2508.04969](https://arxiv.org/abs/2508.04969) gives a primal-dual model with certifiable proximity bounds — for a DECODER, under a given model. Krishnamoorthy et al. [2608.25545](https://arxiv.org/abs/2608.25545) certify inside a sampler. The estimated-DEM papers (Takou [2606.11496](https://arxiv.org/abs/2606.11496), Stein [2601.16123](https://arxiv.org/abs/2601.16123), [2504.20212](https://arxiv.org/abs/2504.20212)) carry no per-shot witness at all | **NOT FOUND in the 2026-09-16 search.** The certificate is the repository's existing claim (row 12); what is new here is that the MODEL it certifies against is learned from the same run |
| 30 | **A fitted latent chain LOWERED to a static, certifiable matching graph** (R65, R68a): three numbers per position become same-position pair edges at every span in closed form, `p(s) = pi1(pi1 + (1-pi1)rho^s)p1^2` | Leakage-aware decoding reweights edges and adds edges in response to HERALD events (Riverlane LCD notes; Varbanov [2002.07119](https://arxiv.org/abs/2002.07119) post-selects). PAEMS [2603.29439](https://arxiv.org/abs/2603.29439) models leakage propagation inside an error model. Every one found prices PER SHOT or per herald | **NOT FOUND.** The nearest are per-shot/heralded; a static analytic lowering of a fitted chain, certifiable because it is static, was not found. **This repository's own R27/R28 killed the per-shot version**, which is the sharper comparison |
| 31 | **A compression claim: three numbers per position carry most of what hundreds of estimated edges carry** (R68b) — DEM+chain repairs 89-98% of the DEM's held-out errors; the 168-573-edge measured graph still wins every record | No estimated-DEM paper found reports a parameter-count comparison; Takou et al. extend to hyperedges (more parameters), PAEMS adds sectors | **NOT FOUND**, and it is a PARSIMONY result, never a performance one: the measured graph decodes better on 6 of 6 |
| 32 | **The burst LOCUS decided per device by likelihood** (R66): data-qubit chains win on `ibm_kingston` by 979 and 2,745 nats and carry 43-45% of the cross-position family; on `ibm_fez` the same fit switches the shared qubit off and loses by 3,312 nats at d=5 | That data-qubit and ancilla leakage have different syndrome signatures is textbook: Fowler et al. [1306.0925](https://arxiv.org/abs/1306.0925) (leakage and paralysis), Varbanov [2002.07119](https://arxiv.org/abs/2002.07119), [2510.25661](https://arxiv.org/abs/2510.25661) (leakage speculation) | **MECHANISM NOT OURS.** What was not found is the likelihood comparison between the two loci run on the same syndromes per device, with the answer DIFFERING between two machines of one generation. Narrow, and stated as such |
| 28 | Transient growth in capped-iteration quantum BP (R40): the message Jacobian is strongly non-normal (sigma_max/rho 2.9-9.4) and the transient regime covers 88-100% of shots -- and rho still beats every transient quantity on both the asymptotic and the budget-relevant target | Transient growth, pseudospectra and the Kreiss constant are classical (Trefethen-Embree, *Spectra and Pseudospectra*, 2005). BP stability analyses are norm-based sufficient conditions or rho-based local stability, both asymptotic. FTQC_ESTATE_MAP #7 located no transient analysis of quantum BP under a fixed iteration budget | **NOT NOVEL as linear algebra.** The measurement is new and its result is NEGATIVE: the mechanism is real, is not a re-description of syndrome weight, and does not predict the outcome better than rho. Reported as a negative result, not as an open question |

---

## What was NOT searched, named so the gap is auditable

- **The classical covering-bound literature in depth.** One search ran
  (row 4). The covering-radius/covering-code literature is large and was
  not swept.
- **The LRC bound literature in depth.** One paper found (row 3). The
  locality/availability bound literature was not swept.
- **Constraint-programming distance work.** Searched at the survey level
  only (row 5). CP-specific venues (CP, CPAIOR) were not searched.
- **Correspondence with the 2024 / 2026 authors**, which
  `docs/OPEN_WORK.md` B4 names as the thing that would settle it
  fastest. Not attempted.
- **Non-English and pre-arXiv literature.** Not attempted, and this is
  where a 1990s covering-bound result would most plausibly hide.
- **Rows 14-22 (the estimator campaign, 2026-09-08):** eight web-index
  queries, abstracts and (for 2606.11496, 2601.16123, 2603.29439) full
  text read. NOT SEARCHED: Google Scholar; the Google Willow / Sycamore
  open datasets re-analysed for the space-like overstatement (row 16);
  correspondence with Takou et al. or Stein et al., whose IBM data is
  public and would settle row 16 fastest; any query for a decoder prior
  that gets WORSE when made generative (row 21).

---

## The searches, verbatim

Databases, dates, queries and inclusion criteria — the appendix
`PAPER2_EXTERNAL_AUDIT.md` §10 requires before the word "first" may be
used at all.

| date (UTC) | database | query | included |
|---|---|---|---|
| 2026-07-31 | arXiv API | 8 queries, recorded verbatim in `experiments/m0_prior_art/arxiv_sweep.json` | all results, incl. the 6 that returned 0 |
| 2026-08-21 | web index | `quantum code minimum distance constraint programming exact algorithm 2025 2026` | survey + hardness results (rows 5, 6) |
| 2026-08-21 | web index | `covering bound information sets linear code minimum distance integer programming aggregation lower bound` | LRC IP bound, LP-via-covering (rows 3, 4) |
| 2026-08-21 | web index | `certifying decoder quantum error correction per-shot optimality certificate LP duality witness verifiable` | **HyperBlossom** (row 1) |
| 2026-08-21 | web index | `logical error rate decomposition decoder failure attribution search ambiguity model residual quantum error correction` | nothing four-way (row 7) |
| 2026-08-22 | web index | `certified maximum-likelihood decoding quantum error correction per-shot certificate coset partition function` | MLD-as-partition-function and #P-hardness confirmed; coset ensemble decoder; no certificate (row 12) |
| 2026-08-22 | web index | `rigorous upper lower bounds tensor network contraction partition function certified truncation error` | cluster-corrected BP with rigorous but CONDITIONAL error bounds; nothing per-instance checkable (row 12) |
| 2026-08-22 | web index | `soft-output decoder surface code complementary gap magic state cultivation posterior bound` | complementary gap, soft-output, GNN confidence — all proxies (row 13) |
| 2026-08-22 | web index | `mini-bucket elimination weighted Holder bounds partition function upper lower bound anytime` | mini-bucket, weighted mini-bucket, gauged MBE (row 10) |
| 2026-08-22 | web index | `"certified" OR "verifiable" certificate partition function counting #P checkable proof independent verifier exact arithmetic` | CPOG certified knowledge compilation; certified approximate model counting (row 12) |
| 2026-08-22 | web index | `beam search truncation rigorous lower bound partition function anytime upper bound discarded mass residual constraint pricing` | **Dechter & Dechter, and Lou–Dechter–Ihler — the row-10 narrowing.** Found by searching for the mechanism rather than for the application, which is how a novelty claim gets refuted before it ships |
| 2026-08-22 | web index | `quantum decoder certificate proof of optimality verifier refuses degenerate maximum likelihood enclosure interval` | DQMLD and its promise-gap formulation; nothing on certificates (rows 12, 13) |
| 2026-09-08 | web index | `Spitz Tarasinski Beenakker O'Brien adaptive weight estimator quantum error correction time-dependent environment detector correlations pij` | 1712.02360 (row 14) |
| 2026-09-08 | web index | `pij method estimating detector error model edge probabilities from detection event correlations Google surface code unphysical negative probabilities` | 2606.11496, 2504.14643 (rows 14, 21) |
| 2026-09-08 | web index | `IBM Heron repetition code detector error model calibration versus data-driven learned error model decoding accuracy 2025 2026` | 2601.16123, 2406.02700 (row 15) |
| 2026-09-08 | web index | `leakage induced correlated errors multi-round syndrome correlations repetition code superconducting qubits leakage removal reset` | 2102.06131 (rows 18, 19) |
| 2026-09-08 | web index | `measuring spatially correlated dephasing noise between superconducting qubits decoherence-free subspace hardware bound crosstalk correlated noise 2024 2025` | 2502.09706, 2603.19373 (row 22) |
| 2026-09-08 | web index | `learning detector error model from syndrome data beyond pairwise correlations hyperedges maximum likelihood "detector error model" estimation 2024 2025 arXiv` | 2602.19722 (row 15) |
| 2026-09-08 | web index | `IBM calibration-based error model inaccurate decoding weights data-driven weights improve logical error repetition code heavy-hex Eagle Heron "detector error model" experiment` | 2601.16123 only; nothing per-family (row 16) |
| 2026-09-08 | web index | `time-correlated burst errors detection events superconducting quantum error correction stuck ancilla leakage multi-round correlation "repetition code" 2023 2024 2025 arXiv` | PAEMS 2603.29439; HMM leakage detection 2002.07119 (rows 18, 20) |
| 2026-09-08 | corpus (not a search) | Google Willow corpus re-read by family, `experiments/dual_observable/willow_family_ratio.py` | row 23; narrows row 16 |
| — | Google Scholar, IEEE Xplore, ACM DL, DBLP | **NOT SEARCHED** | — |
| — | CP / CPAIOR proceedings | **NOT SEARCHED** | — |
| — | non-English, pre-arXiv | **NOT SEARCHED** | — |

**Inclusion criterion:** a result is a row only if its own abstract or
body was read for what it CLAIMS, not matched on title resemblance. Row
1 was fetched and read; rows 3–6 are recorded at the strength of a
search-result summary and are marked accordingly by carrying their URL
rather than a quotation.

**Zero results are recorded, not dropped.** Six of the eight 2026-07-31
arXiv queries returned nothing; a sweep that reported only its hits
would look like a wider search than it was.

---

## How to use this file

A claim in a paper, a README or a passport **must cite its row**. A
mechanism with no row here has not been checked and must be presented as
`UNREVIEWED`, never as novel. `tools/prior_art_gate.py` enforces exactly
that: it reads the register, and any distinctiveness word in the prose
that does not trace to a row is a red.

`docs/OISC_PRIOR_ART.md` remains the deeper treatment of row 2 and is
cited, not superseded.
