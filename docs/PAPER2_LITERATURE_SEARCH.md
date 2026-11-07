# Paper 2 — structured literature search log

**Evidence class:** [EXTERNAL] a structured search log over outside literature, kept so the paper's knowledge claim can be verified rather than asserted.


Kept so the paper's "to our knowledge after a structured literature
search" is a checkable statement, not a shrug. Every "first" claim in
§8 is conditioned on this log; anything it missed is a correction, not
a scandal — send it.

## Rounds and dates

- **2026-11-06, after external audit 3 (which named arXiv:2610.28008).** The
  previous round was 2026-09-15, before that paper appeared (2026-10-20).
  Run 2026-11-07 04:00-04:20 UTC (2026-11-06 evening EST).
  - **arXiv itself, advanced search** (`arxiv.org/search/advanced`), field
    = abstract, terms ANDed, cross-lists included, submission date
    2026-09-01 to 2026-11-07, 200 results per page. Sixteen queries,
    verbatim, with the hit counts arXiv returned:
    `certified AND decoding AND quantum` 15;
    `certificate AND decoder AND LDPC` 0;
    `"bivariate bicycle" AND decoder` 31;
    `"bivariate bicycle" AND decoding` 31;
    `"maximum likelihood" AND "quantum LDPC"` 1;
    `"linear programming" AND decoding AND quantum` 6;
    `"branch and bound" AND decoding` 3;
    `optimality AND "ordered statistics"` 37 (nearly all statistics, not
    decoding); `"minimum weight" AND decoding AND certificate` 1;
    `"gross code"` 11; `"lower bound" AND decoder AND "logical error"` 2;
    `verifiable AND decoding AND quantum` 15;
    `benchmark AND decoder AND "maximum likelihood" AND quantum` 1;
    `"BP-OSD"` 13; `"optimal decoding" AND quantum AND LDPC` 1;
    `rational AND certificate AND decoding` 5. 118 distinct papers; every
    title was read, 11 of them were opened (listed below), and the other
    107 were judged from their titles to be off-topic (most matched
    "optimality" in statistics) or already recorded (2508.04769,
    2609.07035, 2507.00254).
    (The arXiv API, `export.arxiv.org`, answered HTTP 429 and then a
    redirect loop throughout; the web search was used instead.)
  - **Web index**, three queries, verbatim: `certified optimality decoding
    quantum LDPC bivariate bicycle certificate arXiv October 2026`;
    `exact rational certificate minimum weight decoding BP-OSD lower bound
    dual arXiv 2026`; `maximum likelihood benchmark quantum LDPC decoder
    branch and bound certified lower bound gross code 2026`. They found
    2610.28008 again and two classical-code papers the arXiv queries
    could not (no "quantum" in their abstracts), 2610.27805 and 2610.32171,
    and one earlier paper, 2603.19062. Fourteen abstracts were read in
    all; 2610.28008 in full, and of the other three now cited the
    sections each citation rests on (named below).
  - **Found, read, and now cited in paper 2:**
    - **Banfi, Ferrari, Favano, Tarable, Barletta, "Certified
      Non-Degenerate Maximum-Likelihood Benchmarking of Quantum LDPC
      Decoders", arXiv:2610.28008 (2026-10-20).** Read in full. Per-shot
      certified success / failure / undecided of NON-DEGENERATE ML on
      [[144,12,12]] (3 rounds) and [[72,12,6]] (6 rounds), full
      circuit-level noise; upper bounds from verified witnesses, lower
      bounds from disjoint-detector and charging bounds, best-first
      branch-and-bound, an adaptive Feldman/Taghavi-Siegel parity-cut LP,
      HiGHS MIP dual bounds and infeasibility tests; App. C.7: "Costs,
      charges, and solver bounds use floating-point arithmetic. The
      current results therefore rely on the solver and its tolerances for
      numerical lower bounds and for the infeasibility claims"; validation
      by small-instance tests and by re-evaluating every certified failure
      with Tesseract's exact IP decoder. No exported proof of the lower
      bounds is described. THE CLOSEST WORK: same code, same per-shot
      benchmarking purpose, an overlapping toolbox. It narrows paper 2's
      novelty sentences (Secs. I and VIII now name it) to what it always
      meant: EXACT, independently re-checkable proofs of a declared
      integer objective's minimum, decided without a solver. Compared on
      four axes in Sec. VIII.
    - **Hwang, Won, Hwang, "Certified Adaptive List Decoding for
      Error-Correcting Codes", arXiv:2610.27805 (2026-10-20).** Classical
      block codes; an LP dual bound and a distance test certify per frame
      that a neural decoder's candidate is a global ML solution, in
      floating point with a margin 1e-4 under which a dual-certified frame
      is "epsilon-optimal in likelihood cost rather than exactly optimal"
      (their App. A.5). Same mechanism as FWK's ML certificate; not
      quantum, not exact, not a separate checker. Cited beside Wu et al.
    - **Fregona, Loebl, Brown, Kastoryano, Paesani, "Dealing with
      low-weight errors that lead BP OSD to fail", arXiv:2610.30305
      (2026-10-21).** Proves Tanner-graph "reflections" keep parallel
      product-sum BP from converging on some weight-3 errors of BB72,
      BB144, BB288 at code capacity; reports NaN from opposite infinite
      messages in Roffe's BP-OSD; repairs by Noisy BP (bounded random
      perturbation) and Detect-Hold BP. Concurrent with paper 2's Sec.
      VI D (exact cancellation in unscaled min-sum); cited there. No
      certificates.
    - **Cheng, Guo, Pan, "Certifying quantum code and circuit distances at
      scale", arXiv:2611.08247 (2026-11-05).** Certifies code and circuit
      distances by proved lower bounds paired with logical errors of
      matching weight; "With one noisy cycle, the certified circuit
      distance is 10 in both bases for the gross code" (standard
      schedule); two-gross certified at 16, not the assumed 18. Cited in
      paper 2's distance paragraph. MATERIAL FOR PAPER 3 (circuit-distance
      certificates): reported to the owner of that paper.
  - **Read, not cited (adjacent, no displacement):** Aliasgari,
    Khodaiemehr, Bagheri, arXiv:2610.32171 (exact Nullstellensatz
    certificates of syndrome exclusion for classical LDPC matrices, two
    verifiers; not decoding optimality); Liu, Zeng, Wu, Lao,
    arXiv:2609.03928 (approximate ML via truncated free energies over a
    BP-OSD candidate pool, incl. [[144,12,12]]; no certificate); Shang,
    arXiv:2610.24503 (logical-space theory of OSD lists on the gross
    code); Lin, arXiv:2611.01360 (logical-class diagnostic layer;
    "certified faithful window" on the Steane code, a protocol for the
    gross code; not optimality); Javed, Garcia-Herrero, Vasic, Flanagan,
    arXiv:2611.03128 (BP-guided Tesseract); Wang, Coveney,
    arXiv:2609.37629 (adaptive BP-OSD routed by decoder disagreement);
    Bilgin et al., arXiv:2610.41398 (learned message passing with beam
    search); Wayo et al., arXiv:2607.19447 (syndrome-to-decoder mapping
    audit); Xie et al., arXiv:2601.17757 (post-selection by argument
    reweighting); Pandey, arXiv:2603.19062 (BB baselines on the erasure
    channel; no certificates).
  - Not searched this round: Google Scholar, Semantic Scholar, IEEE
    Xplore, ACM DL; non-English venues. Not-found is not absent.

- **2026-09-15, pre-review refresh.** Web index, two queries, verbatim:
  `"certified decoding" OR "optimality certificate" quantum LDPC bivariate
  bicycle BP-OSD arXiv September 2026`; `linear programming decoding quantum
  LDPC exact dual bound branch and bound certificate gross code 144 12 12
  2026 arXiv`. Returned arXiv:2608.25545 (Krishnamoorthy et al.),
  arXiv:2606.20513 (Leverrier–Urbanke), arXiv:2508.04769 (Gu et al.),
  arXiv:2311.18488 (Javed et al.) and BP-variant decoders without
  certificates (arXiv:2507.00254, 2605.25777, 2605.14170). **Nothing found
  that the 2026-09-12 round had not already recorded and the manuscript
  does not already cite.** Not searched: Google Scholar, Semantic Scholar,
  IEEE Xplore; non-English venues.

- **2026-09-12, pre-disclosure refresh (six weeks after the last
  round; a "first" claim is only as fresh as its last search).** Web
  index, three queries, verbatim: `certified optimality qLDPC decoding
  LP dual certificate bivariate bicycle BP-OSD 2026 arXiv`;
  `verifiable per-shot decoder certificate quantum LDPC "minimum
  weight" proof checker arXiv August September 2026`; `linear
  programming decoding quantum LDPC codes exact optimality witness
  branch and bound gross code 144 12 12 2026`. **Three papers found
  that the 2026-08-02 log could not have seen or did not record, and
  the first narrows §8(iii):**
  - **Krishnamoorthy, Gerhardt, Knaute, Klir, Maschek, Schulze, Maras,
    Dotterweich, Lee, Bauckhage, Piatkowski (Fraunhofer IAIS / PwC /
    Bull), "Certified decoding of quantum LDPC codes",
    arXiv:2608.25545, posted 2026-08-26** — the same two BB codes
    ([[72,12,6]], [[144,12,12]]), the same three noise models, compared
    against BP+OSD, "attach to every decision a certificate of
    optimality." NEAREST FOUND to (iii), and the distinctions are
    exact: their object is the **degenerate-coset ML class** (a strictly
    stronger target than minimum declared weight), certified either
    STATISTICALLY (paired bootstrap over CRN-coupled annealed
    importance-sampling chains, Bonferroni-corrected) or EXACTLY via
    their Lemma 1 composed with a constant-factor partition-function
    estimator (WISH); the proof system is partition-function /
    Bethe-free-energy, with no LP, no dual, and no minimum-weight
    formulation; verifying a certificate requires the sampler's own
    paired estimates or the estimator's constant-factor guarantee, so
    it is not re-checkable by a third party from (H, w, s) alone; they
    audit BP+OSD but do not repair it; code "available upon request";
    no prior certificate-carrying decoder is cited. So (iii) survives
    narrowed to what it always meant — INDEPENDENTLY re-checkable,
    exact-rational, minimum-weight receipts on this code — and must
    now say so against this paper by name.
  - **Shen, Zhong, "Capability-Gated Conformance Testing of Quantum
    Error-Correction Decoder Libraries", arXiv:2609.07035, posted
    2026-09-07** — conformance testing of five public decoder
    libraries; certificates "ship as bundles a standalone verifier
    re-derives from first principles"; a minimum-weight check via two
    feasible corrections of different weight under two presentations
    of one instance. Adjacent in spirit (verifier-side, producer
    untrusted), different object (library conformance, not per-shot
    optimality; no LP duals, no BB code).
  - **Leverrier, Urbanke, "Approximating optimal decoding of quantum
    LDPC codes with narrow frontiers", arXiv:2606.20513, posted
    2026-06-18** — predates the August round and was not recorded
    then: an exact-ordered-inference decoder (pruned) with results on
    the gross code at circuit level; no certificate, no optimality
    guarantee once pruned. Related work for "exact inference on this
    code", to be cited.
  Not searched this round: Google Scholar, Semantic Scholar, IEEE
  Xplore; non-English venues. Not-found ≠ absent.

- **2026-08-01/02, internal review rounds 1–2:** referee-driven search
  for the closest prior art to each novelty claim. Confirmed and cited:
  Feldman–Wainwright–Karger (IEEE-IT 2005), Taghavi–Siegel (IEEE-IT
  2008, adaptive LP + exact O(deg) separation), Zhang–Siegel (IEEE-IT
  2012, adaptive cuts), Li–Vontobel (ISIT 2018, quantum LP),
  Landahl et al. (arXiv:1108.5738, ILP decoding), Mehlhorn et al.
  (CSR 2011, certifying-algorithms survey), Blum–Kannan (JACM 1995),
  Bravyi et al. (Nature 627, BB codes), Panteleev–Kalachev / Roffe
  (BP-OSD), Gidney (Stim), Google QEC below threshold (Nature 638),
  Shutty–Newman–Villalonga (arXiv:2401.12434, ensembling confidence).
- **2026-08-02, external adversarial audit response:** the audit named
  four further lines; each was verified to exist before citing:
  - Halabi–Even, local-optimality certificates for Tanner codes —
    arXiv:1104.4646 (query: `Halabi Even "local optimality"
    certificates Tanner codes LP decoding`).
  - Javed–Garcia-Herrero–Flanagan, LP decoding of qLDPC —
    arXiv:2311.18488 (query: `"linear programming" post-processing BP
    decoder quantum LDPC Javed`).
  - Gu et al., power and limitations of LP decoding for qLDPC —
    arXiv:2508.04769 (same query family).
  - HyperBlossom/Hyperion, minimum-weight parity factor certifying
    primal-dual — arXiv:2508.04969 (query: `HyperBlossom Hyperion
    certifying primal-dual quantum decoder`).

## Inclusion criterion

A work displaces a novelty claim if it (a) attaches a
decoder-independent, stranger-checkable optimality witness to a
third-party heuristic decoder's answers, (b) measures certified-repair
coverage end-to-end on a named quantum code, or (c) reports per-shot
optimality receipts for a bivariate-bicycle code under stated noise
models. Nothing found to date does (a) or (b). Since 2026-11-06, (c)
holds only as narrowed: Banfi et al. (arXiv:2610.28008) report per-shot
CERTIFIED OUTCOMES of non-degenerate ML on [[144,12,12]] under
circuit-level noise, with solver-dependent floating-point lower bounds and
no exported proof; nothing found reports EXACT, independently re-checkable
per-shot receipts of a declared objective's minimum for a
bivariate-bicycle code. The lines above are the nearest neighbours and
are cited as such in §8.

## Known limits of this search

English-language, arXiv/IEEE/Nature-indexed, through 2026-11-07 (the
arXiv rounds; the venue rounds stopped at 2026-08-02). No
systematic sweep of conference workshops or of the Chinese-language
literature; the "to our knowledge" qualifier carries this limit.
