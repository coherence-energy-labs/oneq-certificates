# Paper 2 — structured literature search log

**Evidence class:** [EXTERNAL] a structured search log over outside literature, kept so the paper's knowledge claim can be verified rather than asserted.


Kept so the paper's "to our knowledge after a structured literature
search" is a checkable statement, not a shrug. Every "first" claim in
§8 is conditioned on this log; anything it missed is a correction, not
a scandal — send it.

## Rounds and dates

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
models. Nothing found to date does (a), (b), or (c); the four lines
above are the nearest neighbors and are cited as such in §8.

## Known limits of this search

English-language, arXiv/IEEE/Nature-indexed, through 2026-08-02. No
systematic sweep of conference workshops or of the Chinese-language
literature; the "to our knowledge" qualifier carries this limit.
