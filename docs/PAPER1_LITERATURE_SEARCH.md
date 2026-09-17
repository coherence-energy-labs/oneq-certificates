# Paper 1 — structured literature search log

**Evidence class:** [EXTERNAL] a structured search log over outside literature, kept so the paper's knowledge claim can be verified rather than asserted.

Paper 1 (*Certifying quantum error-correction decoders*, `paper/certifying-decoders/main.tex`)
makes one knowledge claim, in its related-work section: that its searches did not find the
forgery battery, the upstream-fault detection experiment, or hardware certification rates
elsewhere, and that not finding them is not proof of absence. This file is the log that
sentence refers to. Everything a "first" would rest on is conditioned on it; anything it
missed is a correction, not a scandal — send it.

Paper 1 claims no method novelty. The observation that decoding has a checkable dual is
prior art and the manuscript says so (Wu et al., arXiv:2508.04969, primal–dual with
certifiable proximity bounds; the LP structure of matching and T-joins is Edmonds 1965 and
Edmonds–Johnson 1973). Its contribution is the assurance engineering and measurement around
that observation.

## Rounds and dates

- **2026-09-16, the round behind Sec.~"Certifying against a model measured from the run itself".**
  That section added a knowledge claim to the manuscript, so the searches behind it are logged here
  and its verdict is register row 29. Web index, twelve queries, verbatim:
  `two-point correlation pij method estimating decoding graph edge probabilities from detector data
  surface code`; `leakage induced correlated errors superconducting quantum error correction leakage
  lifetime rounds`; `hidden Markov model leakage detection decoder surface code transmon`; `data
  qubit leakage versus ancilla leakage distinguishing syndrome signature repetition code`; `learned
  noise model beats vendor detector error model decoding logical error rate IBM experiment 2026`;
  `temporally correlated errors beyond detector error model repetition code experiment long range
  detector correlations`; `leakage aware decoder hidden Markov state inserted as decoding graph edges
  weights improve logical error rate`; `arXiv 2026 correlated flipping adjacent detectors consecutive
  rounds repetition code Google high distance noise floor`; `PAEMS precise adaptive error model
  superconducting quantum processors detector error model refit`; `per-shot optimality certificate
  decoder learned noise model proof carrying quantum error correction verifiable minimum weight`;
  `two-state Markov leakage model closed form pairwise edge probabilities all round separations
  matching graph analytic lowering`; `leakage localization data qubit versus ancilla likelihood
  comparison different devices IBM measured syndrome statistics 2026`. Three sources were fetched in
  full: arXiv:2512.10814 (DEM estimation on Willow, long-range detector correlations),
  arXiv:2504.20212 (estimating decoding graphs and hypergraphs), arXiv:2601.16123 (Stein et al.).
  **What the round FOUND, and it narrowed the manuscript before anything was written:** the
  estimator is prior art (Spitz et al., arXiv:1712.02360; Google's analysis, arXiv:2102.06132);
  estimated models beating device-informed ones on IBM is prior art (Takou et al., arXiv:2606.11496)
  and is done *better* on our own two processors by Stein et al. (arXiv:2601.16123, 11.1x at d=11
  against calibration-weighted matching, held out on unseen chains). The section states both, and
  claims no priority for its decoding margin. **What it did NOT find:** a per-shot,
  third-party-checkable optimality witness over a model estimated from the shots being decoded.
  The nearest are HyperBlossom (arXiv:2508.04969, certifiable proximity bounds for a decoder under a
  given model) and Krishnamoorthy et al. (arXiv:2608.25545, certificates internal to a sampler);
  none of the estimated-DEM papers carries a per-shot witness at all. Not searched this round:
  Google Scholar, Semantic Scholar, IEEE Xplore, non-English venues, and any venue's 2026 Q3
  proceedings not yet indexed by the web index used.

- **2026-09-15, pre-review refresh.** Web index, two queries, verbatim:
  `certified optimality decoder quantum error correction dual certificate minimum weight
  perfect matching receipt arXiv 2026`; `verifiable decoder output syndrome decoding
  certificate checker "surface code" PyMatching optimality proof 2026`. Returned HyperBlossom
  (arXiv:2508.04969, already cited), Krishnamoorthy et al. (arXiv:2608.25545, already cited),
  neural and learned MWPM variants without certificates (arXiv:2601.00242, 2512.08914,
  2509.06756, 2604.12841), PyMatching itself, and surveys. **Nothing found that the
  manuscript does not already cite.** Not searched this round: Google Scholar, Semantic
  Scholar, IEEE Xplore; non-English venues.

- **2026-09-12 to 2026-09-14, audit response.** The 2026-09-12 refresh recorded for paper 2
  (`docs/PAPER2_LITERATURE_SEARCH.md`) found two preprints that bear on this paper as well and
  are now cited as concurrent work: Krishnamoorthy et al., *Certified decoding of quantum LDPC
  codes*, arXiv:2608.25545 (posted 2026-08-26; certificates internal to a sampler, degenerate-
  coset maximum likelihood), and Shen and Zhong, *Capability-Gated Conformance Testing of
  Quantum Error-Correction Decoder Libraries*, arXiv:2609.07035 (posted 2026-09-07; verifier-
  side conformance of libraries, not per-shot optimality). The external audit of 2026-09-14
  (`docs/EXTERNAL_AUDIT_2026-09-14.md`, C-01..C-03) asked for Sahay, Williamson and Brown,
  arXiv:2602.22770, and for sharper comparisons with the two above; Ehatamm et al., *End-to-End
  Formalization of Quantum Error Correction*, arXiv:2605.16523, was added in the same revision
  as the proof-assistant end of the same spectrum. Every entry is verified against its source
  by `paper/tools/verify_refs.py` (`paper/refs_verified.json`, fetched 2026-09-14).

- **2026-08-21 to 2026-08-22, prior-art register sweep.** Web index; the verbatim queries are
  rows of the table in `docs/PRIOR_ART_REGISTER.md` §"The searches, verbatim". The query
  `certifying decoder quantum error correction per-shot optimality certificate LP duality
  witness verifiable` returned HyperBlossom, which narrowed the claim as recorded there: a
  primal–dual LP model with certifiable bounds for QEC decoding is not novel and is never
  presented as such. The queries of 2026-08-22 on certified maximum-likelihood decoding,
  soft-output confidence and per-shot proxies found no third-party-checkable per-shot
  optimality receipt for a matching decoder.

- **2026-07-31, arXiv API sweep.** Eight queries recorded verbatim, with all results including
  the six that returned nothing, in `experiments/m0_prior_art/arxiv_sweep.json`.

## Inclusion criterion

A work displaces the knowledge claim if it (a) attaches a decoder-independent, stranger-
checkable optimality witness to a matching decoder's per-shot output and reports acceptance
rates, (b) reports an adversarial forgery battery against such a checker with every attack
verified false independently of the checker, (c) reports detection of seeded decoder faults or
of an upstream calibration fault by such a witness with a false-alarm base rate, or (d)
reports per-shot certification rates on syndromes from real hardware. Nothing found to date
does (a)–(d); the nearest neighbours are cited in the manuscript's related-work section as
such.

## Known limits of this search

English-language, arXiv- and web-index-based, through 2026-09-15. No systematic sweep of
Google Scholar, Semantic Scholar, IEEE Xplore, conference workshops, or the Chinese-language
literature; the manuscript's "not finding them is not proof of absence" carries this limit.
