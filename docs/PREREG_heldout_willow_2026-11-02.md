# Preregistration: a held-out Willow sample for paper 1's hardware rate (2026-11-02)

**Why.** Paper 1's hardware rates are measured on the shots the producer was developed on: each producer generation was
introduced against the shots its predecessor left open, and the epsilon rerun re-certified those same shots. The
first independent referee (2026-11-02, finding 4) asked for a rate on shots no generation ever saw, with the producer
frozen. This file fixes that measurement before any of its shots is certified; it is committed, alone with its runner
and inputs, ahead of the run, and the commit's timestamp is the record.

**Data (never used in development).** Google's Willow surface-code archive, Zenodo 10.5281/zenodo.13273331, the
configurations at **50 rounds** on the same two qubit placements paper 1 used at 13 and 30 rounds:

| configuration | basis | rounds |
|---|---|---|
| `d3_at_q4_5` | X, Z | 50 |
| `d5_at_q8_7` | X, Z | 50 |

No producer generation, rerun, test or analysis of this project has read a 50-round configuration (the corpus fetched
for paper 1 held rounds 13 and 30 only; `evidence/google_corpus_manifest.json`). The four configurations were fetched on
2026-11-02 by `tools/fetch_google_corpus.py --rounds 50 --distances 3 5`, with the experimenters' graphlike model
`correlated_matching_decoder_with_si1000_prior`; their bytes are pinned in
`evidence/google_corpus_manifest_heldout_r50.json`. Distance 7 is excluded in advance: at 30 rounds it needed the full
ladder at tens of seconds per shot, which this box cannot spend now.

**Sample.** The first **5,000 shots** of each configuration, in 20 chunks of 250 (shots [0, 5000)), all of them; no
exclusion, no re-run, every chunk retried until it completes, as in the epsilon rerun.

**Producer (frozen).** `CertifyingDecoder(edges, epsilon_max=1e-6).certify` -- the base ladder, exactly the call
`experiments/m0_prior_art/epsilon_rerun.py` makes for the ten configurations it counts at generation 4 -- at the commit
that carries this file; decoder: PyMatching on the experimenters' model, as for every Willow configuration in paper 1.

**Checker.** `matching_cert_epsilon.check_epsilon`, eps_max = 1e-6, exact rationals. Every receipt kept
(`data/epsilon_receipts/heldout_<configuration>/`, per-chunk sha256 in `evidence/epsilon_rerun/heldout_<configuration>/`).

**Reported, whatever it shows.** Per configuration and pooled: nontrivial shots, accepted (proven within eps_max),
exactly optimal, not proven, the largest accepted epsilon, and the exact two-sided 95% Clopper-Pearson interval on the
pooled acceptance rate. The forgery battery runs as in the rerun and its acceptances are reported. Paper 1 reports this
pooled rate beside, not in place of, the development-sample rate, and says which is which.

**Runner.** `python experiments/m0_prior_art/heldout_willow.py --workers 3`, then `--merge`.
