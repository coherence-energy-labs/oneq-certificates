# Superactivation in two channel uses — re-runnable bundle

This is a SEPARATE line of work from the decoder certificates released beside
it: quantum capacity, not error-correction decoding. It accompanies the
manuscript by Josh Philbrick (Coherence Energy Labs), whose LaTeX source is in
`paper/superactivation/`:

* *Superactivation in two channel uses for private Horodecki channels with
  large shields: exact certificates*

The same release tag carries the decoder bundle (`oneq-certifying-decoder`,
three manuscripts on certifying decoders). Each bundle is signed and
time-stamped on its own, and neither needs the other. In the public repository
the decoder bundle is the root of the tree and this bundle is the subdirectory
`superactivation/`; every command below runs from INSIDE this directory
(`cd superactivation`), and its `SHA256SUMS` lists this directory's files only.

## What is here

* `experiments/superactivation/` — the lane: the code families, the n-use
  certifiers, the reductions and the hardware designs.
* `evidence/superactivation/` — every certificate, witness, code and decoder
  the manuscript's numbers come from, each certificate bound by sha256 to the
  bytes it proved.
* `src/oneq/superactivation*.py` — the channel model and the certificate
  checker (exact integers and ball arithmetic; no solver).
* `tests/` — the lane's tests, including the recomputation of stored pairs.
* `docs/PAPER4_LITERATURE_SEARCH.md` and `docs/reviews/` — the literature
  record the novelty wording is held to, and the referee reports.

## Reproduce

    python -m pip install -r requirements.txt
    python paper/tools/make_numbers_superactivation.py
    python paper/tools/build.py superactivation
    python tools/verify_r142_rational_witness.py evidence/superactivation/shield_family/certificate_two_use_d20.json evidence/superactivation/shield_family/rational_recovery_d20.json
    python -m pytest tests/ -q

(and the same for d21 and d22_closed with their rational_recovery_d21/d22
witnesses). `make_numbers_superactivation.py` re-proves every n-use certificate exactly,
recomputes the closed forms and their thresholds, and refuses when a
certificate no longer binds to its bytes; every number in the manuscript is a
macro it writes. `verify_r142_rational_witness.py` re-checks the d = 20, 21, 22
two-use bounds in exact rationals with no producer code. The two-group and
symmetric certificates took hours of exact arithmetic and are not re-run by the
generator: each is bound by sha256 to its witness and decoder bytes, and the
tests recompute the stored pairs. Building the PDF needs a TeX distribution
with REVTeX 4.2. The see-saw producers that found the codes need an SDP solver
(cvxpy) that is not pinned here: their outputs are what is checked.

`python tools/verify_release.py --bundle superactivation` checks a signed
release against the issuer pinned in `ISSUERS.json`: put the three
`oneq-superactivation` release assets in `release/` first.
