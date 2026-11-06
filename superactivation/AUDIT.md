# Auditing the superactivation bundle

This bundle (`oneq-superactivation`) carries one manuscript, *Superactivation in two channel uses: exact certificates
for a family of PPT channels*, with every certificate, record and line of code its numbers come from. It is released
beside the decoder bundle (`oneq-certifying-decoder`) under the same tag, and is signed and time-stamped on its own.

## 0. Is this the tree that was signed

```
sha256sum -c SHA256SUMS                                    # 1055 files
python tools/verify_release.py --bundle superactivation   # needs the three oneq-superactivation release assets in release/
```

The first command checks every file against the list; the count on its line is held by the bundle builder to the
number of files `SHA256SUMS` lists, so a file added or dropped after minting shows here. The second checks the
signature over the archive, the in-toto statement it binds, every subject's digest, and each `.ots` timestamp beside
the release files.

## 1. Are the certificates true

```
python -m pip install -r requirements.txt
python paper/tools/make_numbers_superactivation.py
python tools/verify_r142_rational_witness.py evidence/superactivation/shield_family/certificate_two_use_d20.json evidence/superactivation/shield_family/rational_recovery_d20.json
python -m pytest tests/ -q
```

The generator re-proves every n-use certificate in exact arithmetic and refuses when a certificate no longer binds to
the code and decoder bytes it names; it then writes every number the manuscript prints. The rational checker re-checks
the d = 20, 21, 22 two-use bounds with fractions only. The tests recompute the stored two-group and symmetric pairs.

## What this bundle does not establish

That any device superactivates (the hardware runs are an outlook, and the recorded device run is below 3/4); the onset
for Smith and Yard's pair (a certified code bounds it from above); that the see-saw producers re-run here (they need an
SDP solver this bundle does not pin).
