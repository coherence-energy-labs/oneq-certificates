# Held-out campaign — results and disclosures

**Evidence class:** [ESTATE-MEASURED] the observed-results record of the held-out campaign; every number here was measured after the protocol was sealed.


*Separated from `HELD_OUT_PREDECLARATION.md` per audit round 6; the preregistration is the immutable protocol record, this file is the observed-results record.*

---

# RESULTS (as observed; run of record, seeds from freeze aba2aec)

| rung | N nontrivial | tier 1 | tier 2 | tier BD | exact-certified | open |
|---|---|---|---|---|---|---|
| uniform p=0.01 | 300 | 300 | 0 | 0 | **300/300** | 0 |
| uniform p=0.02 | 470 | 450 | 20 | 0 | **470/470** | 0 |
| heterogeneous p=0.02 | 300 | 299 | 1 | 0 | **300/300** | 0 |
| phenomenological p=0.01 | 150 | 150 | 0 | 0 | **150/150** | 0 |
| phenomenological p=0.02 | 150 | 149 | 0 | 1 | **150/150** | 0 |
| X-sector p=0.02 | 200 | 194 | 6 | 0 | **200/200** | 0 |
| stress p=0.05 | 300 | 122 | 164 | 14 | **300/300** | 0 |
| reference-schedule circuit | 149 | 116 | 31 | 2 | **149/149** | 0 |

**2,019/2,019 nontrivial shots exact-certified. Zero solver-tier
receipts. Zero float/exact disagreements. Full reference-checker
agreement on every flat certificate. Reference-schedule observable
agreement 149/149.** The confirmatory campaign REPLICATES the
canonical rates on unseen data under frozen code; the branch-dual
closer resolved every residual (17 across three rungs). No re-runs, no
exclusions, no changes after opening. Held-out rates are now the
papers' headline rates; canonical rates are development data.

Artifacts: `evidence/qldpc_heldout.json`,
`evidence/per_shot/heldout_*.jsonl`,
`evidence/heldout_qldpc_bb_refsched_gate.json`.

---

# POST-CAMPAIGN INTEGRITY-AUDIT DISCLOSURES (2026-08-03)

An independent adversarial reproducibility audit verified the campaign
mechanically end-to-end (pinned hashes match the tagged code; freeze-set
files byte-identical through the results commit; all 9 seeds re-derive
exactly; 2,019 records reconcile cell-for-cell; single freeze, no
discarded attempts, no stray artifacts). Its findings, disclosed:

- **F1 (fixed):** the predeclared held-out stress paired 2×2 went
  unreported in the first results write-up. Computed from the shipped
  records: 125 / 170 / 0 / 5 (BP-OSD wrong 175 vs pipeline 5;
  discordants one-sided again) — favorable and late, now in
  `evidence/qldpc_heldout_addendum.json` and paper §5.0.
- **F2 (fixed):** the 17 tier-BD closures carried no re-checkable
  payload in the records, and "every exact certificate through the
  pinned reference checker" silently narrowed to flat certificates
  (the reference implements flat only). All 17 trees are regenerated
  deterministically from the shipped records, re-verified, and STORED
  with node counts in the addendum; the scope narrowing is now stated
  wherever the claim appears.
- **F3:** the runner mods the refsched seed by 2^31−1 for stim — frozen
  in the pinned runner before data, deterministic, recorded; disclosed
  here because the prose seed rule was silent on it.
- **F4 (wording corrected):** the seed source is a SELF-REFERENTIAL
  commitment (the freeze commit's own sha), not an external public
  beacon; local history shows a single freeze and the tag is on the
  remote, but freeze-shopping cannot be cryptographically excluded
  from local evidence alone. Future campaigns should bind seeds to an
  external beacon (NIST/drand).
- **F5:** refsched's "150" was a shot count, not a nontrivial target;
  the run yielded 149 nontrivial and is reported as 149/149.
- **F6:** pinned hashes were computed over checked-out bytes with
  mixed EOL conventions; re-verifiers should normalize line endings
  (3 of 6 pins are CRLF-byte hashes).
- **F7:** the "~30–60 minutes" compute estimate was off (~100 s actual)
  — the tight freeze-to-results window is corroborating evidence for a
  genuine post-freeze run, not against it.
