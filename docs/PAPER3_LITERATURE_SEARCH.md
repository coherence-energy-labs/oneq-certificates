# Paper 3 (circuit-distance certificates): literature search log

**Evidence class:** [OBSERVED] what searches on the stated dates returned. Not finding a result is not proof that it
does not exist; the novelty wording in the manuscript (`paper/circuit-distance/main.tex`) is held to this record.
Nothing here is a claim about who found a value first: only public release dates are stated.

Earlier searches: 2026-10-26 to 2026-11-03 by the drafting session and three AI referees
(`docs/reviews/PAPER3_REFEREE*_2026-1*.md`). They read Pan's README (2026-11-02, two referees) and his manuscript
`release/arxiv/main.tex` at v2.0 (2026-11-03) but NOT his research record (`research/`): a `git grep` for `E-017`,
`E-019` or `gross-code-distance-12` over this repository found only the referee reports and refs.bib. The README's
sentence "Results not in the paper, such as the morphing-circuit analysis, are recorded there as work in progress" was
read as "not claimed". That was the miss external audit 2 (finding F11) found.

## 2026-11-03

### Pan's repository, read in full where it matters

| Query (verbatim) | Engine | What it returned that matters |
|---|---|---|
| `GET /repos/GavinNull/gross-code-distance-12/releases/tags/v2.0` | GitHub API | v2.0 published `2026-10-24T00:09:59Z`, created `2026-10-24T00:07:21Z`; tag `v2.0` is a lightweight tag on commit `5ae848ab86d25e5235f7db4fe3c8df17f071fe60`. Assets: `draft.pdf`, `arxiv.tar.gz`, `zenodo.tar.gz` (304,378,360 bytes, sha256 `797747b7...e8e9`). |
| `GET .../releases` | GitHub API | v1.0 published `2026-10-20T17:36:32Z` on commit `1ba1efa2dc53fbdd36305193ae97cd4ce9bda288`. |
| `GET /repos/GavinNull/gross-code-distance-12` | GitHub API | created `2026-10-13T12:07:29Z`, public. (When it became public is not recorded by the API.) |
| `GET .../git/trees/{v1.0, v2.0 commit}?recursive=1` | GitHub API | `research/audits/2026-10-08-E-017.md` (blob `eab30a71`) and `research/audits/2026-10-08-E-019.md` (blob `25afd539`) are in BOTH trees, byte-identical. Pan's git history dates both files 2026-10-08 (commits `37bec5aa40`, `ac1a4cbbee`); a committer date is the committer's claim. |
| `codeload .../tar.gz/5ae848ab` (3,768,727 bytes, sha256 `fafedc29...`), every text file grepped as data | download | 1,175 text files. `[[42,6,6]]`: 2 hits, both the code catalog (WM codes, code distance; not a morphing circuit). `[[30,8,5]]`: 0. `[[48,4,8]]`: 2 hits, a literature note on another paper's code (code distance). `2604.09797`: cited as the source of Prop. 1 only; no 2BGA morphing circuit of it is computed. `paper/claims_map.md` lists claims C1-C9 (the morphing results of E-017/E-019) as "moved to a separate paper"; C4 reads "to our knowledge first rigorous d_circ lower bound for any morphing circuit" (search date 2026-10-08). |

**Read in full: E-017** (English). The Shaw-Terhal circuit from the [[288,12,18]] mother code, homomorphism f_x,
repository layer order (the circuit of our `morph144`), end-cycle [[144,12,12]]:
- Sec. 1: gate-identical to the authors' builder for 3 homomorphisms x 2 layer orders x 2 bases x 2 starting rounds.
- Sec. 2: projected SD6 instance; Stim DEM terms equal the auditor's fault signatures, **864 at T = 1 and 1,584 at
  T = 2** (our projected counts are the same); every round's instance equivalent to round 1's.
- Sec. 3: the fold premise of 2604.09797 Prop. 1 checked by computation in a basis-free form, r = 2..T, T = 2..5, both
  bases, noisy preparation and readout; "arbitrary T" = these checks plus periodicity, "a machine-checked hypothesis
  plus a written argument".
- Sec. 4.1: W = 11 exhaustive DFS, two independently written programs, two detector bases, 0 solutions.
- Sec. 4.2: end-cycle code distance exactly 12; minimum stabiliser weight exactly 9 (gross: 6), so "not equivalent to
  the gross code" under qubit permutations and local Cliffords; weight-12 witnesses injected, every T.
- Sec. 8, verdicts: d_circ = 12 for every T, ideal and noisy boundaries, both bases: CONFIRMED. No certificate.

**Read in full: E-019** (Chinese). All 80 two-round morphing circuits of 2407.16336 (10 (mother code, hom) x Table
IV/VIII x 2 x 2 term orders), rebuilt from the paper's Tables IV/VIII/IX by the auditor, 80/80 gate-identical to the
experimenter's:
- Sec. 2.1: the 56 small-mother circuits solved exactly on the whole T = 4 circuit (noisy preparation and readout),
  both bases, SD6, by a new DFS (`mdfs.c`): the remaining [[144,12,12]]-mother circuits, end-cycle [[72,12,6]]
  (among them the repository order, i.e. our `morph72`): **d_circ = 6 / 6**.
- Sec. 2.1-2.3: CNOT/term order matters: [[108,8,10]] f_y -> [[54,8,8]] gives 6, 6, 7, 7 for its four legal term
  orders; [[72,12,8]] variants give 6, others 5 or 7.
- Sec. 5: every-T rests on the experimenter's per-circuit fold computation (160/160 flags true, read in code, not
  re-implemented by the auditor).
- Scope: three-round morphing and the non-Abelian 2BGA codes of App. C are out of scope (Sec. 7, item 6).

**Read: STATE.md, E-022 journal and yaml.** E-022 (2026-10-11): an exact criterion for when two-round morphing
circuits lose distance, a theorem plus a preregistered test on 288 new circuits (576 instances, all equal to the
prediction), mother codes [[180,8,16]], [[124,10,10]], [[120,8,12]], [[108,12,6]]; "loss <= 2 (344 circuits)" as an
observation. No 2604.09797 2BGA codes.

**Read: `release/arxiv/main.tex` at v2.0**, lines 94-99: the Shaw-Terhal circuit has "d_circ <= 12 [ShawTerhal2025]"
(the manuscript states only the upper bound); Wills et al. believe a morphing idling circuit has 12; the end-cycle code
is not the gross code (supplement section `sm:morph`). Line 55: IBM's depth-8 circuit "has circuit distance exactly 10
under SD6 for R <= 3, by exact enumeration". Lines 260-265: Fig. 12 = 11 for every R from SBB's Prop. 3/Cor. 1 and
d_ext = 11 by two exhaustive searches; supplement 824-826: "d_ext >= 9 ... is certified; d_ext = 11 ... rest on two
independent exhaustive searches".

### arXiv

| Query (verbatim) | Engine | What it returned that matters |
|---|---|---|
| `abs:morphing AND abs:circuit`, newest first, 40 | arXiv API | QEC entries: 2606.12724 (Chao, block algebra), 2604.09797v2 (Shaw-Terhal), 2407.16336v3; the rest are other fields or other QEC topics. |
| `all:"circuit-level distance"`, newest first, 40 | arXiv API | 2610.33347 (colour code), 2608.11516 (trapping sets), 2603.28852 (colour code), 2509.00850, 2507.06096. None on BB or morphing circuits' exact distances. |
| `all:"circuit distance" AND all:bicycle` | arXiv API | 0 entries. |
| `abs:"morphing circuits"` | arXiv API | 2606.12724, 2604.09797, 2503.18657 (magic-state cultivation), 2407.16336. |
| `all:"gross code" AND all:"circuit distance"` | arXiv API | 2603.05481v2 (SBB), 2601.09911 (time-dynamic circuits; BP-OSD upper bounds). |
| `all:"syndrome extraction" AND all:"circuit-level distance" AND all:exact` | arXiv API | 0 entries. |
| `abs:"two-block group algebra" AND abs:circuit` | arXiv API | 2610.13880, 2610.05307, 2607.22498, 2606.17268, 2606.12724, 2604.09797, 2603.05496, 2603.00228, 2509.26494, 2502.19406, 2407.16336. |
| `all:"circuit distance" AND all:certificate` | arXiv API | 2609.37231 (amplified qLDPC memories). |

Sources grepped as data (e-print tarballs) for `[[42,6,6]]`, `[[30,8,5]]`, `[[48,4,8]]`, `[[72,12,6]]`, `morphing`,
`circuit-level distance`, `d_circ`:
- **2610.13880** (Bicycle Race): no hit for any of them.
- **2606.12724** (Chao): 25 "morphing" hits; no code-parameter hit; Table II distances are code distances certified by
  an exact code-distance solver (per the abstract page and Pan's note), not circuit distances.
- **2507.19430** (directional codes): circuit-level distances of its own codes via 2604.09797 Prop. 1; none of P3's
  instances.
- **2606.17268** (coset codes): BP-OSD upper bounds on d_circ only.
- **2603.00228** (stairway codes): ILP circuit distances of its own Floquet codes, upper bounds where unfinished.

Version checks (arXiv abstract pages and e-prints):
- **2603.05481** (SBB): v1 2026-03-05, v2 2026-09-14, "Corrected figures in Appendix H, expanded introduction".
  `figures/three-color-circuits-compressed.pdf` (Fig. 12) DIFFERS between v1 and v2. v2's figure gives every CNOT,
  preparation and measurement time; v1's has Z-check labels one lower and other X-check labels, and no preparation or
  measurement times. Figure numbers 10-12 are the same in both. The lane's `SCHEDULE_3C` equals v2's timetable entry
  for entry. The appendix still reads "only 3 tiling combinations of residual errors for which the extended-code
  distance upper bound was 11" and "we conjecture this bound is tight" (v2 08-appendices.tex lines 381, 408), so
  "11 (conj.)" is accurate.
- **2604.09797** (Shaw-Terhal): v1 2026-04-10, v2 2026-04-20, "Fixed typos and corrected an error in Section 6.3.2".
  Def. 10, Prop. 1, Fig. 25 and App. C.4 read the same in both. App. C.4: "we did not check the circuit-level distance
  of the bare-ancilla or the morphing circuits."
- **2605.21898** (Wills, Beverland, Bishop, Gambetta, Rall, Siddhu, Cross; v2 2026-06-23), `05-appendices.tex`: a new
  idling circuit for the gross code "is believed to have circuit distance 12, matching that of the code, and is
  constructed using Shaw-Terhal morphing", simulated by Monte Carlo. Not cited by P3 before 2026-11-03.
- **2610.04306** (An, Yang, abstract): exact distances of a phenomenological repeated-readout model for logical Pauli
  readout. Not a syndrome-circuit memory distance.

### Web and GitHub

| Query (verbatim) | Engine | What it returned that matters |
|---|---|---|
| `morphing circuit circuit-level distance exact "two-block group algebra" codes 2026` | web | 2604.09797, 2606.12724, 2407.16336, 2603.00228; none with exact circuit distances of 2BGA morphing circuits. |
| `Shaw Terhal morphing circuits circuit distance 12 proof [[144,12,12]] end-cycle code` | web | 2407.16336 (BP-OSD <= 12), Pan's repository, Error Correction Zoo pages; no proof other than Pan's record. |
| `"circuit distance" bivariate bicycle morphing exact lower bound certificate` | web | our own release v2026.11.03; 2407.16336 (upper bounds only); 2509.26494 (no circuit distance claims). |
| `"[[42,6,6]]" OR "[[30,8,5]]" OR "[[48,4,8]]" morphing circuit distance` | web | 2604.09797 (end-cycle parameters, "we did not check the circuit-level distance"), 2606.12724, 2507.19430. |
| `gross code IBM syndrome circuit circuit-level distance 10 every number of rounds proof 2026` | web | Error Correction Zoo PR #420 (Crocker): 12 noisy + 2 noiseless rounds, "a search result, without a machine-checkable certificate" (merged 2026-10-26); Pan's v1.0/v2.0; SBB. No every-round proof for IBM's cycle. |
| repositories `morphing circuit distance`, `circuit-level distance`, `morphing_qec_circuits`, `circuit distance bivariate bicycle`, `gross code circuit distance` | GitHub search API | `shohamjac/ILP-circuit-distance` (a generic ILP tool), `Mac-Shaw/morphing_qec_circuits`, `GavinNull/gross-code-distance-12`. |
| code `"morphing" "circuit distance" in:file`; `"get_standard_morphing_specification"`; `"[[42,6,6]]"`; `"[[48,4,8]]"`; `"[[30,8,5]]"` | GitHub code search (authenticated) | Hits outside this project: `unitaryfoundation/qldpc-challenge` (code-distance records and layouts for [[42,6,6]], [[48,4,8]], [[30,8,5]]; read: no circuit distance), `qiskit-community/qcode-discovery` (a survey; morphing reaches degree 5 with weight-9 stabilizers), `elder-plinius/V3R1T4S` (an open-problem note; "Morphing circuits ... do not count"), `dylanneve1/qsim-lab` (a research note listing morphing circuits as prior art). No circuit distance of any P3 2026 instance. |

**The first pass above searched Pan's RELEASES, not his live branch, and grepped only text files.** It missed his
research records on IBM's cycle, committed to main after v2.0 (final referee of 2026-11-03, finding N1). The second
pass below replaces its conclusions.

## 2026-11-03, second pass: every cited repository at its LIVE default-branch HEAD, every file type

### Pan, `GavinNull/gross-code-distance-12`, main HEAD `41c8f978bf344552b41fe7553a057a5437c6909b` (2026-11-01T01:24:36Z)

| Query (verbatim) | Engine | What it returned that matters |
|---|---|---|
| `GET .../commits/main`; `GET .../compare/v2.0...main` | GitHub API | HEAD as above; 170 commits after v2.0 (2026-10-24T00:09:31Z to 2026-11-01T01:24:36Z); `pushed_at` 2026-11-01T01:25:27Z; releases still only v1.0 and v2.0. |
| `codeload .../tar.gz/41c8f978` (4,470,591 bytes, sha256 `7a6ab0ad...`), EVERY member grepped as data whatever its type | download | 1,609 files: md 142, py 1,121, json 47, yaml 49, txt 50, jsonl 9, tex 2, sh 117, c 32, log 7, ... 16 binary or gzip members (PDF, PNG) listed, not grepped. |
| `GET .../commits?path=research/audits/2026-10-26-E-037.md&sha=main` | GitHub API | one commit, `34ffcb140e76bec35430429957c6fed7d51e74e7`, 2026-10-25T21:39:52Z ("E-037 audit: CONFIRMED"); blob `40b60973` at HEAD (unchanged). |
| `GET .../commits?path=research/audits/2026-10-31-E-043.md&sha=main` | GitHub API | one commit, `18e47dae3076138503fba99220c357db5087ffc5`, 2026-10-31T08:55:19Z; blob `609cca0f` at HEAD (unchanged). |
| `GET .../commits/d3782f4b14` | GitHub API | 2026-10-25T21:41:11Z, "Paper: IBM depth-8 d_circ=10 certified for R<=12 (E-037, audited)". |

**Read: E-037** (audit, Chinese). IBM's depth-8 circuit, gross code: no undetectable logical of weight <= 9 for every
R = 1..12, both bases, under SD6 (R noisy rounds and a noisy transversal readout) and under Pan's re-implementation of
IBM's `decoder_setup.py` fault model (R noisy cycles + 2 noiseless; X and Z faults counted separately); [[108,8,10]]:
none <= 7 for R = 1..10. 88 VeriPB 3.0.2 certificates, each re-generated byte for byte by the auditor. The certified
formula is STRENGTHENED (minimality constraints on silent trivial sets, orbit symmetry breaking); the strengthening
"rests on a short written argument, not on the checker", which the auditor proves. An independent plain exhaustive
search (no added constraints) reproduces the eight R = 12 / R = 10 target statements. Scope stated by the auditor: "only
the listed R ..., not arbitrary R".

**Read: E-043** (audit, Chinese) and **STATE.md item 29**. IBM's depth-8 gross circuit has d_circ = 10 for every R,
both bases, SD6 ("generalised from E-037's R <= 12"): lower bound = his Theorem 5 (a bulk lifting criterion) with "two
solvers, 0 lifts" (CP-SAT and Bellman-Ford; no certificates) plus a direct DEM search for R = 1..3; upper bound = a
weight-10 witness at R = 1 and a persistence argument checked by Stim injection to R = 30. Verdict CONFIRMED.

**Read: `paper/draft.tex` at HEAD**, lines 318-324: the remaining circuits "satisfy the criterion ... by two independent
solvers and a direct search for R <= 2, 3 or 4; their upper bounds are explicit witnesses at R = 1 that persist to every
R ... In particular IBM's circuits have d_circ = 10 and 8 for every R". Lines 1080-1086, Fig. 12: "two independent
exhaustive searches give d_ext = 11; hence d_circ = 11"; no certificate of 11.

**Other items at HEAD:** `paper/claims_map.md` rows L-7 ("IBM gross depth 8 in T(8), d_circ = 10 every R", E-037 +
E-043) and L-8 (Fig. 12, "2026-10-05 E-014"); C58: "Fig. 12 d_ext >= 9 certified". E-040's g8_d11 is a DIFFERENT
period-8 circuit (translation invariant; d = 11 from Theorem 5, solvers and a direct search). Morphing: nothing after
E-017/E-019/E-022. `[[42,6,6]]`, `[[30,8,5]]`, `[[48,4,8]]`: only the code catalog and a literature note (code distance),
as before. `2604.09797` with 2BGA: Prop. 1 citations and the E-019 scope note only. `Lean`: no use.

### The other cited repositories and records

| Source | What was checked | Result |
|---|---|---|
| `Mac-Shaw/morphing_qec_circuits` | default branch `main` HEAD | `1ee705f466` (2025-08-29), the commit paper 3 cites; no later commit. |
| `sbravyi/BivariateBicycleCodes` | default branch `main` HEAD | `fa77e3333d` (2024-04-29); `decoder_setup.py` last changed 2024-02-06; nothing later. |
| Crocker, Zenodo 10.5281/zenodo.23452754 | `GET /api/records/23452754`, `/versions?allversions=true`, concept 23452753 | ONE version (published 2026-10-18; record metadata updated 2026-10-25). Description: d_circ = 10 "for 12 noisy syndrome cycles followed by 2 noiseless ones", both bases, two separately written programs; "There is no machine-checkable certificate, so this is a search-based result". Cites Cross, He, Rall, Yoder (arXiv:2407.18393) for numerical evidence of 10. |

### Every cited preprint at its latest arXiv version (abstract pages, 2026-11-03)

| Entry | Latest version | Pinned in refs.bib now | Cited claim re-read in the latest version |
|---|---|---|---|
| strikis2026high 2603.05481 | v2 2026-09-14 | v2 | Fig. 12 timetable; "we conjecture this bound is tight" |
| shaw2026optimising 2604.09797 | v2 2026-04-20 | v2 | Def. 10, Prop. 1, Fig. 25, App. C.4 |
| wills2026concatenating 2605.21898 | v2 2026-06-23 | v2 | App. C belief, no stated basis |
| webster2026distance 2603.22532 | v2 2026-08-14 | v2 | "completed within 8 hours only for the first member of the family" (Sec. "Bivariate Bicycle Code Circuits"; its number 5.5.3 not re-derived) |
| an2026certificate 2610.03214 | v2 2026-10-05 | v2 | abstract: code parameters by kernel-checked certificates |
| huang2025veriqec 2504.07732 | v3 2025-10-29 | v3 | abstract: assertion and program logic for QEC programs |
| jacoby2026stairway 2603.00228 | v1 2026-02-27 | v1 | ILP circuit distances, upper bound where unfinished |
| chen2026satqldpc 2606.12445 | v1 2026-05-29 | v1 | exact CODE distance by SAT/MaxSAT/SMT |
| beni2025tesseract 2503.10988 | v2 2025-08-06 | unpinned (shared with paper 2) | MaxSAT: [[72,12,6]] = 6, [[90,8,10]] = 8, read in v2's source |
| ehatamm2026end 2605.16523 | v1 2026-05-15 | unpinned (shared with papers 1, 2) | code distances only |
| shaw2024lowering 2407.16336, panteleev2021degenerate, roffe2020decoding, gidney2021stim | journal versions (DOI) | by DOI | not re-read |

### What survives for paper 3 (replaces the first pass's list)

- **Not new:** the morphing values (Pan E-017, E-019; releases v1.0 and v2.0), without certificates; witness injection
  for them is also Pan's (E-017 Secs. 4.2-4.3, E-019 Sec. 3). The inequivalence of the morph144 end-cycle code to the
  gross code (E-017 Sec. 4.2). That the CNOT order changes a morphing circuit's distance (E-019, E-022).
- **Not new:** IBM's 10 for every R. Pan's E-043 (main, 2026-10-31) states it, from a theorem, two solvers without
  certificates and a direct search to R = 3; his E-037 (main, 2026-10-25) CERTIFIES no weight <= 9 at every R = 1..12
  (VeriPB, strengthened formula). Crocker: 12 + 2 rounds, search only.
- **Not new as a value:** the three-colour 11 for every R (Pan, from SBB's reduction and exhaustive search).
- **Ours, as far as these searches reach:** a lower bound for IBM's cycle at EVERY T in which every SEARCH result is
  certified (22 time-shift certificates for T = 1..11, checked under a Lean-verified calculus), joined by the
  hand-proved Theorem 1 whose finite hypotheses (the identities fixing h and e, P1 and the boundary windows at T_1, the
  pairs below T_1) are checked by script, not certified; Pan's certificates stop at R = 12 and his every-R step rests on
  uncertified solver runs (E-043's heading calls it "认证", certification; it produces no proof log). (Corrected
  2026-11-03 after the final referee's residual 1: the earlier wording "whose finite inputs are all certificates"
  overstated.) The first
  certified lower bound of 11 for the three-colour cycle found (Pan certifies d_ext >= 9 only; nothing on main HEAD
  certifies 11). The first machine-checkable certificates for the two morphing values found. Exact circuit distances of
  the 14 circuits with the 2604.09797 parameters (no source computes any) and their 4 kernel-checked one-round
  distances. The Lean-verified calculus; the fold theorem as a theorem; the identity of our projected mechanism sets
  with Pan's own circuits (records/pan_crosscheck.jsonl).
- **Search scope and its limit.** The live HEADs and commit histories of the four cited repositories, Crocker's record
  and all its versions, and the latest arXiv version of every cited preprint, on 2026-11-03. A result published after
  that, or outside these sources and the queries of the first pass, is not covered.
