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

## 2026-11-07: Cheng, Guo and Feng Pan (arXiv:2611.08247v1), Guancheng Pan's live branch, and arXiv from 2026-11-01

Trigger: arXiv:2611.08247 appeared after the 2026-11-03 search. Every download below went into its own new, empty
directory and was read as data with `python -I`; nothing downloaded was executed.

### arXiv:2611.08247v1, read from its e-print source

| Item | Value |
|---|---|
| Abstract page | `https://arxiv.org/abs/2611.08247v1`: "[v1] Thu, 5 Nov 2026 16:25:02 UTC (511 KB)"; 73 pages (18 main, 55 SI), 8 figures; quant-ph |
| e-print | `https://arxiv.org/e-print/2611.08247v1`, gzip tar, sha256 `6ce317c3bc16e958...`; `ncs.tex` (3,727 lines, sha256 `b174740f...`) plus `tables/si/*.tex`, read wherever they touch a circuit or claim of paper 3 |
| HTML | `https://arxiv.org/html/2611.08247v1` (sha256 `b68fb8e2...`), used only to confirm the numbering of Supplementary sections and tables cited below |
| Authors | Bin Cheng (1), Naixu Guo (1), Feng Pan (1, 2, corresponding): (1) Centre for Quantum Technologies, National University of Singapore; (2) Singapore University of Technology and Design; `feng_pan@sutd.edu.sg` (`ncs.tex` lines 139-143) |
| Code | `github.com/Fanerst/tropidist`, MIT, created 2026-10-07, default branch `main` at `b03ec2d4077f` (2026-11-06); no release, no tag |

**Feng Pan is not Guancheng Pan.** Evidence, each read from its source: (i) affiliations and addresses: Feng Pan, CQT
(NUS) and SUTD, `feng_pan@sutd.edu.sg`; Guancheng Pan, "Zhejiang University, Hangzhou, China", `3250101086@zju.edu.cn`
(his manuscript `release/arxiv/main.tex` at tag v2.0, lines 21-23, and again at main HEAD `2d5131b1`). (ii) GitHub
accounts (`gh api users/...`, 2026-11-07): `Fanerst`, name "Feng Pan", company "Singapore University of Technology and
Design", bio "PhD in theoretical physics", created 2017-09-26, which hosts tropidist and is cited by 2611.08247 as
"F. Pan, tropidist"; `GavinNull`, no name, created 2026-03-17, which hosts gross-code-distance-12. (iii) 2611.08247
cites none of Guancheng Pan's work, Crocker, Shaw and Terhal, Wills et al. or paper 3 (`ncs.bbl`, `bu1.bbl` grepped);
its own prior work is "Cheng, B. & Pan, F., Exact maximum likelihood decoding beyond treewidth via rank-decomposition
dynamic programming", arXiv:2609.39556.

**What "certified" means there.** SI preamble (`ncs.tex` 868-869): "a distance is certified when a lower bound proved by a
complete evaluator meets the weight of an explicit error that has been checked against the matrices". SI 12.2
(`ncs.tex` 3673-3674): "Lower bounds come only from completed thresholds of an evaluator whose completeness is proved in"
SI 3-5. Completeness is proved on paper (SI 5.2 Theorem "Certified weight-bounded search", SI 5.5). The witness is
checked against the matrices and replayed in the noiseless Stim circuit as deterministic Pauli errors (Methods,
`ncs.tex` 642). There is no proof object for a lower bound. What a third party receives (SI 12.3, and
`results/README.md` at `b03ec2d4`): "every result record (`results/`), each with its certified bounds, its threshold log
and its error" -- a JSON record per run with the threshold-by-threshold log (`K`, status, node count, seconds) and the
witness with its replay; re-checking a lower bound means re-running a search. Floating point: the search kernel's
covering bound is computed in `double` with tolerance `lim = R + 1e-9` and the dual ascent with slack threshold
`1e-12` (`cdist/gmitm.c`, `cover_prune`, at `b03ec2d4`); the paper proves the bound over the reals (SI 5.3, 5.5.1).
Soundness defects fixed during the work (SI 12.2): until 2026-10-20 a labeled-mode table path "can report exhaustion
wrongly" (SI 5.5.2); on 2026-10-30 "four further paths that could end a threshold early were fixed"; the paper states
the affected values were recomputed or are unaffected.

**Method, against paper 3's.** Sector reduction (detectors of one memory basis; exactness checked per circuit) = our
same-type projection. Search branches on an unsatisfied row (firing detector), anchors on symmetry orbits (lattice
translations), prunes with a covering (dual LP) bound, closes with completion tables (Methods, SI 5.1-5.3). Time
collapse sums each check's detectors over the rounds; for "identical syndrome rounds that carry every Pauli error on the
data qubits through unchanged ... and a final transversal measurement", "the collapsed problem is the same for every
number of rounds, and [the relaxation inequality] bounds all of them" (SI 5.4, `ncs.tex` 2135-2142); "For single-ancilla
CSS circuits it is the extended-code bound of Strikis, Browne and Beverland" (Methods, `ncs.tex` 743-745). Lifting the
collapsed minimizers raises the bound by one per exhausted weight, and "Lifting is for a fixed number of rounds"
(Methods, `ncs.tex` 752).

**Per circuit of paper 3.**

| Paper-3 circuit | What 2611.08247 states | Where |
|---|---|---|
| IBM depth-8 gross cycle, standard schedule | (a) Memory instruction of IBM's architecture study (noise-free preparation and syndrome round, `C` noisy cycles, noise-free final round; CNOT two-qubit depolarizing, idle depolarizing, preparation and measurement flips), `C = 1`: "the certified circuit distance is 10 in both bases for the gross code, as assumed"; "the time-collapsed bound gives 8 (X basis) and 9 (Z basis), the sector search and lifting raise it to 10". (b) Circuits of Bravyi et al.'s public code (`d` noisy cycles, then noise-free ones): "With 12 cycles the gross code's circuit has distance 10 in both bases, as stated." (c) Six-cycle memory circuits in their own noise model: the standard schedule's distance, "the smaller of the two" bases, is 10 (Supp. Table 9). (d) The tesseract-decoder Stim file of the schedule (CZ gates, SI1000), 12 rounds: the bound 10 of 2308.07915 "is exact" (Supp. Table 8, provenance P; repository instance `bb144_r12`, memory-Z, collapsed distance 9). (e) Gross-code chips at three noisy cycles: a perfect chip at 10 (Fig. 4c, SI 11). **No statement for every number of rounds at 10**: the every-round bound, the time collapse, is 8 / 9. | main text "The error-correction layer of IBM's planned Starling processor"; Methods; SI 7.2, 8.1, 9.1, 9.2; Supp. Tables 8, 9, 11 |
| Three-colour cycle (SBB Fig. 12) | "We confirm two conjectured circuit distances: 11 for a three-color schedule of the gross code"; Supp. Table 8: "11 (conjecture)" -> **11**, "conjectured; 11 in both bases", provenance R (rebuilt from the paper's text). The paper gives no number of rounds and no every-round statement for it. Repository: `papers/lrgross.py` rebuilds "arXiv:2603.05481v2, Appendix H Fig. 12" (the version paper 3 uses); `docs/notes/PAPERS.md` gives "Six-cycle memory experiments, both bases" 11 / 11; `bench/instances/lrgross/meta.json` (memory-Z, 6 rounds) records the collapsed (extended-code) distance as 11, from `results/papers3/lrgross/server_lr11_Z.json`, which is not in the repository. Not compared with our circuit. | main text, audit paragraph; Supp. Table 8; repository files named |
| Shaw-Terhal morphing circuits (2407.16336) and the 2026 follow-up (2604.09797) | Not treated: no "morph", "Shaw" or "Terhal" anywhere in `ncs.tex` or its bibliographies; the repository's `docs/AUDIT.md` sec. 5: "morphing circuits 2604.09797: not attempted". | -- |
| IBM-style [[72,12,6]] (Table I, last row) | 6, the published bound "is exact" (Supp. Table 8; SI 7.2), consistent with Tesseract's 6. | SI 7.2 |

**The audit of 26 papers** (SI 7, Supp. Table 8): mitten codes, IBM bivariate bicycle codes (2308.07915), the
distance-finding benchmark (2603.22532), Cornucopia, routing, shift-automorphism, Walking Cat, left-right circuits
(2603.05481), vine, amplified, tile, planar, multivariate multicycle, weight-8 BB, modular hyperbolic, balanced-product,
and, in "Further values confirmed", STAR, self-dual BB, GALA, pair-partition, clustered lifted-product, LLM-guided BB,
planar families. Cited by paper 3: `bravyi2024high`, `strikis2026high`, `webster2026distance`. Their values that paper 3
states (the gross-code bound 10, the three-colour conjecture 11; Tesseract's 6 and 8) are confirmed. The corrected
values that touch works paper 3 cites are for the two-gross [[288,12,18]] circuit (published <= 18 and best known 24,
certified 16), which paper 3 does not treat.

**Other close work found through it: dist-m4ri's benchmark.** `QEC-pages/dist-m4ri`, `benchmark/BENCHMARK.md`, Sec. 2.4
(read at master `bed621ea46`, 2026-10-15): `gross_144_12_12_si1000_r12`, memory-X: **10**, "certified by CC"
(connected-cluster enumeration, version 0.14.0, "179 s for gross"); memory-Z: not computed. The row first appears in
commit `205658e8ae` (2026-10-13). A search result without a proof object, on the tesseract SI1000 file of IBM's
schedule at 12 rounds. Not cited by paper 3 (no bib entry; outside this pass's permissions to add).

### Guancheng Pan's repository, main HEAD `2d5131b197f0c71555fde00ebb9311cc8d54d6b8` (2026-11-06T09:04:34Z)

`compare/41c8f978...2d5131b1`: 21 commits, 73 changed paths. Releases still only v1.0 and v2.0; the README calls main
"revision P", with a release and a Zenodo version 3.0 "in preparation".

- **E-045** (`research/audits/2026-11-04-E-045.md`, one commit `488c18392ce80d036668c993be54887db4fc2cf4`,
  2026-11-04T06:50:21Z, blob `e22383e5`; journal `research/journal/2026-11-04-E-045-crocker.md`), audit verdicts
  CONFIRMED: (A) the tesseract CZ / SI1000 gross file, r = 12, both bases: projected, a VeriPB 3.0.2 certificate of no
  undetectable logical of weight <= 9, an unconstrained exhaustive search agreeing, weight-10 witnesses (his and
  Crocker's) on the unprojected circuit; the projected DEM is column-isomorphic to his SD6 R = 12 model. (B) **his
  re-implementation of IBM's decoder_setup.py model (R noisy cycles followed by two noiseless ones): gross 10 and
  [[108,8,10]] 8 for every R >= 1, both bases**, by a written translation lemma, an exact periodicity check and a
  "connected compression" argument that reduce every R to the certified range R' = 11 (resp. 9) of E-037. Kind of
  certificate: the search results are E-037's VeriPB 3.0.2 proofs (R = 1..12 gross, 1..10 [[108,8,10]], both bases;
  a formula strengthened by minimality and orbit symmetry-breaking constraints whose soundness rests on a written
  argument the E-037 auditor proved); the every-R step itself is a written proof whose finite premises (a = w = 1,
  N_min <= 3, the periodicity criterion E_2) are checked by exact frame simulation -- no certificate for the step, no
  Lean or other formalization. The audit (B1-B4) re-proves the three steps and reproduces the premises with its own
  frame simulator. Scope: Pan's decoder_setup.py model only; for SD6 his every-R statement is still E-043's. (C) a
  monotonicity argument in R for a class of schedules. Revision P's manuscript (`release/arxiv/main.tex` at HEAD, lines
  904-914): "In our re-implementation of the single-fault model of IBM's decoder (R noisy cycles followed by two
  noiseless ones), the circuit distance is 10 for the gross code ... for every $R\ge1$ in both bases; a written
  time-translation argument, whose premises we check by exact computation, reduces arbitrary $R$ to the certified range
  $R\le11$".
- Fig. 12 (the three-colour cycle): unchanged, "two independent exhaustive searches give $\dext=11$; hence $\dcirc=11$"
  (HEAD `main.tex` 1055-1057): the lower bound 11 still has no certificate.
- **`search/experiments/paperP_audit/item1_fig12_drat.py`** (added in commit
  `ba0b75e557ee98638abc8bfb99010f493f1d0db2`, 2026-11-06T09:03:13Z, "Audit of revision P leftovers"; report
  `research/audits/2026-11-06-paperP-leftovers.md` sec. 1, verdict CONFIRMED 50/50). Read in full at HEAD: it takes
  every record whose name starts `fig12_retimings` from the release package's `certificates_proofs.jsonl` (25
  full-logical-family CNFs and 25 `len2only` CNFs), checks each CNF's sha256 against its record, re-solves it with
  CaDiCaL writing a DRAT proof, and checks the proof with drat-trim. What those CNFs encode (E-027, journal
  `research/journal/2026-10-14-E-027-certificates.md`, audit `research/audits/2026-10-15-E-027.md`, commit `9af2f4d2de`,
  2026-10-15; manuscript HEAD lines 1058-1059 "Its orders cannot reach 12 by retiming within several symmetric
  families (bulk statements, each certified by DRAT)"; claims map C16): every legal deterministic RE-TIMING of the
  circuit that keeps Fig. 12's CNOT orders, within the listed symmetric families and periods, has a bulk weight-11
  undetected logical, i.e. an UPPER bound of 11 "for sufficiently many rounds" on those retimed circuits. For Fig. 12
  itself the same lift encoding is the positive control: 4224/4224 instances SAT at R = 1 (weight-11 lifts exist,
  the upper-bound side). No R-indexed lower bound for Fig. 12 is certified by these proofs. **So the script does not
  touch paper 3's three-colour claim**, which concerns the lower bound 11 for the circuit of Fig. 12 at every number
  of rounds.
- Morphing: no changed path under E-017, E-019 or E-022.

### arXiv API, `submittedDate:[202611010000 TO 202611072359]`, sorted by submission date, run 2026-11-07

| Query (verbatim, AND-ed with the date range) | Hits | What matters |
|---|---|---|
| `all:"circuit distance"` | 1 | 2611.08247 |
| `all:"circuit-level distance"` | 0 | -- |
| `all:"circuit level distance"` | 0 | -- |
| `all:"bivariate bicycle"` | 5 | 2611.07144 (dithered BP decoding), 2611.07062 (logically augmented QLDPC codes), 2611.04179 (RL decoder scheduling hardware), 2611.03128 (BP-informed Tesseract decoding), 2611.02972 (routing for qLDPC on spin qubits): decoding or hardware, no circuit distance |
| `all:"gross code"` | 2 | 2611.08247; 2611.02972 (shuttling strategies for the gross code; logical error rates) |
| `all:"morphing circuit" OR all:"morphing circuits"` | 0 | -- |
| `all:"syndrome extraction" AND all:distance` | 0 | -- |
| `all:certificate AND all:distance AND cat:quant-ph` | 0 | -- |
| `all:certified AND all:distance AND cat:quant-ph` | 1 | 2611.08247 |
| `(all:SAT OR all:MaxSAT OR all:VeriPB OR all:"pseudo-Boolean") AND all:distance AND cat:quant-ph` | 0 | -- |
| `all:"detector error model"` | 1 | 2611.06605 (data-driven DEM augmentation for decoding) |
| `all:"fault distance"` | 1 | 2611.05867 (tracking errors in universal circuits) |
| `all:"two-block group algebra"` | 0 | -- |
| `all:"qLDPC" AND all:distance` | 2 | 2611.07144, 2611.02972 |
| `all:"Lean" AND all:"error correction"` | 2 | **2611.05047** (QECert: a Lean library for QEC theory: stabilizer and CSS codes, product constructions, BPT bounds; no circuit distance or certificate checker); 2611.03329 (agentic pipelines) |
| `abs:circuit AND abs:distance AND cat:quant-ph` | 8 | 2611.08247; the others decoding, sampling, magic-state verification (2611.04909, Comb: fault-tolerance verification of CNOT+T circuits), complexity |
| `all:"exact distance"` | 4 | none in quant-ph |
| `all:"minimum distance" AND cat:quant-ph` | 1 | 2611.05489 (stabiliser codes over prime fields as graphs) |
| `all:"hook error" OR all:"hook errors"` | 0 | -- |
| `all:"syndrome extraction circuit" OR all:"syndrome extraction circuits"` | 1 | 2611.08488 (provable circuit-level thresholds, non-abelian phases) |
| `all:"distance certificate" OR ... OR all:"certified distances"` | 1 | 2611.08247 |
| `all:"Shaw" AND all:"Terhal"` | 0 | -- |
| `all:"three-colour" OR all:"three-color" AND all:gross` | 0 | -- |
| `all:"logical error" AND all:"lower bound" AND cat:quant-ph` | 1 | 2611.08247 |
| `all:"tropical"` | 11 | 2611.08247; the rest other fields |
| `all:"dead check" OR all:"dead checks" OR all:"defective chips"` | 1 | 2611.08247 |
| `all:"branch-and-bound" AND cat:quant-ph` | 0 | -- |
| `all:"Stim" AND all:distance` | 0 | -- |

Raw Atom responses were kept in the session scratchpad, not in the repository.

### What changes in paper 3 (sentences revised 2026-11-07)

- **IBM every-T "first" withdrawn.** "the first lower bound for every number of rounds in which every search result is
  certified, among the works we found" is FALSE since Guancheng Pan's E-045 (B) (main, 2026-11-04): for his model of
  IBM's decoder_setup.py, an every-R lower bound of 10 whose search results are all VeriPB-certified (E-037), joined by
  a written argument with exactly checked premises -- the same shape as ours. 2611.08247 does not affect it (its
  every-round bound for this cycle is 8 / 9). The intro bullet now states what is specific (certificates re-derived by
  a separately written checker, Theorem 1, script-checked hypotheses) and no "first". The descriptions of Pan's
  every-R work now say "record E-043 ... under SD6", which is what that record covers; **E-045 itself is not yet cited:
  adding it needs a refs.bib entry, outside this pass's permissions.**
- **Three-colour "first" narrowed.** "a lower bound certified here for the first time among the works we found" is FALSE
  as worded: 2611.08247 certifies 11 (solver output plus replayed witness). Now: the first lower bound of 11 for it at
  every number of rounds that rests on certificates a separate checker re-derives. Survives: Guancheng Pan certifies
  d_ext >= 9 only; 2611.08247's paper states no every-round value, and its lower bounds carry no proof object.
  **Corrected the same day (caller's check of Fanerst/tropidist at b03ec2d4, 2026-11-07):** the record IS shipped:
  `results/papers3/lrgross/lr11_{Z,X}.json` and `server_lr11_{Z,X}.json` (six rounds) all give `collapsed.distance =
  11` and `certified_lower_bound = 11`, and their Methods (v1 text, lines 612-618 of the scratch copy) say the collapsed
  distance "is a lower bound for any number of rounds, as proved in Supplementary Section 5.4". So Cheng, Guo and Pan
  hold an every-round lower bound of 11 for the three-colour cycle in both bases, as solver output without a proof
  object. Paper 3 now says so in the abstract ("the second also for every number of rounds"), the intro and Sec. VI.B,
  and its "first" is only for a lower bound resting on certificates a separate checker re-derives, with Theorem 1.
- **Incomplete, now completed:** the abstract (both values certified by Cheng, Guo and F. Pan at fixed numbers of
  rounds; "G. Pan" and "F. Pan" disambiguated), the intro's account of gross-code-scale work, Table I's caption, Secs.
  VI.A and VI.B (results), and the related-work comparison of Sec. VIII (method, meaning of "certified", the audit,
  what remains specific).

### What survives for paper 3, as far as these searches reach

- The every-T lower bounds of 10 and 11 resting on certificates re-checked by a separately written checker under a
  Lean-verified calculus, with the window theorem's finite hypotheses checked by script (a "first" only for the
  three-colour cycle, see above).
- The first machine-checkable certificates for the two morphing values (2611.08247 does not treat morphing circuits;
  Guancheng Pan's E-017 / E-019 have none; nothing new on his branch).
- Exact values for the 14 circuits with the 2604.09797 parameters (no source computes any; 2611.08247 did not attempt
  2604.09797), and their 4 kernel-checked one-round distances.

### Recommended follow-ups outside this pass's permissions

- Cite Guancheng Pan's E-045 (commit `488c1839`, file `research/audits/2026-11-04-E-045.md`) in the intro and Sec.
  VI.A: "his record E-045 extends the certified range to every number of rounds for his model of IBM's
  decoder_setup.py (R noisy cycles followed by two noiseless ones), by a written time-translation and compression
  argument whose premises are checked by exact computation".
- Cite dist-m4ri's benchmark (memory-X, 10 at 12 rounds of the tesseract SI1000 file, connected-cluster enumeration,
  since 2026-10-13) beside Crocker.
- Consider QECert (arXiv:2611.05047) in the formal-verification sentence of the related work.
- Search scope and its limit: the arXiv queries above for 2026-11-01 to 2026-11-07, 2611.08247 and its repository,
  Guancheng Pan's main branch at `2d5131b1`, and dist-m4ri's benchmark page; nothing else was re-searched on this date.
