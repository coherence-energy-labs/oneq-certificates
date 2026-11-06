# Paper 4 (superactivation at finite blocklength): literature search log

**Evidence class:** [OBSERVED] what searches on the stated dates returned. Not finding a result is not proof that it
does not exist; the novelty wording in the manuscript is held to this record.

Earlier searches: 2026-10-13 and 2026-10-14 (ledger R140, R142), which found no certified onset below the published
n* <= 17 and no unassisted two-use result.

## 2026-11-03

| Query (verbatim) | Engine | What it returned that matters |
|---|---|---|
| `superactivation quantum capacity finite number of channel uses 2026 arXiv` | web | arXiv:2604.27042 (Parentin, Bergh, Datta, Wilde; v2 2026-05-04): n* <= 17 for the Smith–Yard pair. arXiv:2606.10744 (NMR; superposed depolarizing channels, a different mechanism). arXiv:1408.5115 (Cubitt et al.; unbounded number of uses may be needed). |
| `"superactivation" quantum capacity "two uses" OR "two channel uses" PPT erasure Smith Yard` | web | Smith–Yard 2008; Cubitt et al. (Nat. Commun. 6, 6739); arXiv:1107.4385; arXiv:2609.20133 (private capacity, not quantum capacity). No two-use quantum-capacity result. |
| `Parentin Bergh Datta Wilde superactivation finite blocklength` | web | arXiv:2604.27042 and its companion arXiv:2604.27040 (permutation-invariant optimization); no later version or follow-up by the authors. |
| `"Onset of superactivation" quantum capacity cites OR follow-up 2026 n* uses shield dimension` | web | No citing paper found. |
| `arXiv 2026 superactivation quantum capacity certified fidelity channel uses PPT shield dimension Horodecki channel finite` | web | arXiv:2607.24693 (Zhu, Wang; v3 2026-10-05 "added quantum superactivation results"), arXiv:2609.10520 (private capacity), arXiv:2610.03706 (private capacity, channel orders). |
| `all:superactivation`, newest first, 20 results | arXiv API | Newest quantum-capacity entries: 2607.24693v3, 2606.10744, 2604.27042, 2604.27040. The others concern nonlocality, multipartite entanglement, private capacity, incompatibility or unrelated fields. |
| `abs:"quantum capacity" AND abs:"channel uses"`, newest first, 15 results | arXiv API | arXiv:2610.02176 (Khanian, Leung, Smith, 2026-10-01: permutation-invariant inputs reach the single-use coherent information at finite block length; no superactivation result), 2608.01308, 2609.16608, 2610.13854 (converses and bosonic channels). |

**Read in full or at the abstract:**
- **arXiv:2609.20133** (Uthirakalyani, Halder, Elkouss, 2026-09-17): two Pauli channels with zero PRIVATE capacity
  activate private communication with a simple binary code. This concerns private capacity, not quantum capacity.
- **arXiv:2610.02176** (Khanian, Leung, Smith): no superactivation result. "finite-block superadditivity remains
  possible, while the asymptotic optimization reduces to a single channel use."
- **arXiv:2607.24693v3** (Zhu, Wang): a fixed qutrit channel superactivates every finite-dimensional channel with zero
  quantum but positive private capacity, P(M) > 0 implies Q(M (x) Lambda) > 0. This is the ASYMPTOTIC quantum capacity:
  "some finite block has positive coherent information", the proof "is also existential", and "the required
  private-code block and auxiliary dimension can be large". No number of uses is given, and the main text does not
  cite arXiv:2604.27042.

**What survives for paper 4:**
- No certified onset below n* = 17 for the Smith–Yard pair was found.
- No unassisted superactivation of quantum capacity in two uses, in the fidelity-threshold sense of arXiv:2604.27042,
  was found.
- No onset-versus-shield-dimension result was found.
- "Symmetric codes are not optimal" is NOT new: arXiv:2604.27042 Fig. 2 shows it (ledger R140, A10).

## 2026-11-03, addendum (the referee's finding F12)

Records read (arXiv abstract pages and Crossref, fetched 2026-11-03; no new web search):

| Query (verbatim) | Engine | What it returned that matters |
|---|---|---|
| `au:Smolin AND au:Yard AND au:Smith` | arXiv API | arXiv:1102.4580 (Smith, Smolin, Yard), "Gaussian bosonic synergy: quantum communication via realistic channels of zero quantum capacity"; published as "Quantum communication with Gaussian channels of zero quantum capacity", Nature Photonics 5, 624-627 (2011), doi 10.1038/nphoton.2011.203. |
| `ti:Gaussian AND ti:zero AND ti:capacity AND au:Smith` | arXiv API | The same record. (`ti:"Gaussian channels of zero quantum capacity"` returned nothing: the arXiv title differs from the journal's.) |

**Read at the abstract:**
- **arXiv:2510.04527** (Wu, Wang, 2025-10-06), "Quantum capacity amplification via privacy": superadditivity of quantum
  capacity through private channels whose Choi operators are private states; a sufficient condition for capacity
  amplification in terms of the assisting channel's Holevo information, with thresholds for erasure and depolarizing
  assisting channels; under the spin-alignment conjecture, a single-letter capacity for a family of private channels;
  and, for approximate private channels, "an alternative proof of superactivation that extends its validity to a
  broader parameter regime". All statements concern capacities (the asymptotic regime); no number of uses is given.
- **arXiv:1107.4385** (Brandao, Oppenheim, Strelchuk; Phys. Rev. Lett. 108, 040501 (2012), "When Does Noise Increase
  the Quantum Capacity?"): superactivation "exists for a wide class of inequivalent channels, none of which can
  simulate each other", including an example with the depolarizing channel. Asymptotic capacities; no number of uses.
- **arXiv:1102.4580** (Smith, Smolin, Yard): superactivation for optical channels with attenuation and Gaussian noise.
  Asymptotic capacities; no number of uses.

**What survives:** unchanged. None of the three gives a finite number of uses or a fidelity-threshold statement.

## 2026-11-03, provenance audit (the final referee's MAJOR 1)

The referee named the origin of the family P_d; this section records how every construction the paper uses was traced
to its source. No keyword search: the records were fetched by identifier and their LaTeX sources read.

| Query (verbatim) | Engine | What it returned that matters |
|---|---|---|
| `id_list=1309.7992,quant-ph/0506203,2510.04527,2609.20133` | arXiv API | quant-ph/0506203v1, K. Horodecki, Pankowski, M. Horodecki, P. Horodecki, "Low dimensional bound entanglement with one-way distillable cryptographic key", journal_ref IEEE Trans. Inf. Theory 54, 2621 (2008), doi 10.1109/TIT.2008.921709 (HPHH). 1309.7992v2, Badziag, K. Horodecki, M. Horodecki, Jenkinson, Szarek, "Bound entangled states with extremal properties", Phys. Rev. A 90, 012301 (2014). 2609.20133: first author "Uthirakalyani G". |
| `id_list=0807.4935,quant-ph/0607039,2604.27040` | arXiv API | Smith, Yard (Science 321, 1812); Smith, Smolin, Winter (IEEE TIT 54, 4208); Bergh, Parentin v2 (2026-05-04). |
| `works/10.1109/TIT.2008.921709`, `works/10.1103/PhysRevA.90.012301` | Crossref | HPHH: vol. 54, issue 6, pp. 2621-2625, 2008. Badziag et al.: vol. 90, issue 1, art. 012301, 2014; Crossref's family name for the first author is mis-encoded ("Badzia" + U+00B8 + "g"), see below. |
| e-print sources of all seven, read with `grep` for `Observation`, `unimodular`, `u_{ij}`, `sqrt{d_s}`, `Smith2008`, `horodecki2008low`, `mathfrak{S}_k`, `subgroup`, `0.75013` | arxiv.org/e-print | what follows |

**Read in the sources:**
- **HPHH (quant-ph/0506203, the `Construction of small dimensional bound entangled states` paragraph).** A mixture of
  two private bits in X-form with X1 = W_U / ||W_U||, W_U = sum_ij u_ij |ij><ji| for a unitary U, X2 = W_U^Gamma / d,
  weights p1 = ||W_U|| / (||W_U|| + d), p2 = d / (||W_U|| + d); "rho with parameters defined like above is PPT
  invariant" for any unitary U. Observation 1: the ratio p1/p2 is largest, sqrt d, for unimodular U (|u_ij| =
  1/sqrt d), i.e. p2 = 1/(1 + sqrt d). Observation 2: rho_H (d = 2, U = H) is extremal in the PPT set; "any mixture
  with the weight of gamma_1 different then value p_1 ... leads to NPT state. In fact the same argument proves
  extremity ... if only X is hermitian operator". Smith and Yard's four-dimensional channel is cited to this paper
  ("from \cite{HPHH05S}" in 0807.4935).
- **Badziag et al. (1309.7992, Eq. `eq:state` and App. A `sec:private_states`).** rho = (1 - p) gamma(X) + p
  sigma_x gamma(Y) sigma_x, X = (d_s sqrt d_s)^-1 sum u_ij |ij><ji|, Y = sqrt d_s X^Gamma = d_s^-1 sum u_ij |ii><jj|,
  "U ... (arbitrary) unitary ... with |u_ij| = 1/sqrt d_s", "for definiteness ... quantum Fourier transform",
  p = 1/(1 + sqrt d_s); "(i) it is PPT, since by construction it is invariant under the partial transpose";
  "the construction itself was provided in \cite{smallkey}" (= HPHH).
- **Our Eq. (8) against HPHH, numerically** (`tests/test_superactivation_family_origin.py`): equal (to the 1e-9
  noise of eigh square roots) with U = conj(u)/sqrt d for the symmetric twists tried (Hadamard d = 2, Fourier
  d = 3, 4, Sylvester d = 4); for a non-symmetric complex Hadamard (a row-permuted, column-rephased Fourier matrix,
  d = 3) the two terms carry X1 from u^dagger and X2 from conj(u), and no single U among u, conj u, u^T, u^dagger
  reproduces it (differences > 1e-2). So for symmetric u, P_d is HPHH's state; for non-symmetric u it is a variant.
  [Too strong: that very u is locally equivalent to HPHH's state. Corrected in the next section.]
- **PBDW (2604.27042v2).** The remark (`\begin{comment}` in supplemental.tex, near Lemma `choi-as-mixture`) places the
  channel in "a broader family ... a d_s x d_s unitary u with unimodular entries"; they cite HPHH (`horodecki2008low`)
  for private Horodecki channels. The text prints the published code's fidelity as "0.75013 > 3/4"
  (onset_superactivation.tex l. 159, l. 171; supplemental.tex l. 2351); the ancillary `fidelity_verified.npy` holds
  0.750131163826669. Their pre- and post-processing "is inspired by the argument in \cite{Smith2008}". The decoder for
  k flagged uses is covariant under G_k = S_k x S_{n-k} (supplement, "Exploiting the Classical-Quantum Output
  Structure").
- **Smith and Yard (0807.4935).** Their Theorem sends a purification of the PPT channel's input ensemble through a 50%
  erasure channel whose input dimension is the sum of the ensemble's ranks: the coupling of our Lemma 4's encoder.
- **Bergh and Parentin (2604.27040v2, Sec. IV, remark `rem:general_group_invariance`).** The seesaw "extends to any
  finite group G"; footnote: for N_1^{(x)k} (x) N_2^{(x)n-k} "one can restrict to encoders and decoders that are
  invariant under ... G = S_k x S_{n-k}. A similar case arises for the channels involved in superactivation".

**What is prior, and what the manuscript now says:**
- P_d and its PPT point: HPHH (for symmetric u exactly; non-symmetric u is a variant). The "only if": HPHH for U = H
  and asserted for Hermitian W_U; new here for non-Hermitian twists (Fourier, d >= 3) and for the non-symmetric
  variant [the last clause withdrawn in the next section]. Theorem 3 is titled accordingly; the abstract, Intro item 2, Table I and the "Origin" paragraph credit HPHH
  and Badziag et al. The complex-Hadamard condition is the published construction's |u_ij| = 1/sqrt d, not a
  correction of ours; only the commented-out remark in PBDW's source says "unimodular".
- The coupling in Lemma 4: Smith and Yard; its decoder and the d = 2 statement: PBDW Prop. 1 (already credited).
- The flagged channel N~: PBDW Prop. 1 (already credited).
- The two-group form: the restriction to a product of symmetric groups is suggested by Bergh and Parentin for these
  channels, and PBDW's decoders are S_k x S_{n-k} covariant; the G = 1 reduction is the symmetric-subspace case of the
  permutation-invariant evaluation of PBDW and Bergh-Parentin. New: a certified two-group code, and Lemma 2 for G
  groups.

**A tool defect found here:** `paper/tools/verify_refs.py`'s `norm_text` applies NFKD, which turns a SPACING diacritic
(Unicode category Sk: U+00B8 cedilla, U+02DB ogonek, ...) into a space plus a combining mark, so Crossref's mis-encoded
"Badzia" + U+00B8 + "g" normalizes to "badzia g" and the correct bib name "Badzi{\k{a}}g" is reported as a mismatch.
The root fix is one line in that shared tool (drop category-Sk characters before NFKD); it is outside paper 4's
files, so it is reported, not applied. [Applied by the coordinator in 3959d729; `verify_refs.py` now agrees on all
99 entries.]

## 2026-11-03, corrections after the final referee's re-check

No new search; the same sources, re-read.

- **The non-symmetric twist is not a new state (withdrawn claim).** For u = the d = 3 Fourier matrix with rows
  cyclically permuted and columns rephased (the test's twist), Q_ij = conj(u_ij)/conj(u_ji) = r_i/r_j (factorization
  error 7e-16, our check), and the product of diag(sqrt r_i) on R'' and its conjugate on B'' -- one unitary per party,
  independent of the key -- maps Eq. (8) onto HPHH's state with U = conj(u)/sqrt 3 (difference 4.6e-10 after, 6.9e-2
  before; the conjugate map is a control at 5.3e-2). It holds in general by inspection: conjugation by D (x) conj(D)
  multiplies the (ij) entry of W_{u^dagger} by r_i/r_j and fixes W_{conj u}^Gamma, which lives on span{|ii>}. So when
  conj(u)^T is diagonally similar to conj(u), the "if" direction is HPHH's; whether every non-symmetric u is locally
  equivalent to an HPHH state is not settled, and the manuscript claims nothing for it beyond the "only if" for every
  twist. The test `test_that_non_symmetric_twist_is_locally_equivalent_to_the_hphh_state` holds this.
- **HPHH, verbatim (quant-ph/0506203, the construction paragraph and Observation 2):** "We get immediately that rho
  with parameters defined like above is PPT invariant."; on rho_H: "It is straightforward to check, that any mixture
  with the weight of gamma_1 different then value p_1 = sqrt 2/(1 + sqrt 2) leads to NPT state. In fact the same
  argument proves extremity of the state from our construction with X = W_U, if only X is hermitian operator, and
  either X or X^Gamma has some positive eigenvalue." Their gamma_1 and p_1 are our gamma^0 and p_0. The extension is
  stated as extremality, under the two conditions quoted; the manuscript now quotes it so.
- **Bergh and Parentin's remark, exact location.** arXiv:2604.27040v2, compiled PDF p. 36: Sec. IV ("Symmetric Seesaw
  Iteration Method"), IV.C ("Symmetric Seesaw Method"), sub-subsection "Permutation-Invariant Codes"; in the LaTeX,
  Chapters/chapt4_Seesaw.tex, `\begin{remark}\label{rem:general_group_invariance}`. Verbatim: "Remark. Lemma IV.9
  extends to any finite group G: if N_n is G-covariant and the seeds are G-invariant, then the entire iteration can be
  restricted to the G-invariant subspace, giving rise to a G-invariant seesaw method", footnote 3: "For example, if
  N_n = N_1^{(x)k} (x) N_2^{(x)n-k} for some k in [n], then one can restrict to encoders and decoders that are
  invariant under the action of the subgroup G = S_k x S_{n-k} <= S_n. A similar case arises for the channels involved
  in superactivation [1, Section 5.2]." (The arXiv HTML rendering ends inside Sec. III.4, which is why the referee could
  not find it there.) The footnote concerns a channel covariant only under the subgroup; N~^{(x)n} is covariant under
  all of S_n, so the manuscript now says that two-group encoders are a choice of code family, not a symmetry of the
  channel, and quotes the footnote.
