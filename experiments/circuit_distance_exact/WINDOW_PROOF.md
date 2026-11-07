# The window lemma for every T: the IBM depth-8 cycle and the 3-colour cycle (R198 lane convention)

**Scope.** The two non-morphing gross-code memory experiments of the R198 lane, `bb_circuits.ibm_cycle` (Bravyi et
al. 2024) and `bb_circuits.three_colour_cycle` (arXiv 2603.05481, Fig. 12), memory-Z and memory-X, in the lane's
convention: noisy ancilla resets and measurements, noisy readout (and for the 3-colour cycle, a noisy data
preparation), the lane's detectors (no meta-checks), standard depolarizing circuit noise, the same-type projection.

**Result.** Four finite, T-independent hypotheses on the blocks of the circuit (section 2) give, for EVERY T,

- an exact formula for the T-round model (Theorem 1): `M_T = S u U_{i=j_s}^{T-1-U} tau_i(I) u tau_T(End)` -- a fixed
  start part, T - j_s - U time-shifted copies of one interior profile set, and a fixed end part;
- the window structure the lane's rules W and S need (Corollary 1): every mechanism spans at most two consecutive
  detector rounds (P1); the windows of start rounds 1..T-2 are one and the same shape set (P2); the windows of start
  rounds 0, T-1 and T do not depend on T (P3) -- for every consecutive pair (T, T+1), except that in memory-Z the
  pair (1, 2) satisfies only the time-shift form (P3_final: the 1-round model's last round IS its start round;
  rule S needs no more -- section 6, note 3);
- determinism of every detector and observable for every T (Corollary 2);
- the lane's every-T claims (Corollary 3): **IBM cycle d_circ = 10 for every T >= 2; 3-colour cycle d_circ = 11 for
  every T >= 2**, each from the lane's finite premises (section 4.4 says exactly which).

The circuits are NOT `prep . body^T . readout` in general: the 3-colour cycle is pipelined (a round's X checks finish
during the next round's Z checks), so its template is `C_T = P . F . B^(T-1) . E . R` with a first block F (no
previous round to finish) and a tail E (the last round's X checks finishing). The IBM cycle is the special case
F = B, E empty.

**Evidence class.**

- Theorem 1, Corollaries 1-2, Lemma W: **PROVEN** by hand in this file; not machine-verified; survived one independent
  adversarial audit (2026-10-22, section 8), which refuted nothing.
- The hypotheses for both circuits, both bases: **MACHINE-CHECKED** by `window_uniform.py`, every one with a negative
  control that fails (section 5).
- The formula itself is also checked against stim's DEM **alone** at T = 16, 24, 32 (projected) and 16, 24 (full),
  beyond every T the lane computed: equal in all 20 comparisons; both formula controls disagree.
- Corollary 3: IBM -- **PROVEN from MACHINE-CHECKED premises** (the 22 time-shift certificates of
  `ibm_recert_20261026/`, T = 1..11, both bases, every proof shipped) plus Appendix A's code reading. 3-colour -- the every-T step is **PROVEN**, and its finite premises, the lane's
  time-shift sweep for T = 1..12, are **MACHINE-CHECKED**: 24/24 time-shift certificates re-proved by the committed
  prover and VERIFIED by the committed, Lean-modelled checker in the 2026-11-02 re-certification
  (`recert_20261102/manifest.json`; the original records, `records/3c_shift_certificates_w10.jsonl`, were checked by
  the checker of their day, some before the Lean-prompted fixes), so 3-colour d_circ = 11 for every T >= 2 is
  likewise **PROVEN from MACHINE-CHECKED premises**.

The lane's phrase "MACHINE-CHECKED for T <= 11 (resp. 12), DERIVED beyond" -- whose DERIVED step rested on "the
builder's time-periodicity" -- is replaced by Theorem 1 for both circuits.

---

## 1. Setting

### 1.1 Frames, the frame rule (F) and the gauge rule (G)
Qubits Q: data 0..143 (the left and right blocks of the gross code), X ancilla of check k at 144 + k, Z ancilla of
check k at 216 + k (both builders). Paulis are taken modulo phase, `P = X^x Z^z`, `(x, z) in V := F_2^{2|Q|}`
(dimension 576).

**(F) frame rule.** In a circuit of Clifford unitaries (H, CX), single-qubit resets (R, RX) and single-qubit
measurements (M, MX), a Pauli A inserted at a location flips the outcome of a later measurement of `B_q` iff its
*frame* -- A conjugated through every later unitary, its q-part deleted at every reset of q, unchanged by
measurements -- anticommutes with `B_q` just before that measurement. A parity that is deterministic without noise
is flipped iff an odd number of its outcomes are. (Standard; what stim's error analysis and the lane's
`dem_independent.py` compute.)

**(G) gauge rule.** Insert, right after every reset and every measurement, the Pauli of its own basis (Z after R / M,
X after RX / MX), and Z on every qubit at the start (the initial |0>): call these the *gauges*. The noiseless outcome
vector is uniform on `m_ref + span{gauge flip vectors}` (the Pauli-frame simulation of stabilizer circuits: a gauge
does not change the state, and the gauges generate all randomness). Hence a parity is deterministic iff no gauge
flips it. (Cited, not re-proved; corroborated in section 5: on both circuits, both bases, T = 1 and 2, the GF(2) rank
of the gauge flip vectors equals stim's tableau count of random outcomes, 132 = 132, and dropping the preparation's
gauges breaks the equality, 66 != 132.)

### 1.2 The template
A *block* is a finite instruction list. The template is

    C_T  :=  P . F . B . B ... B . E . R         (T - 1 copies of B),      positions -1, 0, 1..T-1, T, T+1.

| circuit | P | F (position 0) | B (positions 1..T-1) | E (position T) | R (position T+1) |
|---|---|---|---|---|---|
| IBM cycle | R on all qubits (+ H on data, memory-X); no noise | = B | one round: 9 TICK-segments (ancilla reset + noise, H + noise; 7 CX layers + DEPOLARIZE2 + idle DEPOLARIZE1; H + noise, flips, M) | empty | (H), X_ERROR on data, M on data |
| 3-colour cycle | R on data (+ H), data flip noise | time steps 1..8 of round 0 only | time steps 8j+1..8j+8: round j's steps 1..8 and round j-1's steps 9..16 | time steps 8T+1..8T+5: round T-1's steps 9..13 | (H), X_ERROR on data, M on data |

`window_uniform.py` reads P, F, E, R off C_1 and B off C_2 (checking that P, F, E, R read off C_2 agree), and
checks the builder's instruction stream against the template at T = 1..9, 16, 24, 32 (Appendix A proves it for every
T by reading the code).

### 1.3 Labels and the closed-form detectors
**LBL** (a finite condition, section 2): P measures nothing; B measures every ancilla exactly once; F measures a
subset of the ancillas once each; E measures exactly the other ancillas once each; R measures every data qubit once.
Set `delta_a = 0` if F measures ancilla a, else `delta_a = -1`. Then each ancilla is measured exactly T times in C_T:
the measurement of a in position j (1 <= j <= T-1) is a's round `j + delta_a` measurement, F's are round 0, E's are
round T-1. Write `m_s(a)` (s = 0..T-1) and `d(q)` for the readout of data qubit q. (IBM: delta = 0 for all 144
ancillas. 3-colour: delta = 0 for the 48 Z checks of colours 0, 1; delta = -1 for the 24 colour-2 Z checks and all 72
X checks.)

The detectors and observables of C_T, with `S^b_k` the k-th row of `H_b` and `L_j` the memory logicals
(`BB.logicals`, as the builders take them), b the memory type:

    D(k, t, s) = m_s(a(t,k)) + m_{s-1}(a(t,k))          1 <= s <= T-1, both types t     coordinates (k, t, s)
    D(k, b, 0) = m_0(a(b,k))                              memory type only
    D(k, b, T) = XOR_{q in S^b_k} d(q) + m_{T-1}(a(b,k))  memory type only
    Obs_j      = XOR_{q in L_j} d(q)

### 1.4 Block maps
For a block X with `m_X` measurements: `A_X : V -> V` (a frame entering X -> its frame at X's end) and
`C_X : V -> F_2^{m_X}` (the flips of X's measurements), both linear (F). Every noise component kappa of X (a location
and a Pauli of its channel's component set) has a *direct effect* `(x_kappa, y_kappa)`: its frame at X's end and the
flips of X's measurements after it. Gauges likewise. Write `A, C` for `A_B, C_B`. `window_uniform.py` computes all of
these by one backward pass per block (`wu_core.Block`), an implementation independent of stim's analysis.

### 1.5 Models and windows
`M_T` (full) is the set of distinct nonzero symptoms (detectors, observables) of the components of C_T; `Proj_T` keeps
the memory-type detectors (what `export_model.project_same_type` makes of stim's DEM). `tau_i` translates detector
coordinates by i rounds. A mechanism's *start round* is its smallest detector round; its *shape* is its symptom
written relative to the start round; `win_T[r]` is the set of shapes starting at r. P1-P3 are the lane's
(`tools_cde.window_preconditions`, drift 0): P1 every mechanism spans <= 2 consecutive rounds; P2 `win[r]` equal for
all 1 <= r <= T-2 in both models of a pair; P3 `win_T[0] = win_{T+1}[0]`, `win_T[T-1] = win_{T+1}[T]`,
`win_T[T] = win_{T+1}[T+1]`; `time_shift_ok` = P1, P2, P3_final and (T = 1 or P3_last).

## 2. The hypotheses (finite, T-independent)

| name | statement | where it is read | negative control (must fail; does) |
|---|---|---|---|
| **TPL** | the builder emits C_T with the closed-form detectors / observables | Appendix A (every T); mechanically at T = 1..9, 16, 24, 32 | one CNOT removed in block 3; a detector missing one record; an observable missing one record |
| **LBL** | section 1.3 | P, F, B, E, R | E (IBM: B) loses one measurement |
| **FIR(h)** | `C (A + I) A^h = 0` (rows: memory-type checks for the projected model, all checks for the full model) | B | a CNOT between two data qubits inside B (fails for every h <= 8); for the 3-colour cycle also h - 1 = 0 (fails: 240 / 384 basis vectors memory-Z, 144 / 384 memory-X) |
| **END(e)** | `N A^e = 0` and `N_obs (A + I) A^e = 0`, where for a frame x' entering the LAST B, `N(x')` lists the detectors that read E / R records or close the last round (`D(k,.,T-1)` for delta = -1 checks: `(C_E A x')_a + (C x')_a`; `D(k,b,T)`: `S^b_k . C_R A_E A x' + m_{T-1}`-flip) and `N_obs(x') = (L_j . C_R A_E A x')_j` | B, E, R | a CNOT between two data qubits in R before the readout (fails for every e <= 8); for the 3-colour cycle also e - 1 = 0 (fails) |
| **DET** | every gauge of C_T flips nothing: the gauge sets S_g, I_g, End_g of Theorem 1 are empty, and no gauge of C_T flips anything for T < T_0 | P, F, B, E, R | an other-type round-0 detector added (a gauge flips it) |

Measured values: IBM h = e = 0 (U = 1); 3-colour h = e = 1 (U = 2), both bases, both row sets.

Each hypothesis is a statement about the five blocks; none mentions T. FIR and END are single matrix identities on
the 576-dimensional frame space: `C (A+I) A^t = C (A+I) A^h A^(t-h)` and `N A^t = N A^e A^(t-e)`, so they hold for
every t >= h (resp. e) once they hold at h (resp. e).

**Not hypotheses** (the proof does not use them): a pure-part (PURE) condition -- `d_circ = d_proj` falls out of
Corollary 3 instead; idle-noise uniformity; any property of the code beyond the closed-form detectors.

## 3. Lemmas

Throughout, `U := max(h, e) + 1`; `j_s := 1` for the projected model, and for the full model `j_s := 2` if some
OTHER-type ancilla has delta = -1 (the 3-colour cycle, both bases), else 1 (IBM); `T_0 := j_s + U`.

**Lemma 1 (block decomposition).** Let kappa sit in position p of C_T with direct effect `(x, y)`. The records it
flips are those flipped by y (position p's own measurements) and, for every later position p' <= T+1, those flipped
by `C_{X_p'} x^(p')`, where `x^(p+1) = x` and `x^(p'+1) = A_{X_p'} x^(p')`. *Proof.* (F): the frame at the end of
position p is x, and the frame rule composes block by block. ∎

**Lemma 2 (suffix covariance).** Let kappa be a component of B at position i, `j_s <= i <= T-1`, or a component of
E or R. Then its symptom in C_T is `tau_i(Psi_kappa(T-i))` (resp. `tau_T(Psi_kappa)`), where `Psi_kappa(u)` is the
symptom computed in the abstract suffix `B^u . E . R` (kappa in its first B) with round labels relative to its start.
The same holds for a frame entering position i. *Proof.* By Lemma 1 the flipped records depend only on the blocks
from position i on, which are `B^(T-i) . E . R` in every C_T. By LBL their labels are the relative labels plus i
(ancilla rounds) or unchanged (data). A flipped record has round >= i - 1. The detectors containing it are, by the
closed form, the translates by i of the relative ones: the difference detectors `D(k,t,s)`, s >= 1, always; the
memory-type `D(k,b,0) = m_0` behaves as `m_0 + m_{-1}` with m_{-1} never flipped (projected model, i >= 1); the final
detectors `D(k,b,T)` translate with T. In the full model the other type has no round-0 detector, so a flipped
round-0 other-type record breaks covariance; that needs i - 1 = 0 and delta = -1, excluded by i >= j_s. ∎

**Lemma 3 (stabilization).** Assume FIR(h), END(e). For a frame x entering position 0 of `B^u . E . R` let `rho(x, u)`
be its relative symptom. Then `rho(x, u) = rho(x, U)` for every u >= U. *Proof.* The output stream is
`y_t = C A^t x` (t = 0..u-1), `y_E = C_E A^u x`, `z = C_R A_E A^u x`. Split the detectors:
(a) those whose records lie in B positions -1..u-1 -- the head (records at relative positions -1, 0: flip `(C x)_a`)
and the differences of consecutive B positions, flip `(C (A+I) A^t x)_a` at a relative round fixed by t and delta_a,
t = 0..u-2; by FIR these vanish for t >= h, so for u >= h + 1 the set of nonzero ones is the same for every u;
(b) the terminal ones, `N(A^(u-1) x)`, which vanish for u - 1 >= e by END; (c) the observables `N_obs(A^(u-1) x)`,
equal to `N_obs(A^e x)` for every u - 1 >= e by END. (For delta = 0 the detector closing round u-1 is a difference in
(a); for delta = -1 it is terminal, in (b).) None of the surviving relative rounds depends on u. ∎

**Lemma 4 (the start part).** A component at a position p < j_s (in P, F, or for the full 3-colour model B at position
1), or an initial-state gauge, has the same symptom in every C_T with T >= T_0. *Proof.* Its records before position
j_s and their detectors are fixed (rounds <= j_s < T, never the final layer); what it flips from position j_s on is,
by Lemma 2, `tau_{j_s}(rho(x', T - j_s))` for its frame x' entering position j_s, and `T - j_s >= U` makes this
`tau_{j_s}(rho(x', U))` by Lemma 3. ∎

## 4. Theorem 1 and its corollaries

**Theorem 1 (window formula).** Assume TPL, LBL, FIR(h), END(e). For every T >= T_0, as sets of nonzero symptoms
(memory-type detectors only, for the projected model),

    M_T  =  S  u  U_{i = j_s}^{T-1-U} tau_i(I)  u  tau_T(End)

where S = the symptoms of the components at positions < j_s (Lemma 4; T-independent), I = `{pi_kappa : kappa in B}`
with `pi_kappa := D_rel(y_kappa) + tau_1 rho(x_kappa, U)` (D_rel maps a body block's own records to their relative
detectors), and End = the symptoms, relative to T, of the components of B at positions T-U..T-1 and of E and R.

*Proof.* Every component of C_T is in exactly one of: positions < j_s (S, Lemma 4); B at a position i with
`j_s <= i <= T-1-U`; B at `T-U <= i <= T-1`, or E, or R. In the second case Lemma 2 gives `tau_i(Psi_kappa(T-i))`;
kappa's own records map by D_rel (i <= T-2, so its rounds i-1, i, i+1 are difference or round-0 detectors), and its
frame enters position i+1 with `T-i-1 >= U` blocks left, so Lemma 3 gives `Psi_kappa(T-i) = pi_kappa`. In the third
case Lemma 2 gives `tau_T` of a set that depends only on `T - i in [1, U]` (or on nothing, for E and R), which needs
`i >= j_s`, i.e. `T >= j_s + U = T_0`. ∎

The sets are finite and computed once from `C_{T_ref}` (`T_ref = T_0 + 3`, interior position `i_0 = j_s + 1`).
Measured sizes (S / I / End): IBM projected 864 / 864 / 864 (both bases); IBM full 4248 / 6336 / 4896 (Z),
4896 / 6336 / 4320 (X); 3-colour projected 768 / 960 / 1776 (Z), 624 / 936 / 1896 (X); 3-colour full
7248 / 6576 / 13104 (Z), 9696 / 6576 / 10656 (X). Recomputing them from `C_{T_ref+1}` at `i_0 + 1` gives the same
sets (all eight).

**Corollary 1 (windows, every T).** Let rho_S be the last round an S-mechanism touches, `o_min, o_max` the start
offsets of I relative to i, and rho_E the largest `T - start` over End. Put `R1 = max(rho_S + 1, j_s + o_max)`,
`R2 = max(rho_E + 1, U + 1 - o_min)`, `T_1 = max(T_0, R1 + R2, R1 - o_min + U, j_s + o_max + R2 - 1) + 1`. For every
T >= T_1: `win_T[r]` is a fixed set Start_r for r < R1, the interior shape set for R1 <= r <= T - R2, and a fixed set
End_{T-r} for r > T - R2. *Proof.* Read off Theorem 1: a start round r < R1 receives S-mechanisms and translates
`tau_i(I)` with `i = r - offset >= j_s`, all T-independent once no End-mechanism reaches r and every needed i is
<= T-1-U (the conditions in T_1); symmetrically at the end; in between every profile of I appears exactly with its
own shape. ∎

Checked at T_1 (projected): P1 holds for every mechanism; the windows at r = 1..R1-1 and at r = T - v, v = 2..R2-1,
equal the interior set. So for every T >= T_1, P1 holds and only the windows 0, T-1, T differ from the interior one,
and those three are T-independent: **P1-P3 hold for every pair (T, T+1) with T >= T_1**. The pairs below T_1 are
checked on the exact models (block iteration, = stim at T <= 3 and = the formula from T_0 on):

| | T_0 | (R1, R2) | T_1 | interior / start / last / final shapes | exact pairs (T, T+1), T = 1..T_1 |
|---|---|---|---|---|---|
| IBM memory-Z | 2 | (2, 2) | 5 | 720 / 648 / 720 / 216 | T=1 time-shift only (P3_last: the 1-round model's last round is its start round); T=2..5 all P1-P3 |
| IBM memory-X | 2 | (2, 2) | 5 | 720 / 720 / 720 / 144 | T=1..5 all P1-P3 |
| 3-colour memory-Z | 3 | (2, 4) | 7 | 720 / 504 / 720 / 360 | T=1 time-shift only (as recorded in `records/3c_window.jsonl`); T=2..7 all P1-P3 |
| 3-colour memory-X | 3 | (2, 4) | 7 | 720 / 720 / 720 / 144 | T=1..7 all P1-P3 |

(The full models need not satisfy P2 -- their window at r = 1 differs, because the other type has no round-0
detector -- and nothing below uses it.)

**Corollary 2 (determinism, every T).** Theorem 1's proof applies to any located Paulis, in particular the gauges
(the initial-state gauges sit in S). DET says their S, I, End are empty and no gauge flips anything for T < T_0. So no
gauge flips a detector or observable of any C_T, and by (G) every detector and observable of every C_T is
deterministic. ∎

### 4.1 Lemma W (the lane's rules W and S, made explicit)
Let w >= 1 with `w + 2 >= T_1` (IBM: w = 9, 11 >= 5; 3-colour: w = 10, 12 >= 7). Let X be a minimum-weight
undetectable logical set of the projected model with `|X| <= w`.

*(i) No gaps.* If no member of X starts at round r, strictly between the smallest and the largest start round, the
members starting before r touch rounds <= r and those starting after touch rounds >= r+1 (P1); each part is
undetectable, one of them flips a logical, and it is smaller -- contradiction. So the start rounds form an interval
`[r1, r2]` with `r2 - r1 <= w - 1`.

*(ii) Rule W.* For T > w+2: X cannot touch both ends (that would need T <= w). If `r2 <= T-2`, keep it in place
(r1 = 0) or translate it by `-(r1 - 1)`; if it touches the end, `r1 >= 3`, translate it by `-(T - w - 2)`. Either way
every window used is the start, an interior, or an end window of both `M_T` and `M_{w+2}` (Corollary 1: both are
>= T_1), so the translate consists of mechanisms of `M_{w+2}` with the same shapes, is undetectable (translation is a
bijection of the touched detectors) and logical (drift 0: shapes include the observables), and has the same weight.
Hence "NONE <= w at T = 1..w+2" gives "NONE <= w at every T".

*(iii) Rule S.* If `r1 >= 2`, chain the pairs (T'', T''+1), T'' = T-r1+1, ..., T-1, each `time_shift_ok`
(Corollary 1 for T'' >= T_1, the exact pairs below): each pair gives `win_{T''+1}[r+1] = win_{T''}[r]` for every
r >= 1 (interior by P2, r = T''-1 by P3_last, r = T'' by P3_final; for T'' = 1 only r = 1 = T'', P3_final). So X is a
translate of a solution starting in round 1 of `M_{T-r1+1}`. If that model is beyond w+2, the solution touches only
rounds <= w+1 <= (T-r1+1) - 2 and embeds into `M_{w+2}` as in (ii). Hence "NONE <= w starting in round 0 or 1 at
T = 1..w+2" gives "NONE <= w at every T". The premise may be weakened to "no IRREDUCIBLE solution of weight <= w
(no NONEMPTY proper subset undetectable; equivalently, no proper subset a solution, the Lean `Irreducible`) starts
in round 0 or 1", which is what the time-shift certificates state: X has
minimum weight, so it is irreducible, and a translate of an irreducible set is irreducible (translation is a
bijection of mechanisms and detectors), so every set this argument produces is irreducible.

### 4.2 Corollary 3 (the lane's claims)
- **IBM cycle, memory-Z and memory-X: d_circ = 10 for every T >= 2.** Lower bound: the 22 time-shift certificates
  of `evidence/circuit_distance_exact/ibm_recert_20261026/` (CDP5, written by the committed `cdist_prove.c` and
  VERIFIED by the committed `ccheck.c`, the version `lean/CertR198` models; T = 1..11, both bases; every proof and
  model shipped, each model byte-identical to a fresh export, `records/ibm_recert_reexport.jsonl`) state "no
  irreducible undetectable logical error of weight <= 9 whose earliest round is <= 1"; with Lemma W(iii), w = 9, they
  give `d_proj(T) >= 10`, and `d_circ >= d_proj`. (A second, older route: the campaign's 22 certificates "no
  undetectable logical error of weight <= 9", `records/ibm_proofs_w9.jsonl`, with Lemma W(ii). Their checker version
  was never committed and their proofs are not kept, so the paper does not rest on them.) Upper bound: the recorded weight-10 full-DEM witness (T = 2,
  `results.json`) lies in `tau_2(End)` of the full formula (T_0 = 2), so `tau_{T-2}` of it lies in `tau_T(End)`
  inside `M_T` for every T >= 2; it is undetectable, logical, and no single deletion keeps it so (checked directly
  at T = 2..6 as well). Hence d_circ(T) = d_proj(T) = 10 for every T >= 2.
- **3-colour cycle, memory-Z and memory-X: d_circ = 11 for every T >= 2.** Lower bound: the 24 time-shift
  certificates "no irreducible undetectable logical error of weight <= 10 whose earliest round is <= 1", T = 1..12,
  both bases, re-proved and VERIFIED by the committed prover and checker in the 2026-11-02 re-certification
  (`recert_20261102/manifest.json`, every re-exported model byte-identical to its record's; models re-exported here
  also hash to the records at T = 1, 2, 6, 11, 12 and equal Theorem 1's model), + Lemma W(iii) with w = 10. Upper bound: every mechanism of the
  recorded weight-11 full witness lies in S of the full formula (start type; all eleven, in both bases -- none needs
  an interior translate `tau_i(I)`), hence in `M_T` for every T >= T_0 = 4 (full model, j_s = 2), and directly at
  T = 2..9. Hence d_circ(T) = 11 for every T >= 2, PROVEN from MACHINE-CHECKED
  premises.

## 5. The checks: `window_uniform.py`
`python -B window_uniform.py` (single process, single-threaded; 6 min; peak 208 MB IBM, 256 MB 3-colour; run log
`run_window_uniform.log`, JSON rows `window_uniform_run.jsonl`): **OVERALL PASS, 4 (circuit, basis) runs.** Per run:

- TPL 12/12 T agree (1..9, 16, 24, 32); LBL PASS; FIR / END minimal h, e as in section 2 (rows mem and all).
- The block machinery equals stim's DEM at T = 1, 2, 3, full and projected (validates this package's frame rule).
- DET PASS (no gauge flips anything at T < T_0; gauge sets of Theorem 1 empty); G: rank 132 = tableau 132 at T = 1, 2.
- Theorem 1 = the exact block iteration at T = T_0..T_1+1, and **= stim's DEM alone** at T = 16, 24, 32 projected
  (11,664 / 17,424 / 23,184 mechanisms, all four) and T = 16, 24 full (IBM 91,368 / 138,600 Z, 91,440 / 138,672 X;
  3-colour 90,528 / 137,760); sets stable across T_ref.
- Corollary 1's structure at T_1 and the pairs below it, as in the table of section 4.
- SIZES: every recorded model size (IBM sweep T = 1..11, 3-colour sweep T = 1..12) = Theorem 1's: 46 rows, 0 differ.
- BIND: re-exported models (the lane's `export_model.export`, written to this directory and deleted) hash to the
  recorded sha256 and their mechanism sets equal Theorem 1's: IBM T = 1, 2, 6, 11 (both bases), 3-colour T = 2
  (certificates) and T = 1, 2, 6, 11, 12 time-ordered (sweep), both bases: 20 rows, all equal.
- WITNESS: IBM end type (every T >= 2), 3-colour start type (every T >= 4 by the formula, T = 2..9 directly).
- Controls, every one fails as required (60 in all: 14 per IBM run, 16 per 3-colour run): TPL x3; LBL; FIR (data-data
  CNOT in B; and h-1 for the 3-colour cycle); END (data-data CNOT in R; and e-1 for the 3-colour cycle); DET (a
  random round-0 detector); G (preparation gauges dropped: 66 != 132); FORMULA (end part one round early vs stim at
  T = 16; stim of C_16 with one interior block's noise deleted vs the formula); WINDOWS (one interior mechanism
  removed: the pair check fails); SIZES (a recorded size off by one); WITNESS (one mechanism dropped; one mechanism
  one round off).

A control that was dark and was replaced, recorded so nobody reuses it: translating the WHOLE IBM end-type witness by
a wrong number of rounds is not a control -- the IBM final-window shapes are interior shapes too, so the witness is
also a valid weight-10 set one round earlier (measured 2026-10-22).

## 6. Adversarial notes
1. **The pipelined boundary.** The 3-colour first block F lacks the previous round's X-check tails (and the data
   idle noise differs there); the tail E finishes the last round. Both are finite blocks; their effect is the start
   part S and the end part End, which Lemma 4 / Lemma 2 make T-independent. The full model is covariant only from
   position 2 (j_s = 2), which is why its start part is thicker and its window at r = 1 differs.
2. **What FIR and END mean physically.** IBM (h = e = 0): every block resets all ancillas first, data Paulis pass a
   round unchanged and flip each check by its syndrome, so `C A = C` and the readout closes the last round exactly.
   3-colour (h = e = 1): a frame on a live ancilla at a block cut spreads (hooks) onto data during the next block; one
   further block settles it to "data Pauli + the partial syndromes the live ancillas have collected". FIR(0) / END(0)
   FAIL at the record level (240 / 384 basis vectors), but -- as the audit found -- those failures cancel in
   the DETECTORS: the detector-level response is already stable after one block, and the 3-colour formula also
   holds with U = 1 (T_0 = 2 projected, 3 full; equal to stim at T = 2, 3, 7, 13, 20). The theorem with U = 2
   stays true; U = 2 is sufficient, not tight, and the controls show only that the sufficient identity is.
3. **T = 1.** In a 1-round model the last round is the start round, so P3_last fails for (1, 2) in memory-Z; rule S
   needs only P3_final there (as `tools_cde` already encodes), and Lemma W(ii) never uses T = 1 windows.
4. **What the theorem does not cover.** Circuits off the template (TPL), noise that is not Pauli or is correlated
   across blocks, leakage, any change of the detector definitions (e.g. meta-checks -- extra detectors can only
   remove solutions, so the lower bounds survive them, but Theorem 1 as stated is for the lane's detectors). For any
   variant, rerun the script; each hypothesis's control shows it can fail.

## 7. Uncertainty flush
- (F) and (G) are cited, not re-proved. (G) is corroborated by the rank equality at T = 1, 2 only.
- **Theorem 1 is a hand proof**, audited once independently (section 8). Its machine side (hypotheses, the formula against stim at fresh T, the window structure) is strong; the hand steps
  most worth attacking are Lemma 2's detector-covariance case analysis (which detectors exist at the boundaries) and
  Lemma 3's split of detectors into (a) / (b).
- The block machinery is this package's own frame-rule implementation; it is cross-checked by stim (exact equality at
  T <= 3 and, through the formula, at T = 16, 24, 32), not by a third implementation.
- Appendix A is a proof by code reading of `bb_circuits.py` at blob `bfa38bb2`, confirmed mechanically at 12 values of
  T up to 32 for all four (circuit, basis) pairs.
- The 3-colour every-T claim rests on the 24 time-shift certificates (records/3c_shift_certificates_w10.jsonl), not on
  the uncertified sweep that preceded them.
- Pinned inputs (ONE_Q HEAD `828e475d3c98`): `bb_circuits.py` blob `bfa38bb2`, `tools_cde.py` `42aae650`,
  `export_model.py` `3e2b8752`, `dem_independent.py` `435d84a4` (read, not imported by the checks); stim 1.16.0.

## 8. Independent adversarial review (2026-10-22): CONFIRMED

An adversary agent that did not write this proof attacked Theorem 1 with its own code (stim's DEM, stim's
FlipSimulator, and its own forward propagator; nothing shared with `wu_*`), archived as
`records/window_review_*.jsonl`.

- **The closed form against stim at fresh T** (`window_review_formula.jsonl`): S, I and End rebuilt from stim alone
  at a different reference (T_ref = T_0 + 6, i_0 = j_s + 3; again at + 3 / + 2: identical) and compared with stim's
  DEM: 102 set comparisons, 0 differences -- all four (circuit, basis) pairs, projected at every T from T_0 to 13 and
  at 20 and 27, full at T_0..13 and 20 (IBM memory-Z projected at T = 27: 19,584 mechanisms). Controls fail as
  required: End one round early (8/8), C_20 with block 10's noise deleted (8/8), j_s - 1 in the full models (4/4).
- **Lemmas 2 and 3 at the frame level** (`window_review_frames.jsonl`): covariance holds for i >= j_s on all 576
  basis frames in all 8 variants and breaks exactly where Lemma 2 says (IBM full at i = 0; 3-colour full at i = 1).
- **FIR / END** recomputed with its own propagator, terminal detectors read from C_6's own records
  (`window_review_fir_end.jsonl`): the author's numbers exactly; its propagator equals stim's FlipSimulator bit for
  bit once resets are cleared (and differs on 144 frames without that, so the comparison can fail).
- **Template** (Appendix A) at T = 1, 2, 4, 13, 20, 27, 40 for all four pairs, and by reading `bb_circuits.py`;
  **P1-P3** from stim's projected models alone at T = 1..12; **witnesses** checked in stim's full DEM at T = 3, 13, 20.
- **Reading**: Lemma 2's boundary cases complete; Lemma 3's split covers every detector reading a suffix record;
  Lemma 4 and the theorem's cases partition the components.

Its two narrower findings are fixed above (D1: section 6 note 2 overstated what FIR(0)'s failure means -- the
3-colour formula also holds with U = 1; D2: the summary's "every consecutive pair" now names the (1, 2) memory-Z
exception). D3 (informational): for the IBM projected model j_s = 0 also works, since F = B. Not re-attacked: the
cited gauge rule (G) (only corroborated: stim refused no detector up to T = 27 projected / 20 full) and Corollary
3's finite premises.

## Appendix A: the builders emit C_T for every T (code reading, `bb_circuits.py` blob bfa38bb2)
- **`ibm_cycle` (lines 193-251).** Before the loop: R on all qubits, H on data for memory-X, TICK -- P. The loop body
  for round r (lines 207-249) emits the same instructions for every r: ancilla R + X_ERROR, H + DEPOLARIZE1 on X
  ancillas, TICK; for t = 0..6 the CX layer built from SX[t], SZ[t] (independent of r) with DEPOLARIZE2 and idle
  DEPOLARIZE1, TICK; H + DEPOLARIZE1, X_ERROR, M on all ancillas; detectors; TICK. Only the detectors depend on r:
  r = 0 emits the memory type's single-record detector, r >= 1 both types' pair `(cur, last)` -- the closed form.
  `_final_readout` (lines 173-190): H (memory-X), X_ERROR, M on data, the final detectors (row of H_b plus the last
  round's record) and the observables (logical rows) -- R and the closed form. So C_T = P . B^T . R with F = B.
- **`three_colour_cycle` (lines 254-333).** Events are generated per round r at absolute times `8r + s` with s from
  the fixed schedule SCHEDULE_3C (s = 1..13), so the event set at time `8j + s'` (s' = 1..8) is round j's events with
  s = s' and round j-1's with s = s' + 8; for 1 <= j <= T-1 both exist and the set is the same up to the round label;
  for j = 0 only round 0's (F); times `8T + 1..8T + 5` hold only round T-1's events with s = 9..13 (E). Within a time
  step the instructions are emitted in a fixed order from the event list (preps, CX pairs in event order -- rounds in
  increasing order, so round j-1's before round j's in every block --, Z then X measurements, idle DEPOLARIZE1 on the
  sorted idle data, detectors), which depends on the round only through the round labels. Every time step 1..8T+5 is
  occupied, so each block is exactly 8 TICK-segments. Detectors: `r == 0` gives the memory type's single record,
  r >= 1 the pair with round r-1 -- the closed form; `_final_readout` as above.
- Mechanical confirmation: `wu_core.template_check` compares the stripped instruction stream and the detector /
  observable label sets at T = 1..9, 16, 24, 32 for all four (circuit, basis) pairs.

## Appendix B: reproduce
```
python -B window_uniform.py                                   # exit 0 = all PASS -> run_window_uniform.log
python -B window_uniform.py --circuit 3c --basis Z --fresh-T 16 40 --fresh-T-full 16
```
From `experiments/circuit_distance_exact/`. Exported models are written to a temporary directory and deleted
after hashing; the archived run is `records/window_uniform_run.jsonl`.
