# Finite-blocklength superactivation on IBM Heron: a circuit-level resource estimate

Item E of the research plan: compiling the full physical experiment of Parentin, Bergh, Datta, Wilde
(arXiv 2604.27042) to IBM Heron's gate set {CZ, sx, rz, x}, with mid-circuit measurement and
feed-forward. This is an estimate built from explicit constructions that were verified by exact
simulation. Nothing was run on hardware and no IBM network API was called. The ibm_kingston
coupling map comes from the local `FakeKingston` snapshot.

Directory: `C:\Users\Josh\Projects\_worktrees\superact_compile\`. Every number below is printed by
a script in this directory; the commands are listed in [Reproduce](#reproduce) and in `run_all.sh`.

## Bottom line

The code being run is ours: the certified n = 17 and n = 24 codes, with the encoder truncated to an
exact isometry and the decoders replaced by minimum-rank decoders of the same fidelity. Error model:
first order, eps = 0.0063 x (number of CZ). The budget is delta = margin / (F - 1/4), taken from the
certified lower bound of the compiled code.

| | n = 17 | n = 24 |
|---|---|---|
| certified F of the **compiled** code (`certify_compiled.py`) | >= 0.750203335560 | >= 0.751030733687 |
| budget delta | 4.07e-4 | 2.06e-3 |
| logical CZ per shot, expected (worst) | **1122** (1455) | **2002** (3061) |
| - encoder | 592 (53%) | 1173 (59%) |
| - n uses of E~, P (x) A, D~ | 62 (5.5%) | 87 (4.3%) |
| - decoder, expected (worst) | 466 (759) | 740 (1742) |
| eps = 0.0063 x CZ | 7.07 | 12.6 |
| **infeasibility factor eps/delta, logical** | **17,400x** (worst 22,500x) | **6,130x** (worst 9,370x) |
| heavy-hex routed CZ, best layout (20 sampled shots, mean +- sem) | 2815 +- 113 | 5407 +- 113 |
| **infeasibility factor, routed** | **43,600x** | **16,600x** |
| encoder-only gate floor used before ((n-1) CZ) | 249x | 70x |
| CZ error that would fit the budget (logical / routed) | 3.6e-7 / 1.4e-7 | 1.0e-6 / 3.8e-7 |
| P(no CZ error in a shot) at 0.63% per CZ, logical | 8e-4 | 3e-6 |

Profile: P and A run as randomized channels and D~ uses measurement-based untwisting, which is the
cheapest honest implementation (see [Profiles](#5-per-run-totals-and-the-infeasibility-factor)).
The full circuit is 70x (n = 17) and 88x (n = 24) further from feasible than the encoder-only floor
used in ledger R133, and 176x and 237x further after routing.

**For these two codes the conclusion holds even if the encoder and decoder were free.** The channel
layer alone (P (x) A with E~ and D~ as compiled here, n times, plus the Bell pair and the Bell
readout) costs 63.6 CZ at n = 17 and 88.9 CZ at n = 24. That is still **985x** and **272x** over
budget (DERIVED from the measured counts). The scope of that claim:
- It is about the paper's reduction (E~, D~) at the blocklengths where superactivation has been
  certified.
- It is not a statement about every possible code, or about other pre/post-processing of P (x) A.

Unlike the encoder-only floor, n = 24 comes out about 3x more feasible than n = 17. Its budget is 5x
larger, and its circuit only 1.8x bigger. Whether larger n keeps improving is open: the budget grows
with the fidelity gap, while the gate count grows about 2n^2. Answering it needs certified codes and
fidelities beyond n = 24, which were not computed here (future work).

Evidence classes:
- **PROVEN:** the compiled code's fidelity bound, by ONE_Q's exact-arithmetic certificate. It imports
  the paper's Proposition 1 and Result 1.
- **MEASURED:** exact-simulation residuals and every gate count, with the construction counted in
  full.
- **DERIVED:** the decompositions below.

Counts are **upper bounds** on the optimal compilation. Minimality was searched for the small gates
only, and that search is evidence, not proof.

## 1. P and A as circuits (`channels.py`, `verify_channels.py`, `check_paper_objects.py`)

### What the supplement defines

These were extracted from `supplemental.tex`, not guessed.

**P, the private Horodecki channel:** C^4 -> C^4, with C^4 = key (x) shield.
- Choi state, eq. (choi-state-horo), l.333-361: sum_i q_i psi^i_{R'B'} (x) rho^(i)_{R''B''}, with
  p = 1/(1+sqrt 2).
- Six Kraus operators, l.365-387: K1 = sqrt((1-p)/2) 1 (x) |0><0|, K2 = sqrt((1-p)/4) 1 (x) X,
  K3 = sqrt((1-p)/2) Z (x) |1><1|, K4 = sqrt((1-p)/4) Z (x) Y, K5 = sqrt(p) X (x) M0,
  K6 = sqrt(p) Y (x) M1.
- M0 = diag(c, s) and M1 = diag(s, -c), where c = cos(pi/8) and s = sin(pi/8).

Checks (`check_paper_objects.log`):
- The Kraus operators reproduce the Choi state to 2.8e-16.
- With Smith-Yard's M1 sign (the misprint the footnote at l.381 flags) the error is 0.29.
- Choi rank 6, trace preserving.
- PPT: the minimum eigenvalue of the partial transpose is -2e-16.
- Lemma choi-as-mixture holds to 4e-17.

**A, the 50% erasure channel:** d_A = 4 and d_B = 5 (l.311-316). The isometry is
V|psi> = (|psi>_B |e>_E + |e>_B |psi>_E)/sqrt 2 (l.317).

### New, DERIVED: P is mixed-unitary

P = sum_j w_j U_j . U_j^dag, where:

| U_j | weight |
|---|---|
| CZ | (1-p)/4 |
| CZ Z_s | (1-p)/4 |
| X_s | (1-p)/4 |
| Z_k Y_s | (1-p)/4 |
| X_k D+ | p/2 |
| X_k D- | p/2 |

Here D+- = 1 (x) M0 +- i Z (x) M1. It is diagonal with phases (pi/8, -3pi/8, -pi/8, 3pi/8), so D+- is
P(-+pi/4)_k S^(-+1)_s CZ, which costs 1 CZ.

How it was derived:
- {K1, K3} is CZ followed by shield dephasing.
- {K2, K4} is a mixture of 1 (x) X and Z (x) Y.
- {K5, K6} is Ad_{X(x)1} composed with {1 (x) M0, i Z (x) M1}. The Kraus combinations
  1 (x) M0 +- i Z (x) M1 are unitary because |c +- is| = 1.

Checked to 1.1e-16. This gives two implementations of P:

| P implementation | qubits | CZ | check (Choi trace distance, compiled) |
|---|---|---|---|
| randomized: controller draws j, runs U_j | 2 (no environment) | 1 or 0 per draw, expected (1+p)/2 = 0.7071 | 7.7e-16 |
| coherent Stinespring, phase-polynomial select (`pp_synth.py`) | 2 + 3 env | **20** | 8.6e-15 |
| coherent Stinespring, gate-by-gate select | 2 + 3 env | 26 | 1.3e-14 |
| coherent, qiskit generic Isometry (comparison) | 2 + 3 env | 104 | 4.4e-14 |

Three environment qubits is the minimum, because the Choi rank is 6.

The coherent dilation works as follows:
- Environment coin: ea = 1 with probability (1+p)/2, eb uniform, and ec = 1 with probability
  2p/(1+p) given ea = 1, using a 1-CZ split.
- Select: X_s^(1-ea) X_k^(ec) exp(i phi).
- phi = pi eb s + pi ea k s + pi eb k - pi ea eb k + (pi/2) eb ec k - (pi/4) ec k - (pi/2) ec s.
  It is this sparse because the environment value (ea=0, ec=1) never occurs.
- phi has 14 parity terms, 10 of them non-linear.

### A as a circuit

**Coherent:** qubits (k2, s2) plus flag f plus environment (fE, e1, e2). The circuit is
Ry(pi/2) on f, then CNOT f -> fE, then two controlled moves |f, x, 0> -> f ? |1, 0, x> : |0, x, 0>.
- Each move is a relative-phase Toffoli plus a CNOT. The relative phase is exact here because e
  enters in |0>.
- **4 CZ per move; 3 CZ was not reachable on the reachable inputs** (`min_cz_search.log`: c = 3 leaves
  residual 0.076).
- Total **9 CZ**, verified against A with |e> embedded as |1>_f|00>, to 3.7e-15.

**Randomized:** the flag is a classical coin, and on erasure the A2 qubits are reset before Bob acts.
This costs **0 CZ**.

## 2. E~ and D~ of Proposition 1 (`channels.py`)

**E~** (eq. encoder_protocol): Phi^{d_s} on (A1'', A2'') with H + CNOT, then CNOT A1' -> A2'.
**2 CZ.** It needs no measurement.

**D~, paper construction:**
1. Erasure measurement Pi_{B2}: a mid-circuit read of the flag.
2. If z = 0: the untwisting V^dag, discard the shields, the parity measurement with outcome l, X^l on
   B1' by **feed-forward**, then inverse CNOT and discard A2'.
3. If z = 1: discard B1'', A2', A2''.

So the construction **does require mid-circuit measurement and feed-forward**, and it is implemented
that way: `d0_paper` uses `if_test`.

The untwisting V^dag:
- Synthesized generically as a 4-qubit unitary it costs 44 CZ.
- Its structure (`untwist_structured`, `check_untwist.log`) cuts this to **17 CZ**, operator-equal
  to 3e-15. Both U^{1,0} = SWAP.CZ and U^{1,1} act as SWAP on the odd shield-parity subspace, and as
  Z and H respectively on the even one. So after CNOT(s2 -> s1), V^dag is one uniformly controlled
  single-qubit gate on s2, controlled by s1, k1 and k2.
- The paper's D^0 then costs **18 CZ**, 1 read and 1 feed-forward X.

**D~, measurement-based untwisting (new, verified equal as a channel):** the shields are discarded
right after V^dag, so Bob only needs the +-1 eigenvalue of the twist U^{1,l}. He measures it and
applies Z to the key if it is -1.
- The coherence survives exactly: sum_+- |Tr(P_+- sigma^l U^{1,l dag})| = 1. This holds because
  sigma^0 is proportional to I, and the support of sigma^1 is invariant under U^{1,1}, which acts as
  H there.
- Circuit: CNOT k1 -> k2, read l, X^l; CNOT s2 -> s1, CH s1 -> s2, [l = 1: Ry(-pi/4)], read m, Z^m.
- Cost: **3 CZ**, 2 reads, 3 conditional single-qubit gates.
- A measurement-free version costs 7 CZ.

Verification: the whole block D~ o (P (x) A) o E~ was checked against N~ of Prop. 1 by exact branch
simulation (flag as a classical output) for every combination of P/A profile and D~ variant. The
largest block Choi trace distance is **8.7e-15**, raw and compiled. Over all 14 rows of
`verify_channels.log` (P, A and the blocks) the largest is 1.3e-14, and every row passes.

Per-use CZ after compilation:

| profile | per controller draw (P_j, erased) | expected per use | worst |
|---|---|---|---|
| randomized + measurement-based D~ | U1-U6 x {kept, erased} = 4/3, 4/3, 3/2, 3/2, 6/3, 6/3 | **3.62** = 3 + 1.5p | 6 |
| randomized + measurement-free D~ | | 5.62 | 10 |
| randomized + paper D~ | | 11.7 | 21 |
| coherent channels + measurement-based D~ | 31 + (z=0) 3 | 32.5 | 34 |
| coherent channels + paper D~ | 31 + (z=0) 18 | 40.0 | 49 |

When P's unitary is diagonal on the key, E~'s CNOT A1' -> A2' and D^0's parity CNOT cancel. That
cancellation is where the 3 in 3 + 1.5p comes from.

## 3. Encoder V: C^2 -> Sym^n (`dicke.py`, `verify_encoder.py`, `verify_encoder_lnn.py`)

**The witness.** The witness X has Kraus rank 1 up to the SDP residue: its second eigenvalue is
8.3e-5 (n = 17) and 4.0e-5 (n = 24). We keep the top eigenvector and take its polar factor, which
gives an exact isometry. The fidelity goes *up* slightly, to 0.7502033361 and 0.7510307345 in float,
and is **re-certified**: n = 17 F >= 0.750203335560, n = 24 F >= 0.751030733687
(`certify_compiled.py`, exact flint arithmetic).

**The construction (representation: unary to Dicke).** Write |unary_w> = |0^{n-w} 1^w>. The input
qubit sits at position n, so |r> (x) |0^{n-1}> = |unary_r>.
1. **W, the amplitude-loading network.** Adjacent-level Givens rotations map |unary_r> to
   sum_w V[w,r] |unary_w>. It is a QR in the order column 0 then column 1. Each rotation is
   uncontrolled, singly controlled, or doubly controlled, depending on which levels are populated at
   that moment (support tracking). A rotation into an empty top level costs 1 CZ. Complex phases
   become single-qubit prefix phases.
   W costs **5n - 5** CZ, made up of n-1 one-CZ rotations, 2 two-CZ rotations and n-2 four-CZ
   rotations.
2. **U_n, the Baertschi-Eidenbenz unitary** (arXiv:1904.07358). It maps |unary_w> -> |D^n_w> for all
   w at once. Each gate is compiled on its reachable inputs only:
   - Gate (i) is a Givens completion: **2 CZ**. 1 CZ was not reachable in the search.
   - Gate (ii) is **4 CZ**. 3 CZ was not reached in 27 placements x 6 restarts with the strict
     specification, nor with per-level phase freedom (`min_cz_search.log`, `min_cz_relaxed.log`).
     The literal BE gate costs 6.
   - Per-(l, ell) angles are tabulated in `gate_ii_num4.json` and polished to amplitude error
     6.4e-16.
   - U_n costs **2(n-1)^2**, against (n-1)(3n-2) for literal BE.

Encoder CZ(n) = 2n^2 + n - 3.

| | n = 17 | n = 24 |
|---|---|---|
| CZ (compiled = closed form) | **592** (W 80 + U 512) | **1173** (W 115 + U 1058) |
| literal BE gates | 864 | 1725 |
| statevector error vs sum_w V[w,r] |D_w> (2^n vector, both r, one global phase) | **1.1e-15** | **3.1e-16** |
| 2q-depth | 173 | 250 |
| heavy-hex (Sabre, level 3) | 1642 CZ | 3611 CZ |
| **nearest-neighbour version on a heavy-hex path** | **997 CZ, routed 997 (zero SWAP)** | **1998, routed 1998** |

Other checks:
- Random complex isometries, n = 3..6, 5 draws each, raw and compiled: worst error 5.1e-15.
- U_n alone maps every |unary_w> to |D_w> with one common phase, for n = 2..7.

**The nearest-neighbour encoder.** Gate (ii) is fused with the swap that moves the travelling qubit,
using CZ only on line neighbours: **7 CZ; 6 was not reachable in the search** (`min_cz_lnn.log`). One extra SWAP
per SCS parks the travelling qubit, so W + U_n^LNN = 997 / 1998 CZ.
- Verified by statevector at n = 3..7 and n = 17 (error 9.4e-16, also with qiskit's Statevector).
- Every 2q gate acts on adjacent qubits.
- On the ibm_kingston snapshot it is placed on a simple heavy-hex path, and routing adds nothing.
- At n = 24 it was built, counted and routed (`verify_encoder_lnn_n24count.log`). The full 2^24
  statevector check was stopped: about 30k single-qubit operations at roughly 0.3 s each.
  - What does cover l <= 24: every one of the 253 tabulated gates is exact to 8.9e-16 on its
    reachable inputs (`gate_tables_lnn.log`), and the composition is verified up to n = 17.
  - The all-to-all n = 24 encoder, which carries the logical counts, was verified in full at 2^24.

Logical counts are lower with the all-to-all version; routed counts are lower with this one.

## 4. Decoders D_{k,b} on Sym^{n-k} (`decoder.py`, `verify_decoder.py`, `verify_decoder_n17_spot.py`, `decoder_counts.py`)

Bob first reads the k flagged qubits in Z; their weight b is a sufficient statistic, per the module.
The decoder for (k, b) then acts on the m = n - k unflagged qubits:

1. **U_m^dag:** Sym^m -> unary levels, **2(m-1)^2** CZ.
2. *(route "compressed" only)* An inverse unary Givens network maps the support S of Bob's state
   (dimension s <= min(2(k+1), m+1)) onto levels 0..s-1.
3. **Unary to binary register (q = ceil(log2 L) qubits).** Bit j of w is the parity of the
   thermometer bits t_i with 2^j | i, which costs sum_j floor((L-1)/2^j) CZ.
   - Then **every thermometer qubit is read in X**, which gives L - 1 mid-circuit reads.
   - The kicked-back phase (-1)^{sum_i s_i [w >= i]} is undone on B by a fixed Gray-code CNOT
     skeleton of **2^q - 2** CZ. Its Rz angles are switched on by feed-forward: conditional
     single-qubit gates only, 75 per shot expected at n = 17 and 117 at n = 24.
4. **The decoder's Stinespring** on (B, out), completed to a unitary and synthesized by QSD. QSD of
   the completion was measured cheaper than qiskit's column-by-column Isometry: 3/4/5/6 qubits cost
   19/95/423/1783 CZ against 24/122/546/2300. B is discarded, and `out` is the decoded qubit.

**Fixed-fidelity rank reduction (new).** The witness decoders have Kraus rank up to about 2s, which
would force an extra ancilla and a 4x larger unitary. Only D restricted to S matters, and the
Barvinok-Pataki argument applies: moving along D(t) = Y(I + tH)Y^dag inside the null space of
{trace preservation, branch fidelity} lowers the rank exactly, down to r <= s.
- On the real n = 17 branches the rank halves (for example 24 -> 12), with branch fidelity changing
  by <= 1e-15 and TP error <= 9e-14 (`test_min_rank.py`).
- The re-certified bound above is for these decoders.

**Verification (exact branch simulation of the compiled dynamic circuits):**
- n = 3..6, random complex codes with module decoders (`decode_power`), **all 74 branches x 5
  constructions**:
  - The full construction equals the module decoder on all of Sym^m.
  - Every construction equals its target on S.
  - Worst Choi distance 1.4e-12.
  - Compiled CZ equals the component formula in every branch and mode.
- **End to end at n = 3..6:** compiled encoder, then exact N~ on every qubit for all 2^n flag
  patterns, then the compiled decoder on the unflagged qubits in their physical positions. The
  entanglement fidelity equals the module's `fidelity(X, decoders)` to **<= 1.3e-14**. This is the
  check that the Dicke basis, the Choi orientation, Kraus conjugation and qubit order agree with the
  certified reduction.
- **The real n = 17 decoders**, every branch with m <= 7 (k >= 10, 116 branches), simulated exactly:
  worst Choi distance 9.3e-13, and compiled CZ equals the formula.

Per-k costs and the full tables are in `report_tables.md`.

| | n = 17 | n = 24 |
|---|---|---|
| expected decoder CZ (k ~ Bin(n,1/2), b ~ Pr(b\|k) of this code) | **466** = U^dag 121 + Givens 2 + unary->binary 15 + phase fix 11 + decoder unitary 317 | **740** = 254 + 39 + 21 + 14 + 412 |
| worst branch | 759 at (k,b) = (2,0) | 1742 at (4,0) |
| typical branch (k = n/2) | 581 (k = 8) | 701 (k = 12) |
| k = 0 (decoder = inverse encoder, s = 2) | 596 | 1176 |
| expected reads (X-reads + flagged Z-reads) | 17.0 | 23.6 |
| max decoder qubits (m + q + 1) | 19 | 26 |

## 5. Per-run totals and the infeasibility factor (`totals.py`, `full_run.py`)

A shot runs as follows:
1. Bell pair (R, input), 1 CZ.
2. Encoder.
3. n x [E~, P_j, A erases?, D~].
4. Z-read of the flagged qubits.
5. Decoder (k, b).
6. Bell measurement (R, out), 1 CZ.

| profile (per-use block, see section 2) | n = 17 CZ E (worst) | factor | n = 24 CZ E (worst) | factor |
|---|---|---|---|---|
| **randomized channels + measurement-based D~** | **1122 (1455)** | **17,387** | **2002 (3061)** | **6,129** |
| randomized + measurement-free D~ | 1156 (1523) | 17,914 | 2050 (3157) | 6,276 |
| randomized + paper D~ | 1259 (1710) | 19,518 | 2196 (3421) | 6,724 |
| coherent Stinespring channels + measurement-based D~ | 1613 (1931) | 24,996 | 2695 (3733) | 8,252 |
| coherent Stinespring channels + paper D~ | 1740 (2186) | 26,972 | 2875 (4093) | 8,803 |

**Whole shots, built and routed.** `full_run.py` draws 20 shots per n from the actual distribution
(flags, P draws, b given k), builds the full dynamic circuit (69 qubits at n = 17, 97 at n = 24) and
transpiles it to the ibm_kingston snapshot at level 3 with exact synthesis:

| | n = 17 | n = 24 |
|---|---|---|
| logical CZ, sample mean +- sd (matches the component total) | 1116 +- 214 | 2029 +- 178 |
| logical 2q-depth | 566 | 810 |
| Sabre layout + routing: CZ (mean +- sem), depth, 2q-depth | 3348 +- 110, 5300, 1708 (3.0x) | 6621 +- 112, 8414, 2774 (3.3x) |
| nearest-neighbour encoder on a heavy-hex path + nearest-free ancillas | **2815 +- 113**, 3994, 1160 (1.8x of its logical count) | **5407 +- 113**, 6342, 1889 |
| routed infeasibility factor (best layout) | **43,630** | **16,558** |

Routing was not extrapolated: every routed number is a full transpilation of a full shot. The
decoder's U_m^dag and the decoder unitary are still Sabre-routed; a nearest-neighbour decoder with
classical compaction of the unflagged qubits is the next lever.

**Mid-circuit measurements per shot (randomized profile):**
- 1 per use on average (2 per unflagged use, from the parity and twist reads).
- 17.0 (n = 17) and 23.6 (n = 24) in the decoder.
- Plus 2 terminal reads.
- Totals 36 (n = 17) and 50 (n = 24), which matches the sampled shots.

**Feed-forward per shot:**
- 2 conditional blocks per unflagged use.
- About 75 (n = 17) and 117 (n = 24) conditional Rz in the decoder's phase fix.
- One classical (k+1)-way selection of the decoder unitary by b.

**Qubits:**

| layout | n = 17 | n = 24 | notes |
|---|---|---|---|
| uses in parallel, randomized profile | 69 | 97 | 4n + 1 logical; fits the 156-qubit Heron |
| minimum width: uses sequential, ancillas reset and reused | 21 | 28 | |
| coherent channel profile, in parallel | 188 | 265 | does not fit 156; needs sequential reuse, which gives n + 11 = 28 and 35 |

**Profiles.** In the randomized profile, the erasure flags are the controller's own random draws, so
the controller knows z before the shot. Bob's z-dependent operations (D~, which qubits are unflagged,
k) are therefore compiled in, and Alice's operations do not depend on z. Only b is a run-time
outcome, and it selects among k+1 final decoder unitaries.

With coherent channels, z is measured at run time, and the decoder would need real-time
qubit-addressing over 2^n flag patterns. That is a control-system blocker, recorded below.

## Independent pass and dark dimensions

**Second simulator.** Everything above was verified with this directory's branch simulator. As a
guard against a common-mode error, qiskit's own simulators re-checked the key artifacts
(`crosscheck_qiskit.py`):
- Compiled n = 17 encoders, all-to-all and nearest-neighbour: Statevector error 1.1e-15 and 9.4e-16.
- The per-use block with randomized channels and measurement-free D~, averaged over the 12 draws:
  DensityMatrix with resets, Choi distance to N~ 9.8e-16.

**Checks that cannot fail in one direction.** The Choi matrices of N~ and of P are both symmetric
under exchanging reference and output. So comparisons against them cannot detect a reference/output
swap. Where that convention is exercised:
- It is fixed by construction in `channel_choi`, inputs first.
- It is tested where it is detectable: A's 4 -> 8 Choi matrix, and the decoders' end-to-end test,
  where a wrong orientation or conjugation would move the fidelity away from the module value.

## Ambiguities in the supplement, and the reading used

| # | where | issue | reading used / evidence |
|---|---|---|---|
| A1 | eq. untwisting_unitary, l.1530 | the key projector \|i><i\| is printed on A2'' (a shield) | read as A2' (the key copy), which matches eq. twisting-unitaries-shift and the proof (l.1843-1849); with it the block equals N~ to 1e-14 |
| A2 | l.1531-1533 | tensor order of U^{i,l} on (A2'', B1'') is not fixed | immaterial: all four U's are invariant under exchanging the two shields; both orders verified to give the same V^dag (`check_untwist.log`) |
| A3 | l.1230, l.1235 | sector-1 antidiagonal constraint written with sigma^0; "\overline{\Gamma^2,}" typo | read as sigma^1; Lemma choi-as-mixture checked numerically with sigma^1 (4e-17) |
| A4 | l.371-387 | tensor-factor order of the Kraus operators; M1 sign | first factor = key; the paper's M1 reproduces its Choi state (2.8e-16), Smith-Yard's sign does not (0.29) |
| A5 | l.311-316 | 5-dim erasure output on qubits | (flag, 2 data) with \|e> := \|1>\|00> (isometric embedding, verified); randomized profile: \|e> represented by the classical flag, which is equivalent up to a reversible relabeling |
| A6 | l.1537-1561 | "parity measurement, X^l, inverse CNOT, discard A2'" | CNOT B1'->A2', destructive read of A2', X^l on B1'; equal as a channel, because A2' is product \|l> after the CNOT |
| A7 | witness convention | Choi orientation, V vs conj V | X = sum \|r><s\| (x) E(\|r><s\|), R first; fixed by the end-to-end test against the module (1e-14) |

## Instrument defects found (measured, and handled; not hidden)

1. **qiskit 2.5 transpile at optimization_level 2-3 perturbs amplitudes by about 6e-8**, even with
   approximation_degree = 1 (2q resynthesis). It showed up as the first encoder check failing at
   1.8e-7 (compiled) while the raw circuit was exact to 1e-15. Every verified artifact is compiled at
   level 1, which is exact. Level-3 counts are used only for routing and the per-use blocks, whose
   level-3 outputs were themselves Choi-checked.
2. **qiskit's TwoQubitBasisDecomposer returned operator errors of 4e-6 to 2.8e-5** on some 2-qubit
   decoder unitaries, with approximate=False and at any optimization level (`debug_kak.py`). Every
   decoder unitary's compiled operator is checked. Failures get a new free completion, or an exact
   LM-fitted circuit of at most 3 CZ.
3. **qiskit's synth_cnot_phase_aam output differed from its own input phase polynomial by pi on a
   parity** (`debug_aam.py`). The residual is local (a product of Z), so it is cancelled exactly and
   re-verified.
4. The fidelity objective used in the gate search certifies amplitudes only to about 1e-7
   (residual 1e-14 is roughly eps^2/2). So the tables were polished on the amplitude residual, to
   6e-16, before use.

## What is NOT modeled

- **Idle errors and decoherence.** The reference qubit R and the encoded qubits wait through
  2q-depths of 566-938 logical and 1160-2774 routed. T1/T2 were not used.
- **Measurement and reset errors.** 36-51 mid-circuit reads and about 3n resets per shot.
- **Feed-forward latency.** The decoder alone has about 75 (n = 17) and 117 (n = 24) conditional
  gates per shot; the idle error during each was not modeled.
- **Single-qubit gate errors.** sx and x were not counted; rz is virtual.
- **Crosstalk, leakage, non-Markovian and coherent error.** The 0.63% figure is the *twirled* rate,
  so twirling of every layer is implicitly assumed.
- **The error model is a first-order sum.** eps >> 1 here, so it is a scale, not a probability; the
  error-free-shot probability (1-0.0063)^N is in the bottom-line table. The budget formula assumes
  global depolarizing at the output.
- **Classical control.** Each shot needs its flag pattern compiled in, which works in the randomized
  profile because the flags are the controller's own draws. b must be computed in real time (a
  popcount of k reads) to select among k+1 decoder unitaries. IBM dynamic circuits expose
  bit-condition if/else, not an integer popcount switch, so this was not modeled. With coherent
  channels the decoder would need run-time qubit addressing over 2^n patterns, which is not feasible
  as a single compiled program.
- **Routing.** It was modeled by full transpilation of sampled shots on the Kingston snapshot (Sabre,
  seed 11, one seed). Better layouts may exist; the nearest-neighbour encoder already recovered 16-18%.
- **Optimality.** The decoder unitaries are generic QSD. A channel-specific synthesis could be cheaper
  by up to about 2x on parameter count. The encoder's O(n^2) is the known ancilla-free scaling for
  arbitrary symmetric states (see the Baertschi-Eidenbenz and divide-and-conquer papers in
  Sources). Constant-depth adaptive constructions exist, but at a polynomial ancilla cost. None of
  this changes the conclusion for n = 17 and 24: the channel layer alone is 985x / 272x over budget.

## Reproduce

Python is `C:\Users\Josh\AppData\Local\Python\pythoncore-3.14-64\python.exe`, with qiskit 2.5.0,
qiskit-ibm-runtime 0.47 (fake backend only), numpy 2.3.5, scipy 1.17 and python-flint. ONE_Q is read
from `C:\Users\Josh\Projects\_worktrees\ONE_Q-superact` (8b8b6247). `run_all.sh` runs the full chain
in dependency order; the main scripts are:

| script | produces |
|---|---|
| `check_paper_objects.py` | P/A/N~ transcription checks |
| `min_cz_search.py`, `min_cz_relaxed.py`, `min_cz_lnn.py` | minimal-CZ searches |
| `gate_tables.py` + `polish_gates.py`, `gate_tables_lnn.py` | gate tables |
| `check_untwist.py`, `p_phasepoly.py` | structured V^dag, phase-polynomial P |
| `verify_channels.py` | deliverables 1-2 |
| `verify_encoder.py --n24`, `verify_encoder_lnn.py [--count-only]` | deliverable 3 |
| `verify_decoder.py 3 4 5 6`, `verify_decoder_n17_spot.py 17 7`, `test_min_rank.py` | deliverable 4 verification |
| `decoder_counts.py 17 24`, `certify_compiled.py 17 24` | deliverable 4 counts, exact certificate |
| `full_run.py {17,24} 20 --route [--lnn]`, `totals.py`, `report_tables.py` | deliverable 5 |

## Sources (literature consulted for the encoder scaling)

- Baertschi, Eidenbenz, Deterministic preparation of Dicke states, [arXiv:1904.07358](https://arxiv.org/pdf/1904.07358.pdf)
- A divide-and-conquer approach to Dicke state preparation, [arXiv:2112.12435](https://arxiv.org/pdf/2112.12435)
- Constant-depth adaptive preparation of Dicke and symmetric states, [arXiv:2608.01144](https://arxiv.org/pdf/2608.01144)
- Global variational quantum circuits for arbitrary symmetric state preparation, [arXiv:2312.05060](https://arxiv.org/pdf/2312.05060)
