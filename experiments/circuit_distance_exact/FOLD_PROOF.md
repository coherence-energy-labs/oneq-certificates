# The fold lemma for every T: two-round morphing memory experiments, R198 lane convention

**Scope.** The two morphing memory experiments of the R198 lane (`bb_circuits.morphing_cycle`: `morph72`,
`morph144`, homomorphism `x`), memory-Z and memory-X, in the lane's convention: noisy preparation and readout, the
lane's derived detectors (no meta-checks), the same-type projection, and standard depolarizing circuit noise.

**Result.** Under six hypotheses on finite data that do not depend on T (section 2), Theorem 1 gives, for every
T >= 1,

    d_circ(T) = d_proj(T) = d_proj(1) = d_circ(1).

All six hypotheses are checked mechanically for both circuits, both bases and all three homomorphisms. For each
hypothesis there is at least one doctored input that must fail it, and it does (section 5).

**The proof is an explicit formula, not an induction on "the pattern continues".** Lemma 3 says that a fault whose
mid-cycle image is the Pauli P at mid-cycle point t has projected symptom `(iota_t(sigma_S(P)), sigma_L(P))`:

- `sigma_S(P)` is P's syndrome against the memory-type generators;
- `sigma_L(P)` is its flip pattern on the memory logicals;
- `iota_t` places generator g's bit on the one detector of g whose time window contains t.

Adding up each generator's detectors (the map kappa) undoes every `iota_t` at once. This is the "round collapse"
map the adversary found empirically. Separately, `iota_0` is injective.

With the lane's 1-round certificates and witnesses (Corollary 3):

- the **[[144,12,12]] morphing circuit has d_circ = 12 for every T >= 1**, memory-Z and memory-X;
- the [[72,12,6]] circuit has d_circ = 6 for every T >= 1.

**Evidence class.**

- **Theorem 1: PROVEN** by hand in this file; not machine-verified. It survived one independent adversarial audit
  (2026-10-22, section 8), which refuted nothing.
- **The hypotheses for the lane's circuits: MACHINE-CHECKED** by `fold_uniform.py`, with negative controls.
- **Corollary 3: PROVEN from MACHINE-CHECKED premises and one argued premise.** It inherits the trusted base of
  the T = 1 certificates (`ccheck.c`, whose certificate calculus is PROVEN sound in Lean, `lean/CertR198`; the
  exporter's projection), and Appendix A's premise that the builder emits C_T for every T: a proof by reading the
  code, confirmed mechanically at 10 values of T up to 32 and, against stim alone, at 18 more (section 8).

The lane's previous phrase "MACHINE-CHECKED for T <= 13, DERIVED beyond" is retired for both morphing circuits.

---

## 1. Setting

### 1.1 Paulis and CNOT layers
Qubits `Q = L u R`, with `L = {0..n-1}` (data, "left") and `R = {n..2n-1}` ("right"); `n = lm` (72 or 144). Paulis
are taken modulo phase: `P = X^x Z^z` with `x, z in F_2^Q`. The symplectic form `<P, P'> = x.z' + z.x' (mod 2)` is 1
exactly when P and P' anticommute. A Pauli is X-type if z = 0 and Z-type if x = 0. For a qubit set S and a type
`b in {X, Z}`, `b_S` is the b-type Pauli supported on S.

A CNOT layer G is a set of (control, target) pairs. If the qubits of distinct pairs are distinct (hypothesis H1), G
is a unitary involution, and conjugation `G(P) := G P G` acts by `x_t <- x_t + x_c` and `z_c <- z_c + z_t` for every
pair. Two facts are used throughout:

- **(CSS)** The new x depends only on x, and the new z only on z. A product of X's (of Z's) goes to a product of X's
  (of Z's), with sign +1.
- **(SYM)** `<U(P), U(P')> = <P, P'>` for every Clifford U.

### 1.2 The finite data D
- **Contraction circuits** `F_r = (G_{r,0}, G_{r,1}, G_{r,2})` for `r in {0,1}`. `F_r(P) := G_{r,2}(G_{r,1}(G_{r,0}(P)))`
  is P carried forward through the contraction. Define `Phi_{r,k}(P) := G_{r,0}(G_{r,1}(...G_{r,k}(P)))` for
  `k = -1, 0, 1, 2`, with `Phi_{r,-1} = id`. Under H1, `Phi_{r,2} = F_r^{-1}`.
- **Measurement split** `R = M^X_r u M^Z_r`: the right qubits measured, and later reset, in X and in Z after `F_r`.
- **Memory basis** b, with b' the other basis. The flip of b is `f_b` (X when b = Z, Z when b = X).
- **Generators of the mid-cycle code.** `S^X_h` are the rows of H_X and `S^Z_g` the rows of H_Z, for `g, h in [n]`.
  The memory-type generators are `S_g := S^b_g`. The memory logicals are `L_1..L_K` (K = 12): b-type, a basis of the
  b-type logicals (`BB.logicals`, as the builder takes them).
- **Splits.** For an operator O of type t and a parity r, by (CSS) `F_r(O) = O^r_R . O^r_L` with
  `O^r_R = t_{supp F_r(O) n R}` and `O^r_L = t_{supp F_r(O) n L}`.
- **Contraction parities.** `Gamma_O := {r : O^r_L = I}`, the parities in which O contracts onto right qubits only.
- **Noise.** The component sets placed at each location (section 1.3).

In the lane, D comes from `bb_circuits.morphing_schedule` and `bb_circuits.BB`.

### 1.3 The circuit C_T
Write `c(s) := s mod 2` (so `c(-1) = 1`). For T >= 1, C_T is:

- **PREP.** Prepare every `q in L` in the +1 eigenstate of `b_q`. Noise `N_prep` (lane: `f_b` on every q in L).
- **ROUND t = 0, ..., T-1**, with `a := c(t-1)` and `c := c(t)`:
  1. **Reset** `M^X_a` in the X basis and `M^Z_a` in the Z basis. Noise `N_reset(a)` (lane: Z_q for q in `M^X_a`,
     X_q for q in `M^Z_a`, DEPOLARIZE1 on L).
  2. **Expansion.** For j = 0, 1, 2: apply `G_{a,2-j}`, then noise `N_exp(a, j)` (lane: `N(G_{a,2-j})`).
  3. **Contraction.** For k = 0, 1, 2: apply `G_{c,k}`, then noise `N_con(c, k)` (lane: `N(G_{c,k})`).
  4. **Measurement.** Noise `N_meas(c)`, then measure `M^X_c` in X and `M^Z_c` in Z. Lane placement: Z_q before
     the MX of `M^X_c`; X_q before the M of `M^Z_c` (which comes after the MX); DEPOLARIZE1 on L after both.
- **READOUT.** Noise `N_ro` (lane: `f_b` on every q in L), then measure every q in L in the b basis.

Here `N(G)` is DEPOLARIZE2 on every pair of G (its 15 non-identity two-qubit Paulis) plus DEPOLARIZE1 (X, Y, Z) on
every qubit outside G. In both lane circuits every layer covers all 2n qubits, so there is no idle noise.

The **mid-cycle point** `mu_t` is between steps 2 and 3 of round t. Under H1 the expansion of round t is `F_a^{-1}`,
the inverse of the contraction measured in round t-1. So around every measurement the circuit reads: contract
`F_r`, measure `M_r` | reset `M_r` in the same bases, expand `F_r^{-1}`. This is the palindrome.

**Records.** `m_s(q)` is the round-s outcome of `q in R`; `d(q)` is the readout outcome of `q in L`. For a b-type O,
`m_s(O; r) := XOR of m_s(q) over q in supp O^r_R`, and `d(O; r) := XOR of d(q) over q in supp O^r_L`.

### 1.4 Detectors and observables (closed form)
For a memory-type generator `S_g`, its **checkpoints** `s_1 < ... < s_m` are the rounds `s in [0, T-1]` with
`c(s) in Gamma_g`. Set `s_0 := -1`. C_T declares the following:

    D_{g,s_j} := XOR_{s_{j-1} < s <= s_j} m_s(S_g; c(s))                          coordinates (g, b, s_j),  j = 1..m
    D_{g,T}   := XOR_{s_m < s <= T-1} m_s(S_g; c(s))  XOR  d(S_g; c(T-1))          coordinates (g, b, T),   iff s_m != T-1
    Obs_k     := XOR_{0 <= s <= T-1} m_s(L_k; c(s))   XOR  d(L_k; c(T-1))

**Windows.** The window of `D_{g,s_j}` is the integer interval `(s_{j-1}, s_j]`, and the window of `D_{g,T}` is
`(s_m, T-1]`. The windows of g partition `{0, ..., T-1}`. `W_g(t)` denotes the coordinate round of the detector of
g whose window contains t.

**The b'-type detectors** are the lane's analogous parities for the b'-type generators. They are emitted only once a
generator's value is known, which is why at round 0 some are missing. The proof uses one property of them: each is a
parity of outcomes of qubits measured in the b' basis.

Appendix A proves, by reading the code, that `bb_circuits.morphing_cycle(kind, basis, T)` emits exactly C_T with these
detectors for every T. `fold_uniform.py` confirms this instruction for instruction and detector for detector at
T in {1, 2, 3, 4, 14, 15, 16, 17, 31, 32}.

### 1.5 Faults, symptoms, models, distances
A **noise component** is a pair (location l, Pauli A) taken from the component set of the channel at l. Its
**symptom** is the set of detectors and observables it flips, given by the standard Pauli-frame rule:

> **(F)** Consider a circuit of Clifford unitaries, single-qubit resets and single-qubit Pauli measurements, and a
> parity Q of outcomes that is deterministic without noise. Inserting A at l flips Q iff it flips an odd number of
> Q's outcomes. The outcome of measuring `B_q` is flipped iff the *frame* anticommutes with `B_q` just before that
> measurement. The frame is A carried forward from l: conjugated through every unitary, its q-part deleted at every
> reset of q, and left unchanged by measurements.

stim's error analysis and the lane's backward pass (`dem_independent.pauli_components`) both compute (F). They agree
exactly on every model compared, including T = 16, 17, 31 and 32 here (`records/fold_uniform_stim_vs_backward.jsonl`).

Symptoms are additive. The **full model** `M_T` is the set of distinct nonzero full symptoms. The **projected model**
`Proj_T` is the set of distinct nonzero (b-type detector part, observable part) of the component symptoms; this is
what `export_model.project_same_type` makes of the DEM.

`d_circ(T)` (respectively `d_proj(T)`) is `min |Y|` over subsets Y of `M_T` (respectively `Proj_T`) whose detector
parts sum to 0 and whose observable parts do not sum to 0. Such a set is "undetectable and logical"; the minimum is
infinity if none exists.

`d_circ` counts DEM MECHANISMS; Shaw & Terhal count FAULTS. The two minima are equal here. A set of faults
produces at most as many mechanisms as it has members (equal symptoms cancel in pairs). Conversely, a set of
mechanisms is realized by at most as many faults: one component per mechanism, and two components at one location
combine into their product, which that location's channel also contains, because every noise location is a single
Pauli or a channel whose component set is closed under products (DEPOLARIZE1/2; the boundary flips are single
Paulis).

## 2. The hypotheses (finite, T-independent)

| name | statement | read from | negative control (must fail; does) |
|---|---|---|---|
| **H1** involutive layers | in every `G_{r,k}`, distinct pairs use distinct qubits | schedule | a layer reusing a qubit |
| **H2** all measured, all reset | `M^X_r u M^Z_r = R` disjointly, r = 0, 1 | schedule | one right qubit never measured |
| **H3** trackability | for every `S^X_h`, every `S^Z_g`, every `L_k`, r = 0, 1: `supp F_r(O) n R` lies in `M^{type(O)}_r` | schedule + code | a contracted qubit moved to the other basis |
| **PAL** palindrome | for r = 0, 1: images (Lemma 2) of `N_exp(r, j)` equal those of `N_con(r, 2-j)`, j = 0, 1, 2; images of `N_reset(r)` equal those of `N_meas(r)` | C_2 (round 0: a = 1, c = 0; round 1: a = 0, c = 1) | expansion not the reverse of the contraction |
| PAL root | each layer's noise set is invariant under conjugation by that layer | C_2 | one layer's noise keeps only X on controls |
| **BND** boundaries | images of `N_prep` lie among those of `N_reset(1)`; images of `N_ro` after a round with `c(T-1) = r` lie among those of `N_meas(r)` | C_1, C_2 | correlated two-qubit readout flips |
| **PURE** pure parts | at every location: if A is a component and `pi(A) != I`, then `pi(A)` is a component, where `pi(A)` is the `f_b`-part of A (X part in memory-Z, Z part in memory-X) | C_1, C_2 | one location loses its pure component |

**Consequences the proof uses, also checked directly:**

- **COV**: the image set of C_2 lies inside that of C_1. Checked both as Paulis and as signatures.
- **TWIN**: the image set of C_1 lies inside the round-0 image set of C_2.
- Controls for these: the hom-y 1-round table; C_1 without its contraction noise; correlated readout flips.

**Not a hypothesis.** The proof does not use the condition "each generator contracts in exactly one parity"
(`|Gamma_g| = 1`, windows of at most 2 rounds). It holds here (INFO in the script, with its own control), but the
argument works for any `Gamma_g` contained in {0, 1}.

Each hypothesis is a statement about D and about the noise of one round body of each parity plus the two
boundaries. All of these are contained in C_1 and C_2, so nothing in the list depends on T.

## 3. Lemmas

**Lemma 0 (determinism).** Assume H1-H3. In the noiseless C_T, let O be any b-type operator among the `S_g` and the
`L_k`. There are bits `v_0, ..., v_{T-1}` (functions of the outcomes) such that:

- `v_0 = 0`;
- the state at `mu_s` is an eigenstate of O with eigenvalue `(-1)^{v_s}`;
- `v_{s+1} = v_s XOR m_s(O; c(s))` for s <= T-2;
- `d(O; c(T-1)) = v_{T-1} XOR m_{T-1}(O; c(T-1))`;
- if `c(s) in Gamma_O`, then `m_s(O; c(s)) = v_s` and `v_{s+1} = 0`.

Hence every b-type detector and every `Obs_k` is 0 with certainty.

*Proof.*

*Base case.* Before round 0's expansion the state is a +1 eigenstate of `b_q` for every q in L (preparation) and for
every q in `M^b_1` (round 0 resets `M^X_1` in X and `M^Z_1` in Z). By H3 and (CSS), `F_1(O) = O^1_R O^1_L` is a
+-signed b-type product supported on `L u M^b_1`. So the state is a +1 eigenstate of `F_1(O)`, and after the
expansion `F_1^{-1}` (H1) a +1 eigenstate of O. Thus `v_0 = 0`.

*Step.* At `mu_s` the state is an eigenstate of O with eigenvalue `(-1)^{v_s}`; let c := c(s).

- After the contraction it is an eigenstate of `F_c(O) = O^c_R O^c_L`.
- `O^c_R` is a product of `b_q` with q in `M^b_c` (H3). So `F_c(O)` commutes with every round-s measurement, and
  afterwards the state is still its eigenstate with the same eigenvalue. It is also an eigenstate of each measured
  `b_q` with eigenvalue `(-1)^{m_s(q)}`. Hence it is an eigenstate of `O^c_L` with eigenvalue
  `(-1)^{v_s XOR m_s(O; c)}`.
- If s = T-1, the readout measures `b_q` on L, which gives the d-identity.
- If s <= T-2, round s+1 resets all of R (H2), in the same bases. This leaves `O^c_L` alone and puts every q in
  `M^b_c` into the +1 eigenstate of `b_q`. So the state is an eigenstate of `F_c(O)` with eigenvalue
  `(-1)^{v_s XOR m_s}`, and after `F_c^{-1}` an eigenstate of O. This gives the recursion.
- If `c in Gamma_O`, then `F_c(O) = O^c_R`, whose eigenvalue the round-s outcomes reveal. So `m_s = v_s` and
  `v_{s+1} = 0`.

*Values of the parities.*

- `Obs_k = XOR_{s<=T-2}(v_s XOR v_{s+1}) XOR m_{T-1} XOR (v_{T-1} XOR m_{T-1}) = v_0 = 0`.
- For `D_{g,s_j}`: `v_{s_{j-1}+1} = 0` (the round after a checkpoint; `v_0 = 0` when j = 1) and `m_{s_j} = v_{s_j}`.
  So `D_{g,s_j} = XOR_{s_{j-1} < s < s_j}(v_s XOR v_{s+1}) XOR v_{s_j} = v_{s_{j-1}+1} = 0`.
- `D_{g,T} = v_{s_m+1} = 0` in the same way. ∎

**Lemma 1 (one round of a frame).** Assume H1, H2. Let the only fault be a Pauli A at `mu_s`. Let c := c(s) and
write `F_c(A) = A_R . A_L` (its restrictions to R and L). Then:

- no outcome of a round before s changes;
- in round s, the outcome of q in `M^Z_c` (respectively `M^X_c`) flips iff `A_R` has X or Y (respectively Z or Y) on q;
- if s <= T-2, the frame at `mu_{s+1}` is `Pi_c(A) := F_c^{-1}(A_L)`;
- if s = T-1, the readout outcome of q in L flips iff `A_L` anticommutes with `b_q`.

In particular, for a b-type O that satisfies H3, the parity `m_s(O; c)` flips by `<A_R, O^c_R>`.

*Proof.* By (F). Between `mu_s` and the round-s measurements there are only the contraction layers, so the frame there
is `F_c(A)`; measurements leave it unchanged. If s <= T-2, the next operations on R are round s+1's resets. They act
on all of R (H2) and delete `A_R`. The expansion `F_c^{-1}` (H1) then gives `F_c^{-1}(A_L)`. If s = T-1, nothing
acts on L before the readout. The last sentence holds because `O^c_R` is the product of `b_q` over q in `M^b_c`. ∎

**Lemma 2 (localization).** Assume H1. For a component A at a location of round t (prep: t = 0; readout: t = T-1),
its **image** `I(A)` at `mu_t` is:

| phase | location | image |
|---|---|---|
| prep, reset | before round t's expansion, with a = c(t-1) | `Phi_{a,2}(A)`, i.e. forward through `F_a^{-1}` |
| expansion | after expansion layer j (the layer `G_{a,2-j}`) | `Phi_{a,1-j}(A)`, forward through the remaining layers |
| contraction | after `G_{c,k}` | `Phi_{c,k}(A)`, backward through `G_{c,k}, ..., G_{c,0}` |
| measurement | round t, c = c(t) | `Phi_{c,2}(A)`, backward through all of `F_c` |
| readout | c = c(T-1) | `Phi_{c,2}(A)`, backward through all of `F_c` |

A at its location and `I(A)` at `mu_t` have the same symptom.

*Proof, forward cases.* Between the location and `mu_t` there are only expansion layers, plus for prep the round-0
resets of R, which do not touch A (A lies on L). Carrying A forward gives `I(A)` at `mu_t` with nothing measured in
between, so by (F) the two have the same symptom.

*Proof, backward cases.* Carry `I(A)` forward from `mu_t`.

- Contraction location: passing `G_{c,0}, ..., G_{c,k}` gives A, because the layers are involutions.
- Measurement and readout locations: passing all of `F_c` gives A.
  - The X_q on `M^Z_c` sits after the MX of `M^X_c`, which does not contain q.
  - DEPOLARIZE1 on L, and the readout flips, sit after round-c measurements that act on R only.
  - So the frame arrives at the location unchanged and having flipped nothing on the way. From there on, the two
    faults have the same frame. ∎

**Lemma 3 (exact symptom formula).** Assume H1-H3. Let the only fault be a Pauli P at `mu_t`. Then:

- **(a)** P flips the b-type detector `D_{g,w}` iff `w = W_g(t)` and `<P, S_g> = 1`;
- **(b)** P flips `Obs_k` iff `<P, L_k> = 1`.

So the projected symptom of P is `(iota_t(sigma_S(P)), sigma_L(P))`, where:

- `sigma_S(P) := (<P, S_g>)_g in F_2^n`;
- `sigma_L(P) := (<P, L_k>)_k in F_2^K`;
- `iota_t(e_g) := e_{(g, W_g(t))}`.

*Proof.* Fix O in `{S_g} u {L_k}`.

*Setup.* Frames: `P_t := P` and `P_{s+1} := Pi_{c(s)}(P_s)` (Lemma 1). Bits: `a_s := <P_s, O>` for t <= s <= T-1,
and `a_T := <(F_{c(T-1)}(P_{T-1}))_L, O^{c(T-1)}_L>`.

*Claim.* For t <= s <= T-1, `m_s(O; c(s))` flips by `a_s XOR a_{s+1}`; it does not flip for s < t; and
`d(O; c(T-1))` flips by `a_T`.

*Proof of the claim.* Let c = c(s) and `F_c(P_s) = A_R A_L`. By (SYM),
`a_s = <F_c(P_s), F_c(O)> = <A_R, O^c_R> XOR <A_L, O^c_L>`.

- The first term is the flip of `m_s(O; c)` (Lemma 1).
- The second term is `a_{s+1}`. For s <= T-2, `a_{s+1} = <F_c^{-1}(A_L), O> = <A_L, O^c_R O^c_L> = <A_L, O^c_L>`,
  by (SYM) and because `A_L` has no support on R. For s = T-1 it is the definition of `a_T`.
- The readout part is Lemma 1.

Moreover:

> **(\*)** if s >= t and `c(s) in Gamma_O`, then `a_{s+1} = <A_L, I> = 0`.

*(b).* The flip of `Obs_k` is `XOR_{s=t}^{T-1}(a_s XOR a_{s+1}) XOR a_T = a_t XOR a_T XOR a_T = a_t = <P, L_k>`.

*(a).* Take a detector of g with window (u, v], where v = s_j, or v = T-1 for the final detector. Its flip is the sum
of `a_s XOR a_{s+1}` over s in (u, v] with s >= t, plus `a_T` for the final detector.

- **t > v:** the sum is empty. (The final detector's window contains T-1 >= t, so this case only arises for
  checkpoint detectors.)
- **u < t <= v:** the sum telescopes to `a_t XOR a_{v+1}` (plus `a_T` for the final). For a checkpoint detector,
  `a_{v+1} = 0` by (\*). For the final detector, `a_{v+1} = a_T` cancels the extra `a_T`. Either way the flip is
  `a_t = <P, S_g>`.
- **t <= u:** then u is a checkpoint >= t. The sum telescopes to `a_{u+1} XOR a_{v+1}` (plus `a_T` for the final),
  which is 0 by (\*) at u and at v (for the final, `a_{v+1} = a_T` cancels).

Exactly one window contains t, the one at coordinate `W_g(t)`. ∎

*Remark (T = 1).* Every g has exactly one b-type detector: the checkpoint at round 0 if 0 is in `Gamma_g`, otherwise
the final detector. So `iota_0` identifies `F_2^n` with the 1-round detector space, and the projected symptom of a
fault at `mu_0` is its signature `sigma(P) := (sigma_S(P), sigma_L(P))`.

**Lemma 4 (every mid-cycle point sees the same fault images).** Assume H1, PAL, BND.

- Let `K^exp_r` be the image set of the reset and expansion components of a round with `c(t-1) = r`.
- Let `K^con_r` be the image set of the contraction and measurement components of a round with `c(t) = r`.
- These depend only on r, since a round body depends only on `(c(t-1), c(t))`. PAL says `K^exp_r = K^con_r =: K_r`.
- Let `K := K_0 u K_1`.

Then for every T >= 1 and every t <= T-1, the image set of the components of C_T located at `mu_t` is exactly K.
These components are round t's body, plus prep if t = 0, plus readout if t = T-1.

*Proof.* Round t's body contributes `K_{c(t-1)} u K_{c(t)} = K_0 u K_1`, because `{c(t-1), c(t)} = {0, 1}`. Prep
adds nothing new (BND: it lies inside `K_1`). Readout adds nothing new (BND: it lies inside `K_{c(T-1)}`). ∎

*Why PAL holds for the lane's noise (the root).* `N(G)` is invariant under conjugation by G: G permutes the 15
non-identity Paulis of each of its pairs and fixes the single-qubit Paulis off G.

- The expansion location after `G_{r,k}` (j = 2-k) has image set
  `Phi_{r,k-1}(N(G_{r,k})) = Phi_{r,k-1}(G_{r,k}(N(G_{r,k}))) = Phi_{r,k}(N(G_{r,k}))`, which is the image set of the
  contraction location after `G_{r,k}`.
- Reset and measurement noise of parity r are the same Pauli set, with the same map `Phi_{r,2}`.

*Why BND holds for the lane's noise.* The prep and readout flips `f_b` on L are among the DEPOLARIZE1 components on L
in the reset (respectively measurement) phase, with the same map.

**Lemma 5 (pure parts).** For any Pauli A at any location:

- **(i)** A and `pi(A)` flip the same b-basis outcomes;
- **(ii)** `pi(A)` flips no b'-basis outcome.

*Proof.* By (CSS), the X and Z parts of a frame evolve independently through CNOTs. Resets delete both parts on a
qubit, and measurements leave the frame unchanged. So the frame of `pi(A)` is always the `f_b` part of A's frame and
has no b' part. ∎

*Consequence under H3.* The b-type detectors and the observables read only b-basis outcomes, and the b'-type detectors
read only b'-basis outcomes. So `pi(A)` has A's projected symptom and touches no b'-type detector.

## 4. Theorem 1 and its proof

**Theorem 1.** Assume H1-H3, PAL, BND and PURE. Then for every T >= 1:

- **(A)** `d_proj(T) >= d_proj(1)`. Uses H1-H3, PAL, BND.
- **(B)** `d_proj(T) <= d_proj(1)`. Uses H1-H3 and BND only.
- **(C)** `d_circ(T) = d_proj(T)`. Uses H1-H3 and PURE.

Hence `d_circ(T) = d_proj(T) = d_proj(1) = d_circ(1)`.

*Proof.* By Lemma 2, every component of C_T has the symptom of its image P at its mid-cycle point t. By Lemma 3 that
symptom is `(iota_t(sigma_S(P)), sigma_L(P))`.

Let `kappa` send each b-detector `e_{(g,w)}` of C_T to `e_g`; it adds up each generator's detectors. Then
`kappa o iota_t = id` for every t, and `iota_0` is injective.

*(A).* If `Proj_T` has no undetectable logical set, `d_proj(T)` is infinite and there is nothing to prove. Otherwise
let Y be an undetectable logical subset of `Proj_T`.

1. For each y in Y, pick a component that realizes y, with image `P_y` at `mu_{t_y}`. Then
   `y = (iota_{t_y}(sigma_S(P_y)), sigma_L(P_y))`.
2. Define `phi(y) := (kappa(y_det), y_obs)`. This equals `sigma(P_y)`, which is nonzero because y is.
3. `phi(y)` is in `Proj_1`: by Lemma 4, `P_y` lies in K, which is the image set of C_1. So some component of C_1 has
   image `P_y`, and by the T = 1 remark its projected symptom is `sigma(P_y)`.
4. `phi` is linear, so `sum phi(y) = (kappa(sum y_det), sum y_obs) = (0, nonzero)`.
5. Cancel equal pairs in the list `(phi(y))_{y in Y}`. What remains is a set Y' contained in `Proj_1`, with
   `|Y'| <= |Y|` and the same sum. Y' is undetectable and logical, so `d_proj(1) <= |Y|`.

*(B).* Let Y' be an undetectable logical subset of `Proj_1`.

1. For each y' in Y', pick a component of C_1 that realizes it, with image P at `mu_0`. Then `y' = sigma(P)`.
2. The image set of C_1 is: prep, plus `K^exp_1`, plus `K^con_0`, plus the readout after round 0. By BND it lies
   inside the round-0 image set of C_T, which is prep plus `K^exp_1` plus `K^con_0`. (For T >= 2, round 0 of C_T is
   the same body as round 0 of C_1.) So some component of C_T at `mu_0` has image P.
3. That component's projected symptom is `psi(y') := (iota_0(sigma_S(P)), sigma_L(P))`.
4. `psi` is linear and injective. So the `psi(y')` are |Y'| distinct mechanisms of `Proj_T` with sum
   `(0, nonzero)`, and `d_proj(T) <= |Y'|`.

*(C), the bound `d_circ(T) >= d_proj(T)`.* Take a full undetectable logical set. Project its members, drop the ones
that become zero and cancel equal pairs. The observable parts are untouched and still sum to nonzero, so the result is
a non-empty undetectable logical set in `Proj_T`, no larger than the original.

*(C), the bound `d_circ(T) <= d_proj(T)`.* Let Y be an undetectable logical subset of `Proj_T`, and for each y in Y
pick a component A that realizes y.

- `pi(A) != I`: otherwise, by Lemma 5(i), A's projected symptom would be 0.
- By PURE, `pi(A)` is a component at the same location. By Lemma 5 it has projected symptom y and touches no b'-type
  detector.
- So the full symptoms of these components are |Y| distinct mechanisms of `M_T`, summing to `(0, nonzero)`. ∎

**Corollary 2 (structure of every T-round model).** Under H1-H3, PAL and BND, the T-round projected model is the union
of the shifted 1-round models:

    Proj_T  =  union over t = 0..T-1 of  { (iota_t(s_S), s_L) : (s_S, s_L) in Proj_1 }

where `Proj_1` is identified with the nonzero signatures of K. The detector count is
`sum over g of (#checkpoints + [no checkpoint at round T-1])`.

So the whole T-round model, and its size, are functions of the 1-round model and the `Gamma_g` alone (Lemmas 2-4).

**Corollary 3 (the lane).**

- **morph144, both bases.** At T = 1, the committed `ccheck` VERIFIED "no undetectable logical error of weight <= 11"
  on a proof re-written by the committed prover in the 2026-11-02 re-certification (`recert_20261102/manifest.json`;
  the original record, `morph144_certificates_w11.jsonl`, was checked by the checker of its day; model sha256
  `4c21e750...` for Z and `f3e9e6af...` for X, byte-identical in both). So
  `d_proj(1) >= 12`. The recorded weight-12 witnesses live in the T = 1 projected model (`morph144_witnesses.jsonl`,
  `t1_projected`), so `d_proj(1) <= 12`. By Theorem 1, **d_circ(T) = 12 for every T >= 1, memory-Z and memory-X**.
- **morph72, both bases.** The T = 1 time-shift certificate, re-proved and VERIFIED by the committed prover and
  checker in the 2026-11-02 re-certification (`recert_20261102/manifest.json`), excludes every irreducible
  undetectable logical error of weight <= 5 with earliest round <= 1. In the 1-round model every mechanism's detectors
  lie in rounds 0 and 1, so it excludes every irreducible -- hence every minimum-weight -- solution of weight <= 5:
  `d_proj(1) >= 6`. The recorded weight-6 full witnesses at T = 2 give `d_proj(2) <= 6`, and (A) then gives
  `d_proj(1) <= 6`. So **d_circ(T) = 6 for every T >= 1**.

## 5. The checks: `fold_uniform.py` (run log `records/fold_uniform_hom_x.jsonl`)

`python fold_uniform.py --template-T 1 2 3 4 --fresh-T 14 15 16 17 31 32` (single core, 83 s) gives the
following for morph72 and morph144, memory-Z and memory-X:

- **Hypotheses:** H1, H2, H3 (all 2n generators and the 12 logicals, both parities), PAL, PAL root, BND and PURE:
  **PASS** in all 4 runs.
- **Consequences:** COV and TWIN, each at the Pauli level and at the signature level: **PASS** in all 4 runs.
- **Image sets:**
  - morph72: `|K| = 4320` Paulis and 433 signatures (432 nonzero, equal to the recorded 1-round model size).
  - morph144: `|K| = 8640` Paulis and 865 signatures (864 nonzero, equal to the recorded model size).
- **INFO:** every generator of both types contracts in exactly one parity.
- **Template identification** at T = 1, 2, 3, 4, 14, 15, 16, 17, 31, 32 (4 runs x 10 T): **all agree**.
  - The instruction skeleton is identical to the template's.
  - The b-type detectors and the observables equal the closed form of section 1.4.
  - Every b'-type detector reads b'-basis records only.
  - **Every component's projected symptom equals Lemma 3's formula**, against the lane's independent backward-pass
    masks: 0 mismatches over 5,724,000 components, up to 451,872 per model at T = 32.
  - The pure lift works for every component.
  - Every image lies in K.
  - The model equals Corollary 2's union exactly. For example, morph144 T = 32 has 23,184 mechanisms.
- **Cross-check against the per-T fold rows** in `records/fold_lemma.jsonl` (52 rows, both circuits, both bases,
  T = 1..13): **52/52 agree** in every field (the 4 rows at T = 1 fold a model onto itself, so their consistency
  fields cannot fail; their count predictions can). The theorem predicts:
  - `components = 2n + 98nT`;
  - `mapped = 2n + 73nT` (per round body, 73n of 98n components have nonzero signature);
  - `missing = 0` (from Lemma 4, since the keys are parts of Paulis in K);
  - `conflicts = 0` (from the T = 1 remark: a 1-round symptom is a function of the key);
  - `consistent` (the map is kappa plus 0, by Lemma 3);
  - `holds`.
- **Cross-check against every recorded exported projected model** (morph72 T = 1..7, morph144 T = 1, 2, both bases;
  18 rows): the mechanism and detector counts predicted from the 1-round model alone (Corollary 2) **agree 18/18**.
  Examples: 432/72, 792/108, ..., 2592/288; 864/144, 1584/216.
- **Negative controls:** 20 per run, all **fail as required** in all 4 runs.
  - H1: a layer reuses a qubit.
  - H2: a right qubit is never measured.
  - H3: a contracted qubit is moved to the other basis.
  - PAL: the expansion is not the reverse of the contraction.
  - PAL root: one layer's noise keeps only X on the controls.
  - BND: correlated readout flips.
  - TWIN: correlated readout flips, at the Pauli level and at the signature level.
  - COV: the hom-y 1-round table (Pauli level); C_1 without its contraction noise (signature level).
  - PURE: one location loses its pure component.
  - Template: a detector loses one record (closed form, and symptom formula); an observable loses one record; a
    b'-type detector also reads a b-basis record; one CNOT is removed (skeleton, and symptom formula).
  - INFO: one CNOT is removed from `F_0`.
  - Records: one fold row's component count is off by one; one recorded model size is off by one.
- **Other homomorphisms.** `--hom y` and `--hom xy`, at T = 1..4: **PASS**, with all 18 applicable controls failing
  as required (the two record controls apply to hom x only; `records/fold_uniform_hom_y.jsonl`,
  `fold_uniform_hom_xy.jsonl`). The theorem therefore covers all three morphing schedules, not only the lane's.

**Further corroboration, outside the theorem's premises:**

- `records/fold_uniform_stim_vs_backward.jsonl`: stim's DEM equals the backward-pass mechanism set at T = 17, 16, 32
  and 31 (88k-96k mechanisms each).
- `fold_twin.py --T 32`: the lane's weight-12 1-round witnesses, sent through `iota_0` (Theorem (B)), are
  undetectable and logical in the T = 32 projected model. Their pure-part lifts (Theorem (C)) are undetectable logical
  full-DEM sets of weight 12 at T = 32, in both bases. The control with one mechanism dropped fails
  (`records/fold_uniform_witness_twin_T32.jsonl`).

## 6. Adversarial notes: where could it fail, and why it does not

1. **The boundary at even T.** When T is even, the readout follows an odd round. Its images
   `F_1^{-1}(f_b on L)` are not readout images of C_1. They are covered only because they equal measurement-noise
   images of an odd round, and those equal reset-noise images of C_1's round 0 (BND plus PAL). If the readout channel
   had something with no twin, the check would fail; the correlated-readout control shows exactly that (BND and TWIN
   fail). That there is no counterexample at any T is Theorem 1's conclusion; the tests probe its finite
   hypotheses and sampled T, not every T. The exact formula holds on every component at the 10 T values tested,
   and on the 52 recorded rows.
2. **b'-type generators start unknown.** The b'-type generators that contract in round 0 have no value at `mu_0`, so
   they lack a first detector, and their detectors do not telescope. This is why the fold is stated for the projected
   model. Part (C) reaches `d_circ` through PURE, not through the b'-detectors.
3. **Meta-checks.** The lane omits the n - rank(H) = 6 meta-checks on round-0 b'-type outcomes
   (`tools_cde.deterministic_gap`). Adding them changes nothing:
   - they are parities of b'-basis records, so the pure lifts of (C) flip none of them;
   - extra detectors can only remove solutions from the lower bound.

   So Theorem 1 also holds in the convention with meta-checks. That convention is closer to that of Shaw & Terhal,
   though theirs also has ideal preparation and readout.
4. **Relation to Shaw & Terhal, "Optimising Quantum Error Correction Using Morphing Circuits", arXiv 2604.09797,
   Prop. 1 and App. B.**
   - Their setting: ideal preparation and readout of an end-cycle code; regular and meta-check detectors of duration
     one and two; folding by repeated time-reversal of one round of errors at a time.
   - This proof: noisy boundaries, no meta-checks, the projected model, and a single explicit formula (Lemma 3) with
     a single linear map kappa instead of a sequence of reflections.
   - Both reach `d_circ(T) = d_circ(1)`.
5. **What is not assumed.** "Each stabilizer contracts exactly once per two rounds", and windows of at most 2 rounds.
6. **What the theorem does not cover.**
   - Noise that breaks PAL, for example an expansion layer whose noise differs from its mirrored contraction layer.
   - Noise that breaks PURE, for example Y-only channels.
   - Correlated noise across locations, leakage, and non-Pauli noise.
   - Rounds that do not measure and reset every right qubit (H2).

   For any such variant, rerun the script; the controls show that each condition can fail.

## 7. Uncertainty flush
- **(F), the Pauli-frame rule, is cited, not re-proved.** It is the standard Heisenberg-picture propagation that
  stim's error analysis and the lane's backward pass implement. The two agree on every model compared, here up to
  T = 32.
- **Theorem 1 is a hand proof.** It has no formal (Lean) version. One independent adversarial pass (section 8)
  attacked Lemma 0, Lemmas 1-4, the full/projected reduction and the T-independence of the hypotheses, and refuted
  nothing.
- **Builder and template.** Theorem 1 is about C_T. It reads the builder only at T = 1 (where the hypotheses are read
  and the certified model is exported) and T = 2; both are checked mechanically. That `morphing_cycle` emits C_T for
  every T is Appendix A, a proof by code reading, confirmed mechanically at 10 values of T up to 32.
- **Corollary 3 inherits the trusted base of the T = 1 certificates**: the soundness of `ccheck.c` (its certificate
  calculus is PROVEN in Lean, `lean/CertR198`; that the C implements that calculus is argued, `SPEC.md` gap G1), the
  exporter's projection, and the witness verification. The audit re-exported all four 1-round models from `main`:
  their sha256 equal the certificate records'.
- **Pinned inputs**, read on 2026-10-21 at ONE_Q HEAD `743940840771`:

  | file | git blob |
  |---|---|
  | `bb_circuits.py` | `bfa38bb2` |
  | `fold_lemma.py` | `7280bc0d` |
  | `dem_independent.py` | `435d84a4` |
  | `export_model.py` | `3e2b8752` |
  | `records/fold_lemma.jsonl` | `a9026d63` |

## 8. Independent adversarial review (2026-10-22): CONFIRMED

An adversary agent that did not write this proof attacked Theorem 1 from five directions; none broke it. Its
outputs are archived as `records/fold_review_*.jsonl`; nothing it ran wrote into the lane.

- **Lemma 0 (determinism).** Each step re-derived against the builder (the round-0 resets of `M^X_1 u M^Z_1`, H3
  for the base case, the reset bases of round s+1, the checkpoint clause and both telescopes). Independently, stim
  built all 30 DEMs it was asked for, and stim refuses a DEM with a non-deterministic detector.
- **Lemmas 1-3 (the symptom formula, `iota_t`, the collapse).** All three window cases by hand (including `t <= u`,
  the final detector's cancellation and generators whose first checkpoint is round 1); every boundary (preparation,
  the round-0 reset noise, the measurement-noise ordering, the readout); faults on qubits measured in the other basis
  (zero projected symptom, as predicted); `kappa . iota_t = id`.
- **Corollary 2 against stim, sharing none of the proof's machinery** (`fold_review_cor2.jsonl`): the T-round projected model
  read from stim's DEM equals `{(iota_t(s_S), s_L) : t < T, (s_S, s_L) in Proj_1}` built from stim's 1-round DEM and
  the 2-round circuit's detector coordinates alone, at 18 fresh points: morph72-Z T = 5, 6, 9, 10, 20, 33; morph72-X
  T = 5, 8, 11; morph144-Z T = 5, 6, 7; morph144-X T = 5, 8; morph72-Z hom y T = 6, 7; morph144-X hom xy T = 5, 6.
  Both controls disagree as required (the window parity flipped; one round's expansion noise deleted: 1800 vs 1872
  for morph72, 3600 vs 3744 for morph144). The suite now runs this check at two points with both controls
  (`test_fold_corollary2_matches_stim_at_fresh_T`).
- **Full versus projected (part C, Lemma 5).** The projection only relaxes; the pure lift needs only CNOTs, H3 and
  PURE. At T = 1 there are no other-type detectors, so the full and projected 1-round models are the same 864
  mechanisms and the T = 1 certificates are on the full model.
- **T-independence of the hypotheses.** A round body depends only on the parities of the previous and current
  round; the round number enters only through record indices, coordinates and running record sets. C_2 holds both
  bodies; C_1 has the readout after an even round, C_2 after an odd one; preparation is the same in every C_T.
- **The checker's controls are live.** The set comparisons of PAL, BND and PURE would pass vacuously on a misnamed
  phase tag (empty-set defaults); each matching control fails, which shows the sets are populated.
- **Distribution shift** (`fold_review_b_n18.jsonl`, `fold_review_b_n18ctl.jsonl`, `fold_review_b_n36.jsonl`): 8 small bivariate-bicycle morphing codes
  through the unmodified builder, never seen by the proof. All six hypotheses pass; exact distances from stim's DEM
  are constant in T (n = 18: d = 3 for T = 1..6 in three codes, d = 2 in one; n = 36: d = 4 for T = 1..4 on t12x3,
  d = 3 on t6x6 for homs x, y, xy) and d_circ = d_proj in every case. Sensitivity control: correlated flips added to
  the resets of odd rounds only break PAL and give d(1) = 3 but d(T >= 2) = 2 (and 4 -> 2 on t12x3), so the
  instrument sees a T-dependence when a hypothesis fails.
- **Corollary 3's binding.** All four certified 1-round models re-exported from `main` hash to the recorded sha256;
  both weight-12 witnesses re-verify from stim's DEM and no single deletion leaves a solution.

Its four narrower findings are fixed in this file: section 6's "no counterexample at any T" now reads as the
theorem's conclusion; the 52/52 notes its 4 rows that cannot fail; section 1.5 states why counting mechanisms equals
counting faults here; and Corollary 3's evidence class names Appendix A's code-reading premise. Not run: whether
deleting preparation and readout noise leaves `Proj_T` unchanged (BND predicts it), and the small codes with d = 5
and 6 at T >= 2 (an exact search too heavy in Python).

## Appendix A: `bb_circuits.morphing_cycle` emits C_T for every T
The code was read at blob `bfa38bb2`, lines 402-524.

- **Splits** (`carry`, lines 429-438). Computed once per (operator, r) and cached in `split`; independent of T and of
  the round. The assertion at line 433 says carried operators never touch the other type's measured qubits. Together
  with H2 this is H3, and it gives b'-type detectors that read b'-basis records only.
- **PREP** (lines 451-453): `R` or `RX` on L, the flip noise, `TICK`.
- **Loop** (lines 461-504), `for i in range(rounds)` with `prev, cur = (i-1)%2, i%2`:
  - resets of `meas[X][prev]` (RX) and `meas[Z][prev]` (R), their flips, DEPOLARIZE1 on L (463-468);
  - the six layers `reversed(gates[prev]) + gates[cur]`, each followed by DEPOLARIZE2 on its pairs and DEPOLARIZE1
    on idle qubits (469-477);
  - measurement noise, MX, M, DEPOLARIZE1 on L (479-489);
  - detectors (490-500);
  - logical bookkeeping (501-503).

  The instructions emitted in iteration i depend on i only through `(prev, cur)`, the record counter, the coordinate
  i, and the dictionaries `val` and `lval`. This is the body of round i of C_T.
- **Invariant** (induction on i). Before iteration i:
  - `val[(b, g)]` is the set of records of `m_s(S_g; c(s))` for s in (g's last checkpoint before i, i-1]. Its
    initial value is the empty set for the memory type (line 458).
  - `lval[k]` is the set of records of `m_s(L_k; c(s))` for s <= i-1.

  Iteration i sets `now = val XOR rec_i(same)`. If `rest` is empty, that is `c(i) in Gamma_g`, it emits
  `DETECTOR(now)` at `(g, b, i)`, which is `D_{g,i}`, and resets `val` to the empty set. Otherwise it sets
  `val = now`.
- **Readout** (lines 505-519): flip noise, then M or MX on L. For every g whose last-round rest is non-empty it emits
  `val XOR data(rest)` at `(g, b, rounds)`, which is `D_{g,T}`. It then emits `lval XOR data(rest of L_k)`, which is
  `Obs_k`.

Mechanical confirmation: `template_check` compares the skeleton, the closed-form detectors and the observables at
T in {1, 2, 3, 4, 14, 15, 16, 17, 31, 32}, for all four (circuit, basis) pairs.

## Appendix B: reproduce
```
python fold_uniform.py --template-T 1 2 3 4 --fresh-T 14 15 16 17 31 32   # exit 0 = all PASS (-> fold_uniform_hom_x.jsonl)
python fold_uniform.py --hom y                                           # -> fold_uniform_hom_y.jsonl
python fold_uniform.py --hom xy                                          # -> fold_uniform_hom_xy.jsonl
python fold_twin.py --T 32                                               # -> fold_uniform_witness_twin_T32.jsonl
python -m pytest ../../tests/test_r198_circuit_distance_exact.py -k fold  # light form + Corollary 2 vs stim
```
From `experiments/circuit_distance_exact/`. The archived JSON rows are under
`evidence/circuit_distance_exact/records/` as `fold_uniform_*.jsonl`.
