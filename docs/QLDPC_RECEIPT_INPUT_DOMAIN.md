# qLDPC receipt input contract

The exact and floating-screen Python entry points, standalone Python readers,
and JavaScript flat/tree readers share this input contract.
`conformance/qldpc_input_domain_vectors.json` contains synthetic specimens;
`tests/test_qldpc_input_domain.py` exercises all seven entry points.
Agreement across implementations is not an outside human review or a proof
that the specified noise model describes hardware.

## 1. Integers

An **integer slot** holds one of the following, and nothing else is accepted:

* **In JSON** (receipt files, `tree_checker_b.py` documents, JavaScript batches): a number token written as an integer
  literal, `-?(0|[1-9][0-9]*)`. These are refused: `1.0`, `1e0`, `true`, `"1"`, `null`, and any fraction.
  * **The `1.0` policy, and why.** JSON's grammar distinguishes `1` from `1.0` at the token level, but JavaScript's
    `Number` cannot. The receipts are written by Python's `json.dumps` from `int` values, which always emits integer
    literals. A received non-integer token is refused rather than reinterpreted as another value.
  * Python's `json` module reads `1.0` as a `float`, which the Python checkers refuse by type.
  * The JavaScript checkers read the token itself through `JSON.parse` source-text access (use Node 24). On an
    engine without it they refuse to run rather than guess.
  * Integer literals beyond `Number.MAX_SAFE_INTEGER` become exact `BigInt` values. They are legal for weights and
    multipliers, and as indices they are out of range.
* **In the Python API** (`check_qldpc*`): an object accepted by `operator.index` that is not a `bool`. That is `int` and
  NumPy integer scalars, which the producers pass. These are refused: `float` (even when integral), `str`, `bool`, and
  `numpy.bool_`. Bare `int()` is a lossy coercion (`int(0.9) == 0`) and is never applied to an unvalidated value.

## 2. The instance

* **Variables.**
  * JSON formats: the integers `0 .. n-1`, where `n` is the length of the weight vector (or the batch's `n`).
  * Python API: the keys of the `weights` mapping, each a non-negative integer (section 1).
* **Checks.** Each check is a list of distinct variables. An entry that is not an integer, or names a variable that has
  no weight, is refused. A duplicate entry is refused, because over GF(2) it cancels while as a set it does not.
* **Syndrome.**
  * Bit-vector form (Python API, `tree_checker_b.py`, the reference): one entry per check, each exactly the integer
    `0` or `1`. `2`, `-1`, `True`, `1.0` and `"1"` are refused, never reduced modulo 2.
  * Support form (JavaScript batches and the receipt files): a list of distinct row indices in `0 .. m-1`.
* **Weights.**
  * Exact checkers: non-negative exact rationals.
    * Python API: `int` or `fractions.Fraction`.
    * Reference: `{"num", "den"}`, in lowest terms, with `den >= 1`.
    * `tree_checker_b.py` and JavaScript: integers.
  * The floating-point screen, which decides no receipt: finite non-negative real numbers, with `bool` and `str`
    refused.

## 3. The witness

* **Candidate.** A list of distinct variables (the support form). The reference takes it as a bit vector of exactly
  `0`/`1` integers. A repeated entry is refused.
* **Facet.** `(row, F, y)`, or `[row, F, num, den]` in JSON.
  * `row` is either a single check index (an integer) or a GF(2) combination: a list of **at least two distinct**
    check indices. A one-element list is refused, because a single check is written as an integer.
  * `F` is a list of distinct variables. Order is immaterial; the reference's own document format sorts it, and its
    adapter `tools/replay_receipts.py:_reference_doc` canonicalises only the order, never a value.
  * The multiplier is a non-negative exact rational.
    * Python API: `int` or `Fraction`.
    * JSON: integers `num` and `den`, in lowest terms, with `den >= 1`. So `[.., 2, 2]` and `[.., -1, -1]` are
      refused rather than read as 1.
    * Floating-point screen: a finite real.
  * **Every listed facet is validated, including one whose multiplier is 0.** A zero multiplier adds nothing to
    the bound, but the facet is still a statement in the receipt: its row, its `F`, its multiplier and its
    uniqueness against the other facets are checked first, and only then is it skipped.
    The synthetic vectors include malformed zero facets both at the root and at tree depth 1.
* **Tree** (`check_qldpc_bnb`, `tree_checker_b.py`, `verify-qldpc-tree.mjs`):
  * `["branch", k, child_xk_0, child_xk_1]`, where `k` is an integer variable not already fixed on the path.
  * `["bound", [facet, ...]]`.
  * `["infeasible", row]`, where `row` is a check index or a list of at least two distinct indices.
* **Lattice rule and its guard.** With every weight in `(1/D)Z`, a bound `L > U - 1/D` proves a feasible candidate of
  objective `U` minimum.
  * For a valid dual, `L <= OPT <= U` always holds. So **`L > U` is an instrument failure and is refused** wherever
    nothing is fixed: a flat certificate (all four flat implementations) and the root of a tree (all three tree
    implementations).
  * Deeper in a tree, `W_fixed1 + L > U` is exactly what closes a branch.

## 4. What an acceptance says

* **Candidate mode** (flat, or a tree with a candidate): the candidate is feasible and of minimum weight; status
  `EXACT_OPTIMAL`.
* **Bound-only mode** (`check_qldpc_bnb(..., bound_only_threshold=T)`, used by the frame-determinacy and distance
  certificates): there is no candidate, and the acceptance says only that every solution of the system weighs more
  than `T`.
  * The result carries status `LOWER_BOUND_PROVEN`, the threshold `T`, and the lattice-rounded bound
    `(floor(T*D) + 1) / D`.
  * It carries no primal weight and no optimality claim.
  * The statement holds vacuously when the system is infeasible. Feasibility is not established by this mode and is
    reported as not established.

## 5. Changing this contract

A change to any rule above changes the vector file, and every implementation must then agree on every vector. A rule
that one implementation cannot enforce is a rule this contract does not state.
