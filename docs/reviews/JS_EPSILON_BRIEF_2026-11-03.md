# Brief given to the agent that wrote `verifier-js/verify-epsilon.mjs` (2026-11-03)

This file is the complete specification the JavaScript epsilon-checker was written from, kept so that its
independence claim can be audited (external review 1, 2026-11-03: "log the session context"). The writing agent was
a fresh AI agent with no prior context; it was told to read nothing but this brief and the manuscript sections named
below, and to list every file it opened (its list is appended at the end by the coordinator). The agent that wrote
this brief HAD read the Python checkers while auditing them; the brief therefore states only what the manuscript
states, plus the wire format, and deliberately leaves every contract question the manuscript leaves open, so that a
differential fuzzer can find where the two readings part.

## What to build

A single file, `verifier-js/verify-epsilon.mjs`, Node.js (v24), built-ins only (no npm packages), that decides the
**epsilon-claim** of a matching-decoder receipt in exact arithmetic. Write it from this brief and from the manuscript
`paper/certifying-decoders/main.tex`, Sections II-IV (lines 57-130: "Certificates for minimum-weight decoding",
"The checker", "What a receipt proves under floating-point weights"). Do NOT open any other source code in the
repository: nothing under `src/`, `tools/`, `tests/`, `experiments/`, `release/`, `verifier-c/`, `third_party/`, and
no other file in `verifier-js/`. Data files under `evidence/instances/` and `data/epsilon_receipts/` may be opened to
learn the wire format, but do not tune your checker against the verdicts recorded in them; comparing verdicts is the
coordinator's job, done afterwards.

## The mathematics (from the manuscript)

- A decoding graph `G = (V, E, w)` with non-negative edge weights and a virtual boundary vertex `b`, written `-1`.
  Every other vertex is a detector, a non-negative integer.
- The syndrome lists the fired detectors. `T` is that set, with `b` added when the number of fired detectors is odd,
  so `|T|` is even.
- A correction is an edge set `J` of `E` whose odd-degree vertices, ignoring `b`, are exactly the fired detectors.
- A set `S` of vertices is T-odd when `|S intersect T|` is odd (with `T` including `b` as above). `delta(S)` is the set
  of edges with exactly one endpoint in `S`.
- A packing is a map from T-odd sets `S` to values `z_S >= 0`. Load of edge `e`: `l_e = sum of z_S over S with e in
  delta(S)`.
- Lemma: let `theta = min(1, min over edges with l_e > 0 of w_e / l_e)`. Then `theta*z` is feasible and for every
  correction `J`, `w(J) - OPT <= eps(J, z) := w(J) - theta * sum_S z_S`.
- Outcomes, exactly one per receipt:
  - `EXACT_OPTIMAL` if `eps = 0`;
  - `EPSILON_OPTIMAL` if `0 < eps <= eps_max`;
  - `NOT_PROVEN` if `eps > eps_max`;
  - `INVALID_INPUT` for any input outside the declared domain (non-finite or negative values, sets that are not
    T-odd, a correction that does not explain the syndrome, a set listed twice in the packing, and anything else you
    judge malformed from the manuscript; say in a comment why each refusal exists);
  - `VERIFIER_FAILURE` if the bound exceeds the correction's weight (`eps < 0`), impossible for valid input and never
    an acceptance.
- Every double is a rational number. All arithmetic must be exact: parse every weight and every packing value into an
  exact rational (a double is `m * 2^k`), compute loads, `theta` and `eps` exactly (BigInt; compare `w_e / l_e`
  ratios by cross-multiplication, never by floating division). Report `eps` and the lower bound `theta * sum z` as
  exact rationals in lowest terms.

## Wire format

Numbers that are doubles travel as hexadecimal floating literals in the form Python's `float.hex()` writes:
`"0x1.28a3957ae21cdp+2"`, `"-0x1.0000000000000p-1"`, `"0x0.0p+0"`, subnormals like `"0x0.0000000000001p-1022"`;
non-finite values appear as `"inf"`, `"-inf"`, `"nan"`. Parse them exactly yourself (JavaScript's `parseFloat` does
not read hex floats). Decide, and document, what you do with a hex literal that is not exactly a binary64 value.

Rationals (`eps_max`, and your outputs `epsilon`, `lower_bound`) are strings `"p/q"` in lowest terms with `q > 0`, or
`"p"` when the denominator is 1 (`"0"` for zero).

### Batch mode (used by the coordinator's differential fuzzer)

`node verifier-js/verify-epsilon.mjs --batch < cases.jsonl > verdicts.jsonl`

Each input line is one JSON object:

```
{"id": "...", "edges": [[u, v, "hexweight"], ...], "fired": [d, ...],
 "correction": [[u, v], ...], "duals": [[[member, ...], "hexvalue"], ...], "eps_max": "1/1000000"}
```

The fuzzer deliberately sends malformed values in every field (fractional or boolean vertex ids, strings, repeated
entries, non-finite or negative values, parallel edges, self-loops, unknown vertices, empty sets, `-1` in places it
may not belong). Never crash and never stop the batch: every line gets exactly one output line, in input order:

```
{"id": "...", "status": "EXACT_OPTIMAL|EPSILON_OPTIMAL|NOT_PROVEN|INVALID_INPUT|VERIFIER_FAILURE",
 "epsilon": "p/q" or null, "lower_bound": "p/q" or null, "reason": "..."}
```

`epsilon` and `lower_bound` are null for `INVALID_INPUT` and present otherwise.

### Replay mode

`node verifier-js/verify-epsilon.mjs --instance evidence/instances/matching_<name>.json --receipts <chunk>.jsonl.gz
[--eps-max 1/1000000]`

An instance file holds `{"schema", "name", "edges": [[u, v, "hexweight"], ...], "edges_sha256"}`, where
`edges_sha256` is the SHA-256 of the `edges` array serialised as compact JSON (no spaces). Refuse the instance if the
hash does not match. A receipts chunk is gzip-compressed JSON lines, one per shot:
`{"shot", "fired", "correction", "duals", "status", "epsilon", "lower_bound"}`. Print one output line per receipt as in
batch mode (with `"shot"` in place of `"id"`), and nothing else on stdout. Do not compare with the recorded fields.

## Self-test

Include `--selftest`: a handful of vectors you construct BY HAND from the definitions above on graphs of 2-5
vertices (an exactly optimal packing, an epsilon-optimal one with a known rational epsilon, a not-proven one, an
overloaded packing that theta must scale down, a boundary case where `|fired|` is odd and `b` is in `T`, each
INVALID_INPUT reason). Exit 0 only if all pass.

## Rules

- Write the file with your file-writing tool, never through a shell heredoc (this machine's shell corrupts
  backslashes).
- Do not commit, stash or check out anything. Do not modify any file except `verifier-js/verify-epsilon.mjs`.
- When done, report: the file path, its line count, every file you opened, the contract decisions you made where the
  manuscript is silent (one line each), and the self-test result.

## Appendix: what the writing agent reported (2026-11-03, verbatim in substance)

Files opened, complete list as reported: this brief (in full); `paper/certifying-decoders/main.tex` lines 57-130 only;
`evidence/instances/matching_sim_d5_r25.json` (keys, edge count, first and last few hundred characters, hash
convention); `data/epsilon_receipts/sim_d5_r25/chunk_000001.jsonl.gz` (key names, the first receipt's fired,
correction and first three duals; the recorded status, epsilon and lower_bound values were not read); its own file.
Directory listings (names only): `evidence/instances/`, `data/epsilon_receipts/`, `data/epsilon_receipts/sim_d5_r25/`,
`verifier-js/`. It reported opening no source code and grepping only its own file. This is self-reported by an AI
agent; nothing here makes it independent authorship, and the file is the same project's work.

Contract decisions it reported where the manuscript is silent: parallel edges refused (the manuscript says graphs
are simple); self-loops refused; a correction edge listed twice (either orientation) refused; vertices not in the
graph refused (V = b plus every edge endpoint); non-integer, integral-float, boolean, string and null ids refused,
read by its own JSON parser that keeps number text; -1 in the syndrome and a repeated fired detector refused; empty
sets refused as not T-odd, zero-valued T-odd sets accepted; hex literals must match `[+-]0x<hex>[.<hex>]p[+-]<dec>`,
non-canonical but exact spellings accepted, anything not exactly a finite binary64 refused (never rounded),
`inf`/`nan` refused; negative zero read as 0; JSON numbers where hex strings belong refused; eps_max canonical only;
duplicate JSON keys refused; an unexpected internal exception is reported as VERIFIER_FAILURE with null fields.

Self-test it reported: 27,836 checks, 0 failed (hand-built vectors, every INVALID_INPUT reason, a brute-force
optimum on random graphs of 2-5 vertices, 6,000 mutation-fuzz inputs with no crash), and 12 deliberately broken
copies of the checker each caught. Replay of `sim_d5_r25` chunk 1: 250 lines for 250 receipts, 0.6-0.7 ms per receipt
end to end, peak 79 MB.
