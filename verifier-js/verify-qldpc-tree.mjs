#!/usr/bin/env node
// ONE-Q qLDPC branch-dual tree checker -- the THIRD implementation, in a second language.
//
// Paper 2's branch-dual receipts (tier BD) are checked by the production checker
// (src/oneq/qldpc_check.py) and by a second checker written from the schema
// (tools/external/tree_checker_b.py). Both are Python. This one is JavaScript,
// Node built-ins only, exact BigInt rationals, written from the manuscript's
// statement of the tier and the serialized-tree schema, not from either Python
// file. Flat facets are decided exactly as verify-qldpc-flat.mjs decides them,
// restated here rather than imported so that this file stands alone.
//
//   instance  checks N(j) over variables 0..n-1, integer weights w_i >= 0,
//             a syndrome s in GF(2)^m given as its support
//   claim     the candidate e, with He = s, has minimum weight U = w(e)
//   witness   a tree over the binary feasible set:
//               ["branch", k, child_for_xk_eq_0, child_for_xk_eq_1]
//               ["bound", [[row | [rows], [F...], num, den], ...]]
//               ["infeasible", row | [rows]]
//             Branching on x_k partitions the feasible set. Fixing x_k = 0
//             deletes variable k; fixing x_k = 1 deletes it, flips the syndrome
//             bit of every check containing it, and adds w_k to the weight
//             already committed on the path. Every root-to-leaf path must close:
//               bound      the safe-dual bound L of the listed facets on the
//                          REDUCED instance at that node satisfies
//                          W_fixed1 + L > U - 1, so no completion is lighter
//                          than U (weights are integers, so feasible weights
//                          lie on the integer lattice)
//               infeasible the reduced supports of the listed rows XOR to the
//                          empty set while their reduced syndrome bits XOR to 1:
//                          the GF(2) system has no solution on this branch
//   facet     for a reduced check -- or a GF(2) sum of distinct reduced checks --
//             with support S and parity p, and F a subset of S with |F| != p
//             (mod 2), every x satisfying the check obeys
//                 sum_{i in F} x_i - sum_{i in S \ F} x_i <= |F| - 1
//   bound     with load t_i = sum_r y_r a_{r,i} (a = -1 on F, +1 on S \ F) and
//             z_i = max(0, t_i - w_i), (y, z) is dual feasible, so every
//             feasible completion weighs at least L = sum_r y_r (1 - |F_r|) - sum_i z_i
//             on top of the weight already fixed to 1 on the path.
//   verdict   accept iff the candidate is feasible and every leaf closes. At the
//             root (nothing fixed) a bound above U cannot happen for valid input
//             and is reported as a verifier failure, never an acceptance; deeper
//             in the tree a bound above U is exactly what closes a branch.
//
// Representation is strict where the problem is not: a repeated variable in a
// check, a facet or the candidate, a repeated row in a combined check, a
// repeated syndrome index, a branch on a variable already fixed, a non-integer
// anywhere an integer is meant, and a node of unknown kind or arity are all
// refused rather than read charitably.
//
//   node verify-qldpc-tree.mjs batch.json
//
// batch.json
//   {"instances": {name: {"n": n, "checks": [[i, ...], ...]}},
//    "weights":   {key: [w_0, ..., w_{n-1}]},
//    "receipts":  [{"id", "instance", "weights", "syndrome_support", "candidate",
//                   "tree": <tree>, "expect": "accept" | "refuse"}]}
//
// Prints `id <TAB> ACCEPT|REFUSE <TAB> reason` per receipt and a summary line.
// Exit 0 iff every verdict equals its expectation.

import { readFileSync } from "node:fs";

// ---------------------------------------------------------------- exact rationals
function gcd(a, b) {
  if (a < 0n) a = -a;
  if (b < 0n) b = -b;
  while (b !== 0n) [a, b] = [b, a % b];
  return a;
}

function Q(num, den = 1n) {
  if (den === 0n) throw new Error("zero denominator");
  if (den < 0n) { num = -num; den = -den; }
  const g = gcd(num, den);
  return g > 1n ? [num / g, den / g] : [num, den];
}

const add = (x, y) => Q(x[0] * y[1] + y[0] * x[1], x[1] * y[1]);
const sub = (x, y) => Q(x[0] * y[1] - y[0] * x[1], x[1] * y[1]);
const scale = (x, k) => Q(x[0] * k, x[1]);
const cmp = (x, y) => {
  const l = x[0] * y[1];
  const r = y[0] * x[1];
  return l < r ? -1 : l > r ? 1 : 0;
};
const show = (x) => (x[1] === 1n ? `${x[0]}` : `${x[0]}/${x[1]}`);

function int(v, what) {
  if (typeof v !== "number" || !Number.isSafeInteger(v)) {
    throw new Error(`${what} is not an integer: ${JSON.stringify(v)}`);
  }
  return v;
}

// ---------------------------------------------------------------- instance and weights
function validateInstance(name, inst) {
  const n = int(inst.n, `${name}: n`);
  if (n < 0) throw new Error(`${name}: negative n`);
  if (!Array.isArray(inst.checks)) throw new Error(`${name}: checks is not a list`);
  const checks = inst.checks.map((c, j) => {
    if (!Array.isArray(c)) throw new Error(`${name}: check ${j} is not a list`);
    const vars = c.map((v) => int(v, `${name}: check ${j} variable`));
    if (new Set(vars).size !== vars.length) throw new Error(`${name}: check ${j} repeats a variable`);
    for (const i of vars) if (i < 0 || i >= n) throw new Error(`${name}: check ${j} variable ${i} out of range`);
    return vars;
  });
  return { n, checks };
}

function validateWeights(key, w, n) {
  if (!Array.isArray(w) || w.length !== n) throw new Error(`weights ${key}: expected ${n} entries`);
  return w.map((v, i) => {
    const x = int(v, `weights ${key}[${i}]`);
    if (x < 0) throw new Error(`weights ${key}[${i}] is negative`);
    return BigInt(x);
  });
}

// ---------------------------------------------------------------- the decision
class Refusal extends Error {}
const refuse = (reason) => { throw new Refusal(reason); };

function rowsOf(spec, m, what) {
  let rows;
  if (Array.isArray(spec)) {
    rows = spec.map((r) => int(r, `${what} row`));
    if (rows.length < 1) refuse(`${what} lists no row`);
    if (new Set(rows).size !== rows.length) refuse(`${what} repeats a row`);
  } else {
    rows = [int(spec, `${what} row`)];
  }
  for (const r of rows) if (r < 0 || r >= m) refuse(`${what} row ${r} out of range`);
  return rows;
}

// support of the GF(2) sum of `rows` on the reduced instance (fixed variables deleted), and its parity
function combined(rows, checks, syn, fixed) {
  const count = new Map();
  let p = 0;
  for (const r of rows) {
    p ^= syn[r];
    for (const i of checks[r]) if (!fixed[i]) count.set(i, (count.get(i) || 0) + 1);
  }
  const S = new Set();
  for (const [i, c] of count) if (c % 2 === 1) S.add(i);
  return { S, p };
}

// safe-dual bound of the listed facets on the reduced instance
function boundOf(duals, checks, syn, fixed, w, m) {
  if (!Array.isArray(duals)) refuse("bound leaf: duals are not a list");
  const load = new Map();
  let L = Q(0n);
  const seen = new Set();
  for (const f of duals) {
    if (!Array.isArray(f) || f.length !== 4) refuse("bound leaf: malformed facet entry");
    const [rowSpec, Fraw, num, den] = f;
    const y = Q(BigInt(int(num, "multiplier numerator")), BigInt(int(den, "multiplier denominator")));
    if (y[0] < 0n) refuse("bound leaf: negative multiplier");
    if (y[0] === 0n) continue; // contributes nothing to the bound
    const rows = rowsOf(rowSpec, m, "facet");
    const { S, p } = combined(rows, checks, syn, fixed);
    if (!Array.isArray(Fraw)) refuse("bound leaf: facet set is not a list");
    const F = Fraw.map((v) => int(v, "facet variable"));
    const Fset = new Set(F);
    if (Fset.size !== F.length) refuse("bound leaf: facet repeats a variable");
    for (const i of F) if (!S.has(i)) refuse(`bound leaf: facet variable ${i} is outside its reduced check`);
    if (F.length % 2 === p) refuse("bound leaf: facet has the wrong parity: not a face of the relaxation");
    const key = JSON.stringify([[...rows].sort((a, b) => a - b), [...F].sort((a, b) => a - b)]);
    if (seen.has(key)) refuse("bound leaf: facet listed twice");
    seen.add(key);
    for (const i of S) {
      const t = load.get(i) || Q(0n);
      load.set(i, Fset.has(i) ? sub(t, y) : add(t, y));
    }
    L = add(L, scale(y, BigInt(1 - F.length)));
  }
  for (const [i, t] of load) {
    const over = sub(t, Q(w[i]));
    if (over[0] > 0n) L = sub(L, over);
  }
  return L;
}

function decide(rec, inst, w) {
  const { n, checks } = inst;
  const m = checks.length;

  const s = new Uint8Array(m);
  for (const v of rec.syndrome_support) {
    const j = int(v, "syndrome index");
    if (j < 0 || j >= m) refuse(`syndrome index ${j} out of range`);
    if (s[j]) refuse(`syndrome index ${j} repeated`);
    s[j] = 1;
  }

  const inE = new Uint8Array(n);
  let U = 0n;
  for (const v of rec.candidate) {
    const i = int(v, "candidate variable");
    if (i < 0 || i >= n) refuse(`candidate variable ${i} out of range`);
    if (inE[i]) refuse(`candidate repeats variable ${i}`);
    inE[i] = 1;
    U += w[i];
  }
  for (let j = 0; j < m; j++) {
    let parity = 0;
    for (const i of checks[j]) parity ^= inE[i];
    if (parity !== s[j]) refuse(`candidate does not satisfy check ${j}`);
  }
  const target = Q(U - 1n); // a leaf closes when W_fixed1 + L exceeds this

  // rows containing each variable, for the syndrome flip on x_k = 1
  const rowsWith = Array.from({ length: n }, () => []);
  checks.forEach((c, j) => { for (const i of c) rowsWith[i].push(j); });

  let leaves = 0;
  let depthMax = 0;
  const walk = (node, syn, fixed, wFixed, depth, path) => {
    if (depth > depthMax) depthMax = depth;
    if (depth > 100000) refuse("tree deeper than any receipt this checker will read");
    if (!Array.isArray(node) || node.length < 2 || typeof node[0] !== "string") refuse(`${path}: malformed node`);
    const kind = node[0];
    if (kind === "branch") {
      if (node.length !== 4) refuse(`${path}: malformed branch node (arity ${node.length})`);
      const k = int(node[1], `${path}: branch variable`);
      if (k < 0 || k >= n) refuse(`${path}: branch variable ${k} out of range`);
      if (fixed[k]) refuse(`${path}: branch on variable ${k}, already fixed on this path`);
      const f = fixed.slice();
      f[k] = 1;
      walk(node[2], syn, f, wFixed, depth + 1, `${path}/x${k}=0`);
      const syn1 = syn.slice();
      for (const j of rowsWith[k]) syn1[j] ^= 1;
      walk(node[3], syn1, f, wFixed + w[k], depth + 1, `${path}/x${k}=1`);
      return;
    }
    leaves++;
    if (kind === "bound") {
      if (node.length !== 2) refuse(`${path}: malformed bound node`);
      const L = boundOf(node[1], checks, syn, fixed, w, m);
      const total = add(L, Q(wFixed));
      if (depth === 0 && cmp(L, Q(U)) > 0) {
        refuse(`VERIFIER FAILURE: root bound ${show(L)} exceeds the weight ${U} of a valid correction`);
      }
      if (cmp(total, target) <= 0) {
        refuse(`${path}: leaf not proven: fixed ${wFixed} + bound ${show(L)} does not exceed ${U - 1n}`);
      }
      return;
    }
    if (kind === "infeasible") {
      if (node.length !== 2) refuse(`${path}: malformed infeasible node`);
      const rows = rowsOf(node[1], m, `${path}: infeasible leaf`);
      const { S, p } = combined(rows, checks, syn, fixed);
      if (S.size !== 0) refuse(`${path}: infeasible leaf: reduced supports do not XOR to empty (${S.size} variables remain)`);
      if (p !== 1) refuse(`${path}: infeasible leaf: rows XOR to empty with even parity, which is satisfiable`);
      return;
    }
    refuse(`${path}: unknown node kind ${JSON.stringify(kind)}`);
  };
  walk(rec.tree, s, new Uint8Array(n), 0n, 0, "root");
  return `optimal at weight ${U}: ${leaves} leaves closed, depth ${depthMax}`;
}

// ---------------------------------------------------------------- batch driver
function main() {
  const path = process.argv[2];
  if (!path) {
    console.error("usage: node verify-qldpc-tree.mjs batch.json");
    process.exit(2);
  }
  const batch = JSON.parse(readFileSync(path, "utf8"));
  const instances = new Map();
  for (const [name, inst] of Object.entries(batch.instances)) instances.set(name, validateInstance(name, inst));
  const weights = new Map();

  let accepted = 0;
  let refused = 0;
  let mismatches = 0;
  const out = [];
  for (const rec of batch.receipts) {
    let verdict;
    let reason;
    try {
      const inst = instances.get(rec.instance);
      if (!inst) throw new Error(`unknown instance ${rec.instance}`);
      if (!weights.has(rec.weights)) weights.set(rec.weights, validateWeights(rec.weights, batch.weights[rec.weights], inst.n));
      reason = decide(rec, inst, weights.get(rec.weights));
      verdict = "ACCEPT";
    } catch (err) {
      verdict = "REFUSE";
      reason = err instanceof Refusal ? err.message : `malformed: ${err.message}`;
    }
    if (verdict === "ACCEPT") accepted++;
    else refused++;
    const want = rec.expect === "accept" ? "ACCEPT" : "REFUSE";
    if (verdict !== want) mismatches++;
    out.push(`${rec.id}\t${verdict}\t${reason}`);
  }
  process.stdout.write(out.join("\n") + "\n");
  process.stdout.write(`SUMMARY\taccepted=${accepted}\trefused=${refused}\tmismatches=${mismatches}\n`);
  // NOT process.exit(): on POSIX a pipe is written asynchronously, and exiting here abandons
  // whatever has not drained -- an external audit (2026-09-15) measured exactly 65,536 bytes of
  // verdicts delivered, no summary, exit 0. Set the code and let the process end on its own.
  process.exitCode = mismatches === 0 ? 0 : 1;
}

main();
