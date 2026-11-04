#!/usr/bin/env node
// ONE-Q qLDPC flat-receipt checker -- a THIRD implementation, in a second language.
//
// Paper 2's flat receipts are checked by the production checker
// (src/oneq/qldpc_check.py) and by a reference checker written separately
// (tools/external/exact_lp_certificate_reference.py). Both are Python. This one
// is JavaScript, Node built-ins only, exact BigInt rationals, and it was written
// from the manuscript's statement of the problem, not from either Python file.
//
//   instance  checks N(j) over variables 0..n-1, integer weights w_i >= 0,
//             a syndrome s in GF(2)^m given as its support
//   claim     the candidate e, with He = s, has minimum weight
//   witness   multipliers y_r >= 0 on facet inequalities of the Feldman
//             relaxation. For a check -- or a GF(2) sum of at least two distinct
//             checks -- with support S and parity p, and F a subset of S with
//             |F| != p (mod 2), every x satisfying the check obeys
//                 sum_{i in F} x_i - sum_{i in S \ F} x_i <= |F| - 1.
//   bound     with load t_i = sum_r y_r a_{r,i} (a = -1 on F, +1 on S \ F) and
//             z_i = max(0, t_i - w_i), (y, z) is dual feasible, so every
//             feasible e weighs at least
//                 L = sum_r y_r (1 - |F_r|) - sum_i z_i.
//   verdict   weights are integers, hence so is every feasible weight: L > U - 1
//             with U = w(e) proves e optimal. L > U cannot happen for valid input
//             and is reported as a verifier failure, never an acceptance.
//
// Representation is strict where the problem is not: a check or a facet that
// repeats a variable, a combined check that repeats a row, and a candidate
// that repeats a variable are refused rather than read as sets, because a
// repeated element means one thing under set semantics and another over GF(2).
//
// Input domain: docs/QLDPC_RECEIPT_INPUT_DOMAIN.md, the contract all seven qLDPC checkers implement, with the shared
// vectors conformance/qldpc_input_domain_vectors.json. Every integer slot holds a JSON integer literal (`1.0` is
// refused); a multiplier is num/den in lowest terms with den >= 1.
//
//   node verify-qldpc-flat.mjs batch.json
//
// batch.json
//   {"instances": {name: {"n": n, "checks": [[i, ...], ...]}},
//    "weights":   {key: [w_0, ..., w_{n-1}]},
//    "receipts":  [{"id", "instance", "weights", "syndrome_support", "candidate",
//                   "facet_duals": [[row | [rows], [F...], num, den], ...],
//                   "expect": "accept" | "refuse"}]}
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

// ---------------------------------------------------------------- the input domain
// docs/QLDPC_RECEIPT_INPUT_DOMAIN.md (2026-11-03), the contract all seven qLDPC checkers implement. Every number in
// a batch sits in an integer slot, and an integer slot holds a JSON integer LITERAL: `1.0`, `1e0` are refused, as the
// Python checkers refuse the float json.loads makes of them. A JavaScript Number cannot tell 1.0 from 1, so the
// token itself is read, through JSON.parse source-text access (Node >= 21); without it this file will not guess.
// Literals beyond the safe-integer range become exact BigInts: legal as weights and multipliers, out of range as
// indices.
const INTEGER_LEXEME = /^-?(0|[1-9][0-9]*)$/;
class NonInteger {
  constructor(source) { this.source = source; }
}
function strictNumbers(key, value, context) {
  if (typeof value !== "number") return value;
  if (!context || typeof context.source !== "string") {
    throw new Error("this JavaScript engine has no JSON.parse source-text access (Node >= 21 needed); "
      + "refusing to read numbers whose token it cannot see");
  }
  if (!INTEGER_LEXEME.test(context.source)) return new NonInteger(context.source);
  return Number.isSafeInteger(value) ? value : BigInt(context.source);
}
const shown = (v) => (v instanceof NonInteger ? v.source : typeof v === "bigint" ? `${v}` : JSON.stringify(v));

function int(v, what) {
  if (typeof v !== "number" || !Number.isSafeInteger(v)) {
    throw new Error(`${what} is not an integer: ${shown(v)}`);
  }
  return v;
}

// an exact integer of any size, for weights and multipliers
function big(v, what) {
  if (typeof v === "bigint") return v;
  return BigInt(int(v, what));
}

// a multiplier num/den: integers, den >= 1, in lowest terms -- never normalised into something else
function multiplier(num, den) {
  const p = big(num, "multiplier numerator");
  const q = big(den, "multiplier denominator");
  if (q === 0n) throw new Error("zero denominator");
  if (q < 1n) throw new Error(`multiplier denominator ${q} is not >= 1`);
  if (gcd(p, q) !== 1n) throw new Error(`multiplier ${p}/${q} is not in lowest terms`);
  return [p, q];
}

// ---------------------------------------------------------------- instance and weights
function validateInstance(name, inst) {
  const n = int(inst.n, `${name}: n`);
  if (n < 0) throw new Error(`${name}: negative n`);
  if (!Array.isArray(inst.checks)) throw new Error(`${name}: checks is not a list`);
  const checks = inst.checks.map((c, j) => {
    if (!Array.isArray(c)) throw new Error(`${name}: check ${j} is not a list`);
    const vars = c.map((v) => int(v, `${name}: check ${j} variable`));
    const set = new Set(vars);
    if (set.size !== vars.length) throw new Error(`${name}: check ${j} repeats a variable`);
    for (const i of vars) if (i < 0 || i >= n) throw new Error(`${name}: check ${j} variable ${i} out of range`);
    return vars;
  });
  return { n, checks };
}

function validateWeights(key, w, n) {
  if (!Array.isArray(w) || w.length !== n) throw new Error(`weights ${key}: expected ${n} entries`);
  return w.map((v, i) => {
    const x = big(v, `weights ${key}[${i}]`);
    if (x < 0n) throw new Error(`weights ${key}[${i}] is negative`);
    return x;
  });
}

// ---------------------------------------------------------------- the decision
const REFUSE = (reason) => ({ verdict: "REFUSE", reason });

function decide(rec, inst, w) {
  const { n, checks } = inst;
  const m = checks.length;

  const s = new Uint8Array(m);
  for (const v of rec.syndrome_support) {
    const j = int(v, "syndrome index");
    if (j < 0 || j >= m) return REFUSE(`syndrome index ${j} out of range`);
    if (s[j]) return REFUSE(`syndrome index ${j} repeated`);
    s[j] = 1;
  }

  const inE = new Uint8Array(n);
  let U = 0n;
  for (const v of rec.candidate) {
    const i = int(v, "candidate variable");
    if (i < 0 || i >= n) return REFUSE(`candidate variable ${i} out of range`);
    if (inE[i]) return REFUSE(`candidate repeats variable ${i}`);
    inE[i] = 1;
    U += w[i];
  }
  for (let j = 0; j < m; j++) {
    let parity = 0;
    for (const i of checks[j]) parity ^= inE[i];
    if (parity !== s[j]) return REFUSE(`candidate does not satisfy check ${j}`);
  }

  const load = new Map();
  let bound = Q(0n);
  const seen = new Set();
  for (const f of rec.facet_duals) {
    if (!Array.isArray(f) || f.length !== 4) return REFUSE("malformed facet entry");
    const [rowSpec, Fraw, num, den] = f;
    const y = multiplier(num, den);
    if (y[0] < 0n) return REFUSE("negative multiplier");
    // a zero multiplier is skipped only AFTER its facet is validated (final referee, 2026-11-03): every listed
    // facet is a claim, and a malformed one is refused whatever its multiplier

    let rows;
    if (Array.isArray(rowSpec)) {
      rows = rowSpec.map((r) => int(r, "row"));
      if (rows.length < 2) return REFUSE("a combined check needs at least two rows");
      if (new Set(rows).size !== rows.length) return REFUSE("a combined check repeats a row");
    } else {
      rows = [int(rowSpec, "row")];
    }
    const count = new Map();
    let p = 0;
    for (const r of rows) {
      if (r < 0 || r >= m) return REFUSE(`row ${r} out of range`);
      p ^= s[r];
      for (const i of checks[r]) count.set(i, (count.get(i) || 0) + 1);
    }
    const S = new Set();
    for (const [i, c] of count) if (c % 2 === 1) S.add(i);

    if (!Array.isArray(Fraw)) return REFUSE("facet set is not a list");
    const F = Fraw.map((v) => int(v, "facet variable"));
    const Fset = new Set(F);
    if (Fset.size !== F.length) return REFUSE("facet repeats a variable");
    for (const i of F) if (!S.has(i)) return REFUSE(`facet variable ${i} is outside its check`);
    if (F.length % 2 === p) return REFUSE("facet has the wrong parity: not a face of the relaxation");

    const key = JSON.stringify([[...rows].sort((a, b) => a - b), [...F].sort((a, b) => a - b)]);
    if (seen.has(key)) return REFUSE("facet listed twice");
    seen.add(key);
    if (y[0] === 0n) continue; // validated above; contributes nothing to the bound

    for (const i of S) {
      const t = load.get(i) || Q(0n);
      load.set(i, Fset.has(i) ? sub(t, y) : add(t, y));
    }
    bound = add(bound, scale(y, BigInt(1 - F.length)));
  }

  for (const [i, t] of load) {
    const over = sub(t, Q(w[i]));
    if (over[0] > 0n) bound = sub(bound, over);
  }

  const u = Q(U);
  if (cmp(bound, u) > 0) {
    return REFUSE(`VERIFIER FAILURE: bound ${show(bound)} exceeds the weight ${U} of a valid correction`);
  }
  if (cmp(bound, Q(U - 1n)) > 0) {
    return { verdict: "ACCEPT", reason: `optimal: bound ${show(bound)} exceeds ${U - 1n}` };
  }
  return REFUSE(`not proven: bound ${show(bound)} does not exceed ${U - 1n}`);
}

// ---------------------------------------------------------------- batch driver
function main() {
  const path = process.argv[2];
  if (!path) {
    console.error("usage: node verify-qldpc-flat.mjs batch.json");
    process.exit(2);
  }
  const batch = JSON.parse(readFileSync(path, "utf8"), strictNumbers);
  // instances are validated lazily, once each: a malformed instance refuses every receipt that names it, and only
  // those, instead of ending the whole batch with an uncaught error
  const instances = new Map();
  const instance = (name) => {
    if (!instances.has(name)) {
      if (!Object.prototype.hasOwnProperty.call(batch.instances, name)) throw new Error(`unknown instance ${name}`);
      try {
        instances.set(name, { ok: validateInstance(name, batch.instances[name]) });
      } catch (err) {
        instances.set(name, { err });
      }
    }
    const v = instances.get(name);
    if (v.err) throw v.err;
    return v.ok;
  };
  const weights = new Map();

  let accepted = 0;
  let refused = 0;
  let mismatches = 0;
  const out = [];
  for (const rec of batch.receipts) {
    let result;
    try {
      const inst = instance(rec.instance);
      if (!weights.has(rec.weights)) weights.set(rec.weights, validateWeights(rec.weights, batch.weights[rec.weights], inst.n));
      result = decide(rec, inst, weights.get(rec.weights));
    } catch (err) {
      result = REFUSE(`malformed: ${err.message}`);
    }
    if (result.verdict === "ACCEPT") accepted++;
    else refused++;
    const want = rec.expect === "accept" ? "ACCEPT" : "REFUSE";
    if (result.verdict !== want) mismatches++;
    out.push(`${rec.id}\t${result.verdict}\t${result.reason}`);
  }
  process.stdout.write(out.join("\n") + "\n");
  process.stdout.write(`SUMMARY\taccepted=${accepted}\trefused=${refused}\tmismatches=${mismatches}\n`);
  // NOT process.exit(): on POSIX a pipe is written asynchronously, and exiting here abandons
  // whatever has not drained -- an external audit (2026-09-15) measured exactly 65,536 bytes of
  // verdicts delivered, no summary, exit 0. Set the code and let the process end on its own.
  process.exitCode = mismatches === 0 ? 0 : 1;
}

main();
