#!/usr/bin/env node
// ONE-Q decoding-certificate checker -- the SECOND language.
//
// WHY A SECOND IMPLEMENTATION EXISTS
// A checker written by the same hand, in the same language, sharing the same
// serializer, cannot detect a shared misunderstanding. The Python checker has
// already been wrong four times in ways its own tests did not see: it assumed
// PyMatching returned a matching on syndrome nodes when it returns paths, it
// accepted node potentials as a T-join dual (they are the dual of a different
// primal), it mishandled the virtual boundary, and it counted equal-weight
// alternative optima as forgeries. Every one of those was a misreading of the
// PROBLEM, not a typo -- and a port would have inherited all four.
//
// So nothing here is ported. No shared library, no npm packages, Node
// built-ins only. It reads the spec, not the Python.
//
//   node verify-certificate.mjs cert.json --graph graph.json
//   node verify-certificate.mjs --conformance vectors.json
//
// Exit 0 = the certificate PROVES its correction is minimum weight.
// Nonzero = refused, with the reason on stderr.

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const SCHEMA = "oneq-decoding-certificate/1";
const BOUNDARY = -1;
const TOL = 1e-9;

// ---------------------------------------------------------------------------
// Exact floats. The wire format carries every double as its 64-bit pattern
// precisely so this step cannot introduce a discrepancy: a decimal round-trip
// through JSON is not bit-identical across languages, and a checker that
// re-parses a slightly different weight can refuse a valid certificate or
// accept a marginal one.
// ---------------------------------------------------------------------------
function h2f(s) {
  if (typeof s !== "string" || !/^0x[0-9a-f]{16}$/.test(s)) {
    throw new Error(`not a 64-bit float pattern: ${JSON.stringify(s)}`);
  }
  const b = new DataView(new ArrayBuffer(8));
  for (let i = 0; i < 8; i++) b.setUint8(i, parseInt(s.slice(2 + 2 * i, 4 + 2 * i), 16));
  const x = b.getFloat64(0, false); // big-endian, matches struct.pack('>d')
  // EXTERNAL AUDIT 2026-08-04, finding 5 (CONFIRMED and fixed here).
  // 0x7ff8000000000000 decodes to NaN, and every comparison against NaN
  // is false in JS exactly as in Python -- so a NaN walked through every
  // refusal in this file and out the accept path, serialising as null.
  // A checker cannot compare its way out of NaN; it has to test for it,
  // at the boundary, before any comparison sees the value.
  if (!Number.isFinite(x)) {
    throw new Error(`not a finite value: ${JSON.stringify(s)} decodes to ${x}`);
  }
  return x;
}

// Canonical JSON, implemented from the spec: sorted keys, compact separators,
// ASCII-escaped, non-finite refused. Deliberately re-derived rather than
// imported from verify.mjs -- the conformance harness pins both against the
// Python, so a divergence here fails loudly instead of cancelling out.
function canonical(v) {
  if (v === null) return "null";
  const t = typeof v;
  if (t === "boolean") return v ? "true" : "false";
  if (t === "number") {
    if (!Number.isFinite(v)) throw new Error("non-finite number in canonical JSON");
    if (!Number.isInteger(v)) throw new Error("floats are not permitted here");
    return String(v);
  }
  if (t === "string") {
    let out = '"';
    for (const ch of v) {
      const c = ch.codePointAt(0);
      if (ch === '"') out += '\\"';
      else if (ch === "\\") out += "\\\\";
      else if (c === 0x08) out += "\\b";
      else if (c === 0x0c) out += "\\f";
      else if (c === 0x0a) out += "\\n";
      else if (c === 0x0d) out += "\\r";
      else if (c === 0x09) out += "\\t";
      else if (c < 0x20) out += "\\u" + c.toString(16).padStart(4, "0");
      else if (c < 0x7f) out += ch;
      else if (c > 0xffff) {
        const u = c - 0x10000;
        out += "\\u" + (0xd800 + (u >> 10)).toString(16).padStart(4, "0");
        out += "\\u" + (0xdc00 + (u & 0x3ff)).toString(16).padStart(4, "0");
      } else out += "\\u" + c.toString(16).padStart(4, "0");
    }
    return out + '"';
  }
  if (Array.isArray(v)) return "[" + v.map(canonical).join(",") + "]";
  if (t === "object") {
    const keys = Object.keys(v).sort();
    return "{" + keys.map((k) => canonical(k) + ":" + canonical(v[k])).join(",") + "}";
  }
  throw new Error(`cannot canonicalise ${t}`);
}

function canonEdge(u, v) {
  // BOUNDARY IS ALWAYS SECOND. Written boundary-first,
  // u === -1 makes `u <= v` already true, so the swap never
  // fired and (-1,3) and (3,-1) survived as distinct keys --
  // the same edge counted twice, and a valid document refused
  // with GRAPH MISMATCH. Both languages carried it.
  if (u === BOUNDARY) return [v, u];
  return v === BOUNDARY || u <= v ? [u, v] : [v, u];
}
const ekey = (u, v) => { const [a, b] = canonEdge(u, v); return `${a},${b}`; };

// THE GRAPH'S WEIGHTS ARE EXACT TOO, in the same 64-bit hex as the
// certificate's. Accepting a decimal number here would reintroduce exactly the
// round-trip the format exists to eliminate -- and it would do so in the one
// place where a discrepancy is invisible, since a fingerprint mismatch reads
// as "wrong graph" rather than "wrong parser". Refuse anything else outright.
//
// Parallel edges collapse to the cheapest, the same rule the producer and the
// Python checker use. If this kept duplicates, two graphs that behave
// identically would fingerprint differently and valid certificates would be
// refused for a bookkeeping reason.
function cheapest(edges) {
  const best = new Map();
  for (const [u0, v0, w0] of edges) {
    const w = typeof w0 === "string" ? h2f(w0) : (() => {
      throw new Error(`graph weight must be a 64-bit hex pattern, got ${typeof w0}`);
    })();
    const [u, v] = canonEdge(Number(u0), Number(v0));
    const k = ekey(u, v);
    const cur = best.get(k);
    if (cur === undefined || w < cur[2]) best.set(k, [u, v, w]);
  }
  return best;
}

function graphFingerprint(edges) {
  const best = cheapest(edges);
  const rows = [...best.values()]
    .sort((a, b) => a[0] - b[0] || a[1] - b[1])
    .map(([u, v, w]) => {
      const d = new DataView(new ArrayBuffer(8));
      d.setFloat64(0, w, false);
      let hex = "";
      for (let i = 0; i < 8; i++) hex += d.getUint8(i).toString(16).padStart(2, "0");
      return [u, v, "0x" + hex];
    });
  const bytes = canonical({ schema: "oneq-decoding-graph/1", edges: rows });
  return createHash("sha256").update(Buffer.from(bytes, "utf8")).digest("hex");
}

// ---------------------------------------------------------------------------
// The check itself.
// ---------------------------------------------------------------------------
// EXACT DYADIC ARITHMETIC, ported from the Python checker after the
// external audit of 2026-08-04 (finding 4). Every IEEE double is exactly
// n / 2^k, so sums and comparisons can be done in BigInt with no
// rounding at all. This exists because the conformance suite CAUGHT the
// divergence: Python moved to exact acceptance and this file was still
// accepting on `gap > 1e-6`, so the two checkers disagreed on one vector.
// That is precisely what a cross-language suite is for.
function toDyadic(x) {            // exact: x === n / 2^k
  let k = 0n, y = x;
  while (!Number.isInteger(y)) { y *= 2; k += 1n; }
  return [BigInt(y), k];
}
function dyadicSum(xs) {          // exact sum as [numerator, exponent]
  let n = 0n, k = 0n;
  for (const x of xs) {
    const [xn, xk] = toDyadic(x);
    if (xk > k) { n <<= (xk - k); k = xk; }
    n += xn << (k - xk);
  }
  return [n, k];
}
// EXACT TWO-SIDED DUAL REPAIR. The lattice argument this replaces was sound
// but useless over IEEE doubles: 1/D came out near 4.4e-16, FINER than the
// float error in the dual itself, so it could not certify the very
// certificates it was written for. The real problem was never the checker --
// it was that a shipped dual is a float ROUNDING of an exact one, and
// rounding damages it in both directions. `surface-d3-4` overloaded one edge
// by 4.44e-16; `surface-d3-5` fell 2.22e-16 short. Neither is suboptimal.
//
// So: shed any infeasibility (reducing a dual lowers load on every edge of
// its cut and raises it nowhere, so no feasible edge can regress -- and each
// pass clears at least the worst violated edge, which bounds the loop by the
// edge count as a PROOF, not a guess), then absorb the shortfall into a
// variable with enough slack across its whole cut, then VERIFY from scratch.
//
// Step 3 is the entire guarantee; steps 1-2 are a search and are allowed to
// be wrong. And the search cannot launder a bad answer even in principle: by
// LP duality a feasible dual attaining U exists IFF U is optimal, so on a
// suboptimal primal the absorb step provably cannot find the slack.
// MULTI-START. Greedy order decides whether the search finds the dual:
// shedding can take from a variable absorbing then needs back. Trying four
// deterministic orders cannot weaken anything -- every attempt is verified
// from scratch by step 3 -- and it must match Python EXACTLY or the
// cross-language conformance is a lie rather than a check.
const REPAIR_ORDERS = 4;

function exactRepairMulti(W, zs, vals, primalTerms) {
  let last = [false, "no spend ordering produced a feasible attaining dual"];
  for (let o = 0; o < REPAIR_ORDERS; o++) {
    const r = exactRepair(W, zs, vals, primalTerms, o);
    if (r[0]) return r;
    last = r;
  }
  return last;
}

function exactRepair(W, zs, vals, primalTerms, order = 0) {
  const wl = [...W.values()];
  let K = 0n;
  for (const x of [...primalTerms, ...vals, ...wl.map((w) => w[2])]) {
    const [, k] = toDyadic(x); if (k > K) K = k;
  }
  const S = (x) => { const [n, k] = toDyadic(x); return n << (K - k); };
  const P = primalTerms.reduce((a, x) => a + S(x), 0n);
  const wt = wl.map(([u, v, w]) => [u, v, S(w)]);
  const z = vals.map(S);
  const sum = (a) => a.reduce((p, q) => p + q, 0n);
  // The spend order. Index is the tie-break in every case, so all four are
  // total orders and both languages repair identically.
  const ord = z.map((_, j) => j);
  if (order === 1) ord.reverse();
  else if (order === 2) ord.sort((a, b) => (z[b] === z[a] ? a - b : (z[b] > z[a] ? 1 : -1)));
  else if (order === 3) ord.sort((a, b) => (z[a] === z[b] ? a - b : (z[a] > z[b] ? 1 : -1)));
  const inCut = (j, u, v) => zs[j].has(u) !== zs[j].has(v);
  const load = (u, v) => {
    let s = 0n;
    for (let j = 0; j < z.length; j++) if (inCut(j, u, v)) s += z[j];
    return s;
  };

  for (let pass = 0; pass <= wt.length; pass++) {           // 1. SHED
    let worst = -1, over = 0n;
    for (let i = 0; i < wt.length; i++) {
      const d = load(wt[i][0], wt[i][1]) - wt[i][2];
      if (d > over) { over = d; worst = i; }
    }
    if (worst < 0) break;
    const [u, v] = wt[worst];
    for (const j of ord) {
      if (over <= 0n) break;
      if (!inCut(j, u, v)) continue;
      const take = z[j] < over ? z[j] : over;
      if (take <= 0n) continue;
      z[j] -= take; over -= take;
    }
    if (over > 0n) {
      return [false, "cannot restore dual feasibility without driving a dual negative -- certificate is malformed"];
    }
  }

  let short = P - sum(z);                                    // 2. ABSORB
  if (short < 0n) {
    return [false, "dual exceeds primal even after feasibility was restored -- impossible under weak duality"];
  }
  if (short > 0n) {
    // SPREAD the shortfall across variables. Requiring ONE to cover all of
    // it refused 77 of 9,841 hardware shots whose gap measured EXACTLY 0.0:
    // an attaining dual existed and the search failed to build it. Each
    // variable contributes up to the slack across its whole cut.
    for (const j of ord) {
      if (short <= 0n) break;
      let slack = null;
      for (const [u, v, w] of wt) {
        if (!inCut(j, u, v)) continue;
        const s = w - load(u, v);
        if (slack === null || s < slack) slack = s;
      }
      if (slack === null || slack <= 0n) continue;
      const take = slack < short ? slack : short;
      z[j] += take; short -= take;
    }
    if (short > 0n) {
      return [false, "no dual variable has the slack to close the gap, so the shortfall is REAL and this correction is not proven minimum-weight"];
    }
  }

  for (const [u, v, w] of wt) {                              // 3. VERIFY
    if (load(u, v) > w) return [false, "repaired dual is infeasible"];
  }
  for (const zz of z) if (zz < 0n) return [false, "repaired dual has a negative term"];
  if (sum(z) !== P) return [false, "repaired dual does not attain the primal"];
  return [true, "a feasible dual ATTAINS the primal in exact arithmetic, so weak duality proves both optimal"];
}

export function checkCertificate(doc, edges) {
  if (doc.schema !== SCHEMA) return no(`unknown schema ${JSON.stringify(doc.schema)}`);

  // THE GRAPH COMES FROM THE VERIFIER, NOT THE DOCUMENT. A document that
  // supplied both the claim and the graph it is judged against could supply a
  // graph on which any claim is true.
  const fp = graphFingerprint(edges);
  if (doc.graph_sha256 !== fp) {
    return no(`GRAPH MISMATCH: document is for ${doc.graph_sha256}, this graph is ${fp}`);
  }
  if (doc.proven_optimal !== true) return no("document does not claim optimality (degraded receipt)");
  if (!Array.isArray(doc.correction) || !Array.isArray(doc.dual)) {
    return no("a claim of optimality without a witness");
  }

  const W = cheapest(edges);
  const syndrome = new Set(doc.syndrome.map(Number));

  // 1. IS IT EVEN A CORRECTION? The odd-degree vertices of the edge set must
  //    be exactly the fired detectors. This is the T-JOIN condition, and it is
  //    the only part of the claim that a production decoder checks today.
  const deg = new Map();
  let primal = 0;
  const pTerms = [];   // the EXACT terms, for the dyadic check below
  for (const e of doc.correction) {
    const [u, v] = canonEdge(Number(e[0]), Number(e[1]));
    const w = W.get(ekey(u, v));
    if (w === undefined) return no(`correction uses edge (${u},${v}) that is not in the graph`);
    primal += w[2];
    pTerms.push(w[2]);
    for (const x of [u, v]) {
      if (x !== BOUNDARY) deg.set(x, (deg.get(x) || 0) + 1);
    }
  }
  const odd = new Set([...deg.entries()].filter(([, c]) => c % 2 === 1).map(([n]) => n));
  if (odd.size !== syndrome.size || [...odd].some((n) => !syndrome.has(n))) {
    return no("NOT A VALID CORRECTION: odd-degree set != syndrome");
  }

  // The odd-degree target set. When the syndrome has odd parity the virtual
  // boundary joins it -- that is what makes |T| even and the T-join well posed.
  const T = new Set(syndrome);
  if (T.size % 2 === 1) T.add(BOUNDARY);

  // 2. IS THE DUAL ADMISSIBLE? Non-negative, and every set T-ODD. An even set
  //    is not a dual variable at all; admitting one lets the bound climb past
  //    the true optimum, which would let a suboptimal correction "meet" it.
  // A REPEATED MEMBER SET MAKES THE DOCUMENT AMBIGUOUS. A dual is a map from
  // vertex sets to values; a list encodes one only while the keys are
  // distinct. Python built a dict (last value wins, the rest dropped) and this
  // file summed the array -- on the same bytes Python read an objective of 2.0
  // and ACCEPTED where this read 2.4 and refused. Neither reading is right,
  // because the document does not denote one certificate, so both now refuse.
  const seenSets = new Set();
  for (const [members] of doc.dual) {
    const k = [...new Set(members.map(Number))].sort((a, b) => a - b).join(",");
    if (seenSets.has(k)) return no(`AMBIGUOUS DOCUMENT: member set [${k}] appears more than once in the dual`);
    seenSets.add(k);
  }

  const zs = [];
  let objective = 0;
  const dTerms = [];
  for (const [members, hex] of doc.dual) {
    const val = h2f(hex);
    if (val < -TOL) return no(`negative dual z=${val}`);
    const S = new Set(members.map(Number));
    let inter = 0;
    for (const t of T) if (S.has(t)) inter++;
    if (inter % 2 !== 1) return no(`dual on a set with EVEN intersection with T`);
    zs.push(S);
    objective += val;
    dTerms.push(val);
  }

  // 3. IS THE PACKING FEASIBLE? For every edge, the duals whose boundary it
  //    crosses must not exceed its weight. Edge-local, so this is the whole
  //    verification -- no LP, no matching algorithm, nothing to trust.
  const vals = doc.dual.map(([, hex]) => h2f(hex));
  for (const [k, [u, v, w]] of W) {
    void k;
    let sum = 0;
    for (let j = 0; j < zs.length; j++) {
      const a = zs[j].has(u), b = zs[j].has(v);
      if (a !== b) sum += vals[j];
    }
    if (sum > w + TOL) return no(`DUAL INFEASIBLE on edge (${u},${v}): ${sum} > ${w}`);
  }

  // 4. DOES IT CLOSE? Weak duality makes any feasible packing a lower bound on
  //    the minimum-weight T-join. Equality forces both to be optimal -- and
  //    that argument does not care which solver produced either side, which is
  //    the entire reason a certificate is worth more than a test suite.
  const gap = primal - objective;
  const [exact, why] = exactRepairMulti(W, zs, vals, pTerms);
  if (gap > 1e-6 || !exact) return no(`NOT PROVEN OPTIMAL: ${why} -- gap ${gap} (primal ${primal}, bound ${objective})`);
  if (gap < -1e-6) return no(`IMPOSSIBLE: dual ${objective} exceeds primal ${primal}; the packing is not feasible`);

  const stated = h2f(doc.weight);
  if (Math.abs(stated - primal) > TOL) {
    return no(`document states weight ${stated} but the correction weighs ${primal}`);
  }
  return { accepted: true, reason: `OPTIMAL: ${why}, proven in exact dyadic arithmetic with no tolerance`, primal, objective, gap };
}

const no = (reason) => ({ accepted: false, reason });

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
function main(argv) {
  const conf = argv.indexOf("--conformance");
  if (conf >= 0) {
    // AGREEMENT IS THE CONFORMANCE DEFINITION. Each vector records what the
    // Python checker decided; this implementation must reach the same verdict
    // for the same reason class, or the two have diverged and neither number
    // can be quoted.
    const vectors = JSON.parse(readFileSync(argv[conf + 1], "utf8"));
    let pass = 0, fail = 0;
    for (const v of vectors.vectors) {
      let got;
      try {
        got = checkCertificate(v.document, v.graph);
      } catch (e) {
        got = { accepted: false, reason: `threw: ${e.message}` };
      }
      const ok = got.accepted === v.expected_accepted;
      if (ok) pass++;
      else {
        fail++;
        console.error(`  MISMATCH ${v.name}: python=${v.expected_accepted} js=${got.accepted}`);
        console.error(`    python said: ${v.expected_reason}`);
        console.error(`    js     said: ${got.reason}`);
      }
    }
    console.log(`conformance: ${pass}/${pass + fail} agree`);
    return fail === 0 ? 0 : 1;
  }
  const gi = argv.indexOf("--graph");
  if (argv.length < 1 || gi < 0) {
    console.error("usage: verify-certificate.mjs cert.json --graph graph.json");
    console.error("       verify-certificate.mjs --conformance vectors.json");
    return 2;
  }
  const doc = JSON.parse(readFileSync(argv[0], "utf8"));
  const graph = JSON.parse(readFileSync(argv[gi + 1], "utf8"));
  const r = checkCertificate(doc.core ?? doc, graph.edges ?? graph);
  if (argv.includes("--json")) console.log(JSON.stringify(r, null, 2));
  else console.log(r.accepted ? `ACCEPTED: ${r.reason}` : `REFUSED: ${r.reason}`);
  return r.accepted ? 0 : 1;
}

// pathToFileURL, not string surgery: on Windows a path is `C:\...` and the URL
// is `file:///C:/...` with THREE slashes, so a hand-rolled comparison silently
// fails to match and the CLI exits 0 having run nothing. A conformance harness
// that reports success by doing nothing is worse than one that errors.
if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  process.exit(main(process.argv.slice(2)));
}
