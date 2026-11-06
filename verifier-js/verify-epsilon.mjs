#!/usr/bin/env node
/*
 * verify-epsilon.mjs -- an independent checker for the epsilon-claim of a matching-decoder receipt.
 *
 * Written only from docs/reviews/JS_EPSILON_BRIEF_2026-11-03.md and paper/certifying-decoders/main.tex Sections II-IV.
 * Node.js built-ins only. Every weight and packing value is held as an exact integer (BigInt); no floating-point
 * operation touches a weight, a load, theta or epsilon.
 *
 *   node verify-epsilon.mjs --batch < cases.jsonl > verdicts.jsonl
 *   node verify-epsilon.mjs --instance <instance.json> --receipts <chunk.jsonl.gz> [--eps-max 1/1000000]
 *   node verify-epsilon.mjs --selftest
 *
 * THE MATHEMATICS (manuscript Sec. II, IV)
 *   G = (V, E, w), w >= 0, boundary b = -1, detectors are non-negative integers. T = fired, plus b when |fired| is odd.
 *   A correction J is a T-join: its odd-degree vertices are exactly T. (The brief says "ignoring b, exactly the fired
 *   detectors"; the two are equivalent, because a graph has an even number of odd-degree vertices, so b's parity is
 *   forced to |fired| mod 2. We check the full statement, b included.)
 *   A packing z >= 0 is a map from T-odd sets S (|S cap T| odd) to values. l_e = sum of z_S over S with e in delta(S).
 *   theta = min(1, min_{e: l_e > 0} w_e / l_e); eps = w(J) - theta * sum_S z_S, and w(J) - OPT <= eps (the Lemma).
 *   EXACT_OPTIMAL eps = 0 | EPSILON_OPTIMAL 0 < eps <= eps_max | NOT_PROVEN eps > eps_max |
 *   INVALID_INPUT outside the domain | VERIFIER_FAILURE eps < 0 (impossible for valid input; never an acceptance).
 *
 * EXACT REPRESENTATION
 *   Every finite binary64 value is M * 2^k with k >= -1074. Weights are held as integers W_e = w_e * 2^a, where 2^-a is
 *   the finest binary grid any weight of the graph uses (0 <= a <= 1074); a receipt's packing values as Z_S = z_S * 2^b
 *   with its own finest grid b. Both are exact. Loads are integer sums of Z. Ratios w_e / l_e = (W_e / L_e) * 2^(b-a)
 *   share one factor, so the tightest edge is found by cross-multiplication of integers (W_i * L_j < W_j * L_i); only the
 *   comparison with 1 and the final eps align the two grids by an exact shift. eps and the bound are reduced to lowest
 *   terms with gcd. (A fixed grid 2^1074 is equally exact but makes every product ~1100 bits: 4x slower, measured.)
 *
 * CONTRACT DECISIONS where the manuscript and the brief are silent (each refusal is also commented at its site):
 *   C1  Parallel edges (two edges with the same unordered endpoint pair, either orientation, equal weights or not)
 *       -> INVALID_INPUT. The manuscript: every decoding graph is simple, so an edge is named by its endpoints.
 *   C2  Self-loops (u = v, including (-1, -1)) -> INVALID_INPUT. A simple graph has none; a loop is in no delta(S).
 *   C3  An edge of the correction listed twice (either orientation) -> INVALID_INPUT: J is a set; a repeat has two
 *       readings (multiset weight 2w, or parity-cancelled), so the document has no single meaning.
 *   C4  Correction edges are unordered: [u, v] and [v, u] name the same edge. A pair that is not an edge -> INVALID.
 *   C5  V = {b} union {endpoints of edges}; b is a vertex even with no incident edge. A fired detector or a set
 *       member outside V -> INVALID_INPUT (S, T are subsets of V by definition).
 *   C6  Vertex ids must be JSON integer literals (no fraction, no exponent): 1.0, 1e0, true, "1", null -> INVALID.
 *       The JSON text is parsed by our own parser, which keeps number lexemes, so 1.0 is not silently 1 and ids
 *       beyond 2^53 stay distinct (no precision collapse). Ids of any size are accepted. "-0" is the integer 0.
 *       Negative ids other than -1 -> INVALID.
 *   C7  -1 in fired -> INVALID: b is not a detector; it enters T only by the parity of |fired|.
 *   C8  A detector repeated in fired -> INVALID (set vs XOR reading give different T).
 *   C9  A packing set listed twice (same members in any order) -> INVALID (the manuscript's conformance case).
 *       A member repeated inside one set -> INVALID (is [3, 3] the set {3} or a cancelled pair?).
 *   C10 Empty set -> INVALID (|S cap T| = 0 is even, so it is not a T-odd cut). Every LISTED set must be T-odd, even
 *       when its value is 0: a packing is a map whose keys are T-odd sets (checker step (ii): "every S is T-odd").
 *   C11 Zero-valued T-odd sets are accepted (z >= 0); they contribute nothing. S and its complement V \ S may both be
 *       listed: they are distinct T-odd sets (|T| is even) and both are legitimate cut constraints.
 *   C12 Hex literals: grammar [+-]0x<hex>[.<hex>]p[+-]<decimal>, case-insensitive letters, no whitespace. "0x" and the
 *       exponent are required (Python's float.fromhex would read the decimal-looking "1.8" as 0x1.8 = 1.5; we refuse).
 *       Non-canonical but exact spellings (0x2.0p-2, 0x0.8p+0, 0X1P-1, +0x1p+0) are accepted with their exact value.
 *       A literal whose exact value is NOT a finite binary64 value (more than 53 significant bits, below 2^-1074,
 *       above the largest double) -> INVALID: the claims are over double-valued weights, and silently rounding would
 *       change the instance. "inf", "-inf", "nan" (any case, "infinity" too) -> INVALID (non-finite).
 *   C13 Negative zero ("-0x0.0p+0") is the rational 0, which is non-negative -> accepted as 0, for weights and values.
 *   C14 A weight or a packing value given as a JSON number instead of a hex string -> INVALID (decimal round-trips are
 *       not exact across languages; the manuscript requires the 64-bit pattern).
 *   C15 eps_max must be a string in canonical form: "p/q" in lowest terms with q > 1, or "p"; non-negative; no sign, no
 *       leading zeros, no whitespace ("0/5", "2/4", "1/1", "+1/5", "1e-6" -> INVALID). At most 8192 characters
 *       (resource bound). In batch mode eps_max is required (no default); replay defaults to 1/1000000.
 *   C16 Duplicate keys in any JSON object -> INVALID (a last-wins parser and a first-wins parser read different
 *       documents). A byte-order mark, invalid UTF-8, or non-JSON (NaN, Infinity, trailing text) -> INVALID.
 *   C17 Unknown extra fields are ignored. In replay, the receipt's status/epsilon/lower_bound are never read, and any
 *       edges/eps_max inside a receipt are ignored (the instance file and the command line govern).
 *   C18 Empty graph, empty fired, empty correction, empty packing are all in the domain (theta = 1 when no edge is
 *       loaded; an empty packing proves eps = w(J)).
 *   C19 A missing id (or shot) is echoed as null; the id may be any JSON value and is echoed verbatim.
 *   C20 An unexpected internal exception -> VERIFIER_FAILURE with epsilon/lower_bound null and reason "internal error"
 *       (the only deviation from "present otherwise": there is no bound to report). The self-test fuzzes for it.
 *       An internal re-check that theta * z is feasible also reports VERIFIER_FAILURE if it ever fails.
 *   C21 Replay: an instance whose edges_sha256 does not equal SHA-256 of the edges array re-serialised as compact
 *       Python-style JSON (separators , and :, ensure_ascii escapes) is refused, as is an instance whose graph is
 *       invalid: message on stderr, exit 2, nothing on stdout. Receipts may be gzip (detected by magic) or plain.
 *   C22 Batch lines are split on LF only; a trailing CR is JSON whitespace. An empty line is a line (INVALID_INPUT,
 *       id null). A final segment without LF is a line if non-empty.
 *   C23 Nesting deeper than 64 levels -> INVALID (resource bound; the schema needs 4).
 */

import { createHash } from 'node:crypto';
import { createReadStream, openSync, readSync, closeSync, readFileSync } from 'node:fs';
import { createGunzip, gzipSync } from 'node:zlib';
import { Readable } from 'node:stream';
import { once } from 'node:events';

const SCALE_BITS = 1074n; // the finest binary64 grid is 2^-1074
const BOUNDARY = '-1';
const MAX_DEPTH = 64;
const MAX_RATIONAL_CHARS = 8192;
const DEFAULT_EPS_MAX = '1/1000000';
const STATUSES = ['EXACT_OPTIMAL', 'EPSILON_OPTIMAL', 'NOT_PROVEN', 'INVALID_INPUT', 'VERIFIER_FAILURE'];

// ------------------------------------------------------------------------------------------------ errors, helpers

class Invalid extends Error {
  constructor(code, detail) {
    super(`${code}: ${detail}`);
    this.code = code;
  }
}

class JsonError extends Error {
  constructor(code, detail) {
    super(detail);
    this.code = code;
  }
}

function short(s, max = 60) {
  s = String(s);
  return s.length <= max ? s : s.slice(0, max) + '...';
}

function quote(s) {
  return JSON.stringify(short(s));
}

// ------------------------------------------------------------------------------------------------ JSON parser
// Our own parser: numbers keep their lexeme (JNum), objects become Maps, duplicate keys are refused (C16), nesting is
// bounded (C23). Recursion depth is bounded by MAX_DEPTH, so no input can overflow the stack.

class JNum {
  constructor(lex) {
    this.lex = lex;
  }
}

function kindOf(v) {
  if (v === null) return 'null';
  if (typeof v === 'boolean') return 'boolean';
  if (typeof v === 'string') return 'string';
  if (v instanceof JNum) return `number ${short(v.lex, 30)}`;
  if (Array.isArray(v)) return 'array';
  if (v instanceof Map) return 'object';
  return typeof v;
}

const NUM_RE = /-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?/y;
const HEX4_RE = /^[0-9a-fA-F]{4}$/;

function parseJson(text) {
  let i = 0;
  const n = text.length;
  const fail = (msg) => {
    throw new JsonError('json', `${msg} at offset ${i}`);
  };
  const skipWs = () => {
    while (i < n) {
      const c = text.charCodeAt(i);
      if (c === 0x20 || c === 0x09 || c === 0x0a || c === 0x0d) i++;
      else break;
    }
  };
  const parseString = () => {
    i++; // opening quote
    let out = '';
    let start = i;
    for (;;) {
      if (i >= n) fail('unterminated string');
      const c = text.charCodeAt(i);
      if (c === 0x22) {
        out += text.slice(start, i);
        i++;
        return out;
      }
      if (c < 0x20) fail('raw control character in string');
      if (c === 0x5c) {
        out += text.slice(start, i);
        i++;
        if (i >= n) fail('unterminated escape');
        const e = text[i];
        if (e === '"') out += '"';
        else if (e === '\\') out += '\\';
        else if (e === '/') out += '/';
        else if (e === 'b') out += '\b';
        else if (e === 'f') out += '\f';
        else if (e === 'n') out += '\n';
        else if (e === 'r') out += '\r';
        else if (e === 't') out += '\t';
        else if (e === 'u') {
          const h = text.slice(i + 1, i + 5);
          if (!HEX4_RE.test(h)) fail('bad \\u escape');
          out += String.fromCharCode(parseInt(h, 16));
          i += 4;
        } else fail('bad escape');
        i++;
        start = i;
        continue;
      }
      i++;
    }
  };
  const parseValue = (depth) => {
    skipWs();
    if (i >= n) fail('unexpected end of input');
    const c = text[i];
    if (c === '{' || c === '[') {
      if (depth >= MAX_DEPTH) fail(`nesting deeper than ${MAX_DEPTH}`);
      i++;
      skipWs();
      if (c === '[') {
        const arr = [];
        if (text[i] === ']') {
          i++;
          return arr;
        }
        for (;;) {
          arr.push(parseValue(depth + 1));
          skipWs();
          if (text[i] === ',') {
            i++;
            continue;
          }
          if (text[i] === ']') {
            i++;
            return arr;
          }
          fail("expected ',' or ']'");
        }
      }
      const obj = new Map();
      if (text[i] === '}') {
        i++;
        return obj;
      }
      for (;;) {
        skipWs();
        if (text[i] !== '"') fail('expected a string key');
        const k = parseString();
        // REFUSAL (C16): a duplicated key makes the document's meaning depend on the reader's last-wins/first-wins rule.
        if (obj.has(k)) throw new JsonError('duplicate-key', `object key ${quote(k)} appears twice`);
        skipWs();
        if (text[i] !== ':') fail("expected ':'");
        i++;
        obj.set(k, parseValue(depth + 1));
        skipWs();
        if (text[i] === ',') {
          i++;
          continue;
        }
        if (text[i] === '}') {
          i++;
          return obj;
        }
        fail("expected ',' or '}'");
      }
    }
    if (c === '"') return parseString();
    if (c === '-' || (c >= '0' && c <= '9')) {
      NUM_RE.lastIndex = i;
      const m = NUM_RE.exec(text);
      if (!m) fail('malformed number');
      i = NUM_RE.lastIndex;
      return new JNum(m[0]);
    }
    if (text.startsWith('true', i)) {
      i += 4;
      return true;
    }
    if (text.startsWith('false', i)) {
      i += 5;
      return false;
    }
    if (text.startsWith('null', i)) {
      i += 4;
      return null;
    }
    fail(`unexpected character ${JSON.stringify(c)}`);
  };
  const v = parseValue(0);
  skipWs();
  if (i !== n) fail('trailing characters after the JSON value');
  return v;
}

// Verbatim JSON text of a parsed value (for echoing ids).
function toJsonText(v) {
  if (v === null) return 'null';
  if (v === true) return 'true';
  if (v === false) return 'false';
  if (v instanceof JNum) return v.lex;
  if (typeof v === 'string') return JSON.stringify(v);
  if (Array.isArray(v)) return '[' + v.map(toJsonText).join(',') + ']';
  if (v instanceof Map) return '{' + [...v].map(([k, x]) => JSON.stringify(k) + ':' + toJsonText(x)).join(',') + '}';
  return 'null';
}

// Compact JSON as Python's json.dumps(x, separators=(',', ':')) writes it (ensure_ascii), for the instance
// fingerprint (C21). Integers are written canonically ("-0" -> "0"); a non-integer number is written by its lexeme,
// which may differ from Python's float repr and then fails the fingerprint (refusal-safe: such a graph is refused
// by C6 anyway).
function pyString(s) {
  let out = '"';
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    if (c === 0x22) out += '\\"';
    else if (c === 0x5c) out += '\\\\';
    else if (c >= 0x20 && c <= 0x7e) out += s[i];
    else if (c === 0x0a) out += '\\n';
    else if (c === 0x0d) out += '\\r';
    else if (c === 0x09) out += '\\t';
    else if (c === 0x08) out += '\\b';
    else if (c === 0x0c) out += '\\f';
    else out += '\\u' + c.toString(16).padStart(4, '0');
  }
  return out + '"';
}

const INT_RE = /^-?(?:0|[1-9][0-9]*)$/;

function pyCompact(v) {
  if (v === null) return 'null';
  if (v === true) return 'true';
  if (v === false) return 'false';
  if (v instanceof JNum) return v.lex === '-0' ? '0' : v.lex;
  if (typeof v === 'string') return pyString(v);
  if (Array.isArray(v)) return '[' + v.map(pyCompact).join(',') + ']';
  if (v instanceof Map) return '{' + [...v].map(([k, x]) => pyString(k) + ':' + pyCompact(x)).join(',') + '}';
  return 'null';
}

// ------------------------------------------------------------------------------------------------ exact numbers

function gcd(a, b) {
  if (a < 0n) a = -a;
  if (b < 0n) b = -b;
  while (b !== 0n) {
    const t = a % b;
    a = b;
    b = t;
  }
  return a;
}

function ctz(x) {
  // x > 0n: number of trailing zero bits
  return BigInt((x & -x).toString(2).length - 1);
}

// Lowest terms of n/d, d > 0. Common powers of two are removed first (all denominators here are 2^1074 or
// L * 2^1074), then Euclid on what remains.
function reduce(n, d) {
  if (n === 0n) return [0n, 1n];
  const tn = ctz(n < 0n ? -n : n);
  const td = ctz(d);
  const k = tn < td ? tn : td;
  n >>= k; // exact: 2^k divides n
  d >>= k;
  const g = gcd(n, d);
  return g === 1n ? [n, d] : [n / g, d / g];
}

function fmtRational(n, d) {
  const [p, q] = reduce(n, d);
  return q === 1n ? p.toString() : `${p}/${q}`;
}

const RAT_RE = /^(0|[1-9][0-9]*)(?:\/([1-9][0-9]*))?$/;

// eps_max: canonical non-negative rational (C15).
function parseRational(v, where) {
  if (typeof v !== 'string') throw new Invalid('eps-max', `${where} must be a string "p/q" or "p", got ${kindOf(v)}`);
  if (v.length > MAX_RATIONAL_CHARS) throw new Invalid('eps-max', `${where} longer than ${MAX_RATIONAL_CHARS} characters`);
  // REFUSAL: a negative tolerance is not a bound on a slack; the EPSILON_OPTIMAL interval would be empty and meaningless.
  if (v.startsWith('-')) throw new Invalid('eps-max', `${where} ${quote(v)} is negative`);
  const m = RAT_RE.exec(v);
  // REFUSAL: the wire format fixes one spelling per rational; anything else (decimals, exponents, signs, spaces,
  // leading zeros, a zero denominator) is not that format.
  if (!m) throw new Invalid('eps-max', `${where} ${quote(v)} is not a canonical rational "p/q" or "p"`);
  const p = BigInt(m[1]);
  if (m[2] === undefined) return { n: p, d: 1n };
  const q = BigInt(m[2]);
  if (q === 1n) throw new Invalid('eps-max', `${where} ${quote(v)}: denominator 1 must be omitted`);
  if (gcd(p, q) !== 1n) throw new Invalid('eps-max', `${where} ${quote(v)} is not in lowest terms`);
  return { n: p, d: q };
}

// Exact reading of a hexadecimal floating literal (C12). Returns {neg, M, shift} with value (-1)^neg * M * 2^shift and
// M odd (or M = 0), when the value is exactly a finite binary64 value; otherwise {error}.
const HEX_RE = /^([+-]?)0[xX]([0-9a-fA-F]*)(?:\.([0-9a-fA-F]*))?[pP]([+-]?)([0-9]+)$/;
const NONFINITE_RE = /^[+-]?(?:inf|infinity|nan)$/i;

function hexToDyadic(s) {
  if (NONFINITE_RE.test(s)) return { error: `non-finite value ${quote(s)}` };
  const m = HEX_RE.exec(s);
  if (!m) return { error: `${quote(s)} is not a hexadecimal floating literal [+-]0x<hex>[.<hex>]p[+-]<exp>` };
  const neg = m[1] === '-';
  const ip = m[2];
  const fp = m[3] === undefined ? '' : m[3];
  if (ip.length + fp.length === 0) return { error: `${quote(s)} has no hexadecimal digits` };
  const digits = ip + fp;
  let a = 0;
  while (a < digits.length && digits.charCodeAt(a) === 48) a++;
  let b = digits.length;
  while (b > a && digits.charCodeAt(b - 1) === 48) b--;
  if (a === b) return { neg, M: 0n, shift: 0n }; // zero, whatever the exponent
  // 15 or more significant hex digits always span more than 53 bits; the exact test below decides the rest.
  if (b - a > 16) return { error: `${quote(s)} has more significant bits than binary64 holds (not exactly a double)` };
  let ed = m[5];
  let z = 0;
  while (z < ed.length - 1 && ed.charCodeAt(z) === 48) z++;
  ed = ed.slice(z);
  // An exponent of 31+ digits is beyond any shift the digit counts of a real line can compensate.
  if (ed.length > 30) return { error: `${quote(s)} has an exponent outside the binary64 range` };
  let shift = BigInt(ed) * (m[4] === '-' ? -1n : 1n) - 4n * BigInt(fp.length) + 4n * BigInt(digits.length - b);
  let M = BigInt('0x' + digits.slice(a, b));
  while ((M & 1n) === 0n) {
    M >>= 1n;
    shift += 1n;
  }
  const bits = BigInt(M.toString(2).length);
  if (bits > 53n) return { error: `${quote(s)} needs ${bits} significant bits; binary64 has 53 (not exactly a double)` };
  if (shift < -1074n) return { error: `${quote(s)} is finer than the binary64 grid 2^-1074 (not exactly a double)` };
  if (shift + bits - 1n > 1023n) return { error: `${quote(s)} exceeds the largest finite binary64 value` };
  return { neg, M, shift };
}

// A weight or packing value: exact, finite, non-negative; returned as {M, e} with value M * 2^e (M = 0n for zero).
function parseValue(v, code, where) {
  // REFUSAL (C14): a JSON number is a decimal; the manuscript requires the double's exact bit pattern.
  if (typeof v !== 'string') throw new Invalid(code, `${where} must be a hex float string, got ${kindOf(v)}`);
  const r = hexToDyadic(v);
  // REFUSAL (C12): non-finite, malformed, or not exactly a binary64 value.
  if (r.error) throw new Invalid(code, `${where}: ${r.error}`);
  if (r.M === 0n) return { M: 0n, e: 0n }; // C13: -0 is the rational 0
  // REFUSAL: negative weights leave the domain (w >= 0); a negative z_S breaks weak duality (z >= 0).
  if (r.neg) throw new Invalid(code, `${where} is negative (${quote(v)})`);
  return { M: r.M, e: r.shift };
}

// The finest grid of a list of {M, e}: the least b >= 0 with every value * 2^b an integer.
function gridOf(vals) {
  let b = 0n;
  for (const x of vals) if (x.M !== 0n && -x.e > b) b = -x.e;
  return b;
}

// Sign of x / 2^ax - y / 2^ay, by an exact shift onto the finer grid.
function cmpScaled(x, ax, y, ay) {
  const l = ax >= ay ? x : x << (ay - ax);
  const r = ax >= ay ? y << (ax - ay) : y;
  return l < r ? -1 : l > r ? 1 : 0;
}

// ------------------------------------------------------------------------------------------------ graph

function vertexId(v, where) {
  // REFUSAL (C6): booleans, strings, null, arrays are not vertex ids (Python's True == 1 must not become vertex 1).
  if (!(v instanceof JNum)) throw new Invalid('vertex-id', `${where} must be an integer, got ${kindOf(v)}`);
  // REFUSAL (C6): 1.0 or 1e0 is a float in other readers; a vertex is named by an integer literal only.
  if (!INT_RE.test(v.lex)) throw new Invalid('vertex-id', `${where} ${short(v.lex, 30)} is not an integer literal`);
  if (v.lex === '-0') return '0';
  // REFUSAL: detectors are non-negative; -1 is the boundary; any other negative id names nothing.
  if (v.lex.charCodeAt(0) === 45 && v.lex !== BOUNDARY) {
    throw new Invalid('vertex-id', `${where} ${short(v.lex, 30)} is negative; only -1 (the boundary) may be`);
  }
  return v.lex;
}

function pairKey(u, v) {
  return u < v ? u + ' ' + v : v + ' ' + u;
}

function vname(x) {
  return x === BOUNDARY ? 'the boundary b' : `vertex ${short(x, 30)}`;
}

function buildGraph(edgesVal) {
  if (!Array.isArray(edgesVal)) throw new Invalid('edges', `edges must be an array, got ${kindOf(edgesVal)}`);
  const U = [];
  const V = [];
  const W = [];
  const adj = new Map([[BOUNDARY, []]]); // C5: b is always a vertex of G
  const index = new Map();
  for (let k = 0; k < edgesVal.length; k++) {
    const e = edgesVal[k];
    if (!Array.isArray(e) || e.length !== 3) throw new Invalid('edge-shape', `edge #${k} must be [u, v, "hexweight"]`);
    const u = vertexId(e[0], `edge #${k} endpoint u`);
    const v = vertexId(e[1], `edge #${k} endpoint v`);
    // REFUSAL (C2): a self-loop is not an edge of a simple graph and crosses no cut.
    if (u === v) throw new Invalid('self-loop', `edge #${k} joins ${vname(u)} to itself`);
    const w = parseValue(e[2], 'weight', `edge #${k} (${short(u, 30)}, ${short(v, 30)}) weight`);
    const key = pairKey(u, v);
    // REFUSAL (C1): with parallel edges a correction naming (u, v) would have no unambiguous weight.
    if (index.has(key)) {
      throw new Invalid('parallel-edge', `edge #${k} repeats the endpoint pair of edge #${index.get(key)} (${short(u, 30)}, ${short(v, 30)})`);
    }
    index.set(key, k);
    U.push(u);
    V.push(v);
    W.push(w);
    if (!adj.has(u)) adj.set(u, []);
    if (!adj.has(v)) adj.set(v, []);
    adj.get(u).push(k);
    adj.get(v).push(k);
  }
  // Every weight onto the graph's finest grid: W_e = w_e * 2^a, an integer.
  const a = gridOf(W);
  for (let k = 0; k < W.length; k++) W[k] = W[k].M << (W[k].e + a);
  return { U, V, W, a, adj, index };
}

// ------------------------------------------------------------------------------------------------ the decision

function classify(epsN, epsD, epsMax) {
  if (epsN < 0n) return 'VERIFIER_FAILURE';
  if (epsN === 0n) return 'EXACT_OPTIMAL';
  return epsN * epsMax.d <= epsMax.n * epsD ? 'EPSILON_OPTIMAL' : 'NOT_PROVEN';
}

const STATUS_NOTE = {
  EXACT_OPTIMAL: 'eps = 0: the scaled packing value equals w(J)',
  EPSILON_OPTIMAL: '0 < eps <= eps_max',
  NOT_PROVEN: 'eps > eps_max',
  VERIFIER_FAILURE: 'the bound exceeds w(J) (eps < 0), impossible for valid input',
};

function decide(g, firedVal, corrVal, dualsVal, epsMax) {
  // (a) The syndrome and T.
  if (!Array.isArray(firedVal)) throw new Invalid('fired', `fired must be an array, got ${kindOf(firedVal)}`);
  const fired = new Set();
  for (let k = 0; k < firedVal.length; k++) {
    const d = vertexId(firedVal[k], `fired[${k}]`);
    // REFUSAL (C7): b is not a detector; listing it would make T ambiguous.
    if (d === BOUNDARY) throw new Invalid('fired', `fired[${k}] is -1, the boundary; b enters T only by parity`);
    // REFUSAL (C5): T is a subset of V; a detector with no edge can never be explained.
    if (!g.adj.has(d)) throw new Invalid('fired', `fired[${k}] = ${short(d, 30)} is not a vertex of the graph`);
    // REFUSAL (C8): a syndrome is a set.
    if (fired.has(d)) throw new Invalid('fired', `detector ${short(d, 30)} is listed twice in fired`);
    fired.add(d);
  }
  const T = new Set(fired);
  if (fired.size % 2 === 1) T.add(BOUNDARY);

  // (b) The correction: a set of edges of G whose odd-degree vertices are exactly T (checker step (i)).
  if (!Array.isArray(corrVal)) throw new Invalid('correction-shape', `correction must be an array, got ${kindOf(corrVal)}`);
  const inJ = new Set();
  const parity = new Map();
  let WJ = 0n;
  for (let k = 0; k < corrVal.length; k++) {
    const p = corrVal[k];
    if (!Array.isArray(p) || p.length !== 2) throw new Invalid('correction-shape', `correction[${k}] must be a pair [u, v]`);
    const u = vertexId(p[0], `correction[${k}][0]`);
    const v = vertexId(p[1], `correction[${k}][1]`);
    const idx = g.index.get(pairKey(u, v));
    // REFUSAL (C4): J must be a subset of E.
    if (idx === undefined) {
      throw new Invalid('correction-edge', `correction[${k}] = (${short(u, 30)}, ${short(v, 30)}) is not an edge of the graph`);
    }
    // REFUSAL (C3): J is a set of edges.
    if (inJ.has(idx)) throw new Invalid('correction-repeat', `edge (${short(u, 30)}, ${short(v, 30)}) is listed twice in the correction`);
    inJ.add(idx);
    WJ += g.W[idx];
    parity.set(u, (parity.get(u) ?? 0) ^ 1);
    parity.set(v, (parity.get(v) ?? 0) ^ 1);
  }
  // REFUSAL: a correction that does not explain the syndrome is not a T-join; weak duality needs |J cap delta(S)| odd.
  for (const t of T) {
    if (parity.get(t) !== 1) throw new Invalid('not-t-join', `the correction leaves ${vname(t)} (in T) with even degree`);
  }
  for (const [x, p] of parity) {
    if (p === 1 && !T.has(x)) throw new Invalid('not-t-join', `the correction gives ${vname(x)} (not in T) odd degree`);
  }

  // (c) The packing: distinct T-odd sets of V with finite non-negative values (checker step (ii)).
  if (!Array.isArray(dualsVal)) throw new Invalid('set-shape', `duals must be an array, got ${kindOf(dualsVal)}`);
  const seen = new Map();
  const listed = [];
  for (let k = 0; k < dualsVal.length; k++) {
    const d = dualsVal[k];
    if (!Array.isArray(d) || d.length !== 2 || !Array.isArray(d[0])) {
      throw new Invalid('set-shape', `duals[${k}] must be [[member, ...], "hexvalue"]`);
    }
    const mem = d[0];
    const S = new Set();
    let tCount = 0;
    for (let j = 0; j < mem.length; j++) {
      const x = vertexId(mem[j], `duals[${k}] member ${j}`);
      // REFUSAL (C5): S must be a subset of V.
      if (!g.adj.has(x)) throw new Invalid('set-member', `duals[${k}] member ${short(x, 30)} is not a vertex of the graph`);
      // REFUSAL (C9): a set lists each element once.
      if (S.has(x)) throw new Invalid('set-repeat-member', `duals[${k}] lists ${vname(x)} twice`);
      S.add(x);
      if (T.has(x)) tCount++;
    }
    // REFUSAL (C10): the empty set meets T in 0 vertices, an even number; it is not a cut constraint.
    if (S.size === 0) throw new Invalid('set-empty', `duals[${k}] is the empty set, which is not T-odd`);
    // REFUSAL: an even set is not a T-cut constraint; admitting one lets the bound exceed the optimum (manuscript).
    if (tCount % 2 === 0) {
      throw new Invalid('set-not-t-odd', `duals[${k}] meets T in ${tCount} vertices, an even number (T = fired${T.has(BOUNDARY) ? ' + b' : ''})`);
    }
    const z = parseValue(d[1], 'dual-value', `duals[${k}] value`);
    const canon = [...S].sort().join(',');
    // REFUSAL (C9): a packing is a map from sets to values; a list encodes one only while its keys are distinct.
    if (seen.has(canon)) throw new Invalid('set-repeated', `duals[${k}] repeats the set of duals[${seen.get(canon)}]`);
    seen.set(canon, k);
    listed.push([S, z]);
  }
  // Every packing value onto the receipt's finest grid: Z_S = z_S * 2^b, an integer.
  const a = g.a;
  const b = gridOf(listed.map((x) => x[1]));
  const active = [];
  let sumZ = 0n;
  for (const [S, z] of listed) {
    if (z.M === 0n) continue;
    const Z = z.M << (z.e + b);
    sumZ += Z;
    active.push([S, Z]);
  }

  // (d) Loads, walked from each set's boundary (checker step (iii)): an edge with exactly one endpoint in S is reached
  // exactly once, from that endpoint; an edge with both endpoints in S is skipped from both.
  const load = new Map();
  for (const [S, z] of active) {
    for (const u of S) {
      for (const e of g.adj.get(u)) {
        const other = g.U[e] === u ? g.V[e] : g.U[e];
        if (!S.has(other)) load.set(e, (load.get(e) ?? 0n) + z);
      }
    }
  }

  // (e) theta = min(1, min w_e / l_e) (checker step (iv)). Real values: w_e = W_e / 2^a, l_e = L_e / 2^b, so
  // w_e / l_e = (W_e / L_e) * 2^(b - a) for every edge alike, and the tightest edge minimises W_e / L_e, compared by
  // cross-multiplication of integers.
  let bestE = -1;
  let bW = 0n;
  let bL = 1n;
  for (const [e, L] of load) {
    if (L <= 0n) continue;
    const W = g.W[e];
    if (bestE < 0 || W * bL < bW * L) {
      bestE = e;
      bW = W;
      bL = L;
    }
  }
  // theta = 1 unless w_best / l_best = (bW / 2^a) / (bL / 2^b) < 1, i.e. bW / 2^a < bL / 2^b.
  const thetaOne = bestE < 0 || cmpScaled(bW, a, bL, b) >= 0;
  let lbN;
  let lbD;
  let epsN;
  let epsD;
  if (thetaOne) {
    // bound = sumZ / 2^b; eps = WJ / 2^a - sumZ / 2^b, over the finer grid 2^c.
    const c = a > b ? a : b;
    lbN = sumZ;
    lbD = 1n << b;
    epsN = (WJ << (c - a)) - (sumZ << (c - b));
    epsD = 1n << c;
  } else {
    // theta = (bW / 2^a) / (bL / 2^b); bound = theta * sumZ / 2^b = bW * sumZ / (bL * 2^a): the grid b cancels.
    lbN = bW * sumZ;
    lbD = bL << a;
    epsN = WJ * bL - bW * sumZ;
    epsD = bL << a;
  }
  const epsilon = fmtRational(epsN, epsD);
  const lowerBound = fmtRational(lbN, lbD);

  // Internal re-check (C20): theta * l_e <= w_e on every loaded edge.
  for (const [e, L] of load) {
    const feasible = thetaOne ? cmpScaled(L, b, g.W[e], a) <= 0 : bW * L <= g.W[e] * bL;
    if (!feasible) {
      return { status: 'VERIFIER_FAILURE', epsilon, lower_bound: lowerBound, reason: `internal: scaled packing infeasible at edge #${e}` };
    }
  }

  const status = classify(epsN, epsD, epsMax);
  const thetaNote = bestE < 0
    ? 'theta = 1 (no edge carries load)'
    : thetaOne
      ? 'theta = 1 (packing feasible)'
      : `theta < 1 (packing overloaded; tightest edge (${short(g.U[bestE], 30)}, ${short(g.V[bestE], 30)}))`;
  return {
    status,
    epsilon,
    lower_bound: lowerBound,
    reason: `${STATUS_NOTE[status]}; ${thetaNote}; ${dualsVal.length} set(s), ${inJ.size} correction edge(s)`,
  };
}

// ------------------------------------------------------------------------------------------------ records

function verdictLine(idKey, idText, status, epsilon, lowerBound, reason) {
  return `{${JSON.stringify(idKey)}:${idText},"status":${JSON.stringify(status)},"epsilon":${epsilon === null ? 'null' : JSON.stringify(epsilon)},"lower_bound":${lowerBound === null ? 'null' : JSON.stringify(lowerBound)},"reason":${JSON.stringify(reason)}}`;
}

const BATCH_FIELDS = ['edges', 'fired', 'correction', 'duals', 'eps_max'];
const RECEIPT_FIELDS = ['fired', 'correction', 'duals'];

// One input line -> one output line. Never throws.
function processRecord(text, mode, ctx) {
  const idKey = mode === 'batch' ? 'id' : 'shot';
  let idText = 'null';
  try {
    try {
      let doc;
      try {
        doc = parseJson(text);
      } catch (e) {
        if (e instanceof JsonError) throw new Invalid(e.code, e.message);
        throw e;
      }
      if (!(doc instanceof Map)) throw new Invalid('json', `the line is a JSON ${kindOf(doc)}, not an object`);
      if (doc.has(idKey)) idText = toJsonText(doc.get(idKey));
      let g;
      let epsMax;
      if (mode === 'batch') {
        for (const f of BATCH_FIELDS) if (!doc.has(f)) throw new Invalid('missing-field', `field ${JSON.stringify(f)} is missing`);
        epsMax = parseRational(doc.get('eps_max'), 'eps_max');
        g = buildGraph(doc.get('edges'));
      } else {
        for (const f of RECEIPT_FIELDS) if (!doc.has(f)) throw new Invalid('missing-field', `field ${JSON.stringify(f)} is missing`);
        g = ctx.graph;
        epsMax = ctx.epsMax;
      }
      const r = decide(g, doc.get('fired'), doc.get('correction'), doc.get('duals'), epsMax);
      return verdictLine(idKey, idText, r.status, r.epsilon, r.lower_bound, r.reason);
    } catch (e) {
      if (e instanceof Invalid) return verdictLine(idKey, idText, 'INVALID_INPUT', null, null, e.message);
      return verdictLine(idKey, idText, 'VERIFIER_FAILURE', null, null, `internal error: ${short(e && e.message ? e.message : e, 200)}`);
    }
  } catch {
    return `{${JSON.stringify(idKey)}:null,"status":"VERIFIER_FAILURE","epsilon":null,"lower_bound":null,"reason":"internal error"}`;
  }
}

// Split a byte stream on LF (C22).
async function* byteLines(stream) {
  let pending = [];
  for await (const chunk of stream) {
    const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    let start = 0;
    for (;;) {
      const nl = buf.indexOf(10, start);
      if (nl < 0) {
        if (start < buf.length) pending.push(buf.subarray(start));
        break;
      }
      pending.push(buf.subarray(start, nl));
      yield pending.length === 1 ? pending[0] : Buffer.concat(pending);
      pending = [];
      start = nl + 1;
    }
  }
  if (pending.length) {
    const last = Buffer.concat(pending);
    if (last.length) yield last;
  }
}

async function runLines(stream, handle, write, idKey) {
  const dec = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true });
  let count = 0;
  for await (const buf of byteLines(stream)) {
    let out;
    let text = null;
    try {
      text = dec.decode(buf);
    } catch {
      text = null;
    }
    // REFUSAL (C16): bytes that are not UTF-8 are not a JSON text.
    if (text === null) out = verdictLine(idKey, 'null', 'INVALID_INPUT', null, null, 'encoding: the line is not valid UTF-8');
    else out = handle(text);
    count++;
    const p = write(out);
    if (p) await p;
  }
  return count;
}

function stdoutWrite(line) {
  if (!process.stdout.write(line + '\n')) return once(process.stdout, 'drain');
  return null;
}

// ------------------------------------------------------------------------------------------------ replay

function loadInstance(buf) {
  let text;
  try {
    text = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(buf);
  } catch {
    throw new Invalid('instance', 'the instance file is not valid UTF-8');
  }
  let doc;
  try {
    doc = parseJson(text);
  } catch (e) {
    if (e instanceof JsonError) throw new Invalid('instance', e.message);
    throw e;
  }
  if (!(doc instanceof Map)) throw new Invalid('instance', 'the instance is not a JSON object');
  if (!doc.has('edges')) throw new Invalid('instance', 'the instance has no "edges"');
  const claimed = doc.get('edges_sha256');
  if (typeof claimed !== 'string') throw new Invalid('instance', 'the instance has no string "edges_sha256"');
  const got = createHash('sha256').update(pyCompact(doc.get('edges')), 'utf8').digest('hex');
  // REFUSAL (C21): the fingerprint binds a certificate to its graph; a mismatch means the graph is not the one hashed.
  if (got !== claimed) throw new Invalid('fingerprint', `edges_sha256 is ${quote(claimed)} but the edges hash to ${got}`);
  return buildGraph(doc.get('edges'));
}

function openReceipts(path) {
  const fd = openSync(path, 'r');
  const head = Buffer.alloc(2);
  let got;
  try {
    got = readSync(fd, head, 0, 2, 0);
  } finally {
    closeSync(fd);
  }
  const raw = createReadStream(path);
  if (got === 2 && head[0] === 0x1f && head[1] === 0x8b) {
    const gz = createGunzip();
    raw.on('error', (e) => gz.destroy(e));
    return raw.pipe(gz);
  }
  return raw;
}

function die(msg) {
  process.stderr.write(`verify-epsilon: ${msg}\n`);
  process.exitCode = 2;
}

async function replayMain(instPath, recPath, epsStr) {
  let epsMax;
  try {
    epsMax = parseRational(epsStr, '--eps-max');
  } catch (e) {
    return die(`refused: ${e.message}`);
  }
  let graph;
  try {
    graph = loadInstance(readFileSync(instPath));
  } catch (e) {
    return die(`instance ${instPath} refused: ${e.message}`);
  }
  const ctx = { graph, epsMax };
  const t0 = process.hrtime.bigint();
  let n;
  try {
    n = await runLines(openReceipts(recPath), (t) => processRecord(t, 'replay', ctx), stdoutWrite, 'shot');
  } catch (e) {
    return die(`reading ${recPath} failed: ${e && e.message ? e.message : e}`);
  }
  const ms = Number(process.hrtime.bigint() - t0) / 1e6;
  process.stderr.write(`verify-epsilon: replayed ${n} receipt(s) against ${graph.U.length} edges in ${ms.toFixed(1)} ms (${n ? (ms / n).toFixed(3) : '0'} ms/receipt)\n`);
}

// ------------------------------------------------------------------------------------------------ self-test

// Python float.hex() of a double, from its bit pattern (used to build vectors and for the round-trip check).
function floatHex(x) {
  const dv = new DataView(new ArrayBuffer(8));
  dv.setFloat64(0, x);
  const hi = dv.getUint32(0);
  const lo = dv.getUint32(4);
  const sign = hi >>> 31;
  const ex = (hi >>> 20) & 0x7ff;
  const mant = (BigInt(hi & 0xfffff) << 32n) | BigInt(lo);
  if (ex === 0x7ff) return mant ? 'nan' : sign ? '-inf' : 'inf';
  const s = sign ? '-' : '';
  if (ex === 0 && mant === 0n) return `${s}0x0.0p+0`;
  const e = ex === 0 ? -1022 : ex - 1023;
  return `${s}0x${ex === 0 ? '0' : '1'}.${mant.toString(16).padStart(13, '0')}p${e < 0 ? '-' : '+'}${Math.abs(e)}`;
}

function mulberry32(seed) {
  let a = seed >>> 0;
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return { next, int: (k) => Math.floor(next() * k), u32: () => Math.floor(next() * 4294967296) >>> 0 };
}

// Independent exact fractions for the oracle (no shared code with the 2^1074 scaling of the checker).
class Q {
  constructor(n, d = 1n) {
    if (d < 0n) {
      n = -n;
      d = -d;
    }
    const g = gcd(n, d) || 1n;
    this.n = n / g;
    this.d = d / g;
  }
  add(o) { return new Q(this.n * o.d + o.n * this.d, this.d * o.d); }
  sub(o) { return new Q(this.n * o.d - o.n * this.d, this.d * o.d); }
  mul(o) { return new Q(this.n * o.n, this.d * o.d); }
  div(o) { return new Q(this.n * o.d, this.d * o.n); }
  cmp(o) { const x = this.n * o.d - o.n * this.d; return x < 0n ? -1 : x > 0n ? 1 : 0; }
  str() { return this.d === 1n ? `${this.n}` : `${this.n}/${this.d}`; }
  static parse(s) {
    const m = /^(-?[0-9]+)(?:\/([0-9]+))?$/.exec(s);
    return m ? new Q(BigInt(m[1]), m[2] ? BigInt(m[2]) : 1n) : null;
  }
}

async function selftest() {
  let pass = 0;
  const failures = [];
  const ok = (cond, name, detail) => {
    if (cond) pass++;
    else failures.push(detail ? `${name}: ${detail}` : name);
  };
  const groups = [];
  const group = (name, fn) => {
    const before = pass + failures.length;
    const fb = failures.length;
    fn();
    groups.push(`  ${name}: ${pass + failures.length - before} checks, ${failures.length - fb} failed`);
  };
  const run = (input) => {
    const text = typeof input === 'string' ? input : JSON.stringify(input);
    const line = processRecord(text, 'batch', null);
    return { line, out: JSON.parse(line) };
  };
  const vec = (name, input, exp) => {
    let r;
    try {
      r = run(input);
    } catch (e) {
      ok(false, name, `output is not JSON (${e.message})`);
      return null;
    }
    const o = r.out;
    ok(Object.keys(o).join(',') === 'id,status,epsilon,lower_bound,reason', `${name} schema`, Object.keys(o).join(','));
    ok(o.status === exp.status, name, `status ${o.status}, expected ${exp.status} (${o.reason})`);
    if (exp.status === 'INVALID_INPUT') {
      ok(o.epsilon === null && o.lower_bound === null, `${name} nulls`);
      if (exp.code) ok(typeof o.reason === 'string' && o.reason.startsWith(exp.code + ':'), `${name} reason`, `got "${o.reason}", expected code ${exp.code}`);
    } else {
      if (exp.epsilon !== undefined) ok(o.epsilon === exp.epsilon, `${name} epsilon`, `${o.epsilon} != ${exp.epsilon}`);
      if (exp.lower_bound !== undefined) ok(o.lower_bound === exp.lower_bound, `${name} lower_bound`, `${o.lower_bound} != ${exp.lower_bound}`);
    }
    return r;
  };

  const H = {
    z: '0x0.0p+0',
    q: '0x1.0000000000000p-2',
    h: '0x1.0000000000000p-1',
    one: '0x1.0000000000000p+0',
    two: '0x1.0000000000000p+1',
    three: '0x1.8000000000000p+1',
    four: '0x1.0000000000000p+2',
    halfMinus: '0x1.fffffff000000p-2', // 1/2 - 2^-30
    halfPlus: '0x1.0000000800000p-1', // 1/2 + 2^-30
  };
  const E6 = '1/1000000';
  // Graph A: detectors 0, 1 and b; (0,1) weight 1, (0,b) and (1,b) weight 2. fired {0,1}: OPT = 1 via (0,1).
  const GA = [[0, 1, H.one], [0, -1, H.two], [1, -1, H.two]];
  const base = { id: 'A', edges: GA, fired: [0, 1], correction: [[0, 1]], duals: [[[0], H.h], [[1], H.h]], eps_max: E6 };
  const w = (o) => ({ ...base, ...o });
  // Graph D: (0,b) is light, so a packing that loads it must be scaled down by theta.
  const GD = [[0, 1, H.one], [0, -1, H.q], [1, -1, H.four]];
  const GD2 = [[0, 1, H.one], [0, -1, H.h], [1, -1, H.four]];
  // Graph E: (0,1) 1, (1,b) 1, (0,b) 3. fired {0}: T = {0, b}, OPT = 2 via (0,1)+(1,b).
  const GE = [[0, 1, H.one], [1, -1, H.one], [0, -1, H.three]];
  const P1074 = (1n << 1074n).toString();

  group('hand-built vectors', () => {
    vec('A exact: z{0}=z{1}=1/2, theta=1, bound 1 = w(J)', base, { status: 'EXACT_OPTIMAL', epsilon: '0', lower_bound: '1' });
    vec('A2 exact: set {0,b} is T-odd when b is not in T', w({ duals: [[[0, -1], H.h], [[1], H.h]] }), { status: 'EXACT_OPTIMAL', epsilon: '0', lower_bound: '1' });
    vec('A3 exact: a set and its complement both listed', w({ duals: [[[0], H.h], [[1, -1], H.h]] }), { status: 'EXACT_OPTIMAL', epsilon: '0', lower_bound: '1' });
    vec('B epsilon-optimal: z{1} = 1/2 - 2^-30, eps = 2^-30', w({ duals: [[[0], H.h], [[1], H.halfMinus]] }), { status: 'EPSILON_OPTIMAL', epsilon: '1/1073741824', lower_bound: '1073741823/1073741824' });
    vec('B at eps_max = eps exactly (<= is inclusive)', w({ duals: [[[0], H.h], [[1], H.halfMinus]], eps_max: '1/1073741824' }), { status: 'EPSILON_OPTIMAL', epsilon: '1/1073741824' });
    vec('B just above eps_max', w({ duals: [[[0], H.h], [[1], H.halfMinus]], eps_max: '1/1073741825' }), { status: 'NOT_PROVEN', epsilon: '1/1073741824' });
    vec('B with eps_max 0', w({ duals: [[[0], H.h], [[1], H.halfMinus]], eps_max: '0' }), { status: 'NOT_PROVEN' });
    vec('C not proven: z = 1/4 each, eps = 1/2', w({ duals: [[[0], H.q], [[1], H.q]] }), { status: 'NOT_PROVEN', epsilon: '1/2', lower_bound: '1/2' });
    vec('D overloaded: (0,b) w=1/4 carries 1/2, theta = 1/2, eps = 1/2', w({ edges: GD }), { status: 'NOT_PROVEN', epsilon: '1/2', lower_bound: '1/2' });
    vec('D2 overloaded by 2^-30: theta = 2^29/(2^29+1), eps = 1/536870913', w({ edges: GD2, duals: [[[0], H.halfPlus], [[1], H.halfMinus]] }), { status: 'EPSILON_OPTIMAL', epsilon: '1/536870913', lower_bound: '536870912/536870913' });
    vec('theta = 0: a zero-weight edge carries load', w({ edges: [[0, 1, H.one], [0, -1, H.z], [1, -1, H.two]] }), { status: 'NOT_PROVEN', epsilon: '1', lower_bound: '0' });
    vec('E boundary: |fired| odd, b in T, set {b} is T-odd', { id: 'E', edges: GE, fired: [0], correction: [[0, 1], [1, -1]], duals: [[[0], H.one], [[-1], H.one]], eps_max: E6 }, { status: 'EXACT_OPTIMAL', epsilon: '0', lower_bound: '2' });
    vec('E forgery: V = {0,1,b} meets T={0,b} evenly (accepting it would certify w=3 vs OPT=2)', { id: 'Ef', edges: GE, fired: [0], correction: [[0, -1]], duals: [[[0], H.one], [[0, 1, -1], H.two]], eps_max: E6 }, { status: 'INVALID_INPUT', code: 'set-not-t-odd' });
    vec('E forgery 2: {0,b} is odd against fired alone but even against T', { id: 'Ef2', edges: GE, fired: [0], correction: [[0, -1]], duals: [[[0, -1], H.one]], eps_max: E6 }, { status: 'INVALID_INPUT', code: 'set-not-t-odd' });
    vec('E: correction (0,1) alone does not explain fired {0}', { id: 'Ej', edges: GE, fired: [0], correction: [[0, 1]], duals: [], eps_max: E6 }, { status: 'INVALID_INPUT', code: 'not-t-join' });
    vec('E: (0,b) alone is a valid T-join, empty packing proves eps = 3', { id: 'Eb', edges: GE, fired: [0], correction: [[0, -1]], duals: [], eps_max: E6 }, { status: 'NOT_PROVEN', epsilon: '3', lower_bound: '0' });
    vec('empty syndrome, empty correction, empty packing', w({ fired: [], correction: [], duals: [] }), { status: 'EXACT_OPTIMAL', epsilon: '0', lower_bound: '0' });
    vec('empty syndrome, a cycle as correction (all degrees even)', w({ fired: [], correction: [[0, 1], [0, -1], [1, -1]], duals: [] }), { status: 'NOT_PROVEN', epsilon: '5', lower_bound: '0' });
    vec('empty graph', { id: 'g0', edges: [], fired: [], correction: [], duals: [], eps_max: E6 }, { status: 'EXACT_OPTIMAL', epsilon: '0', lower_bound: '0' });
    vec('subnormal weight 2^-1074 on the correction edge', w({ edges: [[0, 1, '0x0.0000000000001p-1022'], [0, -1, H.one], [1, -1, H.one]], duals: [] }), { status: 'EPSILON_OPTIMAL', epsilon: '1/' + P1074, lower_bound: '0' });
    vec('largest finite double as a weight', w({ edges: [[0, 1, H.one], [0, -1, '0x1.fffffffffffffp+1023'], [1, -1, H.two]] }), { status: 'EXACT_OPTIMAL', epsilon: '0' });
    vec('packing value 2^-1074 overloads (0,1): theta = 1/(1+2^-1074), bound exactly 1', w({ duals: [[[0], H.h], [[1], H.h], [[1, -1], '0x0.0000000000001p-1022']] }), { status: 'EXACT_OPTIMAL', epsilon: '0', lower_bound: '1' });
    vec('packing value = largest double: theta = 1/MAX, bound exactly 1', w({ duals: [[[0], '0x1.fffffffffffffp+1023']] }), { status: 'EXACT_OPTIMAL', epsilon: '0', lower_bound: '1' });
    vec('subnormal weight with a coarse packing: grids a = 1074, b = 1', w({ edges: [[0, 1, H.one], [0, -1, '0x0.0000000000001p-1022'], [1, -1, H.two]], duals: [[[1], H.h]] }), { status: 'NOT_PROVEN', epsilon: '1/2', lower_bound: '1/2' });
    vec('correction edge in reversed orientation', w({ correction: [[1, 0]] }), { status: 'EXACT_OPTIMAL', epsilon: '0' });
    vec('zero-valued T-odd set accepted', w({ duals: [[[0], H.h], [[1], H.h], [[1, -1], H.z]] }), { status: 'EXACT_OPTIMAL', lower_bound: '1' });
    vec('negative zero value is the rational 0', w({ duals: [[[0], H.h], [[1], H.h], [[1, -1], '-0x0.0p+0']] }), { status: 'EXACT_OPTIMAL', lower_bound: '1' });
    vec('non-canonical exact hex spellings', w({ duals: [[[0], '0x2.0p-2'], [[1], '0X0.8P+0']] }), { status: 'EXACT_OPTIMAL', lower_bound: '1' });
    vec('unknown extra field ignored', w({ note: 'x' }), { status: 'EXACT_OPTIMAL' });
    vec('id -0 is vertex 0', '{"id":"nz","edges":[[-0,1,"0x1.0000000000000p+0"],[0,-1,"0x1.0000000000000p+1"],[1,-1,"0x1.0000000000000p+1"]],"fired":[0,1],"correction":[[0,1]],"duals":[[[0],"0x1.0000000000000p-1"],[[1],"0x1.0000000000000p-1"]],"eps_max":"1/1000000"}', { status: 'EXACT_OPTIMAL' });
    const big = '123456789012345678901234567890';
    vec('arbitrary-size detector id', `{"id":"big","edges":[[${big},-1,"${H.one}"]],"fired":[${big}],"correction":[[${big},-1]],"duals":[[[${big}],"${H.one}"]],"eps_max":"1/1000000"}`, { status: 'EXACT_OPTIMAL', lower_bound: '1' });
    vec('ids 2^53 and 2^53+1 stay distinct', `{"id":"p53","edges":[[9007199254740992,-1,"${H.one}"],[9007199254740993,-1,"${H.one}"]],"fired":[9007199254740992,9007199254740993],"correction":[[9007199254740992,-1],[9007199254740993,-1]],"duals":[[[9007199254740992],"${H.one}"],[[9007199254740993],"${H.one}"]],"eps_max":"1/1000000"}`, { status: 'EXACT_OPTIMAL', lower_bound: '2' });
    const r1 = vec('numeric id echoed verbatim', w({ id: 5 }), { status: 'EXACT_OPTIMAL' });
    ok(r1 && r1.line.startsWith('{"id":5,'), 'numeric id echo', r1 && r1.line);
    const r2 = vec('missing id echoed as null', (() => { const o = { ...base }; delete o.id; return o; })(), { status: 'EXACT_OPTIMAL' });
    ok(r2 && r2.out.id === null, 'missing id null');
  });

  group('INVALID_INPUT reasons', () => {
    const bad = (name, input, code) => vec(name, input, { status: 'INVALID_INPUT', code });
    const raw = (o) => JSON.stringify(o);
    bad('not JSON', '{"id":"x",', 'json');
    bad('empty line', '', 'json');
    bad('top level is an array', '[1,2]', 'json');
    bad('NaN literal', raw(base).replace('"1/1000000"', 'NaN'), 'json');
    bad('trailing text', raw(base) + ' x', 'json');
    bad('byte-order mark', '\uFEFF' + raw(base), 'json');
    bad('nesting too deep', '['.repeat(100000), 'json');
    bad('duplicate key', raw(base).replace('"fired":[0,1]', '"fired":[0,1],"fired":[0,1]'), 'duplicate-key');
    bad('missing duals', (() => { const o = { ...base }; delete o.duals; return o; })(), 'missing-field');
    bad('missing eps_max', (() => { const o = { ...base }; delete o.eps_max; return o; })(), 'missing-field');
    for (const e of ['1e-6', '2/4', '-1/2', '1/0', '0/5', '1/1', ' 1/5', '+1/5', '01/5', '0.000001', '', 'x']) bad(`eps_max ${JSON.stringify(e)}`, w({ eps_max: e }), 'eps-max');
    bad('eps_max as a JSON number', w({ eps_max: 0.000001 }), 'eps-max');
    bad('eps_max null', w({ eps_max: null }), 'eps-max');
    bad('eps_max too long', w({ eps_max: '1/' + '3'.repeat(9000) }), 'eps-max');
    bad('edges not an array', w({ edges: {} }), 'edges');
    bad('edge of arity 2', w({ edges: [[0, 1], [0, -1, H.two], [1, -1, H.two]] }), 'edge-shape');
    bad('edge of arity 4', w({ edges: [[0, 1, H.one, 5], [0, -1, H.two], [1, -1, H.two]] }), 'edge-shape');
    bad('fractional vertex id 1.5', w({ edges: [[0, 1.5, H.one], [0, -1, H.two], [1, -1, H.two]] }), 'vertex-id');
    bad('vertex id 1.0', raw(base).replace('[0,1,"0x1', '[0,1.0,"0x1'), 'vertex-id');
    bad('vertex id 1e0', raw(base).replace('[0,1,"0x1', '[0,1e0,"0x1'), 'vertex-id');
    bad('boolean vertex id', w({ edges: [[0, true, H.one], [0, -1, H.two], [1, -1, H.two]] }), 'vertex-id');
    bad('string vertex id', w({ edges: [[0, '1', H.one], [0, -1, H.two], [1, -1, H.two]] }), 'vertex-id');
    bad('null vertex id', w({ edges: [[0, null, H.one], [0, -1, H.two], [1, -1, H.two]] }), 'vertex-id');
    bad('negative vertex id -2', w({ edges: [[0, 1, H.one], [0, -2, H.two], [1, -1, H.two]] }), 'vertex-id');
    bad('self-loop', w({ edges: [...GA, [1, 1, H.one]] }), 'self-loop');
    bad('boundary self-loop', w({ edges: [...GA, [-1, -1, H.one]] }), 'self-loop');
    bad('parallel edge, reversed', w({ edges: [...GA, [1, 0, H.one]] }), 'parallel-edge');
    bad('parallel edge, other weight', w({ edges: [...GA, [0, 1, H.two]] }), 'parallel-edge');
    const wBad = ['inf', '-inf', 'nan', 'Infinity', '-0x1.0000000000000p+0', '1.0', '1.8', '0x1.00000000000008p+0', '0x1.0000000000000p+1024',
      '0x1.0000000000000p-1075', '0x1.8p-1074', '0x1p', ' 0x1p+0', '0x1p+0 ', '0x.p+0', '0x1.0', '0x1.0p+0x', '0x1p+0p+0'];
    for (const s of wBad) bad(`weight ${JSON.stringify(s)}`, w({ edges: [[0, 1, s], [0, -1, H.two], [1, -1, H.two]] }), 'weight');
    bad('weight as a JSON number', w({ edges: [[0, 1, 1], [0, -1, H.two], [1, -1, H.two]] }), 'weight');
    bad('fired not an array', w({ fired: '0' }), 'fired');
    bad('-1 in fired', w({ fired: [0, 1, -1] }), 'fired');
    bad('repeated detector', w({ fired: [0, 1, 1] }), 'fired');
    bad('fired detector not in the graph', w({ fired: [0, 7] }), 'fired');
    bad('fired 1.0', raw(base).replace('"fired":[0,1]', '"fired":[0,1.0]'), 'vertex-id');
    bad('fired true', w({ fired: [0, true] }), 'vertex-id');
    bad('correction not an array', w({ correction: null }), 'correction-shape');
    bad('correction pair of arity 3', w({ correction: [[0, 1, 2]] }), 'correction-shape');
    bad('correction edge not in the graph', w({ correction: [[0, 2]] }), 'correction-edge');
    bad('correction self-pair', w({ correction: [[0, 0]] }), 'correction-edge');
    bad('correction edge repeated (reversed)', w({ correction: [[0, 1], [1, 0]] }), 'correction-repeat');
    bad('correction edge repeated', w({ correction: [[0, 1], [0, 1]] }), 'correction-repeat');
    bad('correction does not explain the syndrome', w({ correction: [[0, -1]] }), 'not-t-join');
    bad('correction non-empty for an empty syndrome', w({ fired: [], correction: [[0, 1]], duals: [] }), 'not-t-join');
    bad('duals not an array', w({ duals: {} }), 'set-shape');
    bad('dual of arity 1', w({ duals: [[[0]]] }), 'set-shape');
    bad('dual of arity 3', w({ duals: [[[0], H.h, 1]] }), 'set-shape');
    bad('set not an array', w({ duals: [[0, H.h]] }), 'set-shape');
    bad('empty set', w({ duals: [[[], H.h]] }), 'set-empty');
    bad('T-even set {0,1}', w({ duals: [[[0, 1], H.h]] }), 'set-not-t-odd');
    bad('set {b} with b not in T', w({ duals: [[[-1], H.h]] }), 'set-not-t-odd');
    bad('zero-valued T-even set', w({ duals: [[[0], H.h], [[1], H.h], [[0, 1], H.z]] }), 'set-not-t-odd');
    bad('any set when T is empty', w({ fired: [], correction: [], duals: [[[0], H.h]] }), 'set-not-t-odd');
    bad('repeated set', w({ duals: [[[0], H.h], [[0], H.q]] }), 'set-repeated');
    bad('repeated set in another order', w({ duals: [[[0, -1], H.q], [[-1, 0], H.q]] }), 'set-repeated');
    bad('repeated member', w({ duals: [[[1, 0, 0], H.h]] }), 'set-repeat-member');
    bad('member not in the graph', w({ duals: [[[0, 9], H.h]] }), 'set-member');
    bad('member 0.0', raw(base).replace('[[[0],', '[[[0.0],'), 'vertex-id');
    for (const s of ['-0x1.0000000000000p-1', 'nan', 'inf', '-inf', '0x1.0000000000000p-1075', '0.5']) bad(`dual value ${JSON.stringify(s)}`, w({ duals: [[[0], s], [[1], H.h]] }), 'dual-value');
    bad('dual value as a JSON number', w({ duals: [[[0], 0.5], [[1], H.h]] }), 'dual-value');
  });

  group('classification and arithmetic units', () => {
    const em = { n: 1n, d: 1000000n };
    ok(classify(-1n, 7n, em) === 'VERIFIER_FAILURE', 'eps < 0 is VERIFIER_FAILURE');
    ok(classify(0n, 7n, em) === 'EXACT_OPTIMAL', 'eps = 0 is EXACT_OPTIMAL');
    ok(classify(1n, 1000000n, em) === 'EPSILON_OPTIMAL', 'eps = eps_max is EPSILON_OPTIMAL');
    ok(classify(1000001n, 1000000n * 1000000n, em) === 'NOT_PROVEN', 'eps just above eps_max is NOT_PROVEN');
    ok(fmtRational(6n, 4n) === '3/2' && fmtRational(-6n, 4n) === '-3/2' && fmtRational(0n, 9n) === '0' && fmtRational(8n, 4n) === '2', 'fmtRational lowest terms');
    ok(fmtRational(3n << 1074n, 9n << 1074n) === '1/3', 'reduce strips 2^1074 and odd factors');
    const pr = (s) => { try { const r = parseRational(s, 't'); return `${r.n}/${r.d}`; } catch (e) { return e.code; } };
    ok(pr('1/1000000') === '1/1000000' && pr('0') === '0/1' && pr('5') === '5/1' && pr('2/4') === 'eps-max' && pr('0/1') === 'eps-max', 'parseRational');
    const hx = (s) => { const r = hexToDyadic(s); return r.error ? 'ERR' : (r.M === 0n ? '0' : `${r.neg ? '-' : ''}${r.M}*2^${r.shift}`); };
    ok(hx('0x1.8000000000000p+1') === '3*2^0', 'hex 3', hx('0x1.8000000000000p+1'));
    ok(hx('0x0.0000000000001p-1022') === '1*2^-1074', 'hex min subnormal');
    ok(hx('0x1.fffffffffffffp+1023') === `${(1n << 53n) - 1n}*2^971`, 'hex max double');
    ok(hx('0x2.0p-1075') === '1*2^-1074', 'non-canonical exact min subnormal');
    ok(hx('0x10p-4') === '1*2^0' && hx('+0x1p+0') === '1*2^0' && hx('0x.8p+1') === '1*2^0' && hx('0x1.p+0') === '1*2^0', 'non-canonical exact spellings');
    ok(hx('-0x0.0p+0') === '0' && hx('0x0p+99999999999999999999999999999999999') === '0', 'zeros');
    ok(hx('0x0.8p-1074') === 'ERR' && hx('0x3p-1075') === 'ERR' && hx('0x2.0p+1023') === 'ERR' && hx('0x1.00000000000008p+0') === 'ERR', 'not exactly binary64');
    ok(hx('0x1.0000000000001p+0') === `${(1n << 52n) + 1n}*2^-52`, 'hex 1 + ulp');
    ok(hx('0x1' + '0'.repeat(100000) + 'p-400000') === '1*2^0', 'long exact literal');
    ok(hx('0x1p+' + '9'.repeat(40)) === 'ERR', 'huge exponent refused without huge arithmetic');
  });

  group('hex round trip on random bit patterns', () => {
    const rng = mulberry32(0x5eed1);
    const dv = new DataView(new ArrayBuffer(8));
    let tested = 0;
    for (let k = 0; k < 4000; k++) {
      let hi = rng.u32();
      const lo = rng.u32();
      const pick = rng.next();
      if (pick < 0.15) hi &= 0x800fffff; // subnormal (or zero)
      else if (pick < 0.2) hi = (hi & 0x800fffff) | (2046 << 20); // largest binade
      else if (pick < 0.25) hi = (hi & 0x800fffff) | (1 << 20); // smallest normal binade
      else if (pick < 0.27) hi &= 0x80000000; // zero mantissa high half
      dv.setUint32(0, hi);
      dv.setUint32(4, pick < 0.27 && pick >= 0.25 ? 0 : lo);
      const x = dv.getFloat64(0);
      if (!Number.isFinite(x)) continue;
      const s = floatHex(x);
      const r = hexToDyadic(s);
      if (r.error) {
        ok(false, 'round trip parse', `${s}: ${r.error}`);
        continue;
      }
      const ex = (hi >>> 20) & 0x7ff;
      const mant = (BigInt(hi & 0xfffff) << 32n) | BigInt(dv.getUint32(4));
      const expScaled = ex === 0 ? mant : ((1n << 52n) + mant) << BigInt(ex - 1);
      const gotScaled = r.M === 0n ? 0n : r.M << (r.shift + SCALE_BITS);
      ok(gotScaled === expScaled, 'round trip exact value', s);
      if (r.M !== 0n) {
        ok(r.neg === x < 0, 'round trip sign', s);
        const back = (r.neg ? -1 : 1) * Number(r.M) * 2 ** Number(r.shift);
        ok(back === x, 'round trip against the engine double', `${s} -> ${back} vs ${x}`);
        const extra = rng.int(16);
        const s2 = s.replace('p', extra.toString(16) + 'p');
        const r2 = hexToDyadic(s2);
        if (extra === 0) ok(!r2.error && r2.M === r.M && r2.shift === r.shift, 'appended zero digit keeps value', s2);
        else ok(Boolean(r2.error), 'appended nonzero digit is not a double', s2);
      }
      tested++;
    }
    ok(tested > 3000, 'round trip sample size', String(tested));
    for (const s of [floatHex(Infinity), floatHex(-Infinity), floatHex(NaN)]) ok(Boolean(hexToDyadic(s).error), `non-finite ${s} refused`);
  });

  group('JSON reader and fingerprint serialiser', () => {
    const v = parseJson('{"a":[1,2.5e3,-0,"x\\u0041\\n"],"b":{"c":null,"d":true}}');
    const a = v.get('a');
    ok(a[0].lex === '1' && a[1].lex === '2.5e3' && a[2].lex === '-0' && a[3] === 'xA\n', 'lexemes and escapes kept');
    ok(v.get('b').get('c') === null && v.get('b').get('d') === true, 'nested object');
    const throwsCode = (t) => { try { parseJson(t); return null; } catch (e) { return e.code; } };
    ok(throwsCode('{"a":1,"a":2}') === 'duplicate-key', 'duplicate key');
    ok(throwsCode('{"o":{"k":1,"k":1}}') === 'duplicate-key', 'nested duplicate key');
    for (const t of ['{"a":01}', '{"a":1,}', '[1 2]', '"\\x"', '"a\u0001"', 'NaN', 'Infinity', '-', '1.', '.5', '\uFEFF{}', '{} x', '{"a"}', "{'a':1}", '[' .repeat(65) + ']'.repeat(65)]) {
      ok(throwsCode(t) === 'json', `malformed JSON refused: ${JSON.stringify(t).slice(0, 30)}`);
    }
    ok(throwsCode('['.repeat(64) + ']'.repeat(64)) === null, 'nesting of 64 accepted');
    ok(pyCompact(parseJson('["\\u00e9","\\u007f","/","\\n",-0,[1,"a"],"\\ud83d\\ude00"]')) === '["\\u00e9","\\u007f","/","\\n",0,[1,"a"],"\\ud83d\\ude00"]', 'Python-compatible compact form');
  });

  group('brute-force oracle on random graphs of 2-5 vertices', () => {
    const rng = mulberry32(0xace5);
    const st = { valid: 0, exact: 0, epsopt: 0, notp: 0, thetaLt1: 0, rawAboveOpt: 0, evenRefused: 0, evenWouldForge: 0, probes: 0, probeRefused: 0 };
    const epsChoices = [['0', new Q(0n)], ['1/8', new Q(1n, 8n)], ['1/2', new Q(1n, 2n)], ['1', new Q(1n)]];
    for (let t = 0; t < 800; t++) {
      const nd = 1 + rng.int(4);
      const cand = [-1];
      for (let d = 0; d < nd; d++) cand.push(d);
      const edges = [];
      for (let a = 0; a < cand.length; a++) for (let b = a + 1; b < cand.length; b++) if (rng.next() < 0.7) edges.push([cand[a], cand[b], rng.int(9)]);
      // V = {b} + endpoints (contract C5): an isolated candidate is not a vertex of the graph.
      const verts = cand.filter((x) => x === -1 || edges.some((e) => e[0] === x || e[1] === x));
      const fired = verts.filter((x) => x !== -1 && rng.next() < 0.5);
      const T = new Set(fired);
      if (fired.length % 2) T.add(-1);
      const m = edges.length;
      const isTJ = (mask) => {
        const par = new Map();
        for (let k = 0; k < m; k++) if ((mask >> k) & 1) for (const x of [edges[k][0], edges[k][1]]) par.set(x, (par.get(x) || 0) ^ 1);
        for (const x of verts) if ((par.get(x) || 0) !== (T.has(x) ? 1 : 0)) return false;
        return true;
      };
      const wq = (mask) => { let s = 0; for (let k = 0; k < m; k++) if ((mask >> k) & 1) s += edges[k][2]; return s; };
      const corr = (mask) => {
        const c = [];
        for (let k = 0; k < m; k++) if ((mask >> k) & 1) c.push(rng.next() < 0.5 ? [edges[k][0], edges[k][1]] : [edges[k][1], edges[k][0]]);
        return c;
      };
      const edgesHex = edges.map(([a, b, q]) => [a, b, floatHex(q / 4)]);
      const [epsStr, epsQ] = epsChoices[rng.int(epsChoices.length)];
      // Probe the T-join test against brute force.
      {
        const mask = rng.int(1 << m);
        const o = run({ id: 'probe', edges: edgesHex, fired, correction: corr(mask), duals: [], eps_max: epsStr }).out;
        const refused = o.status === 'INVALID_INPUT';
        ok(refused === !isTJ(mask), 'T-join test agrees with brute force', `trial ${t}: ${o.reason}`);
        if (refused) ok(o.reason.startsWith('not-t-join:'), 'T-join refusal reason', o.reason);
        st.probes++;
        if (refused) st.probeRefused++;
      }
      const tj = [];
      for (let mask = 0; mask < 1 << m; mask++) if (isTJ(mask)) tj.push(mask);
      if (tj.length === 0) continue;
      let opt = Infinity;
      for (const mask of tj) opt = Math.min(opt, wq(mask));
      const OPT = new Q(BigInt(opt), 4n);
      const Jmask = rng.next() < 0.5 ? tj.find((mk) => wq(mk) === opt) : tj[rng.int(tj.length)];
      const WJ = new Q(BigInt(wq(Jmask)), 4n);
      const masks = new Set();
      const k = rng.int(5);
      for (let i = 0; i < k; i++) masks.add(1 + rng.int((1 << verts.length) - 1));
      let sets = [...masks].map((sm) => ({ mem: verts.filter((_, i) => (sm >> i) & 1), zq: rng.int(9) }));
      const tOdd = (s) => s.mem.filter((x) => T.has(x)).length % 2 === 1;
      if (rng.next() < 0.7) sets = sets.filter(tOdd);
      // Oracle: brute-force delta(S), loads in eighths, theta and the bound in independent fractions.
      let theta = new Q(1n);
      let sumZ = new Q(0n);
      for (const s of sets) sumZ = sumZ.add(new Q(BigInt(s.zq), 8n));
      for (const [a, b, q] of edges) {
        let l8 = 0;
        for (const s of sets) if (s.mem.includes(a) !== s.mem.includes(b)) l8 += s.zq;
        if (l8 > 0) {
          const r = new Q(BigInt(2 * q), BigInt(l8)); // (q/4) / (l8/8)
          if (r.cmp(theta) < 0) theta = r;
        }
      }
      const LB = theta.mul(sumZ);
      const o = run({ id: `bf${t}`, edges: edgesHex, fired, correction: corr(Jmask), duals: sets.map((s) => [s.mem, floatHex(s.zq / 8)]), eps_max: epsStr }).out;
      if (!sets.every(tOdd)) {
        ok(o.status === 'INVALID_INPUT' && o.reason.startsWith('set-not-t-odd:'), 'T-even set refused', `trial ${t}: ${o.status} ${o.reason}`);
        st.evenRefused++;
        if (LB.cmp(OPT) > 0) st.evenWouldForge++;
        continue;
      }
      st.valid++;
      ok(['EXACT_OPTIMAL', 'EPSILON_OPTIMAL', 'NOT_PROVEN'].includes(o.status), 'valid input decided', `trial ${t}: ${o.status} ${o.reason}`);
      const lb = Q.parse(o.lower_bound || '');
      const ep = Q.parse(o.epsilon || '');
      ok(lb !== null && lb.cmp(LB) === 0, 'bound equals the oracle', `trial ${t}: ${o.lower_bound} vs ${LB.str()}`);
      ok(ep !== null && ep.cmp(WJ.sub(LB)) === 0, 'eps = w(J) - bound', `trial ${t}`);
      ok(lb !== null && lb.cmp(OPT) <= 0, 'SOUNDNESS: bound <= true optimum', `trial ${t}: ${o.lower_bound} > ${OPT.str()}`);
      const expStatus = ep === null ? '' : ep.n === 0n ? 'EXACT_OPTIMAL' : ep.cmp(epsQ) <= 0 ? 'EPSILON_OPTIMAL' : 'NOT_PROVEN';
      ok(o.status === expStatus, 'status matches eps vs eps_max', `trial ${t}: ${o.status} vs ${expStatus}`);
      if (o.status === 'EXACT_OPTIMAL') {
        st.exact++;
        ok(WJ.cmp(OPT) === 0, 'EXACT_OPTIMAL only for a truly optimal correction', `trial ${t}`);
      }
      if (o.status === 'EPSILON_OPTIMAL') st.epsopt++;
      if (o.status === 'NOT_PROVEN') st.notp++;
      if (theta.cmp(new Q(1n)) < 0) st.thetaLt1++;
      if (sumZ.cmp(OPT) > 0) st.rawAboveOpt++;
    }
    // The battery must contain the cases that make each check load-bearing, or it proves nothing.
    ok(st.valid > 200, 'oracle: enough valid trials', JSON.stringify(st));
    ok(st.exact > 0 && st.epsopt > 0 && st.notp > 0, 'oracle: all three acceptance outcomes occur', JSON.stringify(st));
    ok(st.thetaLt1 > 20, 'oracle: theta < 1 occurs', JSON.stringify(st));
    ok(st.rawAboveOpt > 0, 'oracle: unscaled packings above OPT occur (theta is load-bearing)', JSON.stringify(st));
    ok(st.evenWouldForge > 0, 'oracle: T-even sets that would forge a bound above OPT occur (oddness is load-bearing)', JSON.stringify(st));
    ok(st.probeRefused > 0 && st.probeRefused < st.probes, 'oracle: T-join probes both pass and fail', JSON.stringify(st));
    groups.push(`    oracle stats ${JSON.stringify(st)}`);
  });

  group('never crash: mutation fuzz', () => {
    const rng = mulberry32(0xf022);
    const seeds = [JSON.stringify(base), JSON.stringify(w({ edges: GD2, duals: [[[0], H.halfPlus], [[1], H.halfMinus]] })),
      JSON.stringify({ id: 'E', edges: GE, fired: [0], correction: [[0, 1], [1, -1]], duals: [[[0], H.one], [[-1], H.one]], eps_max: E6 })];
    const palette = '0123456789-.,[]{}":expabcdf+/ tnul\\';
    let internal = 0;
    let decided = 0;
    for (let k = 0; k < 6000; k++) {
      let s = seeds[rng.int(seeds.length)];
      const nm = 1 + rng.int(4);
      for (let j = 0; j < nm; j++) {
        const p = rng.int(s.length + 1);
        const op = rng.int(4);
        if (op === 0) s = s.slice(0, p) + s.slice(p + 1);
        else if (op === 1) s = s.slice(0, p) + palette[rng.int(palette.length)] + s.slice(p);
        else if (op === 2) s = s.slice(0, p) + palette[rng.int(palette.length)] + s.slice(p + 1);
        else { const q = Math.min(s.length, p + rng.int(12)); s = s.slice(0, q) + s.slice(p, q) + s.slice(q); }
      }
      const line = processRecord(s, 'batch', null);
      let o = null;
      try { o = JSON.parse(line); } catch { o = null; }
      if (!o || Object.keys(o).join(',') !== 'id,status,epsilon,lower_bound,reason' || !STATUSES.includes(o.status)) {
        ok(false, 'fuzz output schema', line.slice(0, 200));
        continue;
      }
      if (String(o.reason).startsWith('internal')) internal++;
      if (o.status !== 'INVALID_INPUT') {
        decided++;
        ok(Q.parse(o.epsilon || '') !== null && Q.parse(o.lower_bound || '') !== null, 'fuzz decided lines carry rationals', line.slice(0, 200));
        ok(o.status !== 'VERIFIER_FAILURE', 'fuzz: no VERIFIER_FAILURE', line.slice(0, 200));
      } else ok(o.epsilon === null && o.lower_bound === null, 'fuzz INVALID nulls');
    }
    ok(internal === 0, 'fuzz: no internal errors', String(internal));
    ok(decided > 0, 'fuzz: some mutants still decide', String(decided));
    groups.push(`    fuzz: 6000 mutants, ${decided} decided, ${internal} internal errors`);
  });

  // Asynchronous groups: the byte-level line splitter and the replay pipeline.
  const asyncChecks = [];
  {
    const valid = JSON.stringify(base);
    const parts = [
      Buffer.from(valid + '\n'),
      Buffer.from('\n'),
      Buffer.from(JSON.stringify(w({ id: 'C', duals: [[[0], H.q], [[1], H.q]] })) + '\r\n'),
      Buffer.from([0xff, 0xfe, 0x7b, 0x0a]),
      Buffer.from('\uFEFF' + valid + '\n'),
      Buffer.from('['.repeat(100000) + '\n'),
      Buffer.from('{"id":"bignum","x":' + '9'.repeat(200000) + '}\n'),
      Buffer.from('garbage\n'),
      Buffer.from(valid),
    ];
    const all = Buffer.concat(parts);
    const rng = mulberry32(7);
    const chunks = [];
    for (let p = 0; p < all.length;) {
      const len = 1 + rng.int(5000);
      chunks.push(all.subarray(p, p + len));
      p += len;
    }
    const outs = [];
    const n = await runLines(Readable.from(chunks), (t) => processRecord(t, 'batch', null), (l) => { outs.push(l); return null; }, 'id');
    const expect = ['EXACT_OPTIMAL', 'INVALID_INPUT', 'NOT_PROVEN', 'INVALID_INPUT', 'INVALID_INPUT', 'INVALID_INPUT', 'INVALID_INPUT', 'INVALID_INPUT', 'EXACT_OPTIMAL'];
    asyncChecks.push(['batch splitter: one output per input line', n === 9 && outs.length === 9, `${n} lines`]);
    const got = outs.map((l) => { try { return JSON.parse(l).status; } catch { return 'BAD'; } });
    asyncChecks.push(['batch splitter: statuses in input order', got.join() === expect.join(), got.join()]);
    asyncChecks.push(['batch splitter: invalid UTF-8 refused as encoding', outs[3] && outs[3].includes('encoding:'), outs[3]]);

    const edgesText = JSON.stringify(GA);
    const sha = createHash('sha256').update(edgesText).digest('hex');
    const instText = `{"schema":"oneq-matching-instance/1","name":"selftest","edges":${edgesText},"edges_sha256":"${sha}"}\n`;
    let g = null;
    try { g = loadInstance(Buffer.from(instText)); } catch (e) { asyncChecks.push(['instance accepted', false, e.message]); }
    asyncChecks.push(['instance with matching fingerprint accepted', g !== null && g.U.length === 3, '']);
    let refused = null;
    try { loadInstance(Buffer.from(instText.replace('0x1.0000000000000p+0', '0x1.0000000000001p+0'))); } catch (e) { refused = e.code; }
    asyncChecks.push(['instance with tampered weight refused by fingerprint', refused === 'fingerprint', String(refused)]);
    refused = null;
    const pEdges = JSON.stringify([...GA, [1, 0, H.one]]);
    const pSha = createHash('sha256').update(pEdges).digest('hex');
    try { loadInstance(Buffer.from(`{"edges":${pEdges},"edges_sha256":"${pSha}"}`)); } catch (e) { refused = e.code; }
    asyncChecks.push(['instance with a parallel edge refused', refused === 'parallel-edge', String(refused)]);
    if (g) {
      const rec = (shot, duals) => JSON.stringify({ shot, fired: [0, 1], correction: [[0, 1]], duals, status: 'EXACT_OPTIMAL', epsilon: '0', lower_bound: '1' });
      const text = [rec(0, [[[0], H.h], [[1], H.h]]), rec(1, [[[0], H.q], [[1], H.q]]), 'not json', rec(3, [[[0], H.h], [[0], H.h]])].join('\n') + '\n';
      const outs2 = [];
      const gz = Readable.from([gzipSync(Buffer.from(text))]).pipe(createGunzip());
      const ctx = { graph: g, epsMax: parseRational(DEFAULT_EPS_MAX, 't') };
      await runLines(gz, (t) => processRecord(t, 'replay', ctx), (l) => { outs2.push(l); return null; }, 'shot');
      const p = outs2.map((l) => JSON.parse(l));
      asyncChecks.push(['replay: one line per receipt, keyed by shot', p.length === 4 && Object.keys(p[0]).join(',') === 'shot,status,epsilon,lower_bound,reason', outs2.join(' | ')]);
      asyncChecks.push(['replay: verdicts recomputed, recorded fields ignored', p[0].status === 'EXACT_OPTIMAL' && p[1].status === 'NOT_PROVEN' && p[1].epsilon === '1/2' && p[2].shot === null && p[2].status === 'INVALID_INPUT' && p[3].status === 'INVALID_INPUT' && p[3].shot === 3, outs2.join(' | ')]);
    }
  }
  const fa = failures.length;
  for (const [name, cond, detail] of asyncChecks) ok(Boolean(cond), name, detail);
  groups.push(`  stream splitter, instance fingerprint, replay pipeline: ${asyncChecks.length} checks, ${failures.length - fa} failed`);

  process.stdout.write('verify-epsilon self-test\n' + groups.join('\n') + '\n');
  for (const f of failures) process.stdout.write(`  FAIL ${f}\n`);
  process.stdout.write(failures.length === 0 ? `SELFTEST PASS (${pass} checks)\n` : `SELFTEST FAIL (${failures.length} of ${pass + failures.length} checks failed)\n`);
  return failures.length === 0;
}

// ------------------------------------------------------------------------------------------------ main

function usage() {
  process.stderr.write('usage: verify-epsilon.mjs --batch < cases.jsonl\n' +
    '       verify-epsilon.mjs --instance <instance.json> --receipts <chunk.jsonl.gz> [--eps-max 1/1000000]\n' +
    '       verify-epsilon.mjs --selftest\n');
  process.exitCode = 2;
}

async function main(argv) {
  process.stdout.on('error', (e) => process.exit(e && e.code === 'EPIPE' ? 0 : 2));
  if (argv.length === 1 && argv[0] === '--selftest') {
    process.exitCode = (await selftest()) ? 0 : 1;
    return;
  }
  if (argv.length === 1 && argv[0] === '--batch') {
    await runLines(process.stdin, (t) => processRecord(t, 'batch', null), stdoutWrite, 'id');
    return;
  }
  const opts = {};
  for (let k = 0; k < argv.length; k += 2) {
    const f = argv[k];
    const v = argv[k + 1];
    if (!['--instance', '--receipts', '--eps-max'].includes(f) || v === undefined || f in opts) return usage();
    opts[f] = v;
  }
  if (!opts['--instance'] || !opts['--receipts']) return usage();
  await replayMain(opts['--instance'], opts['--receipts'], opts['--eps-max'] ?? DEFAULT_EPS_MAX);
}

main(process.argv.slice(2)).catch((e) => {
  process.stderr.write(`verify-epsilon: fatal: ${e && e.stack ? e.stack : e}\n`);
  process.exitCode = 2;
});
