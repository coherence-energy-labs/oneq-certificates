/* cdist_prove (R198): PROOF-PRODUCING exact circuit-distance search (the prover half of a certificate).
 *
 * Same search as cdist.c with three changes that make its NONE answers checkable:
 *   - no transposition table (every pruned node carries its own reason),
 *   - children in CANONICAL order: the allowed mechanisms touching the chosen detector, sorted by
 *     (orbit, mechanism id) -- a checker can regenerate the list without being told it,
 *   - a trace: for every node with remaining budget >= 3 and nonzero syndrome, in DFS preorder, ONE BIT:
 *     expanded on the deterministic pick, or pruned by the bound (legal at remaining budget >= 4; at the root,
 *     >= 3). Nodes with remaining budget <= 2 are leaves the checker re-decides itself.
 * Output file (CDP5): "CDP3" w norb maxround nanch; per anchor o < nanch a root code (int8: 0 root pruned,
 * 1 root expanded -> int32 nchild then nchild x (int64 nbits, bits), 2 = w <= 3 -> one (int64 nbits, bits)).
 * The per-child streams let both prover and checker run the children in parallel.
 * Usage: cdist_prove model.txt W proof.bin [MAXROUND]   (compile with -DLIAR only for the negative control)
 * Exit 0 = NONE at weight <= W with a proof written; 1 = FOUND (no proof, witness printed); 2 = error.
 */
#ifdef _OPENMP
#include <omp.h>
#else
#include <time.h>
static double omp_get_wtime(void) { return (double)clock() / CLOCKS_PER_SEC; }
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXW 16
#define MAXDEG 32

static int ND, NM, NO, NORB, GL, GM, W;
static uint64_t *msyn; static uint32_t *mobs; static int *mdeg, *mdet, *morb;
static int *dstart, *dlist; static uint64_t *cot_orbit; static int *anchors;
static int *shash; static uint64_t smask; static int32_t *pa, *pb; static uint64_t pmask;

/* The prover's syndrome hash is LINEAR (XOR of per-detector keys; its own keys, seed and tables -- the checker
   has separate ones), so the walk carries it: hash(s ^ S[m]) = hash(s) ^ mh[m], no pass over the W words. The
   walk also carries |s|. Span masks: a mechanism's detectors sit in <= 3 consecutive words (it touches <= 2
   consecutive rounds); sw0/smk give |s & S[m]| and "S[m] subset of s" in three word operations. A model whose
   mechanisms do not fit (span3 = 0) takes the exact full-width path. (2026-10-21: leaves were ~95% of the time.) */
static uint64_t *dkey, *mh; static int DMAX, span3, *sw0; static uint64_t *smk;
static uint64_t mix64(uint64_t x) { x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; return x ^ (x >> 33); }
static uint64_t hk(const uint64_t *s) { uint64_t h = 0;
    for (int i = 0; i < W; i++) { uint64_t x = s[i]; while (x) { h ^= dkey[i * 64 + __builtin_ctzll(x)]; x &= x - 1; } } return h; }
static int popw(const uint64_t *s) { int c = 0; for (int i = 0; i < W; i++) c += __builtin_popcountll(s[i]); return c; }
static inline int inter(const uint64_t *s, int m) {
    if (!span3) { const uint64_t *ms = msyn + (size_t)m * W; int c = 0; for (int j = 0; j < W; j++) c += __builtin_popcountll(ms[j] & s[j]); return c; }
    const uint64_t *sw = s + sw0[m], *mk = smk + 3 * (size_t)m; int c = __builtin_popcountll(sw[0] & mk[0]);
    if (mk[1]) c += __builtin_popcountll(sw[1] & mk[1]);
    if (mk[2]) c += __builtin_popcountll(sw[2] & mk[2]);
    return c; }
static int eqs(const uint64_t *a, const uint64_t *b) { for (int i = 0; i < W; i++) if (a[i] != b[i]) return 0; return 1; }
/* hbits: one bit per value of the hash's bits 24.. that some mechanism has (the prover's own prefilter; a clear
   bit means no mechanism has that hash, so the table probe is skipped) */
static uint64_t *hbits, hbmask;
static inline int maybe(uint64_t h) { uint64_t k = h >> 24 & hbmask; return (int)(hbits[k >> 6] >> (k & 63) & 1); }
static int single_h(const uint64_t *s, uint64_t h) { if (!maybe(h)) return -1;
    for (uint64_t i = h & smask;; i = (i + 1) & smask) { int m = shash[i]; if (m < 0) return -1;
    if (mh[m] == h && eqs(msyn + (size_t)m * W, s)) return m; } }
static int single_of(const uint64_t *s) { return single_h(s, hk(s)); }
static int lo_allowed(int d, int o) { int lo = dstart[d], hi = dstart[d + 1];
    while (lo < hi) { int mid = (lo + hi) >> 1; if (morb[dlist[mid]] < o) lo = mid + 1; else hi = mid; } return lo; }

/* CDP2: one bit per node with remaining budget >= 3 and nonzero syndrome: 1 = expanded on the detector that
   the DETERMINISTIC pick rule names (fewest allowed mechanisms, ties to the lowest index; the checker
   recomputes it), 0 = pruned by the bound (this prover: at remaining >= 4). */
typedef struct { uint8_t *buf; size_t n, cap; int found, flen, path[64], fpath[64]; } Tr;
static void emit(Tr *t, int bit) {
    if ((t->n >> 3) >= t->cap) {
        size_t old = t->cap, nc = old ? 2 * old : 4096;
        t->buf = realloc(t->buf, nc); memset(t->buf + old, 0, nc - old); t->cap = nc;
    }
    if (bit) t->buf[t->n >> 3] |= (uint8_t)(1u << (t->n & 7));
    t->n++;
}

/* the mechanism b with S[b] = s ^ S[a], via the carried hash h ^ mh[a]; exact word compare on a hash match */
static int single_x(const uint64_t *s, int a, uint64_t h) {
    if (!maybe(h)) return -1;
    const uint64_t *sa = msyn + (size_t)a * W;
    for (uint64_t i = h & smask;; i = (i + 1) & smask) { int m = shash[i]; if (m < 0) return -1; if (mh[m] != h) continue;
        const uint64_t *sm = msyn + (size_t)m * W; int ok = 1;
        for (int j = 0; j < W && ok; j++) ok = sm[j] == (s[j] ^ sa[j]);
        if (ok) return m; } }
static int *lst, *lls;   /* per detector: touching mechanisms by degree, descending (the leaves' disjoint-pair scan) */
static int complete2(const uint64_t *s, uint64_t h, int ps, uint32_t l, int maxk) {
    if (ps <= DMAX) { int m = single_h(s, h); if (m >= 0 && (l ^ mobs[m])) return 1; }
    if (maxk < 2 || ps > 2 * DMAX) return 0;
    for (uint64_t i = h & pmask;; i = (i + 1) & pmask) { int a = pa[i]; if (a < 0) break; int b = pb[i];
        if ((mh[a] ^ mh[b]) != h) continue;
        const uint64_t *sa = msyn + (size_t)a * W, *sb = msyn + (size_t)b * W; int ok = 1;
        for (int j = 0; j < W; j++) if ((sa[j] ^ sb[j]) != s[j]) { ok = 0; break; }
        if (ok && (l ^ mobs[a] ^ mobs[b])) return 1; }
    /* disjoint pairs: the lowest firing detector d is in a, a is a subset of s, the rest is one mechanism of size
       ps - deg(a), which must be 1..DMAX (exact: so deg(a) in [ps - DMAX, ps - 1]; the list is degree-sorted) */
    int d = -1; for (int i = 0; i < W && d < 0; i++) if (s[i]) d = i * 64 + __builtin_ctzll(s[i]);
    for (int t = lst[d]; t < lst[d + 1]; t++) { int a = lls[t], dg = mdeg[a];
        if (dg >= ps) continue;
        if (dg < ps - DMAX) break;
        if (inter(s, a) != dg) continue;
        int b = single_x(s, a, h ^ mh[a]); if (b >= 0 && (l ^ mobs[a] ^ mobs[b])) return 1; }
    return 0;
}
/* Lower bound: orbit-aware detector packing + exact rational dual ascent.
   Let A_m = syndrome(s) intersect syndrome(mechanism m), with orb(m)>=o.
   Any residual completion covers all fired detectors an ODD number of times,
   hence at least once. If y_d>=0 and sum_{d in A_m} y_d<=2520 for every
   allowed m, any completion needs >=ceil(sum_d y_d / 2520) faults.
   The initial y_d=floor(2520 / max_{m: d in A_m}|A_m|) is feasible.
   Coordinate ascent only INCREASES y_d by the minimum remaining slack of
   an allowed mechanism containing d, so feasibility is inductive.
   Trigger the more expensive ascent only when the cheap bound is ONE
   integer away from pruning: the checker proves it independently. */
static int bound(const uint64_t *s, int o, int rem) {
    uint64_t r[MAXW]; memcpy(r, s, 8 * W); int indep = 0;
    for (int i = 0; i < W; i++) while (r[i]) {
        int d = i * 64 + __builtin_ctzll(r[i]); indep++;
        const uint64_t *c = cot_orbit + ((size_t)o * ND + d) * W;
        for (int j = 0; j < W; j++) r[j] &= ~c[j];
    }
    if (indep > rem) return indep;
    unsigned short y[MAXW * 64];
    int fired[MAXW * 64], nf = 0;
    long num = 0;
    for (int i = 0; i < W; i++) {
        uint64_t x = s[i];
        while (x) {
            int d = i * 64 + __builtin_ctzll(x); x &= x - 1;
            int cd = 0;
            for (int t = lo_allowed(d, o); t < dstart[d + 1]; t++) {
                int c = inter(s, dlist[t]); if (c > cd) cd = c;
            }
            if (cd == 0) return 1000;
            fired[nf++] = d;
            y[d] = (unsigned short)(2520 / cd);
            num += y[d];
        }
    }
    int hf = (int)((num + 2519) / 2520);
    if (hf > rem || rem < 4 || hf < rem) return indep > hf ? indep : hf;
    for (int j = 0; j < nf; j++) {
        int d = fired[j], rise = 2520;
        for (int t = lo_allowed(d, o); t < dstart[d + 1]; t++) {
            int m = dlist[t], total = 0;
            for (int k = 0; k < mdeg[m]; k++) {
                int v = mdet[m * MAXDEG + k];
                if ((s[v >> 6] >> (v & 63)) & 1ULL) total += y[v];
            }
            int slack = 2520 - total;
            if (slack < rise) rise = slack;
            if (rise == 0) break;
        }
        if (rise > 0) {
            y[d] = (unsigned short)(y[d] + rise);
            num += rise;
            if (num > (long)rem * 2520) return rem + 1;
        }
    }
    return indep > hf ? indep : hf;
}
static int pick(const uint64_t *s, int o) { int bd = -1, bn = 1 << 30;
    for (int i = 0; i < W; i++) { uint64_t x = s[i]; while (x) { int d = i * 64 + __builtin_ctzll(x); x &= x - 1;
        int n = dstart[d + 1] - lo_allowed(d, o); if (n < bn) { bn = n; bd = d; } } } return bd; }

/* Weight-three terminal decision: no search trace is necessary. For any
   nonzero residual detector d, a solution of <=3 mechanisms must contain
   an allowed mechanism incident to d. Enumerate that first mechanism,
   then decide the residual <=2 problem with the existing exact table.
   Unlike adding a costly depth-three lower bound, this completely DECIDES
   the subtree. CDP5 lets the checker replay it without trusting a bit. */
/* R206: a cheap EXACT terminal feasibility test, before the exhaustive
   three-fault completion. If four currently firing detector vertices can be
   greedily selected so that no ALLOWED mechanism (orbit >= o) touches two,
   every parity-correct completion requires >=4 mechanisms. Consequently
   a <=3-fault tail can be rejected without calling complete2.
   The co-touch suffix graph includes all allowed mechanisms and is computed
   from the same hash-pinned model. Greedy selection need not be maximal:
   reaching four witnesses is enough. This cannot reject a real solution.
   The independent CDP5 checker continues to enumerate every such tail
   with its own method, so pruning here can NEVER promote an invalid proof. */
static int complete3(const uint64_t *s, uint64_t h, int ps, uint32_t l, int o) {
    uint64_t needed[MAXW]; memcpy(needed, s, 8 * W);
    int packed = 0;
    for (int i = 0; i < W; i++) while (needed[i]) {
        int d = i * 64 + __builtin_ctzll(needed[i]);
        if (++packed > 3) return 0;
        const uint64_t *neighbors = cot_orbit + ((size_t)o * ND + d) * W;
        for (int j = 0; j < W; j++) needed[j] &= ~neighbors[j];
    }
    int d = pick(s, o);
    if (d < 0) return 0;
    uint64_t ns[MAXW];
    for (int k = lo_allowed(d, o); k < dstart[d + 1]; k++) {
        int m = dlist[k];
        uint32_t nl = l ^ mobs[m];
        int pn = ps + mdeg[m] - 2 * inter(s, m);
        for (int j = 0; j < W; j++) ns[j] = s[j] ^ msyn[(size_t)m * W + j];
        if (pn == 0) {
            if (nl) return 1;
        } else if (complete2(ns, h ^ mh[m], pn, nl, 2)) return 1;
    }
    return 0;
}

static int dfs(Tr *t, const uint64_t *s, uint64_t h, int ps, uint32_t l, int depth, int budget, int o) {
    int z = ps == 0;
#ifdef LIAR   /* NEGATIVE CONTROL ONLY: a prover that ignores every solution it reaches (tests build it) */
    if (z) return 0;
#else
    if (z) { if (l) { t->flen = depth; memcpy(t->fpath, t->path, 4 * depth); return 1; } return 0; }
#endif
    int rem = budget - depth;
    if (rem <= 0) return 0;
#ifdef LIAR
    if (rem <= 2) return 0;
#else
    if (rem <= 2) return complete2(s, h, ps, l, rem);
#endif
    /* CDP5: depth-three subtrees are decided completely; no trace bit. */
    if (rem == 3) {
#ifdef LIAR
        return 0; /* deliberately unsound negative control */
#else
        return complete3(s, h, ps, l, o);
#endif
    }
    /* prune where the bound exceeds the budget, from remaining 4 up. Measured 2026-10-21: also pruning at
       remaining 3 kills ~46% of those nodes, but computing the bound at every one of them cost the prover more
       than the leaves it saved (14.7 s vs 13.0 s at weight 8), so the proof language stays the original one */
    if (rem >= 4 && bound(s, o, rem) > rem) { emit(t, 0); return 0; }
    int d = pick(s, o); emit(t, 1);
    uint64_t ns[MAXW];
    for (int k = lo_allowed(d, o); k < dstart[d + 1]; k++) { int m = dlist[k]; const uint64_t *ms = msyn + (size_t)m * W;
        int pn = ps + mdeg[m] - 2 * inter(s, m);
        for (int j = 0; j < W; j++) ns[j] = s[j] ^ ms[j];
        t->path[depth] = m;
        if (dfs(t, ns, h ^ mh[m], pn, l ^ mobs[m], depth + 1, budget, o)) return 1; }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: cdist_prove model.txt W proof.bin [MAXROUND]\n"); return 2; }
    FILE *f = fopen(argv[1], "r"); if (!f) return 2; int w = atoi(argv[2]);
    int maxround = argc > 4 ? atoi(argv[4]) : -1;   /* time-shift anchoring, rule S (refused unless time-ordered) */
    if (fscanf(f, "%d %d %d %d %d %d", &ND, &NM, &NO, &NORB, &GL, &GM) != 6) return 2;
    W = (ND + 63) / 64; if (W > MAXW || NO > 32) return 2;
    msyn = calloc((size_t)NM * W, 8); mobs = calloc(NM, 4); mdeg = calloc(NM, 4); mdet = calloc((size_t)NM * MAXDEG, 4); morb = calloc(NM, 4);
    int *cnt = calloc(ND + 1, 4);
    for (int m = 0; m < NM; m++) { int k; unsigned ob; if (fscanf(f, "%d", &k) != 1 || k > MAXDEG) return 2; mdeg[m] = k;
        for (int i = 0; i < k; i++) { int d; if (fscanf(f, "%d", &d) != 1) return 2; mdet[m * MAXDEG + i] = d;
            msyn[(size_t)m * W + d / 64] ^= 1ULL << (d % 64); cnt[d]++; }
        if (fscanf(f, "%u %d", &ob, &morb[m]) != 2) return 2; mobs[m] = ob; }
    anchors = calloc(NORB, 4); for (int i = 0; i < NORB; i++) if (fscanf(f, "%d", &anchors[i]) != 1) return 2;
    int *drnd = calloc(ND, 4);
    for (int d = 0; d < ND; d++) { int s_, t_; if (fscanf(f, "%d %d %d", &s_, &t_, &drnd[d]) != 3) return 2; }
    fclose(f);
    int nanch = NORB;
    if (maxround >= 0) {
        int prev = -1; nanch = 0;
        for (int o = 0; o < NORB; o++) { int m = anchors[o], r = 1 << 30;
            for (int i = 0; i < mdeg[m]; i++) if (drnd[mdet[m * MAXDEG + i]] < r) r = drnd[mdet[m * MAXDEG + i]];
            if (r < prev) { fprintf(stderr, "orbits are not time-ordered: time-shift anchoring refused\n"); return 2; }
            prev = r; if (r <= maxround) nanch = o + 1; }
    }
    dstart = calloc(ND + 1, 4); for (int d = 0; d < ND; d++) dstart[d + 1] = dstart[d] + cnt[d];
    dlist = calloc(dstart[ND] + 1, 4); int *fill = calloc(ND, 4);
    for (int m = 0; m < NM; m++) for (int i = 0; i < mdeg[m]; i++) { int d = mdet[m * MAXDEG + i]; dlist[dstart[d] + fill[d]++] = m; }
    for (int d = 0; d < ND; d++) for (int i = dstart[d] + 1; i < dstart[d + 1]; i++) { int m = dlist[i], j = i - 1;
        while (j >= dstart[d] && (morb[dlist[j]] > morb[m] || (morb[dlist[j]] == morb[m] && dlist[j] > m))) { dlist[j + 1] = dlist[j]; j--; } dlist[j + 1] = m; }
    /* Orbit-conditioned co-touch graphs. A solution in anchor orbit o may use only
       mechanisms in orbits >= o. A set of syndrome detectors with no allowed
       mechanism covering two of them requires at least that many faults.
       Build suffix-unions once; per-node packing costs exactly as before. */
    cot_orbit = calloc((size_t)NORB * ND * W, sizeof(uint64_t));
    if (!cot_orbit) return 2;
    for (int m = 0; m < NM; m++) {
        if (morb[m] < 0 || morb[m] >= NORB) return 2;
        uint64_t *co = cot_orbit + (size_t)morb[m] * ND * W;
        for (int i = 0; i < mdeg[m]; i++) for (int j = 0; j < mdeg[m]; j++) {
            int a = mdet[m * MAXDEG + i], b = mdet[m * MAXDEG + j];
            co[(size_t)a * W + b / 64] |= 1ULL << (b % 64);
        }
    }
    for (int o = NORB - 2; o >= 0; o--) {
        uint64_t *co = cot_orbit + (size_t)o * ND * W;
        const uint64_t *next = co + (size_t)ND * W;
        for (size_t j = 0; j < (size_t)ND * W; j++) co[j] |= next[j];
    }
    /* Include the vertex itself even if no allowed mechanism touches it. */
    for (int o = 0; o < NORB; o++) for (int d = 0; d < ND; d++)
        cot_orbit[((size_t)o * ND + d) * W + d / 64] |= 1ULL << (d % 64);
    /* linear hash keys, mechanism hashes, the widest mechanism, span masks, degree-sorted leaf lists */
    dkey = malloc(8 * (size_t)ND); for (int d = 0; d < ND; d++) dkey[d] = mix64(0x5EED9E37ULL ^ ((uint64_t)d * 0x9E3779B97F4A7C15ULL));
    mh = calloc(NM, 8); DMAX = 0; span3 = 1; sw0 = malloc(4 * (size_t)NM); smk = calloc(3 * (size_t)NM, 8);
    for (int m = 0; m < NM; m++) { int lo = 1 << 30, hi = -1;
        for (int i = 0; i < mdeg[m]; i++) { int d = mdet[m * MAXDEG + i]; mh[m] ^= dkey[d]; if (d / 64 < lo) lo = d / 64; if (d / 64 > hi) hi = d / 64; }
        if (mdeg[m] > DMAX) DMAX = mdeg[m];
        if (hi - lo > 2) span3 = 0;
        sw0[m] = lo; for (int i = 0; i < mdeg[m]; i++) { int d = mdet[m * MAXDEG + i], k = d / 64 - lo; if (k < 3) smk[3 * (size_t)m + k] ^= 1ULL << (d % 64); } }
    lst = calloc(ND + 1, 4); for (int d = 0; d < ND; d++) lst[d + 1] = lst[d] + cnt[d];
    lls = malloc(4 * (size_t)(lst[ND] + 1)); int *lf = calloc(ND, 4);
    for (int dg = MAXDEG; dg >= 1; dg--) for (int m = 0; m < NM; m++) if (mdeg[m] == dg)
        for (int i = 0; i < mdeg[m]; i++) { int d = mdet[m * MAXDEG + i]; lls[lst[d] + lf[d]++] = m; }
    { int hb = 16; while (hb < 18 && (1L << hb) < 16L * NM) hb++; hbmask = (1ULL << hb) - 1; hbits = calloc((size_t)1 << (hb - 6), 8);
      for (int m = 0; m < NM; m++) { uint64_t k = mh[m] >> 24 & hbmask; hbits[k >> 6] |= 1ULL << (k & 63); } }
    int sb = 1; while ((1 << sb) < 4 * NM) sb++; smask = (1ULL << sb) - 1; shash = malloc(sizeof(int) << sb);
    for (uint64_t i = 0; i <= smask; i++) shash[i] = -1;
    for (int m = 0; m < NM; m++) { if (single_of(msyn + (size_t)m * W) >= 0) { fprintf(stderr, "duplicate syndrome\n"); return 2; }
        for (uint64_t i = hk(msyn + (size_t)m * W) & smask;; i = (i + 1) & smask) if (shash[i] < 0) { shash[i] = m; break; } }
    char *mark = calloc(NM, 1); int *nb = malloc(4 * NM); long cap = 0;
    for (int pass = 0; pass < 2; pass++) {
        if (pass == 1) { int pbits = 1; while ((1L << pbits) < 2 * cap) pbits++; pmask = (1ULL << pbits) - 1;
            pa = malloc(4 * (pmask + 1)); pb = malloc(4 * (pmask + 1)); for (uint64_t i = 0; i <= pmask; i++) pa[i] = -1; }
        for (int a = 0; a < NM; a++) { int k = 0;
            for (int i = 0; i < mdeg[a]; i++) { int d = mdet[a * MAXDEG + i];
                for (int t = dstart[d]; t < dstart[d + 1]; t++) { int b = dlist[t]; if (b > a && !mark[b]) { mark[b] = 1; nb[k++] = b; } } }
            for (int i = 0; i < k; i++) { int b = nb[i]; mark[b] = 0;
                if (pass == 0) { cap++; continue; }
                uint64_t tmp[MAXW]; for (int j = 0; j < W; j++) tmp[j] = msyn[(size_t)a * W + j] ^ msyn[(size_t)b * W + j];
                for (uint64_t q = hk(tmp) & pmask;; q = (q + 1) & pmask) if (pa[q] < 0) { pa[q] = a; pb[q] = b; break; } } } }
    /* CDP5: per anchor a root code -- 0 the root is pruned by the bound (remaining w-1), 1 the root is expanded
       on the deterministic pick and each allowed child is a separate task with its own bitstream (children in
       (orbit, id) order), 2 (w <= 3) one bitstream for the whole anchor. Children run in parallel: the anchor-0
       subtree no longer serialises the run. */
    double t0 = omp_get_wtime();
    int8_t *rootcode = calloc(NORB, 1); int *nchild = calloc(NORB, 4), *first = calloc(NORB + 1, 4);
    int ntask = 0;
    for (int o = 0; o < nanch; o++) {
        const uint64_t *s0 = msyn + (size_t)anchors[o] * W; first[o] = ntask;
        if (w <= 3) { rootcode[o] = 2; ntask++; continue; }
        if (bound(s0, o, w - 1) > w - 1) { rootcode[o] = 0; continue; }
        int d = pick(s0, o); rootcode[o] = 1; nchild[o] = dstart[d + 1] - lo_allowed(d, o); ntask += nchild[o];
    }
    first[nanch] = ntask;
    int *task_o = malloc(4 * (size_t)(ntask + 1)), *task_m = malloc(4 * (size_t)(ntask + 1));
    for (int o = 0; o < nanch; o++) {
        if (rootcode[o] == 2) { task_o[first[o]] = o; task_m[first[o]] = -1; }
        if (rootcode[o] != 1) continue;
        const uint64_t *s0 = msyn + (size_t)anchors[o] * W; int d = pick(s0, o), k = first[o];
        for (int t = lo_allowed(d, o); t < dstart[d + 1]; t++) { task_o[k] = o; task_m[k++] = dlist[t]; }
    }
    Tr *tr = calloc(ntask + 1, sizeof(Tr)); volatile int found = 0; int fo = -1;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int k = 0; k < ntask; k++) {
        if (found) continue;
        int o = task_o[k], a = anchors[o], m = task_m[k]; int hit; tr[k].path[0] = a;
        if (m < 0) hit = dfs(&tr[k], msyn + (size_t)a * W, mh[a], mdeg[a], mobs[a], 1, w, o);
        else { uint64_t ns[MAXW]; for (int j = 0; j < W; j++) ns[j] = msyn[(size_t)a * W + j] ^ msyn[(size_t)m * W + j];
               tr[k].path[1] = m; hit = dfs(&tr[k], ns, hk(ns), popw(ns), mobs[a] ^ mobs[m], 2, w, o); }
        if (hit) {
            #pragma omp critical
            { if (!found) { found = 1; fo = o; } } }
    }
    if (found) { printf("FOUND at weight <= %d (orbit %d); no proof written\n", w, fo); return 1; }
    FILE *g = fopen(argv[3], "wb"); fwrite("CDP5", 1, 4, g); int32_t hdr[4] = {w, NORB, maxround, nanch}; fwrite(hdr, 4, 4, g);
    size_t tot = 0;
    for (int o = 0; o < nanch; o++) {
        fwrite(&rootcode[o], 1, 1, g);
        if (rootcode[o] == 1) fwrite(&nchild[o], 4, 1, g);
        for (int k = first[o]; k < first[o + 1]; k++) { int64_t n = (int64_t)tr[k].n; fwrite(&n, 8, 1, g);
            fwrite(tr[k].buf, 1, (tr[k].n + 7) >> 3, g); tot += tr[k].n; }
    }
    fclose(g);
    if (maxround >= 0)
        printf("NONE at weight <= %d with earliest round <= %d: proof written to %s (%zu trace bits, %d tasks, %.1fs)\n", w, maxround, argv[3], tot, ntask, omp_get_wtime() - t0);
    else
        printf("NONE at weight <= %d: proof written to %s (%zu trace bits, %d tasks, %.1fs)\n", w, argv[3], tot, ntask, omp_get_wtime() - t0);
    return 0;
}
