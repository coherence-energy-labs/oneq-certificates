/* ccheck (R198): independent checker for cdist_prove proofs of "no undetectable logical error of weight <= W".
 *
 * The proof (format CDP2) is one bit per search node with remaining budget >= 3 and a nonzero syndrome:
 * 1 = expanded, 0 = pruned by the bound. Everything else is re-derived here from the model file:
 *   (1) the symmetry: for every declared translation it maps detectors by their (site, type, round)
 *       coordinates, checks the mechanism set is invariant, and checks the observables transform AFFINELY,
 *       obs(g m) = M obs(m) + K syn(m), by GF(2) elimination (with every g^-1 in the group this makes
 *       "flips some logical" invariant on undetectable sets); orbits are recomputed and must equal the ids
 *       in the file; anchors must be the first member of each orbit;
 *   (2) the branching detector of every expanded node, by the deterministic rule (the firing detector with
 *       the fewest mechanisms of orbit >= o; ties to the lowest index) -- re-implemented here; ANY firing
 *       detector is sound, and a prover that branched elsewhere desynchronises and is rejected;
 *   (3) the child list: all mechanisms of orbit >= o touching that detector, in (orbit, id) order;
 *   (4) every bound prune, in exact integer arithmetic (independent set; fractional cover over 2520);
 *   (5) every leaf (remaining budget <= 2), with its OWN procedure: the lowest firing detector lies in
 *       exactly one member of any single/pair completion, so try every mechanism touching it and look the
 *       remainder up as a single syndrome (its own Zobrist index with exact sparse confirmation, and exact
 *       popcount filters carried down the walk -- not the prover's tables). 2026-10-21: this replaced a
 *       full-width FNV hash per candidate; leaves were 98% of the checker's time.
 * Rules and their arguments: RESULTS.md (anchoring, toggling semantics, canonical branching, bounds).
 * Usage: ccheck model.txt proof.bin full|c3     Exit 0 = VERIFIED, 1 = REJECTED, 2 = malformed input.
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
static uint64_t *S;       /* NM x W syndromes */
static uint32_t *OB; static int *DEG, *DET, *ORB, *ANC, *dsite, *dtyp, *drnd;
static int *ord;          /* mechanisms sorted by syndrome (for binary search) */
static int *astart, *alist;  /* per detector: touching mechanisms sorted by (orbit, id) */
static uint64_t *CT_ORBIT; /* independently built co-touch suffix graphs */
static int modern_trace; /* 0=CDP3, 1=CDP4, 2=CDP5 with exact weight-3 terminal */

static int cmp_syn(const uint64_t *a, const uint64_t *b) { for (int i = W - 1; i >= 0; i--) { if (a[i] < b[i]) return -1; if (a[i] > b[i]) return 1; } return 0; }
static int cmp_ord(const void *x, const void *y) { int a = *(const int *)x, b = *(const int *)y; int c = cmp_syn(S + (size_t)a * W, S + (size_t)b * W); return c ? c : (a < b ? -1 : a > b); }
/* first index in ord with syndrome >= s */
static int lower(const uint64_t *s) { int lo = 0, hi = NM; while (lo < hi) { int mid = (lo + hi) / 2; if (cmp_syn(S + (size_t)ord[mid] * W, s) < 0) lo = mid + 1; else hi = mid; } return lo; }

/* The checker's own syndrome index (a different method and layout from the prover's). Each detector gets a
   random 64-bit key (splitmix64 of its index, a checker-specific seed) and a syndrome hashes to the XOR of its
   detectors' keys (Zobrist), so the hash of s ^ S[a] is hash(s) ^ hz[a] -- O(1), no pass over the W words.
   The walk also carries |s| (popcount), updated in O(degree) per step. A hash only PROPOSES; every hit is
   confirmed EXACTLY on the candidate's sparse detector list, so collisions cannot change an answer. Every
   mechanism is in the table (equal syndromes with different observables included): linear probing, no
   deletions, so a probe that stops at an empty slot has seen every mechanism with that hash. */
static uint64_t *KEY, *hz; static int *ztab; static uint64_t zmask; static int DMAX;
/* ZB: one bit per value of the hash's high bits that some mechanism has -- a probe whose bit is clear cannot
   match (exact: every mechanism's bit is set), and the bitmap stays in L1/L2 while most probes miss */
static uint64_t *ZB, zbmask;
static uint64_t splitmix(uint64_t x) { x += 0x9E3779B97F4A7C15ULL; x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL; return x ^ (x >> 31); }
static void build_index(void) {
    KEY = malloc(8 * (size_t)ND); for (int d = 0; d < ND; d++) KEY[d] = splitmix(0xC4EC4ULL * 1000003ULL + (uint64_t)d);
    hz = calloc(NM, 8); DMAX = 0;
    for (int m = 0; m < NM; m++) { for (int i = 0; i < DEG[m]; i++) hz[m] ^= KEY[DET[m * MAXDEG + i]]; if (DEG[m] > DMAX) DMAX = DEG[m]; }
    int b = 1; while ((1L << b) < 4L * NM) b++; zmask = (1ULL << b) - 1; ztab = malloc(sizeof(int) << b);
    for (uint64_t i = 0; i <= zmask; i++) ztab[i] = -1;
    for (int m = 0; m < NM; m++) for (uint64_t i = hz[m] & zmask;; i = (i + 1) & zmask) if (ztab[i] < 0) { ztab[i] = m; break; }
    int zb = 16; while (zb < 18 && (1L << zb) < 16L * NM) zb++; zbmask = (1ULL << zb) - 1; ZB = calloc((size_t)1 << (zb - 6), 8);
    for (int m = 0; m < NM; m++) { uint64_t k = hz[m] >> 32 & zbmask; ZB[k >> 6] |= 1ULL << (k & 63); }
}
static inline int bit(const uint64_t *s, int d) { return (int)(s[d >> 6] >> (d & 63) & 1); }
/* Spans: a mechanism touches <= 2 consecutive rounds, so its detectors fit in 3 consecutive words: |s & S[m]| is
   three masked popcounts. LE: per detector, every mechanism touching it (any orbit) as a contiguous record,
   sorted by degree DESCENDING so a leaf can stop as soon as no later candidate can complete it. A model whose
   mechanisms do not fit (span_ok = 0) uses the exact sparse path instead. */
typedef struct { int a, deg, w0; uint64_t m[3]; } LE;
static LE *LL; static int *lstart; static int span_ok;
static int cmp_le(const void *x, const void *y) { const LE *p = x, *q = y;
    return p->deg != q->deg ? q->deg - p->deg : p->a - q->a; }
static void build_spans(void) {
    span_ok = 1;
    for (int m = 0; m < NM; m++) { int lo = 1 << 30, hi = -1;
        for (int i = 0; i < DEG[m]; i++) { int w_ = DET[m * MAXDEG + i] >> 6; if (w_ < lo) lo = w_; if (w_ > hi) hi = w_; }
        if (hi - lo > 2) span_ok = 0; }
    if (getenv("CCHECK_SPARSE")) span_ok = 0;   /* test hook: force the exact sparse path (tests run both) */
    lstart = calloc(ND + 1, 4); for (int m = 0; m < NM; m++) for (int i = 0; i < DEG[m]; i++) lstart[DET[m * MAXDEG + i] + 1]++;
    for (int d = 0; d < ND; d++) lstart[d + 1] += lstart[d];
    LL = calloc((size_t)lstart[ND] + 1, sizeof(LE)); int *fill = calloc(ND, 4);
    for (int m = 0; m < NM; m++) {
        LE e; e.a = m; e.deg = DEG[m]; e.w0 = 1 << 30; e.m[0] = e.m[1] = e.m[2] = 0;
        for (int i = 0; i < DEG[m]; i++) { int w_ = DET[m * MAXDEG + i] >> 6; if (w_ < e.w0) e.w0 = w_; }
        for (int i = 0; i < DEG[m]; i++) { int x = DET[m * MAXDEG + i], k = (x >> 6) - e.w0; if (k < 3) e.m[k] ^= 1ULL << (x & 63); }
        for (int i = 0; i < DEG[m]; i++) { int d = DET[m * MAXDEG + i]; LL[lstart[d] + fill[d]++] = e; }
    }
    for (int d = 0; d < ND; d++) qsort(LL + lstart[d], lstart[d + 1] - lstart[d], sizeof(LE), cmp_le);
    free(fill);
}
static inline int inter_le(const uint64_t *s, const LE *e) {
    const uint64_t *sw = s + e->w0; int c = __builtin_popcountll(sw[0] & e->m[0]);
    if (e->m[1]) c += __builtin_popcountll(sw[1] & e->m[1]);
    if (e->m[2]) c += __builtin_popcountll(sw[2] & e->m[2]);
    return c; }
static uint64_t zob(const uint64_t *s) { uint64_t h = 0;
    for (int i = 0; i < W; i++) { uint64_t x = s[i]; while (x) { h ^= KEY[i * 64 + __builtin_ctzll(x)]; x &= x - 1; } } return h; }
static int popc(const uint64_t *s) { int c = 0; for (int i = 0; i < W; i++) c += __builtin_popcountll(s[i]); return c; }
/* |s ^ S[m]| from |s| in O(deg m) */
static inline int pop_after(const uint64_t *s, int ps, int m) { int in = 0;
    for (int i = 0; i < DEG[m]; i++) in += bit(s, DET[m * MAXDEG + i]); return ps + DEG[m] - 2 * in; }
/* Is there a single mechanism c with S[c] = s ^ S[a] (a = -1: S[c] = s), |that| = pr, hash hr, and the
   observable parity l ^ OB[a] ^ OB[c] nonzero? Exact on the sparse list: |S[c]| = pr and every detector of c
   is in the target set, so the sets are equal. */
static int single_hit(const uint64_t *s, int a, uint64_t hr, int pr, uint32_t l) {
    uint64_t k = hr >> 32 & zbmask; if (!(ZB[k >> 6] >> (k & 63) & 1)) return 0;
    for (uint64_t i = hr & zmask;; i = (i + 1) & zmask) { int c = ztab[i]; if (c < 0) return 0;
        if (hz[c] != hr || DEG[c] != pr || c == a) continue;
        int ok = 1;
        for (int k = 0; k < DEG[c] && ok; k++) { int x = DET[c * MAXDEG + k];
            ok = bit(s, x) ^ (a >= 0 ? bit(S + (size_t)a * W, x) : 0); }
        if (ok && (l ^ (a >= 0 ? OB[a] : 0) ^ OB[c])) return 1; }
}
/* A completion of weight <= maxk (<= 2) flipping a logical? The singles need |s| <= DMAX; a pair has the lowest
   firing detector d in exactly one member a, and the other member's syndrome s ^ S[a] must have 1..DMAX
   detectors -- both filters are exact (no mechanism is empty or wider than DMAX), not heuristics. */
static int leaf_exists(const uint64_t *s, uint64_t hs, int ps, uint32_t l, int maxk) {
    if (ps <= DMAX && single_hit(s, -1, hs, ps, l)) return 1;
    if (maxk < 2 || ps > 2 * DMAX) return 0;
    int d = -1; for (int i = 0; i < W && d < 0; i++) if (s[i]) d = i * 64 + __builtin_ctzll(s[i]);
    if (span_ok) {
        for (const LE *e = LL + lstart[d], *end = LL + lstart[d + 1]; e < end; e++) {
            if (e->deg < ps - DMAX) break;          /* pr >= ps - deg > DMAX for this and every later (smaller) one */
            int pr = ps + e->deg - 2 * inter_le(s, e);
            if (pr < 1 || pr > DMAX) continue;
            if (single_hit(s, e->a, hs ^ hz[e->a], pr, l)) return 1;
        }
        return 0;
    }
    for (int t = astart[d]; t < astart[d + 1]; t++) {
        int a = alist[t]; int pr = pop_after(s, ps, a);
        if (pr < 1 || pr > DMAX) continue;
        if (single_hit(s, a, hs ^ hz[a], pr, l)) return 1;
    }
    return 0;
}
/* the deterministic pick rule, re-implemented: among firing detectors in increasing order, the one with the
   fewest mechanisms of orbit >= o touching it; ties go to the lowest index */
static int allowed_count(int d, int o) { int lo = astart[d], hi = astart[d + 1];
    while (lo < hi) { int mid = (lo + hi) / 2; if (ORB[alist[mid]] < o) lo = mid + 1; else hi = mid; } return astart[d + 1] - lo; }
static int pick_c(const uint64_t *s, int o) { int best = -1, bc = 1 << 30;
    for (int i = 0; i < W; i++) { uint64_t x = s[i]; while (x) { int d = i * 64 + __builtin_ctzll(x); x &= x - 1;
        int c = allowed_count(d, o); if (c < bc) { bc = c; best = d; } } }
    return best; }
/* Independent exact-integer proof of a bounded-depth prune. No witness or
   claimed dual values are trusted: this checker derives each dual constraint
   from its own detector arrays, then performs feasibility-preserving raises.
   If it fails to derive a strong enough bound, the claimed prune is rejected. */
/* CDP5's independently executed, exact weight-three decision.
   A parity-odd detector must be toggled by at least one of the selected
   mechanisms. Check every allowed first mechanism, then invoke the
   checker's separately implemented exact one/two-mechanism oracle. */
static int leaf_exists3(const uint64_t *s, uint64_t hs, int ps, uint32_t l, int o) {
    /* Independent R206 admissible obstruction. This program reconstructed its
       own co-touch graph from checked DET and ORB data, not the prover's
       internal graph. If four fired detectors cannot share an allowed
       mechanism in pairs, >=4 distinct faults are required to toggle them.
       The <=3 subtree therefore contains no solution. */
    uint64_t remaining[MAXW]; memcpy(remaining, s, (size_t)W * sizeof(uint64_t));
    int disjoint_count = 0;
    for (int word = 0; word < W; word++) while (remaining[word]) {
        const int detector = word * 64 + __builtin_ctzll(remaining[word]);
        if (++disjoint_count == 4) return 0;
        const uint64_t *touch = CT_ORBIT + ((size_t)o * ND + detector) * W;
        for (int j = 0; j < W; j++) remaining[j] &= ~touch[j];
    }
    int d = pick_c(s, o);
    if (d < 0) return 0;
    uint64_t residual[MAXW];
    for (int t = astart[d]; t < astart[d + 1]; t++) {
        int a = alist[t];
        if (ORB[a] < o) continue;
        int pop = pop_after(s, ps, a);
        uint32_t logical = l ^ OB[a];
        for (int j = 0; j < W; j++) residual[j] = s[j] ^ S[(size_t)a * W + j];
        if (pop == 0) {
            if (logical) return 1;
        } else if (leaf_exists(residual, hs ^ hz[a], pop, logical, 2)) return 1;
    }
    return 0;
}
static int bound_ok(const uint64_t *s, int o, int rem) {
    uint64_t remaining[MAXW]; memcpy(remaining, s, W * sizeof(uint64_t));
    int packed = 0;
    for (int i = 0; i < W; i++) while (remaining[i]) {
        int d = i * 64 + __builtin_ctzll(remaining[i]);
        packed++;
        const uint64_t *neighbors = CT_ORBIT + ((size_t)(modern_trace ? o : 0) * ND + d) * W;
        for (int k = 0; k < W; k++) remaining[k] &= ~neighbors[k];
    }
    if (packed > rem) return 1;
    unsigned short weight[MAXW * 64];
    int hot[MAXW * 64], count = 0;
    long sum = 0;
    for (int i = 0; i < W; i++) {
        uint64_t hotbits = s[i];
        while (hotbits) {
            int d = i * 64 + __builtin_ctzll(hotbits); hotbits &= hotbits - 1;
            int max_cover = 0;
            for (int t = astart[d]; t < astart[d + 1]; t++) {
                int m = alist[t]; if (ORB[m] < o) continue;
                int cover = 0;
                for (int j = 0; j < DEG[m]; j++)
                    cover += bit(s, DET[m * MAXDEG + j]);
                if (cover > max_cover) max_cover = cover;
            }
            if (!max_cover) return 1;
            hot[count++] = d;
            weight[d] = (unsigned short)(2520 / max_cover);
            sum += weight[d];
        }
    }
    if (sum > 2520L * rem) return 1;
    if (!modern_trace || rem < 4 || sum <= 2520L * (rem - 1)) return 0;
    for (int j = 0; j < count; j++) {
        int d = hot[j];
        int allowance = 2520;
        for (int t = astart[d]; t < astart[d + 1]; t++) {
            int m = alist[t]; if (ORB[m] < o) continue;
            int used = 0;
            for (int k = 0; k < DEG[m]; k++) {
                int v = DET[m * MAXDEG + k];
                if (bit(s, v)) used += weight[v];
            }
            if (2520 - used < allowance) allowance = 2520 - used;
            if (!allowance) break;
        }
        if (allowance > 0) {
            weight[d] = (unsigned short)(weight[d] + allowance);
            sum += allowance;
            if (sum > 2520L * rem) return 1;
        }
    }
    return 0;
}

typedef struct { const uint8_t *tr; int64_t n, p; int bad; } Walk;
/* s with its Zobrist hash hs and popcount ps (both maintained incrementally; ps == 0 iff s == 0) */
static void node(Walk *wk, const uint64_t *s, uint64_t hs, int ps, uint32_t l, int depth, int w, int o) {
    if (wk->bad) return;
    if (ps == 0) { if (l) wk->bad = 1; return; }
    int rem = w - depth;
    if (rem <= 0) return;
    if (rem <= 2) { if (leaf_exists(s, hs, ps, l, rem)) wk->bad = 2; return; }
    if (rem == 3 && modern_trace == 2) {
        if (leaf_exists3(s, hs, ps, l, o)) wk->bad = 2;
        return;
    }
    if (wk->p >= wk->n) { wk->bad = 3; return; }
    int bit_ = wk->tr[wk->p >> 3] >> (wk->p & 7) & 1; wk->p++;
    /* a prune is accepted at remaining >= 4 where the bound (re-proved here in exact integers) exceeds it. A
       bound is valid at remaining 3 too, but the prover gains nothing there (measured 2026-10-21), so the proof
       language stays exactly the original one -- which keeps a flipped bit at a remaining-3 node a rejection */
    if (!bit_) { if (rem < 4 || !bound_ok(s, o, rem)) wk->bad = 4; return; }
    int v = pick_c(s, o);
    if (v < 0 || allowed_count(v, o) == 0) { if (!bound_ok(s, o, rem)) wk->bad = 5; return; }
    uint64_t ns[MAXW];
    for (int t = astart[v]; t < astart[v + 1] && !wk->bad; t++) { int m = alist[t]; if (ORB[m] < o) continue;
        int pn = pop_after(s, ps, m);
        for (int j = 0; j < W; j++) ns[j] = s[j] ^ S[(size_t)m * W + j];
        node(wk, ns, hs ^ hz[m], pn, l ^ OB[m], depth + 1, w, o); }
}


int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ccheck dem3.txt proof.bin full|c3|ex\n"); return 2; }
    FILE *f = fopen(argv[1], "r"); if (!f) return 2;
    if (fscanf(f, "%d %d %d %d %d %d", &ND, &NM, &NO, &NORB, &GL, &GM) != 6) return 2;
    W = (ND + 63) / 64; if (W > MAXW || NO > 32) return 2;
    S = calloc((size_t)NM * W, 8); OB = calloc(NM, 4); DEG = calloc(NM, 4); DET = calloc((size_t)NM * MAXDEG, 4); ORB = calloc(NM, 4);
    for (int m = 0; m < NM; m++) { if (fscanf(f, "%d", &DEG[m]) != 1 || DEG[m] > MAXDEG || DEG[m] < 1) return 2;
        for (int i = 0; i < DEG[m]; i++) { int d; if (fscanf(f, "%d", &d) != 1 || d < 0 || d >= ND) return 2;
            /* a detector listed twice would cancel in the syndrome while DEG still counts it: the leaf filters and
               the empty-syndrome rule then lie (Lean pass 2026-10-21, findings F1/F5) -- malformed, refused */
            if (S[(size_t)m * W + d / 64] >> (d % 64) & 1) { fprintf(stderr, "malformed: mechanism %d repeats detector %d\n", m, d); return 2; }
            DET[m * MAXDEG + i] = d; S[(size_t)m * W + d / 64] ^= 1ULL << (d % 64); }
        unsigned ob; if (fscanf(f, "%u %d", &ob, &ORB[m]) != 2) return 2; OB[m] = ob; }
    ANC = calloc(NORB, 4); for (int i = 0; i < NORB; i++) if (fscanf(f, "%d", &ANC[i]) != 1) return 2;
    dsite = calloc(ND, 4); dtyp = calloc(ND, 4); drnd = calloc(ND, 4);
    for (int d = 0; d < ND; d++) if (fscanf(f, "%d %d %d", &dsite[d], &dtyp[d], &drnd[d]) != 3) return 2;
    fclose(f);
    ord = malloc(4 * NM); for (int m = 0; m < NM; m++) ord[m] = m; qsort(ord, NM, 4, cmp_ord);
    /* ---- (1) symmetry, orbits, anchors ---- */
    int maxr = 0; for (int d = 0; d < ND; d++) if (drnd[d] > maxr) maxr = drnd[d];
    int nsite = GL * GM; int *dk = malloc(4 * (size_t)nsite * 2 * (maxr + 1)); for (int i = 0; i < nsite * 2 * (maxr + 1); i++) dk[i] = -1;
    for (int d = 0; d < ND; d++) { int key = (dsite[d] * 2 + dtyp[d]) * (maxr + 1) + drnd[d]; if (dk[key] != -1) { fprintf(stderr, "duplicate detector coordinate\n"); return 2; } dk[key] = d; }
    /* the declared group, each a SUBGROUP of Z_GL x Z_GM: all translations, the colour-preserving ones (a + b = 0
       mod 3), or the kernel of the morphing homomorphism x -> 1, y -> 0 (a even) */
    int full = strcmp(argv[3], "full") == 0, c3 = strcmp(argv[3], "c3") == 0, ex = strcmp(argv[3], "ex") == 0;
    if (!full && !c3 && !ex) { fprintf(stderr, "unknown group %s\n", argv[3]); return 2; }
    if ((c3 && (GL % 3 || GM % 3)) || (ex && GL % 2)) { fprintf(stderr, "group %s is not a subgroup here\n", argv[3]); return 2; }
    int ng = 0; int ga[4096], gb[4096];
    for (int a = 0; a < GL; a++) for (int b = 0; b < GM; b++)
        if (full || (c3 && (a + b) % 3 == 0) || (ex && a % 2 == 0)) { ga[ng] = a; gb[ng] = b; ng++; }
    int *img = malloc(4 * (size_t)NM * ng);
    /* Augmented rows for the affine observable check: input bits [obs (NO) | syndrome (ND)], output bits obs of
       the image. The relation obs(g m) = M obs(m) + K syn(m) must be CONSISTENT over all mechanisms. With every
       g and its inverse in the group, this makes "flips some logical" invariant for UNDETECTABLE sets:
       obs(X) = M' M obs(X), so obs(X) != 0 implies obs(gX) != 0. */
    int NIN = NO + ND, NTOT = NIN + NO, RW = (NTOT + 63) / 64;
    uint64_t *aug = malloc(8 * (size_t)NM * RW);
    for (int g = 0; g < ng; g++) {
        for (int m = 0; m < NM; m++) {
            uint64_t t[MAXW] = {0};
            for (int i = 0; i < DEG[m]; i++) { int d = DET[m * MAXDEG + i]; int al = dsite[d] / GM, be = dsite[d] % GM;
                int site2 = ((al + ga[g]) % GL) * GM + (be + gb[g]) % GM; int key = (site2 * 2 + dtyp[d]) * (maxr + 1) + drnd[d];
                int d2 = dk[key]; if (d2 < 0) { printf("REJECTED: translation maps a detector outside the model\n"); return 1; }
                t[d2 / 64] ^= 1ULL << (d2 % 64); }
            int i = lower(t); if (i >= NM || cmp_syn(S + (size_t)ord[i] * W, t) != 0) { printf("REJECTED: DEM not invariant under translation (%d,%d)\n", ga[g], gb[g]); return 1; }
            int m2 = ord[i]; img[(size_t)m * ng + g] = m2;
            uint64_t *row = aug + (size_t)m * RW; memset(row, 0, 8 * RW);
            for (int k = 0; k < NO; k++) if (OB[m] >> k & 1) row[k / 64] |= 1ULL << (k % 64);
            for (int d = 0; d < ND; d++) if (S[(size_t)m * W + d / 64] >> (d % 64) & 1) { int b = NO + d; row[b / 64] |= 1ULL << (b % 64); }
            for (int k = 0; k < NO; k++) if (OB[m2] >> k & 1) { int b = NIN + k; row[b / 64] |= 1ULL << (b % 64); }
        }
        /* Gaussian elimination on the input columns; a row whose input part vanishes must have zero output */
        int r = 0;
        for (int col = 0; col < NIN && r < NM; col++) {
            int p = -1; for (int i = r; i < NM; i++) if (aug[(size_t)i * RW + col / 64] >> (col % 64) & 1) { p = i; break; }
            if (p < 0) continue;
            if (p != r) for (int j = 0; j < RW; j++) { uint64_t t = aug[(size_t)p * RW + j]; aug[(size_t)p * RW + j] = aug[(size_t)r * RW + j]; aug[(size_t)r * RW + j] = t; }
            for (int i = 0; i < NM; i++) if (i != r && (aug[(size_t)i * RW + col / 64] >> (col % 64) & 1))
                for (int j = 0; j < RW; j++) aug[(size_t)i * RW + j] ^= aug[(size_t)r * RW + j];
            r++;
        }
        for (int i = r; i < NM; i++) for (int b = NIN; b < NTOT; b++) if (aug[(size_t)i * RW + b / 64] >> (b % 64) & 1) {
            printf("REJECTED: observables are not an affine function of (observables, syndrome) under translation (%d,%d)\n", ga[g], gb[g]); return 1; }
    }
    int *orb2 = malloc(4 * NM); for (int m = 0; m < NM; m++) orb2[m] = -1; int no2 = 0;
    for (int m = 0; m < NM; m++) { if (orb2[m] >= 0) continue; int sz = 0;
        for (int g = 0; g < ng; g++) { int m2 = img[(size_t)m * ng + g]; if (orb2[m2] == -1) { orb2[m2] = no2; sz++; } else if (orb2[m2] != no2) { printf("REJECTED: orbit structure\n"); return 1; } }
        if (sz != ng) { printf("REJECTED: the translation action is not free (orbit %d has %d members)\n", no2, sz); return 1; }
        if (ANC[no2] != m) { printf("REJECTED: anchor of orbit %d is not its first member\n", no2); return 1; }
        no2++; }
    if (no2 != NORB) { printf("REJECTED: orbit count %d != %d\n", no2, NORB); return 1; }
    for (int m = 0; m < NM; m++) if (orb2[m] != ORB[m]) { printf("REJECTED: orbit ids differ\n"); return 1; }
    /* ---- (2) per-detector touching lists in (orbit, id) order; co-touch ---- */
    astart = calloc(ND + 1, 4); for (int m = 0; m < NM; m++) for (int i = 0; i < DEG[m]; i++) astart[DET[m * MAXDEG + i] + 1]++;
    for (int d = 0; d < ND; d++) astart[d + 1] += astart[d];
    alist = malloc(4 * (size_t)astart[ND]); int *fp = calloc(ND, 4);
    for (int o = 0; o < NORB; o++) for (int m = 0; m < NM; m++) if (ORB[m] == o) for (int i = 0; i < DEG[m]; i++) { int d = DET[m * MAXDEG + i]; alist[astart[d] + fp[d]++] = m; }
    /* Checker reconstructs the orbit-restricted co-touch relation from raw DET/ORB,
       not from any prover-produced bound or membership claims. */
    CT_ORBIT = calloc((size_t)NORB * ND * W, sizeof(uint64_t));
    if (!CT_ORBIT) return 2;
    for (int m = 0; m < NM; m++) {
        uint64_t *ct = CT_ORBIT + (size_t)ORB[m] * ND * W;
        for (int i = 0; i < DEG[m]; i++) for (int j = 0; j < DEG[m]; j++) {
            int a = DET[m * MAXDEG + i], b = DET[m * MAXDEG + j];
            ct[(size_t)a * W + b / 64] |= 1ULL << (b % 64);
        }
    }
    for (int o = NORB - 2; o >= 0; o--) {
        uint64_t *ct = CT_ORBIT + (size_t)o * ND * W;
        const uint64_t *next = ct + (size_t)ND * W;
        for (size_t j = 0; j < (size_t)ND * W; j++) ct[j] |= next[j];
    }
    for (int o = 0; o < NORB; o++) for (int d = 0; d < ND; d++)
        CT_ORBIT[((size_t)o * ND + d) * W + d / 64] |= 1ULL << (d % 64);
    /* ---- (3)+(4) walk the proof ---- */
    FILE *g = fopen(argv[2], "rb"); if (!g) return 2; char magic[4]; int32_t hdr[4];
    if (fread(magic, 1, 4, g) != 4) { printf("REJECTED: header\n"); return 1; }
    modern_trace = memcmp(magic, "CDP5", 4) == 0 ? 2 : (memcmp(magic, "CDP4", 4) == 0 ? 1 : 0);
    if ((!modern_trace && memcmp(magic, "CDP3", 4)) || fread(hdr, 4, 4, g) != 4 || hdr[1] != NORB) {
        printf("REJECTED: header\n"); return 1;
    }
    int w = hdr[0], maxround = hdr[2], nanch = hdr[3];
    /* the anchor scope: every orbit (maxround < 0), or -- time-shift anchoring, rule S -- the prefix of orbits whose
       first detector round is <= maxround, which requires orbit rounds to be non-decreasing in orbit id. Both are
       recomputed here from this file's detector coordinates; the proof only states maxround. */
    int want = NORB;
    if (maxround >= 0) {
        int prev = -1; want = 0;
        for (int o = 0; o < NORB; o++) { int m = ANC[o], r = 1 << 30;
            for (int i = 0; i < DEG[m]; i++) if (drnd[DET[m * MAXDEG + i]] < r) r = drnd[DET[m * MAXDEG + i]];
            if (r < prev) { printf("REJECTED: orbits are not time-ordered, so time-shift anchoring does not apply\n"); return 1; }
            prev = r; if (r <= maxround) want = o + 1; }
    }
    if (nanch != want) { printf("REJECTED: the proof covers %d anchors, the scope requires %d\n", nanch, want); return 1; }
    /* per anchor: root code, then streams (see cdist_prove.c); tasks are (anchor, child) */
    int cap = 1024, ntask = 0; int *t_o = malloc(4 * cap), *t_m = malloc(4 * cap); uint8_t **t_tr = malloc(sizeof(uint8_t *) * cap); int64_t *t_n = malloc(8 * cap);
    for (int o = 0; o < nanch; o++) {
        int8_t code; if (fread(&code, 1, 1, g) != 1) { printf("REJECTED: truncated\n"); return 1; }
        const uint64_t *s0 = S + (size_t)ANC[o] * W;
        int nch = 0, v = -1;
        if (code == 0) {        /* the root itself pruned by the bound at remaining w - 1 */
            if (w <= 3 || !bound_ok(s0, o, w - 1)) { printf("REJECTED: orbit %d, root bound does not hold\n", o); return 1; }
            continue;
        } else if (code == 1) {
            if (w <= 3) { printf("REJECTED: orbit %d, root code\n", o); return 1; }
            int32_t claimed; if (fread(&claimed, 4, 1, g) != 1) { printf("REJECTED: truncated\n"); return 1; }
            v = pick_c(s0, o); nch = v < 0 ? 0 : allowed_count(v, o);
            if (claimed != nch) { printf("REJECTED: orbit %d, %d child streams for %d children\n", o, claimed, nch); return 1; }
        } else if (code == 2) {
            if (w > 3) { printf("REJECTED: orbit %d, root code\n", o); return 1; }
            nch = 1;
        } else { printf("REJECTED: orbit %d, root code %d\n", o, code); return 1; }
        for (int k = 0; k < nch; k++) {
            if (ntask == cap) { cap *= 2; t_o = realloc(t_o, 4 * cap); t_m = realloc(t_m, 4 * cap); t_tr = realloc(t_tr, sizeof(uint8_t *) * cap); t_n = realloc(t_n, 8 * cap); }
            int64_t nn; if (fread(&nn, 8, 1, g) != 1 || nn < 0) { printf("REJECTED: truncated\n"); return 1; }
            int64_t nb = (nn + 7) >> 3; uint8_t *buf = malloc(nb + 1); if ((int64_t)fread(buf, 1, nb, g) != nb) { printf("REJECTED: truncated\n"); return 1; }
            t_o[ntask] = o; t_m[ntask] = code == 1 ? alist[astart[v + 1] - nch + k] : -1; t_tr[ntask] = buf; t_n[ntask] = nn; ntask++;
        }
    }
    build_index(); build_spans();
    char extra; if (fread(&extra, 1, 1, g) == 1) { printf("REJECTED: trailing bytes\n"); return 1; }
    fclose(g);
    volatile int bad = 0; int badwhy = 0, bado = -1; double t0 = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic, 1)
    for (int k = 0; k < ntask; k++) {
        if (bad) continue;
        int o = t_o[k], a = ANC[o], m = t_m[k];
        Walk wk = { t_tr[k], t_n[k], 0, 0 };
        if (m < 0) node(&wk, S + (size_t)a * W, zob(S + (size_t)a * W), popc(S + (size_t)a * W), OB[a], 1, w, o);
        else { uint64_t ns[MAXW]; for (int j = 0; j < W; j++) ns[j] = S[(size_t)a * W + j] ^ S[(size_t)m * W + j];
               node(&wk, ns, zob(ns), popc(ns), OB[a] ^ OB[m], 2, w, o); }
        if (!wk.bad && wk.p != wk.n) wk.bad = 6;
        if (wk.bad) {
            #pragma omp critical
            { bad = 1; badwhy = wk.bad; bado = o; } }
    }
    if (bad) { printf("REJECTED: orbit %d, reason %d (1 solution exists, 2 leaf completion exists, 3 trace short, 4 bound does not hold, 5 bad detector, 6 trace too long)\n", bado, badwhy); return 1; }
    /* Scoped proofs establish exactly this: the walk stops at an empty non-logical syndrome, so a solution with an
       undetectable proper subset may be missed when its irreducible logical part starts after maxround. A
       minimum-weight solution is always irreducible, which is all the time-shift lemma uses (Lean pass
       2026-10-21, finding F2; unscoped proofs imply "none at all", since every solution has an irreducible
       logical part of no larger weight). */
    if (maxround >= 0)
        printf("VERIFIED: no irreducible undetectable logical error of weight <= %d whose earliest round is <= %d in this DEM (irreducible: no proper subset is undetectable, as every minimum-weight one is; symmetry group of %d translations, %d orbits, %d anchors; %.1fs)\n", w, maxround, ng, NORB, nanch, omp_get_wtime() - t0);
    else
        printf("VERIFIED: no undetectable logical error of weight <= %d in this DEM (symmetry group of %d translations, %d orbits; %.1fs)\n", w, ng, NORB, omp_get_wtime() - t0);
    return 0;
}
