/* cdist (R198): exact circuit-distance search. Question at budget w: is there a set X of <= w mechanisms with
 * syndrome(X) = 0 and logical(X) != 0? Prints NONE (=> d >= w + 1) or FOUND with a witness, per weight.
 *
 * Soundness rules A (anchoring, affine observable map), I (ordered anchoring by minimal orbit), B (toggling
 * state, transposition table), C (canonical branching on a firing detector), D/G (independent-set and
 * fractional-cover bounds), E (trivial prune) are stated with their arguments in RESULTS.md. Also:
 *  (L) RADIUS-2 PERIMETER, exact and lean. "Is there Y, |Y| <= 2, syn(Y) = s, l ^ obs(Y) != 0?" over ALL
 *      mechanisms (using more mechanisms than the anchor allows can only make the search find more: a
 *      NONE stays sound, and any FOUND is re-verified from the raw DEM by the caller).
 *        single:         exact hash lookup (all syndromes are distinct; checked at load).
 *        overlapping:    table of every pair sharing >= 1 detector, keyed by syn(a)^syn(b), entries
 *                        re-verified by recomputation on every hit (hash collisions cannot lie).
 *        disjoint pair:  the lowest firing detector d lies in exactly one member, whose syndrome is then a
 *                        SUBSET of s; enumerate those, the other is s minus it (single lookup).
 *  (M) At rem == 3 the search branches once (over allowed mechanisms touching the cheapest firing
 *      detector, complete by (C)/(I)) and asks the perimeter. Bound and TT run only at rem >= 4.
 * Per-depth counters report where the work goes.
 * Usage: cdist model.txt WMIN WMAX ttbits      (model from export_model.py)
 */
#ifdef _OPENMP
#include <omp.h>
#else
#include <time.h>
static int omp_get_max_threads(void) { return 1; }
static int omp_get_thread_num(void) { return 0; }
static double omp_get_wtime(void) { return (double)clock() / CLOCKS_PER_SEC; }
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXW 16
#define MAXDEG 32
#define MAXD 40

static int ND, NM, NO, NORB, GL, GM, W;
static uint64_t *msyn; static uint32_t *mobs; static int *mdeg, *mdet, *morb;
static int *dstart, *dlist; static uint64_t *cot; static int *anchors;
static int *shash; static uint64_t smask;
static int32_t *pa, *pb; static uint64_t pmask; static long npairs;   /* overlapping-pair table */

static uint64_t hk(const uint64_t *s) {
    uint64_t h = 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < W; i++) { h ^= s[i] + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2); h *= 0xff51afd7ed558ccdULL; }
    return h ^ (h >> 33);
}
static int eqs(const uint64_t *a, const uint64_t *b) { for (int i = 0; i < W; i++) if (a[i] != b[i]) return 0; return 1; }
static int single_of(const uint64_t *s) {
    for (uint64_t i = hk(s) & smask;; i = (i + 1) & smask) { int m = shash[i]; if (m < 0) return -1;
        if (eqs(msyn + (size_t)m * W, s)) return m; }
}
static int lo_allowed(int d, int o) {
    int lo = dstart[d], hi = dstart[d + 1];
    while (lo < hi) { int mid = (lo + hi) >> 1; if (morb[dlist[mid]] < o) lo = mid + 1; else hi = mid; }
    return lo;
}

typedef struct { uint64_t *key; uint32_t *l; int8_t *rem; int16_t *o; uint64_t mask;
    long long nodes, depth_n[MAXD], bprune[MAXD], tprune[MAXD]; int path[64]; int flen; int fpath[64]; } Ctx;

/* exact: is there Y with |Y| <= 2, syn(Y) = s, l ^ obs(Y) != 0 ?  On success writes Y into path[depth..] */
static int complete2(Ctx *c, const uint64_t *s, uint32_t l, int depth, int maxk) {
    int m = single_of(s);
    if (m >= 0 && (l ^ mobs[m])) { c->path[depth] = m; c->flen = depth + 1; return 1; }
    if (maxk < 2) return 0;   /* only a single fits the remaining budget */
    uint64_t h = hk(s);
    for (uint64_t i = h & pmask;; i = (i + 1) & pmask) {
        int a = pa[i]; if (a < 0) break;
        int b = pb[i]; const uint64_t *sa = msyn + (size_t)a * W, *sb = msyn + (size_t)b * W; int ok = 1;
        for (int j = 0; j < W; j++) if ((sa[j] ^ sb[j]) != s[j]) { ok = 0; break; }
        if (ok && (l ^ mobs[a] ^ mobs[b])) { c->path[depth] = a; c->path[depth + 1] = b; c->flen = depth + 2; return 1; }
    }
    int d = -1; for (int i = 0; i < W && d < 0; i++) if (s[i]) d = i * 64 + __builtin_ctzll(s[i]);
    uint64_t ns[MAXW];
    for (int t = dstart[d]; t < dstart[d + 1]; t++) {
        int a = dlist[t]; const uint64_t *sa = msyn + (size_t)a * W; int sub = 1, eq = 1;
        for (int j = 0; j < W; j++) { if (sa[j] & ~s[j]) { sub = 0; break; } if (sa[j] != s[j]) eq = 0; }
        if (!sub || eq) continue;
        for (int j = 0; j < W; j++) ns[j] = s[j] ^ sa[j];
        int b = single_of(ns);
        if (b >= 0 && (l ^ mobs[a] ^ mobs[b])) { c->path[depth] = a; c->path[depth + 1] = b; c->flen = depth + 2; return 1; }
    }
    return 0;
}

static int bound(const uint64_t *s, int o) {
    uint64_t r[MAXW]; memcpy(r, s, 8 * W);
    int indep = 0;
    for (int i = 0; i < W; i++) while (r[i]) { int d = i * 64 + __builtin_ctzll(r[i]); indep++;
        const uint64_t *cc = cot + (size_t)d * W; for (int j = 0; j < W; j++) r[j] &= ~cc[j]; }
    double frac = 0.0;
    for (int i = 0; i < W; i++) { uint64_t x = s[i]; while (x) { int d = i * 64 + __builtin_ctzll(x); x &= x - 1;
        int cd = 0;
        for (int t = lo_allowed(d, o); t < dstart[d + 1]; t++) {
            const uint64_t *ms = msyn + (size_t)dlist[t] * W; int cnt = 0;
            for (int j = 0; j < W; j++) cnt += __builtin_popcountll(ms[j] & s[j]);
            if (cnt > cd) cd = cnt;
        }
        if (cd == 0) return 1000;
        frac += 1.0 / cd; } }
    int hf = (int)(frac - 1e-9); if ((double)hf < frac - 1e-9) hf++;
    return indep > hf ? indep : hf;
}

static int pick(const uint64_t *s, int o) {
    int bd = -1, bn = 1 << 30;
    for (int i = 0; i < W; i++) { uint64_t x = s[i]; while (x) { int d = i * 64 + __builtin_ctzll(x); x &= x - 1;
        int n = dstart[d + 1] - lo_allowed(d, o); if (n < bn) { bn = n; bd = d; } } }
    return bd;
}

static int dfs(Ctx *c, const uint64_t *s, uint32_t l, int depth, int budget, int o) {
    c->nodes++; c->depth_n[depth]++;
    int z = 1; for (int i = 0; i < W; i++) if (s[i]) { z = 0; break; }
    if (z) { if (l) { c->flen = depth; memcpy(c->fpath, c->path, 4 * depth); return 1; } return 0; }
    int rem = budget - depth;
    if (rem <= 0) return 0;
    if (rem <= 2) { if (complete2(c, s, l, depth, rem)) { memcpy(c->fpath, c->path, 4 * c->flen); return 1; } return 0; }
    uint64_t ns[MAXW];
    if (rem == 3) {
        int d = pick(s, o);
        if (d < 0) return 0;
        for (int t = lo_allowed(d, o); t < dstart[d + 1]; t++) { int m = dlist[t]; const uint64_t *ms = msyn + (size_t)m * W;
            for (int j = 0; j < W; j++) ns[j] = s[j] ^ ms[j];
            c->path[depth] = m;
            int z2 = 1; for (int j = 0; j < W; j++) if (ns[j]) { z2 = 0; break; }
            if (z2) { if (l ^ mobs[m]) { c->flen = depth + 1; memcpy(c->fpath, c->path, 4 * c->flen); return 1; } continue; }
            if (complete2(c, ns, l ^ mobs[m], depth + 1, 2)) { memcpy(c->fpath, c->path, 4 * c->flen); return 1; } }
        return 0;
    }
    if (bound(s, o) > rem) { c->bprune[depth]++; return 0; }
    uint64_t h = hk(s) ^ ((uint64_t)l * 0x9E3779B97F4A7C15ULL);
    uint64_t slot = h & c->mask; uint64_t *k = c->key + slot * W;
    if (c->rem[slot] >= rem && c->o[slot] <= o && c->l[slot] == l && eqs(k, s)) { c->tprune[depth]++; return 0; }
    int d = pick(s, o);
    int a0 = lo_allowed(d, o), n = dstart[d + 1] - a0;
    int order[512], score[512];
    for (int t = 0; t < n; t++) { int m = dlist[a0 + t]; const uint64_t *ms = msyn + (size_t)m * W; int sc = 0;
        for (int j = 0; j < W; j++) sc += __builtin_popcountll(ms[j] & s[j]) * 2 - __builtin_popcountll(ms[j] & ~s[j]);
        order[t] = m; score[t] = sc; }
    for (int i = 1; i < n; i++) { int m = order[i], sc = score[i], j = i - 1;
        while (j >= 0 && score[j] < sc) { order[j + 1] = order[j]; score[j + 1] = score[j]; j--; } order[j + 1] = m; score[j + 1] = sc; }
    for (int t = 0; t < n; t++) { int m = order[t]; const uint64_t *ms = msyn + (size_t)m * W;
        for (int j = 0; j < W; j++) ns[j] = s[j] ^ ms[j];
        c->path[depth] = m;
        if (dfs(c, ns, l ^ mobs[m], depth + 1, budget, o)) return 1; }
    if (c->rem[slot] < rem || (c->rem[slot] == rem && c->o[slot] > o)) {
        memcpy(k, s, 8 * W); c->l[slot] = l; c->rem[slot] = (int8_t)rem; c->o[slot] = (int16_t)o; }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: cdist model.txt WMIN WMAX ttbits [MAXROUND]\n"); return 2; }
    FILE *f = fopen(argv[1], "r"); if (!f) { perror("open"); return 2; }
    int wmin = atoi(argv[2]), wmax = atoi(argv[3]), ttbits = atoi(argv[4]);
    /* TIME-SHIFT ANCHORING (MAXROUND >= 0): search only from orbits whose first detector round is <= MAXROUND.
       The claim printed is then "no solution whose earliest round is <= MAXROUND" -- precisely, no IRREDUCIBLE
       one (no undetectable proper subset), which every minimum-weight solution is. With time-ordered orbits
       (refused otherwise) and the window preconditions P2/P3 chained over consecutive models, a solution whose
       earliest round is r0 >= 2 is a time-translate of a solution starting in round 1 of the (T - r0 + 1)-round
       model, so MAXROUND = 1 for every T = 1..w+2 covers every T (RESULTS.md, rule S). */
    int maxround = argc > 5 ? atoi(argv[5]) : -1;
    if (fscanf(f, "%d %d %d %d %d %d", &ND, &NM, &NO, &NORB, &GL, &GM) != 6) return 2;
    W = (ND + 63) / 64; if (W > MAXW || NO > 32 || wmax >= MAXD) return 2;
    msyn = calloc((size_t)NM * W, 8); mobs = calloc(NM, 4); mdeg = calloc(NM, 4); mdet = calloc((size_t)NM * MAXDEG, 4); morb = calloc(NM, 4);
    int *cnt = calloc(ND + 1, 4);
    for (int m = 0; m < NM; m++) {
        int k; unsigned ob;
        if (fscanf(f, "%d", &k) != 1 || k > MAXDEG || k < 1) return 2;
        mdeg[m] = k;
        for (int i = 0; i < k; i++) { int d; if (fscanf(f, "%d", &d) != 1 || d < 0 || d >= ND) return 2;
            if (msyn[(size_t)m * W + d / 64] >> (d % 64) & 1) { fprintf(stderr, "malformed: mechanism %d repeats detector %d\n", m, d); return 2; }
            mdet[m * MAXDEG + i] = d; msyn[(size_t)m * W + d / 64] ^= 1ULL << (d % 64); cnt[d]++; }
        if (fscanf(f, "%u %d", &ob, &morb[m]) != 2) return 2;
        mobs[m] = ob;
    }
    anchors = calloc(NORB, 4); for (int i = 0; i < NORB; i++) if (fscanf(f, "%d", &anchors[i]) != 1) return 2;
    int *drnd = calloc(ND, 4);
    for (int d = 0; d < ND; d++) { int s_, t_; if (fscanf(f, "%d %d %d", &s_, &t_, &drnd[d]) != 3) return 2; }
    fclose(f);
    for (int o = 0; o < NORB; o++) if (morb[anchors[o]] != o) { fprintf(stderr, "anchor/orbit mismatch\n"); return 2; }
    int nanch = NORB;
    if (maxround >= 0) {
        int prev = -1; nanch = 0;
        for (int o = 0; o < NORB; o++) {
            int m = anchors[o], r = 1 << 30;
            for (int i = 0; i < mdeg[m]; i++) if (drnd[mdet[m * MAXDEG + i]] < r) r = drnd[mdet[m * MAXDEG + i]];
            if (r < prev) { fprintf(stderr, "orbits are not time-ordered: time-shift anchoring refused\n"); return 2; }
            prev = r;
            if (r <= maxround) nanch = o + 1;
        }
    }
    dstart = calloc(ND + 1, 4); for (int d = 0; d < ND; d++) { if (cnt[d] > 512) return 2; dstart[d + 1] = dstart[d] + cnt[d]; }
    dlist = calloc(dstart[ND] + 1, 4); int *fill = calloc(ND, 4);
    for (int m = 0; m < NM; m++) for (int i = 0; i < mdeg[m]; i++) { int d = mdet[m * MAXDEG + i]; dlist[dstart[d] + fill[d]++] = m; }
    for (int d = 0; d < ND; d++)
        for (int i = dstart[d] + 1; i < dstart[d + 1]; i++) { int m = dlist[i], j = i - 1;
            while (j >= dstart[d] && morb[dlist[j]] > morb[m]) { dlist[j + 1] = dlist[j]; j--; } dlist[j + 1] = m; }
    cot = calloc((size_t)ND * W, 8);
    for (int m = 0; m < NM; m++) for (int i = 0; i < mdeg[m]; i++) for (int j = 0; j < mdeg[m]; j++) {
        int a = mdet[m * MAXDEG + i], b = mdet[m * MAXDEG + j]; cot[(size_t)a * W + b / 64] |= 1ULL << (b % 64); }
    int sb = 1; while ((1 << sb) < 4 * NM) sb++; smask = (1ULL << sb) - 1; shash = malloc(sizeof(int) << sb);
    for (uint64_t i = 0; i <= smask; i++) shash[i] = -1;
    for (int m = 0; m < NM; m++) { if (single_of(msyn + (size_t)m * W) >= 0) { fprintf(stderr, "duplicate syndrome\n"); return 3; }
        for (uint64_t i = hk(msyn + (size_t)m * W) & smask;; i = (i + 1) & smask) if (shash[i] < 0) { shash[i] = m; break; } }
    /* overlapping-pair table: every unordered {a,b} sharing a detector, each pair inserted once (a < b) */
    long cap = 0; char *mark = calloc(NM, 1); int *nb = malloc(4 * NM);
    for (int a = 0; a < NM; a++) { int k = 0;
        for (int i = 0; i < mdeg[a]; i++) { int d = mdet[a * MAXDEG + i];
            for (int t = dstart[d]; t < dstart[d + 1]; t++) { int b = dlist[t]; if (b > a && !mark[b]) { mark[b] = 1; nb[k++] = b; } } }
        for (int i = 0; i < k; i++) mark[nb[i]] = 0;
        cap += k; }
    int pbits = 1; while ((1L << pbits) < 2 * cap) pbits++; pmask = (1ULL << pbits) - 1;
    pa = malloc(4 * (pmask + 1)); pb = malloc(4 * (pmask + 1)); for (uint64_t i = 0; i <= pmask; i++) pa[i] = -1;
    uint64_t tmp[MAXW];
    for (int a = 0; a < NM; a++) { int k = 0;
        for (int i = 0; i < mdeg[a]; i++) { int d = mdet[a * MAXDEG + i];
            for (int t = dstart[d]; t < dstart[d + 1]; t++) { int b = dlist[t]; if (b > a && !mark[b]) { mark[b] = 1; nb[k++] = b; } } }
        for (int i = 0; i < k; i++) { int b = nb[i]; mark[b] = 0;
            for (int j = 0; j < W; j++) tmp[j] = msyn[(size_t)a * W + j] ^ msyn[(size_t)b * W + j];
            for (uint64_t q = hk(tmp) & pmask;; q = (q + 1) & pmask) if (pa[q] < 0) { pa[q] = a; pb[q] = b; npairs++; break; } } }
    int T = omp_get_max_threads(); Ctx *ctx = calloc(T, sizeof(Ctx));
    for (int t = 0; t < T; t++) { uint64_t sz = 1ULL << ttbits; ctx[t].key = calloc(sz * W, 8); ctx[t].l = calloc(sz, 4);
        ctx[t].rem = malloc(sz); memset(ctx[t].rem, -1, sz); ctx[t].o = calloc(sz, 2); ctx[t].mask = sz - 1; }
    printf("ND %d NM %d orbits %d threads %d tt 2^%d/thread; overlapping pairs %ld (table 2^%d, %.0f MB)\n",
           ND, NM, NORB, T, ttbits, npairs, pbits, 8.0 * (pmask + 1) / 1e6);
    if (maxround >= 0) printf("time-shift anchoring: anchors in rounds <= %d (%d of %d orbits)\n", maxround, nanch, NORB);
    fflush(stdout);
    /* tasks = (anchor, first child): the root of each anchor is expanded here, its children run in parallel.
       The anchor-0 subtree (every orbit allowed) otherwise dominates the wall time on one thread. */
    int *task_o = malloc(sizeof(int) * (size_t)NORB * 520), *task_m = malloc(sizeof(int) * (size_t)NORB * 520);
    for (int w = wmin; w <= wmax; w++) {
        double t0 = omp_get_wtime(); volatile int found = 0; int fa = -1;
        long long n0 = 0; for (int t = 0; t < T; t++) n0 += ctx[t].nodes;
        long long dn0[MAXD] = {0}, bp0[MAXD] = {0}, tp0[MAXD] = {0};
        for (int t = 0; t < T; t++) for (int q = 0; q < MAXD; q++) { dn0[q] += ctx[t].depth_n[q]; bp0[q] += ctx[t].bprune[q]; tp0[q] += ctx[t].tprune[q]; }
        int ntask = 0;
        for (int o = 0; o < nanch; o++) {
            int a = anchors[o]; const uint64_t *s0 = msyn + (size_t)a * W;
            if (w <= 3) { task_o[ntask] = o; task_m[ntask++] = -1; continue; }
            if (bound(s0, o) > w - 1) continue;                 /* the root itself is pruned (rule D/G) */
            int d = pick(s0, o);
            for (int t = lo_allowed(d, o); t < dstart[d + 1]; t++) { task_o[ntask] = o; task_m[ntask++] = dlist[t]; }
        }
        #pragma omp parallel for schedule(dynamic, 1)
        for (int k = 0; k < ntask; k++) {
            if (found) continue;
            Ctx *c = &ctx[omp_get_thread_num()]; int o = task_o[k], a = anchors[o], m = task_m[k];
            c->path[0] = a; int hit;
            if (m < 0) hit = dfs(c, msyn + (size_t)a * W, mobs[a], 1, w, o);
            else {
                uint64_t ns[MAXW]; for (int j = 0; j < W; j++) ns[j] = msyn[(size_t)a * W + j] ^ msyn[(size_t)m * W + j];
                c->path[1] = m; hit = dfs(c, ns, mobs[a] ^ mobs[m], 2, w, o);
            }
            if (hit) {
                #pragma omp critical
                { if (!found) { found = 1; fa = omp_get_thread_num(); } }
            }
        }
        long long n1 = 0; for (int t = 0; t < T; t++) n1 += ctx[t].nodes;
        double dt = omp_get_wtime() - t0;
        if (found) {
            Ctx *c = &ctx[fa]; printf("w<=%d: FOUND weight %d witness:", w, c->flen);
            for (int i = 0; i < c->flen; i++) printf(" %d", c->fpath[i]);
            printf("  (%.1fs, %lld nodes)\n", dt, n1 - n0); return 0;
        }
        if (maxround >= 0)
            printf("w<=%d: NONE with earliest round <= %d   (%.1fs wall, %lld nodes, %d tasks)\n", w, maxround, dt, n1 - n0, ntask);
        else
            printf("w<=%d: NONE  => d_circ >= %d   (%.1fs wall, %lld nodes, %d tasks)\n", w, w + 1, dt, n1 - n0, ntask);
        printf("   depth: nodes / bound-pruned / tt-pruned\n");
        for (int q = 1; q <= w; q++) { long long dn = -dn0[q], bp = -bp0[q], tp = -tp0[q];
            for (int t = 0; t < T; t++) { dn += ctx[t].depth_n[q]; bp += ctx[t].bprune[q]; tp += ctx[t].tprune[q]; }
            if (dn) printf("   %2d: %lld / %lld / %lld\n", q, dn, bp, tp); }
        fflush(stdout);
    }
    return 1;
}
