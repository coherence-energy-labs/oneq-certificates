/* cdist_prove (R198): PROOF-PRODUCING exact circuit-distance search (the prover half of a certificate).
 *
 * Same search as cdist.c with three changes that make its NONE answers checkable:
 *   - no transposition table (every pruned node carries its own reason),
 *   - children in CANONICAL order: the allowed mechanisms touching the chosen detector, sorted by
 *     (orbit, mechanism id) -- a checker can regenerate the list without being told it,
 *   - a trace: for every node with remaining budget >= 3 and nonzero syndrome, in DFS preorder, ONE BIT:
 *     expanded on the deterministic pick, or pruned by the bound (legal at remaining budget >= 4; at the root,
 *     >= 3). Nodes with remaining budget <= 2 are leaves the checker re-decides itself.
 * Output file (CDP3): "CDP3" w norb maxround nanch; per anchor o < nanch a root code (int8: 0 root pruned,
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
static int *dstart, *dlist; static uint64_t *cot; static int *anchors;
static int *shash; static uint64_t smask; static int32_t *pa, *pb; static uint64_t pmask;

static uint64_t hk(const uint64_t *s) { uint64_t h = 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < W; i++) { h ^= s[i] + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2); h *= 0xff51afd7ed558ccdULL; } return h ^ (h >> 33); }
static int eqs(const uint64_t *a, const uint64_t *b) { for (int i = 0; i < W; i++) if (a[i] != b[i]) return 0; return 1; }
static int single_of(const uint64_t *s) { for (uint64_t i = hk(s) & smask;; i = (i + 1) & smask) { int m = shash[i]; if (m < 0) return -1;
    if (eqs(msyn + (size_t)m * W, s)) return m; } }
static int lo_allowed(int d, int o) { int lo = dstart[d], hi = dstart[d + 1];
    while (lo < hi) { int mid = (lo + hi) >> 1; if (morb[dlist[mid]] < o) lo = mid + 1; else hi = mid; } return lo; }

/* CDP2: one bit per node with remaining budget >= 3 and nonzero syndrome: 1 = expanded on the detector that
   the DETERMINISTIC pick rule names (fewest allowed mechanisms, ties to the lowest index; the checker
   recomputes it), 0 = pruned by the bound (only at remaining >= 4). */
typedef struct { uint8_t *buf; size_t n, cap; int found, flen, path[64], fpath[64]; } Tr;
static void emit(Tr *t, int bit) {
    if ((t->n >> 3) >= t->cap) {
        size_t old = t->cap, nc = old ? 2 * old : 4096;
        t->buf = realloc(t->buf, nc); memset(t->buf + old, 0, nc - old); t->cap = nc;
    }
    if (bit) t->buf[t->n >> 3] |= (uint8_t)(1u << (t->n & 7));
    t->n++;
}

static int complete2(const uint64_t *s, uint32_t l, int maxk) {
    int m = single_of(s); if (m >= 0 && (l ^ mobs[m])) return 1;
    if (maxk < 2) return 0;
    for (uint64_t i = hk(s) & pmask;; i = (i + 1) & pmask) { int a = pa[i]; if (a < 0) break; int b = pb[i];
        const uint64_t *sa = msyn + (size_t)a * W, *sb = msyn + (size_t)b * W; int ok = 1;
        for (int j = 0; j < W; j++) if ((sa[j] ^ sb[j]) != s[j]) { ok = 0; break; }
        if (ok && (l ^ mobs[a] ^ mobs[b])) return 1; }
    int d = -1; for (int i = 0; i < W && d < 0; i++) if (s[i]) d = i * 64 + __builtin_ctzll(s[i]);
    uint64_t ns[MAXW];
    for (int t = dstart[d]; t < dstart[d + 1]; t++) { int a = dlist[t]; const uint64_t *sa = msyn + (size_t)a * W; int sub = 1, eq = 1;
        for (int j = 0; j < W; j++) { if (sa[j] & ~s[j]) { sub = 0; break; } if (sa[j] != s[j]) eq = 0; }
        if (!sub || eq) continue;
        for (int j = 0; j < W; j++) ns[j] = s[j] ^ sa[j];
        int b = single_of(ns); if (b >= 0 && (l ^ mobs[a] ^ mobs[b])) return 1; }
    return 0;
}
static int bound(const uint64_t *s, int o) {
    uint64_t r[MAXW]; memcpy(r, s, 8 * W); int indep = 0;
    for (int i = 0; i < W; i++) while (r[i]) { int d = i * 64 + __builtin_ctzll(r[i]); indep++;
        const uint64_t *c = cot + (size_t)d * W; for (int j = 0; j < W; j++) r[j] &= ~c[j]; }
    long num = 0;   /* exact: sum over firing d of 2520 / c_d */
    for (int i = 0; i < W; i++) { uint64_t x = s[i]; while (x) { int d = i * 64 + __builtin_ctzll(x); x &= x - 1; int cd = 0;
        for (int t = lo_allowed(d, o); t < dstart[d + 1]; t++) { const uint64_t *ms = msyn + (size_t)dlist[t] * W; int c = 0;
            for (int j = 0; j < W; j++) c += __builtin_popcountll(ms[j] & s[j]); if (c > cd) cd = c; }
        if (cd == 0) return 1000; num += 2520 / cd; } }
    int hf = (int)((num + 2519) / 2520);
    return indep > hf ? indep : hf;
}
static int pick(const uint64_t *s, int o) { int bd = -1, bn = 1 << 30;
    for (int i = 0; i < W; i++) { uint64_t x = s[i]; while (x) { int d = i * 64 + __builtin_ctzll(x); x &= x - 1;
        int n = dstart[d + 1] - lo_allowed(d, o); if (n < bn) { bn = n; bd = d; } } } return bd; }

static int dfs(Tr *t, const uint64_t *s, uint32_t l, int depth, int budget, int o) {
    int z = 1; for (int i = 0; i < W; i++) if (s[i]) { z = 0; break; }
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
    if (rem <= 2) return complete2(s, l, rem);
#endif
    if (rem >= 4 && bound(s, o) > rem) { emit(t, 0); return 0; }
    int d = pick(s, o); emit(t, 1);
    uint64_t ns[MAXW];
    for (int k = lo_allowed(d, o); k < dstart[d + 1]; k++) { int m = dlist[k]; const uint64_t *ms = msyn + (size_t)m * W;
        for (int j = 0; j < W; j++) ns[j] = s[j] ^ ms[j];
        t->path[depth] = m;
        if (dfs(t, ns, l ^ mobs[m], depth + 1, budget, o)) return 1; }
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
    cot = calloc((size_t)ND * W, 8);
    for (int m = 0; m < NM; m++) for (int i = 0; i < mdeg[m]; i++) for (int j = 0; j < mdeg[m]; j++) {
        int a = mdet[m * MAXDEG + i], b = mdet[m * MAXDEG + j]; cot[(size_t)a * W + b / 64] |= 1ULL << (b % 64); }
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
    /* CDP3: per anchor a root code -- 0 the root is pruned by the bound (remaining w-1), 1 the root is expanded
       on the deterministic pick and each allowed child is a separate task with its own bitstream (children in
       (orbit, id) order), 2 (w <= 3) one bitstream for the whole anchor. Children run in parallel: the anchor-0
       subtree no longer serialises the run. */
    double t0 = omp_get_wtime();
    int8_t *rootcode = calloc(NORB, 1); int *nchild = calloc(NORB, 4), *first = calloc(NORB + 1, 4);
    int ntask = 0;
    for (int o = 0; o < nanch; o++) {
        const uint64_t *s0 = msyn + (size_t)anchors[o] * W; first[o] = ntask;
        if (w <= 3) { rootcode[o] = 2; ntask++; continue; }
        if (bound(s0, o) > w - 1) { rootcode[o] = 0; continue; }
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
        if (m < 0) hit = dfs(&tr[k], msyn + (size_t)a * W, mobs[a], 1, w, o);
        else { uint64_t ns[MAXW]; for (int j = 0; j < W; j++) ns[j] = msyn[(size_t)a * W + j] ^ msyn[(size_t)m * W + j];
               tr[k].path[1] = m; hit = dfs(&tr[k], ns, mobs[a] ^ mobs[m], 2, w, o); }
        if (hit) {
            #pragma omp critical
            { if (!found) { found = 1; fo = o; } } }
    }
    if (found) { printf("FOUND at weight <= %d (orbit %d); no proof written\n", w, fo); return 1; }
    FILE *g = fopen(argv[3], "wb"); fwrite("CDP3", 1, 4, g); int32_t hdr[4] = {w, NORB, maxround, nanch}; fwrite(hdr, 4, 4, g);
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
