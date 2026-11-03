/* ccheck_oracle (R198 test harness -- NOT part of the trusted base): compiles ccheck.c itself, with its main
 * renamed, and answers queries with the checker's OWN decision procedures, so a test can hold each one against
 * brute force on small models (tests/test_r198_circuit_distance_exact.py) -- and can do the same for a mutated
 * copy of ccheck.c, to show the comparison would catch a wrong rule.
 *
 *   ccheck_oracle model.txt full|c3|ex MODE < queries      MODE 0 = CDP3 rules, 2 = CDP5 rules
 *
 * Each query line is "o rem l k d_1 .. d_k" (anchor orbit, remaining budget, observable mask, the k firing
 * detectors). The answer line is "bound leaf3 leaf2": bound_ok(s, o, rem); leaf_exists3(s, l, o) (is there a
 * completion of weight <= 3 flipping a logical?); leaf_exists(s, l, 2). Exit 2 when the model does not load.
 */
#define main ccheck_main
#include "ccheck.c"
#undef main

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ccheck_oracle model.txt group mode < queries\n"); return 2; }
    char *av[] = { argv[0], argv[1], "<no proof: the oracle reads queries>", argv[2] };
    /* the checker's own loader, symmetry and table construction; it stops at the missing proof file */
    if (ccheck_main(4, av) != 2 || !CT_ORBIT) { fprintf(stderr, "model did not load\n"); return 2; }
    modern_trace = atoi(argv[3]);
    build_index(); build_spans();
    int o, rem, k; unsigned l;
    while (scanf("%d %d %u %d", &o, &rem, &l, &k) == 4) {
        uint64_t s[MAXW] = {0};
        for (int i = 0; i < k; i++) { int d; if (scanf("%d", &d) != 1 || d < 0 || d >= ND) return 2; s[d / 64] ^= 1ULL << (d % 64); }
        if (o < 0 || o >= NORB || !popc(s)) return 2;
        printf("%d %d %d\n", bound_ok(s, o, rem), leaf_exists3(s, zob(s), popc(s), l, o), leaf_exists(s, zob(s), popc(s), l, 2));
    }
    return 0;
}
