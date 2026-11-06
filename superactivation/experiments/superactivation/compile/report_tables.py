"""Print the markdown tables used in RESULTS.md from the JSON outputs (no new computation).

Usage: python report_tables.py > report_tables.md
"""
import json
import math

for n in (17, 24):
    d = json.load(open(f"decoder_counts_n{n}.json"))
    s = d["summary"]
    print(f"\n#### Decoder per flag weight k, n = {n}  (decoder_counts.py {n})\n")
    print("| k | P(k) | m = n-k | support dim s | decoder CZ (min..max over b) | U_m^dag CZ | route | decoder-unitary qubits |")
    print("|---|---|---|---|---|---|---|---|")
    for r in s["per_k"]:
        k = r["k"]
        print(f"| {k} | {r['p_k']:.2e} | {n - k} | {','.join(map(str, r['support_dims']))} | "
              f"{r['cz_min_over_b']}..{r['cz_max_over_b']} | {r['U_dag_cz']} | {'/'.join(r['modes'])} | "
              f"{','.join(map(str, r['dec_unitary_qubits']))} |")
    ce = s["components_expected"]
    print(f"\nExpected decoder CZ {s['decoder_cz_expected']:.1f} = U_m^dag {ce['U_dag_cz']:.1f} + Givens "
          f"{ce['givens_cz']:.1f} + unary->binary {ce['unary_to_binary_cz']:.1f} + phase fix {ce['phase_fix_cz']:.1f}"
          f" + decoder unitary {ce['dec_unitary_cz']:.1f}; worst {s['decoder_cz_worst']} at (k,b) = "
          f"{tuple(s['decoder_cz_worst_branch'])}; expected reads {s['decoder_measurements_expected']:.2f};"
          f" expected feed-forward 1q gates {s['decoder_feedforward_1q_expected']:.1f}; max qubits {s['decoder_qubits_max']}.")
t = json.load(open("totals.json"))
for n, r in t["n"].items():
    print(f"\n#### Totals n = {n}  (totals.py)\n")
    print("| profile | CZ expected | CZ worst | eps = 0.0063 x CZ | factor = eps/delta | factor (worst) | reads expected |")
    print("|---|---|---|---|---|---|---|")
    for prof, x in r["profiles"].items():
        print(f"| {prof} | {x['cz_expected']:.1f} | {x['cz_worst']} | {x['eps_first_order']:.2f} | "
              f"{x['infeasibility_factor']:,.0f} | {x['infeasibility_factor_worst']:,.0f} | "
              f"{x['mid_circuit_measurements_expected']:.1f} |")
    for tag in ("sampled_shots", "sampled_shots_lnn"):
        if tag not in r:
            continue
        ss = r[tag]
        lab = "Sabre layout, all-to-all encoder" if tag == "sampled_shots" else "nearest-neighbour encoder on a heavy-hex path"
        print(f"\nSampled whole shots, {lab} ({ss['count']}): logical CZ {ss['logical_cz_mean']:.1f} +- {ss['logical_cz_sd']:.1f} (sd),"
              f" logical 2q-depth {ss['logical_2q_depth_mean']:.0f}; heavy-hex routed CZ {ss.get('routed_cz_mean', float('nan')):.1f}"
              f" +- {ss.get('routed_cz_sem', float('nan')):.1f} (sem), routed/logical {ss.get('routed_over_logical_mean', float('nan')):.2f},"
              f" routed depth {ss.get('routed_depth_mean', float('nan')):.0f} (2q {ss.get('routed_2q_depth_mean', float('nan')):.0f}),"
              f" routed factor {ss.get('routed_infeasibility_factor', float('nan')):,.0f}")
