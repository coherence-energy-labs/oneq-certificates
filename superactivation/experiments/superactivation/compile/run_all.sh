#!/usr/bin/env bash
# Reproduce every number in RESULTS.md, in dependency order. One heavy job at a time.
# Python: system CPython 3.14 with numpy 2.3.5, scipy 1.17, qiskit 2.5.0, qiskit-ibm-runtime 0.47
# (FakeKingston snapshot only -- no network), python-flint (certificate). No IBM API is called.
# Wall time on this box: ~3.5 h, dominated by the two 2^24 statevector checks and the searches.
set -euo pipefail
PY="/c/Users/Josh/AppData/Local/Python/pythoncore-3.14-64/python.exe"
cd "$(dirname "$0")"

$PY inspect_witness.py              | tee inspect_witness.log        # witness structure (ranks, supports)
$PY check_paper_objects.py          | tee check_paper_objects.log    # P Kraus <-> Choi, PPT, mixed-unitary form
$PY min_cz_search.py 4              | tee min_cz_search.log          # min CZ: BE gates (i),(ii), erasure move
$PY min_cz_relaxed.py               | tee min_cz_relaxed.log         # gate (ii) with per-level phase freedom
$PY min_cz_lnn.py 7                 | tee min_cz_lnn.log             # line-compatible gate (ii)+swap (~40 min)
$PY gate_tables.py 24               | tee gate_tables.log            # 4-CZ gate (ii) tables, l <= 24
$PY polish_gates.py                 | tee polish_gates.log           # amplitude-level polish to 1e-16
$PY gate_tables_lnn.py 24           | tee gate_tables_lnn.log        # 7-CZ line gates, l <= 24
$PY check_untwist.py                | tee check_untwist.log          # paper V^dag: generic 44 vs structured 17 CZ
$PY p_phasepoly.py                  | tee p_phasepoly.log            # coherent P via phase polynomial (20 CZ)
$PY verify_channels.py              | tee verify_channels.log        # deliverables 1-2
$PY verify_encoder.py --n24         | tee verify_encoder_n24.log     # deliverable 3 (2^24 statevector ~40 min)
$PY verify_encoder_lnn.py           | tee verify_encoder_lnn.log     # nearest-neighbour encoder, n <= 17 statevector
$PY verify_encoder_lnn.py --count-only | tee verify_encoder_lnn_n24count.log
$PY test_min_rank.py 17             | tee test_min_rank.log          # fixed-fidelity Kraus-rank reduction
$PY verify_decoder.py 3 4 5 6       | tee verify_decoder.log         # deliverable 4: all branches + end to end
$PY verify_decoder_n17_spot.py 17 7 | tee verify_decoder_spot_n17.log # real n=17 decoders, k >= 10 (~20 min)
$PY decoder_counts.py 17 24         | tee decoder_counts.log         # deliverable 4 counts
$PY certify_compiled.py 17 24       | tee certify_compiled.log       # exact certificate of the compiled code
$PY full_run.py 17 20 --route       | tee full_run_n17.log           # whole shots, Sabre on heavy-hex
$PY full_run.py 24 20 --route       | tee full_run_n24.log
$PY full_run.py 17 20 --route --lnn | tee full_run_n17_lnn.log       # whole shots, LNN encoder on a path
$PY full_run.py 24 20 --route --lnn | tee full_run_n24_lnn.log
$PY totals.py                       | tee totals.log                 # deliverable 5
$PY report_tables.py > report_tables.md
$PY crosscheck_qiskit.py               | tee crosscheck_qiskit.log      # independent simulator (qiskit) cross-check
$PY check_nul.py                                                    # no 0x00 bytes in any source/report
# diagnostics behind the 'instrument defects' section: debug_aam.py, debug_kak.py, debug_k16.py
