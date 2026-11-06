
#### Decoder per flag weight k, n = 17  (decoder_counts.py 17)

| k | P(k) | m = n-k | support dim s | decoder CZ (min..max over b) | U_m^dag CZ | route | decoder-unitary qubits |
|---|---|---|---|---|---|---|---|
| 0 | 7.63e-06 | 17 | 2 | 595..595 | 512 | compressed | 2 |
| 1 | 1.30e-04 | 16 | 4 | 660..660 | 450 | compressed | 3 |
| 2 | 1.04e-03 | 15 | 6 | 759..759 | 392 | compressed | 4 |
| 3 | 5.19e-03 | 14 | 8 | 749..749 | 338 | compressed | 4 |
| 4 | 1.82e-02 | 13 | 10 | 748..748 | 288 | restricted | 5 |
| 5 | 4.72e-02 | 12 | 12 | 701..701 | 242 | restricted | 5 |
| 6 | 9.44e-02 | 11 | 12 | 656..656 | 200 | restricted | 5 |
| 7 | 1.48e-01 | 10 | 11 | 617..617 | 162 | restricted | 5 |
| 8 | 1.85e-01 | 9 | 10 | 581..581 | 128 | restricted | 5 |
| 9 | 1.85e-01 | 8 | 9 | 550..550 | 98 | restricted | 5 |
| 10 | 1.48e-01 | 7 | 8 | 184..184 | 72 | restricted | 4 |
| 11 | 9.44e-02 | 6 | 7 | 161..161 | 50 | restricted | 4 |
| 12 | 4.72e-02 | 5 | 6 | 141..141 | 32 | restricted | 4 |
| 13 | 1.82e-02 | 4 | 5 | 126..126 | 18 | restricted | 4 |
| 14 | 5.19e-03 | 3 | 4 | 33..33 | 8 | restricted | 3 |
| 15 | 1.04e-03 | 2 | 3 | 26..26 | 2 | restricted | 3 |
| 16 | 1.30e-04 | 1 | 2 | 3..4 | 0 | restricted | 2 |
| 17 | 7.63e-06 | 0 | 1 | 0..0 | 0 | restricted | 1 |

Expected decoder CZ 466.3 = U_m^dag 121.0 + Givens 1.8 + unary->binary 14.8 + phase fix 11.4 + decoder unitary 317.3; worst 759 at (k,b) = (2, 0); expected reads 16.95; expected feed-forward 1q gates 75.0; max qubits 19.

#### Decoder per flag weight k, n = 24  (decoder_counts.py 24)

| k | P(k) | m = n-k | support dim s | decoder CZ (min..max over b) | U_m^dag CZ | route | decoder-unitary qubits |
|---|---|---|---|---|---|---|---|
| 0 | 5.96e-08 | 24 | 2 | 1176..1176 | 1058 | compressed | 2 |
| 1 | 1.43e-06 | 23 | 4 | 1269..1269 | 968 | compressed | 3 |
| 2 | 1.65e-05 | 22 | 6 | 1396..1396 | 882 | compressed | 4 |
| 3 | 1.21e-04 | 21 | 8 | 1414..1414 | 800 | compressed | 4 |
| 4 | 6.33e-04 | 20 | 10 | 1742..1742 | 722 | compressed | 5 |
| 5 | 2.53e-03 | 19 | 12 | 1704..1704 | 648 | compressed | 5 |
| 6 | 8.02e-03 | 18 | 13 | 1617..1617 | 578 | compressed | 5 |
| 7 | 2.06e-02 | 17 | 13 | 1502..1502 | 512 | compressed | 5 |
| 8 | 4.38e-02 | 16 | 13 | 1391..1391 | 450 | compressed | 5 |
| 9 | 7.79e-02 | 15 | 13 | 855..855 | 392 | restricted | 5 |
| 10 | 1.17e-01 | 14 | 13 | 800..800 | 338 | restricted | 5 |
| 11 | 1.49e-01 | 13 | 12,13 | 748..748 | 288 | restricted | 5 |
| 12 | 1.61e-01 | 12 | 12 | 701..701 | 242 | restricted | 5 |
| 13 | 1.49e-01 | 11 | 12 | 656..656 | 200 | restricted | 5 |
| 14 | 1.17e-01 | 10 | 11 | 617..617 | 162 | restricted | 5 |
| 15 | 7.79e-02 | 9 | 10 | 581..581 | 128 | restricted | 5 |
| 16 | 4.38e-02 | 8 | 9 | 550..550 | 98 | restricted | 5 |
| 17 | 2.06e-02 | 7 | 8 | 184..184 | 72 | restricted | 4 |
| 18 | 8.02e-03 | 6 | 7 | 161..161 | 50 | restricted | 4 |
| 19 | 2.53e-03 | 5 | 6 | 141..141 | 32 | restricted | 4 |
| 20 | 6.33e-04 | 4 | 5 | 126..126 | 18 | restricted | 4 |
| 21 | 1.21e-04 | 3 | 4 | 33..33 | 8 | restricted | 3 |
| 22 | 1.65e-05 | 2 | 3 | 26..26 | 2 | restricted | 3 |
| 23 | 1.43e-06 | 1 | 2 | 3..4 | 0 | compressed/restricted | 2 |
| 24 | 5.96e-08 | 0 | 1 | 0..0 | 0 | restricted | 1 |

Expected decoder CZ 739.6 = U_m^dag 254.0 + Givens 38.7 + unary->binary 20.7 + phase fix 13.7 + decoder unitary 412.5; worst 1742 at (k,b) = (4, 0); expected reads 23.64; expected feed-forward 1q gates 116.7; max qubits 26.

#### Totals n = 17  (totals.py)

| profile | CZ expected | CZ worst | eps = 0.0063 x CZ | factor = eps/delta | factor (worst) | reads expected |
|---|---|---|---|---|---|---|
| RANDOMIZED channels + measurement-based D~ | 1121.9 | 1455 | 7.07 | 17,387 | 22,549 | 34.0 |
| RANDOMIZED channels + measurement-free D~ | 1155.9 | 1523 | 7.28 | 17,914 | 23,603 | 17.0 |
| RANDOMIZED channels + paper D~ (coherent untwist) | 1259.4 | 1710 | 7.93 | 19,518 | 26,501 | 25.5 |
| COHERENT (Stinespring) channels + measurement-based D~ | 1612.8 | 1931 | 10.16 | 24,996 | 29,927 | 51.0 |
| COHERENT (Stinespring) channels + paper D~ | 1740.3 | 2186 | 10.96 | 26,972 | 33,878 | 42.5 |

Sampled whole shots, Sabre layout, all-to-all encoder (20): logical CZ 1115.6 +- 214.1 (sd), logical 2q-depth 566; heavy-hex routed CZ 3347.7 +- 110.4 (sem), routed/logical 3.03, routed depth 5300 (2q 1708), routed factor 51,882

Sampled whole shots, nearest-neighbour encoder on a heavy-hex path (20): logical CZ 1520.6 +- 214.1 (sd), logical 2q-depth 663; heavy-hex routed CZ 2815.2 +- 113.2 (sem), routed/logical 1.84, routed depth 3994 (2q 1160), routed factor 43,630

#### Totals n = 24  (totals.py)

| profile | CZ expected | CZ worst | eps = 0.0063 x CZ | factor = eps/delta | factor (worst) | reads expected |
|---|---|---|---|---|---|---|
| RANDOMIZED channels + measurement-based D~ | 2001.5 | 3061 | 12.61 | 6,129 | 9,374 | 47.6 |
| RANDOMIZED channels + measurement-free D~ | 2049.5 | 3157 | 12.91 | 6,276 | 9,668 | 23.6 |
| RANDOMIZED channels + paper D~ (coherent untwist) | 2195.6 | 3421 | 13.83 | 6,724 | 10,476 | 35.6 |
| COHERENT (Stinespring) channels + measurement-based D~ | 2694.6 | 3733 | 16.98 | 8,252 | 11,432 | 71.6 |
| COHERENT (Stinespring) channels + paper D~ | 2874.6 | 4093 | 18.11 | 8,803 | 12,534 | 59.6 |

Sampled whole shots, Sabre layout, all-to-all encoder (20): logical CZ 2029.3 +- 178.3 (sd), logical 2q-depth 810; heavy-hex routed CZ 6620.7 +- 111.7 (sem), routed/logical 3.27, routed depth 8414 (2q 2774), routed factor 20,275

Sampled whole shots, nearest-neighbour encoder on a heavy-hex path (20): logical CZ 2854.3 +- 178.3 (sd), logical 2q-depth 938; heavy-hex routed CZ 5406.9 +- 112.5 (sem), routed/logical 1.89, routed depth 6342 (2q 1889), routed factor 16,558
