---
name: superactivation-witness-attacks
description: Attack log for superactivation certificate claims (F_e > 3/4, tilted antidegradability witness, betting test) - what killed, what survived, witnesses with commands
metadata:
  type: project
---

- WORKED (null-set claim, "null = AD"): attack the erasure FLAG pattern actually implemented. Design folds both one-erased orders onto "use 1 erased" (hw_two_use.py:22, r186_faithful_requirements.py:777), so use 0 is a 25%-erasure channel (non-AD). 2026-10-23.
- WITNESS: logical qubit into use 0's A register only -> lam = (3/4,3/4,3/4), F_e 13/16, passes F_e (+0.0625) and tilted (+0.0725) certificates with NO P; iid-flag control arm is AD. Run scratchpad adv_folded.py (session 824a9479).
- WITNESS: "tilted witness certifies the same statement as F_e > 3/4" killed by the claim's own ad_cost.py row spec_helios_1_m5 lam:2 (F_e 0.7464, tilted margin +0.00767).
- FAILED to kill: h(w) = lambda_max((W_BA+W_B'A)/2) is EXACT for every real w (primal twirl construction to 2e-15; Chen-criterion search over general channels, max excess -1e-13). Do not re-attack without a new idea.
- FAILED to kill: per-basis estimator for w >= 0 (error indicator = exact loss on high units; low units at 0 conservative only for w >= 0).
- INSTRUMENT: Gaussian var/margin^2 shot proxy (ad_cost.py) overstates the tilted witness's saving at spec noise (1.7-1.8x claimed vs 1.30-1.41x by exact constant-bet growth with the bet cap, adv_cost.py).
- INSTRUMENT: ad_lmax.py's cross-check shares the twirl step with the claim; it is not independent of step 2.
- OPEN: false-claim simulation for negative-weight witnesses and per-batch witness schedules (derived only); rigorous rational bound on algebraic h(w).
