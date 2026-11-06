"""encoder_exact (dual, 3 parameters) must reproduce the cvxpy encoder SDP's optimum on small cases."""
import numpy as np

import twogroup as tg

for a, b in ((1, 1), (2, 2), (3, 3)):
    rng = np.random.default_rng(a + b)
    X = tg.random_encoder(a, b, rng)
    decs, F = tg.decode(X, a, b, iters=50)
    G = tg.gradient(decs, a, b)
    m = (a + 1) * (b + 1)
    Xs = tg.encoder_step(decs, a, b)                      # the cvxpy SDP (small sizes only)
    Xd, z, r = tg.encoder_exact(G, m)
    vs, vd = float(np.real(np.trace(G @ Xs))), float(np.real(np.trace(G @ Xd)))
    tp = np.abs(Xd.reshape(2, m, 2, m).trace(axis1=1, axis2=3) - np.eye(2)).max()
    print(f"(a,b)=({a},{b}): SDP value {vs:.10f}  dual-route value {vd:.10f}  diff {vd - vs:+.2e}  "
          f"eigenspace dim {r}  TP residual {tp:.1e}  min eig {np.linalg.eigvalsh(Xd).min():.1e}")
