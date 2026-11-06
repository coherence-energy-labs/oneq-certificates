"""Bareiss PD proof must agree with the minor-by-minor Sylvester proof, on PD blocks AND on blocks that are not PD."""
import time

import numpy as np

import twogroup_certificate as tc
from oneq import superactivation_certificate as sc

flint = sc._flint()
rng = np.random.default_rng(3)
agree = 0
for trial in range(40):
    d = int(rng.integers(2, 9))
    A = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    H = A @ A.conj().T
    if trial % 2:                                       # make half of them NOT positive definite
        w, v = np.linalg.eigh(H)
        w[int(rng.integers(0, d))] = -abs(w).max() * rng.uniform(0.01, 0.5)
        H = (v * w) @ v.conj().T
    re, im = sc.exact_block(flint, H, 2.0 ** -34)
    s, f = sc.positive_definite(flint, re, im), tc.positive_definite_bareiss(flint, re, im)
    assert s == f, (trial, s, f)
    agree += 1
print(f"agree on {agree}/40 random blocks (half deliberately indefinite)")
# speed on a big PD block
d = 60
A = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
re, im = sc.exact_block(flint, A @ A.conj().T + np.eye(d), 2.0 ** -34)
t = time.time()
r = tc.positive_definite_bareiss(flint, re, im)
print(f"Bareiss on a {2 * d}x{2 * d} real block: {r} in {time.time() - t:.1f}s")
