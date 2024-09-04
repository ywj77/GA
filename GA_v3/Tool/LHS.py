import numpy as np


def lhs_sample(n, d, l_b, u_b):
    cut = np.linspace(0, 1, n + 1)
    u = np.random.rand(n, d)
    a = cut[:n]
    b = cut[1:n + 1]
    rdpoints = np.zeros_like(u)
    for j in range(d):
        rdpoints[:, j] = u[:, j] * (b - a) + a
    h = rdpoints.copy()
    for j in range(d):
        np.random.shuffle(h[:, j])
    h = l_b + (u_b - l_b) * h
    return h
