"""
cn_baseline.py -- classical time-stepping on the same heat problems as heat_drift.py. CPU only, no training.

Same problem, reference and metric as heat_drift.py:
    u_t = a * laplacian(u) on [0,1]^dim (a = 0.1), u = 1 on the boundary,
    u0 = 0 (clash) or u0 = 1 - prod sin(pi x_i) (compat);
    error = ||u_h - u|| / ||u - 1|| on heat_drift's evaluation grid (201 points in 1D, 41 x 41 in 2D)
    at t = 0.01, 0.05, 0.1, 0.3, 1, against heat_drift.exact (the Fourier series).

Discretization: second-order finite differences on a uniform grid whose nodes include every evaluation point
(h = 1 / (200 m) in 1D, 1 / (40 m) in 2D), so no interpolation enters the error. Time step dt = 0.01 / k, so every
evaluation time is hit exactly. Schemes:
    cn         plain Crank-Nicolson
    rannacher  Rannacher start-up: the first two CN steps are replaced by four backward-Euler half-steps (dt / 2),
               which damps the high-frequency content of the clashing initial data; CN afterwards
    be         backward Euler throughout (first order in time; for reference)
cn and rannacher share one LU factorization (I - dt/2 A); be factors I - dt A.

USAGE (from the PINN folder)
    python cn_baseline.py                                  # 1D and 2D, clash and compat, default refinement
    python cn_baseline.py --dims 1 --cases clash --m 1 2 4 8 16 --k 2 4 8 16
    python cn_baseline.py --schemes rannacher              # one scheme
WRITES  heat_runs/cn_baseline/cn_baseline.json  (one row per dim x case x scheme x m x k) and prints a table.
"""
import argparse, json, os, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
import torch
import heat_drift as hd                    # exact solution, A_DIFF and T_EVAL, so nothing is redefined here

EVAL_N = {1: 200, 2: 40}                   # evaluation grid intervals per dimension (heat_drift.report)

def laplacian(n, h, dim):
    """Dirichlet Laplacian on the n interior nodes per dimension (boundary values carried separately)."""
    L1 = sp.diags([np.ones(n - 1), -2.0 * np.ones(n), np.ones(n - 1)], [-1, 0, 1], format="csc") / h ** 2
    if dim == 1:
        return L1
    I = sp.identity(n, format="csc")
    return (sp.kron(I, L1) + sp.kron(L1, I)).tocsc()

def exact_np(X, case, dim):
    return hd.exact(torch.tensor(X, dtype=torch.float64, device=hd.DEV), case, dim).cpu().numpy()

def eval_points(dim):
    g = np.linspace(0.0, 1.0, EVAL_N[dim] + 1)
    if dim == 1:
        return g[:, None]
    return np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)

def solve(dim, case, m, k, scheme):
    """Returns {t: rel. error} at hd.T_EVAL. Unknown is w = u - 1, which is 0 on the boundary."""
    N = EVAL_N[dim] * m                    # intervals per dimension
    h, n = 1.0 / N, N - 1
    x = np.linspace(h, 1.0 - h, n)         # interior nodes
    if dim == 1:
        Xi = x[:, None]
    else:
        Xi = np.stack(np.meshgrid(x, x, indexing="ij"), -1).reshape(-1, 2)
    w = -np.ones(Xi.shape[0]) if case == "clash" else -np.prod(np.sin(np.pi * Xi), axis=1)
    A = hd.A_DIFF * laplacian(n, h, dim)
    I = sp.identity(A.shape[0], format="csc")
    dt = 0.01 / k
    if scheme in ("cn", "rannacher"):
        lu = splu((I - 0.5 * dt * A).tocsc())               # CN left side = backward Euler at dt / 2
        rhs = (I + 0.5 * dt * A).tocsr()
    else:
        lu = splu((I - dt * A).tocsc())
    if scheme == "rannacher" and k < 2:
        raise ValueError("rannacher needs k >= 2 so the start-up (2 dt) ends before t = 0.01")

    # pick the evaluation nodes out of the full grid (boundary included, u = 1 there)
    Xe = eval_points(dim)
    stride = m

    def full_u(w):
        if dim == 1:
            U = np.ones(N + 1); U[1:-1] = 1.0 + w
            return U[::stride]
        U = np.ones((N + 1, N + 1)); U[1:-1, 1:-1] = 1.0 + w.reshape(n, n)
        return U[::stride, ::stride].reshape(-1)

    out, t_now, step = {}, 0.0, 0
    targets = list(hd.T_EVAL)
    n_total = int(round(targets[-1] / dt))
    while step < n_total:
        if scheme == "rannacher" and step < 2:
            w = lu.solve(w); w = lu.solve(w)                 # two BE half-steps = one step of size dt
        elif scheme == "be":
            w = lu.solve(w)
        else:
            w = lu.solve(rhs @ w)
        step += 1
        t_now = step * dt
        for t in targets:
            if abs(t_now - t) < 1e-12 * max(1.0, t):
                ue = exact_np(np.c_[Xe, np.full(len(Xe), t)], case, dim)
                up = full_u(w)
                out[f"t={t}"] = float(np.linalg.norm(up - ue) / np.linalg.norm(ue - 1.0))
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dims", type=int, nargs="+", default=[1, 2])
    p.add_argument("--cases", nargs="+", default=["clash", "compat"], choices=("clash", "compat"))
    p.add_argument("--schemes", nargs="+", default=["rannacher", "cn", "be"], choices=("rannacher", "cn", "be"))
    p.add_argument("--m", type=int, nargs="+", default=None, help="grid refinement factors (h = 1/(200m) 1D, 1/(40m) 2D)")
    p.add_argument("--k", type=int, nargs="+", default=[2, 4, 8, 16], help="time refinement factors (dt = 0.01/k)")
    p.add_argument("--out", default=os.path.join("heat_runs", "cn_baseline"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for dim in a.dims:
        ms = a.m or ([1, 2, 4, 8] if dim == 1 else [1, 2, 4])
        for case in a.cases:
            for scheme in a.schemes:
                for m in ms:
                    for k in a.k:
                        t0 = time.time()
                        err = solve(dim, case, m, k, scheme)
                        rows.append(dict(dim=dim, case=case, scheme=scheme, m=m, k=k,
                                         h=1.0 / (EVAL_N[dim] * m), dt=0.01 / k, err=err, seconds=time.time() - t0))
                        print(f"d{dim} {case:<6} {scheme:<9} h=1/{EVAL_N[dim] * m:<5} dt=0.01/{k:<3}"
                              + "".join(f"  {t}:{v:.2e}" for t, v in err.items())
                              + f"  ({time.time() - t0:.1f}s)", flush=True)
    path = os.path.join(a.out, "cn_baseline.json")
    json.dump(dict(a_diff=hd.A_DIFF, t_eval=list(hd.T_EVAL), rows=rows), open(path, "w"), indent=2)
    print(f"saved {path}")

if __name__ == "__main__":
    main()
