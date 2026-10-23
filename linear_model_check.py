"""
linear_model_check.py -- numerical check of the weighted-sum proposition on a model that is LINEAR in its parameters.
CPU only, no training: every minimizer is a least-squares solve.

Model (1D heat, clashing IC, same problem and metric as heat_drift.py):
    u(x, t) = 1 + B(x) * sum_ij theta_ij P_i(2x-1) P_j(2t-1),   B(x) = 4 x (1 - x)
so the hard boundary u = 1 holds for all theta, exactly as in heat_drift. All residuals are linear in theta:
    physics   : strong  r = u_t - a u_xx at fixed collocation points
                weak    R_sek = int_e (u_t v_k + a u_x v_k') dx at fixed time samples  (heat_drift.build_weak)
                dual    L^T R, with G^-1 = L L^T the rescaled Gram inverse of heat_drift.build_weak
    IC        : u(x, 0) - 0 at fixed points
    L(theta) = W_PHYS * mean(phys^2) + W_IC * rho * mean(IC^2),   rho = IC-weight multiplier (1 = the default)

CHECKS (all printed; numbers go to heat_runs/linear_check/linear_check.json)
  E1  closed form vs lstsq; the minimizer is unchanged when BOTH weights are scaled
  E2  a weak residual that is exactly s * (strong) gives theta_w(rho) = theta_s(rho / s^2)   (pure algebra)
  E3  IC loss and physics loss against rho: monotone? (weighted-sum trade-off)
  E4  rho -> 0: rank / null space of each physics operator, error of the rho = 0 minimizer (u = 1 => error 1) and of the
      rho -> 0+ limit (differs from u = 1 exactly when the operator has a null space)
  E5  relative error against the exact solution at heat_drift.T_EVAL as a function of rho; U-shaped? argmin per time
  E6  k_lin = ||grad strong physics|| / ||grad weak physics|| at the supervised-start analogue (least-squares fit of the
      exact solution), against mesh and test-function order: compare with the measured k (E^1.25, (p+1)^1.0)

USAGE (from the PINN folder):  python linear_model_check.py
"""
import json, math, os, sys
import numpy as np
from numpy.polynomial import legendre as Lg
from scipy.linalg import lstsq, cholesky
import torch
import heat_drift as hd

A_D, W_PHYS, W_IC = hd.A_DIFF, hd.W_PHYS, hd.W_IC
NX, NT = 14, 10                          # Legendre degrees kept in x and t  -> 140 parameters
RNG = np.random.default_rng(0)

# ----------------------------------------------------------------------------- basis
def leg_cols(s, n, d=0):
    """Columns P_i^(d)(s), i = 0..n-1."""
    out = np.empty((len(s), n))
    for i in range(n):
        c = np.zeros(i + 1); c[i] = 1.0
        out[:, i] = Lg.legval(s, Lg.legder(c, d)) if d else Lg.legval(s, c)
    return out

def gx(x, d=0):
    """d-th x-derivative of g_i(x) = (1 - s^2) P_i(s), s = 2x - 1  (B(x) = 1 - s^2).  Columns i = 0..NX-1."""
    s = 2 * x - 1
    P, P1, P2 = leg_cols(s, NX), leg_cols(s, NX, 1), leg_cols(s, NX, 2)
    if d == 0: return (1 - s ** 2)[:, None] * P
    G1 = -2 * s[:, None] * P + (1 - s ** 2)[:, None] * P1                 # d/ds
    if d == 1: return 2 * G1
    G2 = -2 * P - 4 * s[:, None] * P1 + (1 - s ** 2)[:, None] * P2         # d2/ds2
    return 4 * G2

def ht(t, d=0):
    tau = 2 * t - 1
    return leg_cols(tau, NT, d) * (2 ** d)

def kron_rows(a, b):                     # row-wise Kronecker: [P, na], [P, nb] -> [P, na*nb]
    return (a[:, :, None] * b[:, None, :]).reshape(len(a), -1)

def u_design(x, t):                      # u - 1 = design @ theta
    return kron_rows(gx(x), ht(t))

# ----------------------------------------------------------------------------- operators (rows scaled so loss = ||A theta||^2)
def strong_op(n=6000):
    x, t = RNG.random(n), RNG.random(n)
    r = kron_rows(gx(x), ht(t, 1)) - A_D * kron_rows(gx(x, 2), ht(t))
    return r / math.sqrt(n)

def weak_parts(n_el, p, n_t=40):
    """Raw weak residual operator R[(s,e,k), :] and the Gram inverse, via heat_drift.build_weak."""
    cfg = dict(hd.CFG[1]); cfg["n_el"], cfg["p_deg"] = n_el, p
    cfg["n_q"] = max(cfg["n_q"], p + 2)
    pts_t, V_t, dVs_t, Wq_t, Gi_t = hd.build_weak(1, **cfg)
    pts, V, dV, Wq, Gi = pts_t.cpu().numpy(), V_t.cpu().numpy(), dVs_t[0].cpu().numpy(), Wq_t.cpu().numpy(), Gi_t.cpu().numpy()
    E, Q, _ = pts.shape
    tq = (Lg.leggauss(n_t)[0] + 1) / 2                                         # fixed time samples
    xs = np.repeat(pts[:, :, 0][None], n_t, 0).reshape(-1)                      # [S*E*Q]
    ts = np.repeat(tq[:, None, None], E, 1).repeat(Q, 2).reshape(-1)
    Ut = kron_rows(gx(xs), ht(ts, 1)).reshape(n_t, E, Q, -1)
    Ux = kron_rows(gx(xs, 1), ht(ts)).reshape(n_t, E, Q, -1)
    R = np.einsum("seqn,kq,q->sekn", Ut, V, Wq) + A_D * np.einsum("seqn,kq,q->sekn", Ux, dV, Wq)
    return R, Gi                                                               # R: [S, E, K, n_theta]

def weak_op(n_el=16, p=6, dual=False):
    R, Gi = weak_parts(n_el, p)
    S, E, K, nθ = R.shape
    if not dual:
        return R.reshape(-1, nθ) / math.sqrt(R.shape[0] * E * K)
    L = cholesky(Gi, lower=True)                                               # Gi = L L^T  -> R^T Gi R = |L^T R|^2
    Rd = np.einsum("lk,sekn->seln", L.T, R)
    return Rd.reshape(-1, nθ) / math.sqrt(S * E * K)

def ic_op(n=500):
    x = RNG.random(n)
    C = kron_rows(gx(x), ht(np.zeros(n))) / math.sqrt(n)
    d = -np.ones(n) / math.sqrt(n)                                             # u(x,0) - u0 = (u - 1) + 1  -> C th = -1
    return C, d

# ----------------------------------------------------------------------------- solve / evaluate
def solve(A, C, d, rho, scale=1.0):
    M = np.vstack([math.sqrt(scale * W_PHYS) * A, math.sqrt(scale * W_IC * rho) * C])
    y = np.concatenate([np.zeros(A.shape[0]), math.sqrt(scale * W_IC * rho) * d])
    return lstsq(M, y, cond=1e-13, lapack_driver="gelsd")[0]

def closed_form(A, C, d, rho):
    H = W_PHYS * A.T @ A + W_IC * rho * C.T @ C
    return np.linalg.solve(H, W_IC * rho * C.T @ d)

XG = np.linspace(0, 1, 201)
def errors(theta):
    out = {}
    for t in hd.T_EVAL:
        X = torch.tensor(np.c_[XG, np.full_like(XG, t)], dtype=torch.float64)
        ue = hd.exact(X, "clash", 1).numpy()
        up = 1.0 + u_design(XG, np.full_like(XG, t)) @ theta
        out[f"t={t}"] = float(np.linalg.norm(up - ue) / np.linalg.norm(ue - 1.0))
    return out

def losses(A, C, d, th):
    return float(np.sum((A @ th) ** 2)), float(np.sum((C @ th - d) ** 2))

def supervised_start(n=6000):
    lo = math.log10(hd.T_MIN_FIT)
    x = RNG.random(n)
    t = np.concatenate([hd.T_MIN_FIT + (1 - hd.T_MIN_FIT) * RNG.random(n // 2), 10 ** (lo * RNG.random(n - n // 2))])
    ue = hd.exact(torch.tensor(np.c_[x, t], dtype=torch.float64), "clash", 1).numpy()
    return lstsq(u_design(x, t), ue - 1.0, cond=1e-13, lapack_driver="gelsd")[0]

# ----------------------------------------------------------------------------- main
def main():
    res = {}
    As, Aw, Ad = strong_op(), weak_op(16, 6), weak_op(16, 6, dual=True)
    C, d = ic_op()
    ops = {"strong": As, "weak": Aw, "dual": Ad}
    th0 = supervised_start()
    print(f"parameters: {As.shape[1]}   strong rows {As.shape[0]}, weak rows {Aw.shape[0]}, IC rows {C.shape[0]}")
    print("supervised-start analogue, error vs exact:", {k: round(v, 4) for k, v in errors(th0).items()})

    # E1 ---------------------------------------------------------------------------------------------------------
    print("\nE1  closed form vs lstsq; scale invariance")
    res["E1"] = {}
    for name, A in ops.items():
        t1, t2, t3 = solve(A, C, d, 1.0), closed_form(A, C, d, 1.0), solve(A, C, d, 1.0, scale=7.3)
        nrm = np.linalg.norm(t1)
        res["E1"][name] = dict(closed_vs_lstsq=float(np.linalg.norm(t1 - t2) / nrm), scaled_both=float(np.linalg.norm(t1 - t3) / nrm))
        print(f"  {name:<7} |closed - lstsq|/|theta| = {res['E1'][name]['closed_vs_lstsq']:.1e}   "
              f"|theta - theta(both weights x7.3)|/|theta| = {res['E1'][name]['scaled_both']:.1e}")

    # E2 ---------------------------------------------------------------------------------------------------------
    print("\nE2  exact proportionality  A_w = s A_s  =>  theta_w(rho) = theta_s(rho / s^2)")
    s = 0.05
    worst = 0.0
    for rho in (1e-3, 1.0, 1e3):
        a, b = solve(s * As, C, d, rho), solve(As, C, d, rho / s ** 2)
        worst = max(worst, float(np.linalg.norm(a - b) / np.linalg.norm(b)))
    res["E2"] = dict(s=s, worst_rel_diff=worst); print(f"  s = {s}: worst relative difference over rho in (1e-3, 1, 1e3): {worst:.1e}")

    # E3 + E5 ----------------------------------------------------------------------------------------------------
    rhos = 10.0 ** np.arange(-8, 8.01, 0.5)
    print("\nE3  trade-off along rho (monotone?)   E5  error vs rho")
    res["E3"], res["E5"] = {}, {}
    for name, A in ops.items():
        P, I, ER = [], [], []
        for rho in rhos:
            th = solve(A, C, d, rho); p_, i_ = losses(A, C, d, th); P.append(p_); I.append(i_); ER.append(errors(th))
        P, I = np.array(P), np.array(I)
        mono_p = bool(np.all(np.diff(P) >= -1e-9 * np.maximum(P[1:], 1e-30)))
        mono_i = bool(np.all(np.diff(I) <= 1e-9 * np.maximum(I[:-1], 1e-30)))
        arg = {t: float(rhos[int(np.argmin([e[t] for e in ER]))]) for t in ER[0]}
        best = {t: float(min(e[t] for e in ER)) for t in ER[0]}
        mean_err = np.array([np.mean(list(e.values())) for e in ER])
        res["E3"][name] = dict(physics_nondecreasing=mono_p, ic_nonincreasing=mono_i)
        res["E5"][name] = dict(rho=list(map(float, rhos)), err=ER, argmin_rho_per_time=arg, min_err_per_time=best,
                               argmin_rho_mean=float(rhos[int(np.argmin(mean_err))]))
        print(f"  {name:<7} physics loss non-decreasing in rho: {mono_p}; IC loss non-increasing: {mono_i}")
        print(f"          error at rho=1e-8: {ER[0]['t=1.0']:.3f} (t=1)   rho=1: {ER[16]['t=1.0']:.3f}   rho=1e7: {ER[-1]['t=1.0']:.3f}")
        print("          argmin_rho per time: " + ", ".join(f"{t}:{v:.0e}" for t, v in arg.items())
              + f";  min error " + ", ".join(f"{v:.3f}" for v in best.values()))

    # E4 ---------------------------------------------------------------------------------------------------------
    print("\nE4  null spaces and the rho -> 0 limit")
    res["E4"] = {}
    for name, A in ops.items():
        sv = np.linalg.svd(A, compute_uv=False)
        rank = int(np.sum(sv > sv[0] * 1e-10))
        e0 = errors(np.zeros(A.shape[1]))                                       # rho = 0 minimizer (min-norm) is theta = 0
        e_lim = errors(solve(A, C, d, 1e-14))
        res["E4"][name] = dict(rank=rank, n_params=int(A.shape[1]), cond_sv=float(sv[0] / sv[rank - 1]),
                               err_rho0=e0, err_rho_1e_14=e_lim)
        print(f"  {name:<7} rank {rank}/{A.shape[1]}  (nullity {A.shape[1] - rank});  error t=1: rho=0 -> {e0['t=1.0']:.3f}, "
              f"rho=1e-14 -> {e_lim['t=1.0']:.3f}")
    # a deliberately small test space: 2 elements, order 1 -> a clear null space
    Ats = weak_op(2, 1)
    sv = np.linalg.svd(Ats, compute_uv=False); rk = int(np.sum(sv > sv[0] * 1e-10))
    e_lim = errors(solve(Ats, C, d, 1e-14))
    res["E4"]["weak_2el_p1"] = dict(rank=rk, n_params=int(Ats.shape[1]), err_rho_1e_14=e_lim)
    print(f"  weak 2 elements, p=1: rank {rk}/{Ats.shape[1]}; error t=1 at rho=1e-14 -> {e_lim['t=1.0']:.3f} (rho=0 would give 1.000)")

    # E6 ---------------------------------------------------------------------------------------------------------
    print("\nE6  k_lin = ||grad W_PHYS*strong|| / ||grad W_PHYS*weak|| at the supervised-start analogue")
    g = lambda A, th: 2 * A.T @ (A @ th)
    gs = np.linalg.norm(g(As, th0))
    res["E6"] = {"mesh": {}, "order": {}}
    Es, Ks = [], []
    for E_ in (4, 8, 16, 32):
        kk = float(gs / np.linalg.norm(g(weak_op(E_, 6), th0))); Es.append(E_); Ks.append(kk); res["E6"]["mesh"][E_] = kk
        print(f"  E={E_:<3} p=6   k_lin = {kk:9.2f}")
    sE = float(np.polyfit(np.log(Es), np.log(Ks), 1)[0])
    ps, kp = [], []
    for p_ in (2, 4, 6, 10):
        kk = float(gs / np.linalg.norm(g(weak_op(16, p_), th0))); ps.append(p_); kp.append(kk); res["E6"]["order"][p_] = kk
        print(f"  E=16  p={p_:<3} k_lin = {kk:9.2f}")
    sP = float(np.polyfit(np.log(np.array(ps) + 1), np.log(kp), 1)[0])
    res["E6"]["slope_E"], res["E6"]["slope_p_plus_1"] = sE, sP
    print(f"  log-log slope vs E: {sE:.2f}   vs (p+1): {sP:.2f}      [measured with networks: 1.25 and 1.00]")

    os.makedirs(os.path.join("heat_runs", "linear_check"), exist_ok=True)
    json.dump(res, open(os.path.join("heat_runs", "linear_check", "linear_check.json"), "w"), indent=1, default=float)
    print("\nsaved heat_runs/linear_check/linear_check.json")

if __name__ == "__main__":
    main()
