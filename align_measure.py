"""
align_measure.py -- two checks of Lemma 1 at the supervised starts. Measurement only: no training. heat_drift.py is
imported, not changed, and nothing in its output folders is touched; this writes one new JSON per setting.

  1. Gradient alignment, on the same D draws as heat_drift.py --measure:
       cos_weak_strong   cosine between the weak and the strong physics gradients (Lemma 1: near 1 on fine meshes)
       cos_ic_weak, cos_ic_strong   cosine between the IC gradient and each physics gradient (negative = they pull
                         against each other, which is the clash)
       Gamma             ||mean r grad r|| / mean |r| ||grad r|| over random points: 1 = no cancellation, near 0 =
                         strong cancellation. The lemma's gradient statement needs Gamma not small.
  2. h / ell: at every weak-form element and sampled time, r = u_t - a lap u is fitted at the element's quadrature
     points with Legendre polynomials up to degree 2 (c0 = mean, c1 = slope, c2 = curvature, in element coordinates),
     and h/ell_e = max(2|c1|/|c0|, sqrt(12|c2|/|c0|)) in 1D (tensor version in 2D): how much the residual changes across
     an element relative to its mean. Reported: the residual-energy-weighted mean of (h/ell)^2 (the relative size of the
     lemma's correction term), the median and 90th percentile of h/ell, and the share of residual energy in elements
     with h/ell > 1.

USAGE (from the PINN folder; CPU is fine, it only takes gradients)
    python align_measure.py --dims 1 2 --starts DIR --out heat_align
    python align_measure.py --dims 1 --n-el 4 --starts DIR --out heat_align        # one mesh; repeat for 8 and 32
    python align_measure.py --dims 1 --case compat --starts DIR_COMPAT --out heat_align
DIR is a placeholder for the same --starts folder you give heat_drift.py --measure (default heat_results/ckpt, so
--starts can be left out for the clashing-IC starts; the compat starts need their own folder).
WRITES <out>/align_<case>[_e<N>][_p<P>].json, one file per setting, so runs never overwrite each other or any
heat_drift.py file. kappa_this_device is recomputed for reference only; keep the paper's 100-draw values.
"""
import argparse, json, os
import numpy as np
import torch
from scipy.special import eval_legendre
import heat_drift as hd

def residual(m, X, dim):
    """r = u_t - a lap u at the points X, with the graph kept (heat equation only)."""
    X = X.clone().requires_grad_(True)
    u = m(X)
    g = torch.autograd.grad(u.sum(), X, create_graph=True)[0]
    lap = 0.0
    for i in range(dim):
        lap = lap + torch.autograd.grad(g[:, i].sum(), X, create_graph=True)[0][:, i]
    return g[:, -1] - hd.A_DIFF * lap

def fit_matrix(dim, n_q):
    """Pseudo-inverse of the Legendre basis (degree <= 2 per dimension) at the element's Gauss points.
    2D points are ordered q = i*Q + j with xi = r[i], eta = r[j], as in heat_drift.build_weak."""
    r, _ = np.polynomial.legendre.leggauss(n_q)
    P = [eval_legendre(k, r) for k in range(3)]
    if dim == 1:
        B = np.stack(P, 1)                                                     # c0, c1, c2
    else:
        idx = [(0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1)]                 # c00, c10, c01, c20, c02, c11
        B = np.stack([np.outer(P[a], P[b]).reshape(-1) for a, b in idx], 1)
    return torch.tensor(np.linalg.pinv(B), dtype=torch.float64, device=hd.DEV)

def h_over_ell(c, dim):
    c0 = c[..., 0].abs().clamp_min(1e-300)
    if dim == 1:
        d1 = 2 * c[..., 1].abs() / c0
        d2 = torch.sqrt(12 * c[..., 2].abs() / c0)
    else:
        d1 = 2 * torch.sqrt(c[..., 1] ** 2 + c[..., 2] ** 2) / c0
        d2 = torch.sqrt((12 * (c[..., 3].abs() + c[..., 4].abs()) + 4 * c[..., 5].abs()) / c0)
    return torch.maximum(d1, d2), c[..., 0] ** 2

def ell_stats(m, W, dim, n_t, pinvB, draws):
    pts = W[0]
    E, Q, _ = pts.shape
    hl, w = [], []
    for d in range(draws):
        gen = torch.Generator(device=hd.DEV); gen.manual_seed(3000 + d)
        t = torch.rand(n_t, generator=gen, device=hd.DEV)
        X = torch.cat([pts.unsqueeze(0).expand(n_t, E, Q, dim),
                       t.view(n_t, 1, 1, 1).expand(n_t, E, Q, 1)], -1).reshape(-1, dim + 1)
        r = residual(m, X, dim).detach().reshape(n_t, E, Q)
        x, wt = h_over_ell(torch.einsum("bq,teq->teb", pinvB, r), dim)
        hl.append(x.reshape(-1)); w.append(wt.reshape(-1))
    hl, w = torch.cat(hl), torch.cat(w)
    return dict(hl2_energy_weighted=float((w * hl ** 2).sum() / w.sum()), hl_median=float(hl.median()),
                hl_p90=float(torch.quantile(hl, 0.9)), energy_share_hl_gt_1=float(w[hl > 1].sum() / w.sum()))

def cancellation(m, dim, n_pts, draws):
    """Gamma = ||sum r grad r|| / sum |r| ||grad r|| over random space-time points."""
    params = list(m.parameters())
    S, T = None, 0.0
    for d in range(draws):
        gen = torch.Generator(device=hd.DEV); gen.manual_seed(4000 + d)
        X = torch.rand(n_pts, dim + 1, generator=gen, device=hd.DEV)
        r = residual(m, X, dim)
        for k in range(n_pts):
            gv = torch.cat([g.reshape(-1) for g in torch.autograd.grad(r[k], params, retain_graph=True)])
            rk = r[k].detach()
            S = rk * gv if S is None else S + rk * gv
            T += float(rk.abs() * gv.norm())
    return float(S.norm() / T)

def grads(m, W, dim, cfg, nf, case, D):
    """Averaged gradients on the same draws as heat_drift.measure, and their cosines."""
    acc = {}
    for d in range(D):
        gi = torch.Generator(device=hd.DEV); gi.manual_seed(2000 + d)
        gw = torch.Generator(device=hd.DEV); gw.manual_seed(1000 + d)
        gs = torch.Generator(device=hd.DEV); gs.manual_seed(1000 + d)
        parts = dict(ic=hd.W_IC * hd.ic_loss(m, dim, case, cfg["n_ic"], gi),
                     weak=hd.W_PHYS * hd.weak_residual(m, W, cfg["n_t"], gw),
                     strong=hd.W_PHYS * hd.strong_residual(m, dim, nf, gs))
        for k, L in parts.items():
            acc[k] = acc.get(k, 0) + hd._grad_vec(m, L) / D
    cos = lambda a, b: float(a @ b / (a.norm() * b.norm()))
    return dict(cos_weak_strong=cos(acc["weak"], acc["strong"]), cos_ic_weak=cos(acc["ic"], acc["weak"]),
                cos_ic_strong=cos(acc["ic"], acc["strong"]),
                kappa_this_device=float(acc["strong"].norm() / acc["weak"].norm()))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dims", nargs="+", default=["1", "2"])
    p.add_argument("--case", choices=("clash", "compat"), default="clash")
    p.add_argument("--inits", nargs="+", default=["warm"], choices=("warm", "cold"))
    p.add_argument("--starts", default=os.path.join("heat_results", "ckpt"),
                   help="folder with the supervised starts (as for heat_drift.py --measure); default heat_results/ckpt")
    p.add_argument("--out", default="heat_align")
    p.add_argument("--D", type=int, default=100, help="draws for the averaged gradients (as --measure --D)")
    p.add_argument("--ell-draws", type=int, default=10, help="time draws for h/ell (10 times each)")
    p.add_argument("--points", type=int, default=64, help="random points per draw for Gamma")
    p.add_argument("--cancel-draws", type=int, default=2)
    p.add_argument("--n-el", type=int, default=None, help="weak-form elements per dimension (default 16 in 1D, 8 in 2D)")
    p.add_argument("--p-deg", type=int, default=None, help="weak-form test-function order (default 6 in 1D, 4 in 2D)")
    a = p.parse_args()
    tag = ""
    if a.n_el:                                          # same settings handling as heat_drift.py
        for d in hd.CFG:
            hd.CFG[d]["n_el"] = a.n_el
        tag += f"_e{a.n_el}"
    if a.p_deg:
        for d in hd.CFG:
            hd.CFG[d]["p_deg"] = a.p_deg
            hd.CFG[d]["n_q"] = max(hd.CFG[d]["n_q"], a.p_deg + 2)
        tag += f"_p{a.p_deg}"
    os.makedirs(a.out, exist_ok=True)
    device = torch.cuda.get_device_name(0) if hd.DEV.type == "cuda" else "CPU"
    out = dict(device=device, case=a.case, D=a.D)
    for dim in (int(d) for d in a.dims):
        cfg = hd.CFG[dim]
        W = hd.build_weak(dim, **cfg)
        nf = hd.n_strong(dim, W)
        pinvB = fit_matrix(dim, cfg["n_q"])
        for init in a.inits:
            rows = {}
            for seed in hd.PAPER_SEEDS:
                if init == "warm":
                    m = hd.HardBC(dim).to(hd.DEV)
                    m.load_state_dict(torch.load(hd.start_path(a, dim, a.case, seed), map_location=hd.DEV))
                else:
                    torch.manual_seed(seed); m = hd.HardBC(dim).to(hd.DEV)
                row = grads(m, W, dim, cfg, nf, a.case, a.D)
                row.update(ell_stats(m, W, dim, cfg["n_t"], pinvB, a.ell_draws))
                row["Gamma"] = cancellation(m, dim, a.points, a.cancel_draws)
                rows[seed] = row
                print(f"{dim}D {a.case} {init} s{seed}: cos(weak,strong) {row['cos_weak_strong']:.3f}  "
                      f"cos(IC,weak) {row['cos_ic_weak']:.3f}  Gamma {row['Gamma']:.3f}  "
                      f"(h/ell)^2 {row['hl2_energy_weighted']:.3g}  h/ell median {row['hl_median']:.3g}, "
                      f"p90 {row['hl_p90']:.3g}", flush=True)
            out[f"d{dim}_{init}"] = dict(per_seed=rows, n_el=cfg["n_el"], p_deg=cfg["p_deg"],
                                         lemma_kappa=(cfg["p_deg"] + 1) ** dim * cfg["n_el"] ** (2 * dim))
    path = os.path.join(a.out, f"align_{a.case}{tag}.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"saved {path}")