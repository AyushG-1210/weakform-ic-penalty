"""
bench_cost.py -- computational cost of each loss, for the paper's cost table. Timing only: no training runs, no results.

Times, on whatever device heat_drift.py picks (run it on the A100 with NOTHING else on the GPU):
    one training step (physics term + IC term, backward, Adam) for the strong loss, the weak loss and the dual-norm weak
    loss, in 1D and 2D, on the paper's default meshes and point budgets
    one kappa measurement: D draws of the gradient of each loss term at one point in parameter space (as --measure does)
and reads the finite-difference times from heat_runs/cn_baseline/cn_baseline.json if it is there.

USAGE (from the PINN folder):
    python bench_cost.py                       # 300 timed steps after 30 warm-up steps, D = 100
    python bench_cost.py --steps 100 --D 20    # quicker
WRITES  heat_runs/cost/cost.json  and  heat_runs/cost/cost_table.tex  (copy the .tex next to the paper; the paper
        pulls it in with \\inputif{cost_table})
"""
import argparse, json, os, time
import torch
import heat_drift as hd

def sync():
    if hd.DEV.type == "cuda":
        torch.cuda.synchronize()

def setup(dim):
    torch.manual_seed(0)
    m = hd.HardBC(dim).to(hd.DEV)
    W = hd.build_weak(dim, **hd.CFG[dim])
    return m, W, hd.CFG[dim], hd.n_strong(dim, W)

def time_steps(dim, form, dual, steps, warmup):
    """Mean wall-clock time of one training step, and peak GPU memory, for one loss."""
    m, W, cfg, nf = setup(dim)
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)

    def step():
        phys = hd.weak_residual(m, W, cfg["n_t"], dual=dual) if form == "weak" else hd.strong_residual(m, dim, nf)
        loss = hd.W_PHYS * phys + hd.W_IC * hd.ic_loss(m, dim, "clash", cfg["n_ic"])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    for _ in range(warmup):
        step()
    sync()
    if hd.DEV.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    sync()
    ms = (time.perf_counter() - t0) / steps * 1e3
    mem = torch.cuda.max_memory_allocated() / 2 ** 20 if hd.DEV.type == "cuda" else float("nan")
    pts = W[0].shape[0] * W[0].shape[1] * cfg["n_t"] if form == "weak" else nf      # network points per step
    return dict(ms_per_step=ms, peak_mb=mem, points=int(pts),
                warm_run_min=ms * hd.ITERS["warm"] / 6e4, cold_run_min=ms * hd.ITERS["cold"] / 6e4)

def time_kappa(dim, D):
    """Seconds for one kappa measurement at one seed: D draws of each term's gradient, as in heat_drift.measure."""
    m, W, cfg, nf = setup(dim)
    sync(); t0 = time.perf_counter()
    for d in range(D):
        gi = torch.Generator(device=hd.DEV); gi.manual_seed(2000 + d)
        gw = torch.Generator(device=hd.DEV); gw.manual_seed(1000 + d)
        gs = torch.Generator(device=hd.DEV); gs.manual_seed(1000 + d)
        hd._grad_vec(m, hd.W_IC * hd.ic_loss(m, dim, "clash", cfg["n_ic"], gi))
        hd._grad_vec(m, hd.W_PHYS * hd.weak_residual(m, W, cfg["n_t"], gw))
        hd._grad_vec(m, hd.W_PHYS * hd.strong_residual(m, dim, nf, gs))
    sync()
    return time.perf_counter() - t0

def fd_times(path):
    """Seconds of the finest Rannacher finite-difference run per dim, clashing IC."""
    if not os.path.exists(path):
        return {}
    rows = [r for r in json.load(open(path))["rows"] if r["scheme"] == "rannacher" and r["case"] == "clash"]
    out = {}
    for dim in (1, 2):
        rs = [r for r in rows if r["dim"] == dim]
        if rs:
            r = max(rs, key=lambda r: (r["m"], r["k"]))
            out[dim] = dict(seconds=r["seconds"], h=r["h"], dt=r["dt"])
    return out

def fmt(x, nd=3):
    return "--" if x != x else f"{x:.{nd}g}"

def write_tex(path, res, device, steps, warmup, D):
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{Computational cost on one " + device.replace("_", r"\_") + r", double precision, no other jobs "
             rf"running. Per-step time: mean over {steps} steps after {warmup} warm-up steps, physics and initial-condition "
             r"terms with backward pass and Adam update. Run times are the per-step time multiplied by the steps of a warm "
             r"($10{,}000$) or cold ($15{,}000$) run. $\kappa$: one measurement with $D=" + str(D) + r"$ gradient draws of "
             r"each loss term at one seed. Finite differences: finest Rannacher solve on a CPU (Section~\ref{sec:baseline}).}",
             r"\label{tab:cost}", r"\small", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{llccccc}", r"\toprule",
             r" & Loss & points/step & ms/step & peak MB & warm run (min) & cold run (min) \\", r"\midrule"]
    for dim in (1, 2):
        for key, lab in (("strong", "strong"), ("weak", "weak"), ("dual", "weak, dual norm")):
            r = res["steps"].get(f"d{dim}_{key}")
            if not r:
                continue
            lines.append(f"{dim}D & {lab} & {r['points']:,} & {fmt(r['ms_per_step'])} & {fmt(r['peak_mb'])} & "
                         f"{fmt(r['warm_run_min'])} & {fmt(r['cold_run_min'])} \\\\".replace(",", "{,}"))
        lines.append(r"\midrule" if dim == 1 else r"\midrule")
    for dim in (1, 2):
        k = res["kappa_s"].get(f"d{dim}")
        f = res["fd"].get(dim) or res["fd"].get(str(dim))
        lines.append(f"{dim}D & \\multicolumn{{6}}{{l}}{{$\\kappa$ measurement: {fmt(k)}\\,s"
                     + (f"; finite differences: {fmt(f['seconds'])}\\,s (CPU)" if f else "") + "} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    open(path, "w").write("\n".join(lines))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--D", type=int, default=100)
    ap.add_argument("--out", default=os.path.join("heat_runs", "cost"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = torch.cuda.get_device_name(0) if hd.DEV.type == "cuda" else "CPU"
    if hd.DEV.type == "cuda":
        torch.backends.cudnn.benchmark = False
    print(f"device: {device}", flush=True)
    res = dict(device=device, steps_timed=a.steps, warmup=a.warmup, D=a.D, steps={}, kappa_s={})
    for dim in (1, 2):
        for key, form, dual in (("strong", "strong", False), ("weak", "weak", False), ("dual", "weak", True)):
            r = time_steps(dim, form, dual, a.steps, a.warmup)
            res["steps"][f"d{dim}_{key}"] = r
            print(f"{dim}D {key:<6}: {r['ms_per_step']:.2f} ms/step, {r['points']} points, peak {r['peak_mb']:.0f} MB, "
                  f"warm run {r['warm_run_min']:.1f} min, cold run {r['cold_run_min']:.1f} min", flush=True)
        res["kappa_s"][f"d{dim}"] = time_kappa(dim, a.D)
        print(f"{dim}D kappa measurement (D = {a.D}): {res['kappa_s'][f'd{dim}']:.1f} s", flush=True)
    res["fd"] = fd_times(os.path.join("heat_runs", "cn_baseline", "cn_baseline.json"))
    json.dump(res, open(os.path.join(a.out, "cost.json"), "w"), indent=2)
    write_tex(os.path.join(a.out, "cost_table.tex"), res, device, a.steps, a.warmup, a.D)
    print(f"wrote {a.out}/cost.json and {a.out}/cost_table.tex")