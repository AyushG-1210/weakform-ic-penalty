# Same weight, different loss: the weak-form discretization sets the effective initial-condition weight in physics-informed training

> Ayush Gouda, Swaraag Hebbar N, Aditya Prakash, Sudheendra B S
> RV Institute of Technology and Management, Bengaluru

Code and logged results for the paper (`DigTwin (11).pdf`). It studies what happens when the initial condition clashes with a hard Dirichlet boundary in physics-informed training, for a strong-form and a weak-form (variational) loss, on the transient heat equation (1D, 2D) and advection–diffusion, all with exact solutions.

## Layout

```
heat_drift.py           models, losses, exact solutions, training, κ measurement; every experiment is a mode of this script
run.sh                  batch driver: weak-sweep gaps, annealing, 2D strong sweep, as one pool (GPU recommended)
run_final.py            normalization, test-function order, advection–diffusion (stages A, B, C)
run_mesh.py             weak-form mesh study (4/8/32 elements)
cn_baseline.py          classical Crank–Nicolson / Rannacher baseline (CPU, no training)
linear_model_check.py   Proposition 1 check on a model that is linear in its parameters (CPU, no training)
anneal_weights.py       prints where the annealed IC weight ended up, per run
make_figs.py            figures, tables and per_seed.csv from the stored JSON (CPU, no torch)
pull_heat_tables.py     the appendix .tex tables from the stored JSON
pull_paper_values.py    prints the numbers the text quotes and writes appendix tables (see note below)
heat_runs/              one folder per experiment (ckpt/, logs/, results/), plus heat_figs/, advdiff_figs/, cn_baseline/, linear_check/
results/                generated figures, tables and per_seed.csv
DigTwin (11).pdf        the paper
```

Run everything from this folder (the repo root).

## Reproducing

```
pip install -r requirements.txt
python heat_drift.py --all --jobs 8          # main arms (see the usage block at the top of heat_drift.py for the other modes)
python heat_drift.py --summary               # table from finished runs
python cn_baseline.py                        # classical baseline
python linear_model_check.py                 # linear-model check of Proposition 1
python make_figs.py                          # figures + tables from the stored JSON
python pull_heat_tables.py                   # appendix tables
```

The stored JSON under `heat_runs/*/results/` is enough to regenerate every figure and table without retraining. Finished runs are skipped on restart.

## Notes

- `make_figs.py` and `pull_paper_values.py` came from a shared working directory and still contain an optional electrochemical section that reads `electrochem/results/results.json`. That file is not part of this repo; the section is skipped with a "not found" message and nothing in the paper depends on it. `pull_paper_values.py` writes `app_ec_*.tex` only when that data exists.
- `make_figs.py` writes to `results/paper_figs/`. The figures already in `results/` were copied there flat.
- Training uses double precision, Adam only, seeds 42–44 (2D weak-form defaults use 42–51).
