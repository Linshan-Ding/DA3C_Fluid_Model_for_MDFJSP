# DA3C Fluid Model for MDFJSP

Code for the paper: **Fluid-informed hierarchical dual-policy reinforcement learning for dynamic flexible job-shop scheduling with job multiplicity** (DA3C).

The method embeds a dynamic fluid model into a discrete-event MDFJSP environment. Fluid-derived parameters inform both the state representation and the candidate scheduling rules, and two conditionally connected policy networks (operation-rule policy + machine-rule policy) are trained with an asynchronous advantage double actor-critic algorithm (DA3C) to minimize total tardiness.

## Project Structure

```
agents/                  Main algorithm code (DA3C: dual policy networks + critic, async workers)
environments/            Discrete-event MDFJSP simulation environment
  ├── class_FJSP.py      Base FJSP classes + dynamic fluid model (LP solved at each order arrival)
  ├── SO_DFJSP.py        RL environment (20-dim state, 6 operation rules x 5 machine rules)
  ├── Instance_generate.py  Random dynamic instance generator (DDT, M, S)
  └── solvers.py         Fluid LP solver backend (docplex, scipy fallback)
experiments/             Supplementary experiment entry points (see below)
data/                    The 27 benchmark instances used in the paper (DDT{0.5,1.0,1.5}_M{10,15,20}_S{1,3,5})
data_generalization/     Out-of-distribution instances for the scale-generalization experiment
result/
  ├── models/            Trained checkpoints (actor_task_model.ckpt, actor_machine_model.ckpt)
  ├── csv/               All experiment outputs (raw + summary CSV, ready for the paper)
  └── latex/             LaTeX table fragments generated from the summary CSVs
utilities/               Helper classes (config, CSV logging, plotting)
```

The refactor described in `EXPERIMENT_REFACTOR_PLAN.md` has been applied: package imports match the directory layout, all paths are repo-relative, visdom is optional (`--visdom`), and the fluid LP automatically falls back to `scipy.linprog` when docplex/CPLEX is unavailable.

## Requirements

```bash
Python >= 3.8
torch  >= 1.10
numpy, scipy, pandas
docplex        # optional: IBM CPLEX backend for the fluid LP (falls back to scipy.linprog)
visdom         # optional: live training curves (disabled by default, use --visdom)
```

Install with:

```bash
pip install -r requirements.txt
```

## Basic Usage

**Train DA3C from scratch** (a new random instance is generated every episode):

```bash
python -m agents.DA3C_double_actor --episodes 2000 --seed 0 --result-dir result/models/v0_full
```

**Evaluate a trained policy on one benchmark instance** (30 independent runs):

```bash
python -m experiments.run_retest_27 --instance DDT1.0_M15_S3 --runs 30
```

## Supplementary Experiments

The four experiments below reproduce every table added during the paper revision. Each script writes a `*_raw.csv` (one row per run) and a `*_summary.csv` (one row per instance/method) into `result/csv/`; the summary files are the direct data sources for the LaTeX tables.

### 1. Ablation of the fluid-informed components

Four variants isolate the two injection paths of fluid knowledge:

| Variant | Fluid state features | Fluid rules | Flag |
|---|---|---|---|
| DA3C-Full | yes | yes | `--variant full` |
| DA3C-NS   | no (GAP features zeroed) | yes | `--variant ns` |
| DA3C-NR   | yes | no (rules degrade to non-fluid keys) | `--variant nr` |
| DA3C-NF   | no | no (fluid LP never solved) | `--variant nf` |

```bash
# train each variant (same hyper-parameters as the paper), then evaluate on all 27 instances
python -m experiments.run_ablation --variant full --phase train
python -m experiments.run_ablation --variant full --phase eval --runs 30
# repeat for ns / nr / nf
```

Outputs: `result/csv/ablation_raw.csv`, `result/csv/ablation_summary.csv` (columns `DDT,M,S,variant,mean,std,n_runs`), `result/csv/ablation_wilcoxon.csv`.

### 2. Scale generalization (out-of-distribution instances)

Deploys the trained DA3C-Full policy **without retraining** on larger instances (M∈{25,30,40}, S∈{8,10}), against EDD+SPT, URG+SPT and Random dispatching baselines (run in the non-fluid environment, i.e. plain rule versions):

```bash
python -m experiments.run_generalization --generate   # once: create data_generalization/
python -m experiments.run_generalization --runs 30
```

Outputs: `result/csv/generalization_raw.csv`, `result/csv/generalization_summary.csv` (columns `DDT,M,S,method,mean,std,n_runs`).

### 3. Fluid-LP online solve time

Records the wall-clock time of every fluid-LP solve triggered by an order arrival, across all 27 + 18 instances:

```bash
python -m experiments.run_fluid_lp_time --runs 10
```

Outputs: `result/csv/fluid_lp_time_raw.csv`, `result/csv/fluid_lp_time_summary.csv` (columns `DDT,M,S,mean_ms,p95_ms,max_ms,n_solves`).

### 4. Re-test on the 27 paper instances

Regenerates the authoritative DA3C column for Tables 3–6 of the paper:

```bash
python -m experiments.run_retest_27 --all --runs 30
```

Output: `result/csv/da3c_27instances.csv` (columns `DDT,M,S,mean,std,n_runs,decision_time_ms_mean`).

### Generate LaTeX tables

```bash
python -m experiments.csv_to_latex --input result/csv/ablation_summary.csv --output result/latex/table_ablation.tex
```

Produces a booktabs `tabular` fragment (row-wise best values bolded) that can be `\input{}` directly in the manuscript.

## Reproducibility Notes

- All randomness (`random`, `numpy`, `torch`) is controlled by `--seed`; run *i* of an evaluation uses `seed + i`.
- The paper's reward corresponds to `function_selected = 1` in `environments/SO_DFJSP.py` (tardiness-difference reward); it is the locked default.
- The fluid LP maximizes the minimum processing-rate/workload ratio (equivalent to minimizing the maximum fluid completion time); with docplex unavailable the scipy fallback solves the same LP after the standard max–min linearization.

## Citation

If this code helps you, please cite:

```
@article{ding2024multi,
  title={Multi-policy deep reinforcement learning for multi-objective multiplicity flexible job shop scheduling},
  author={Ding, Linshan and Guan, Zailin and Rauf, Mudassar and Yue, Lei},
  journal={Swarm and Evolutionary Computation},
  volume={87},
  pages={101550},
  year={2024},
  publisher={Elsevier}
}
```

## Contact

If you have any questions, feel free to contact us:

- Email1: linshan_ding@hust.edu.cn
- Email2: linshandingzz@gmail.com
