# Augmented Latent RNNs (AL-RNNs) for Dynamical System Reconstruction

This repository contains a reproducible AL-RNN workflow for Lorenz-63 reconstruction, based on [Brenner et al., NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/40cf27290cc2bd98a428b567ba25075c-Paper-Conference.pdf), with added ablations and multi-seed validation.

## What was added in this project

### 1) No-sparsity vs full-state L1 vs activated L1 (exact behavior)
We compare three regimes by how the L1 term is applied to the latent state $z_t$:

- **No sparsity**: set `lambda_l1 = 0.0`. This removes the L1 term entirely, so the loss is just MSE.
- **Full-state sparsity (`l1_mode='full'`)**: L1 on all latent dimensions. The penalty is:
  $L_{L1} = \mathbb{E}[|z_t|]$ computed over all $M$ latent units.
- **Activation-gated sparsity (`l1_mode='activated'`)**: L1 only on the activated PWL units. The penalty is:
  $L_{L1} = \mathbb{E}[|\phi(z_{t,\,-P:})|]$ where $\phi$ is ReLU and only the last $P$ units are gated.

In code this maps to:
- `l1_mode='none'` + `lambda_l1=0.0` (no sparsity)
- `l1_mode='full'`  + `lambda_l1>0` (full-state)
- `l1_mode='activated'` + `lambda_l1>0` (activated-only)

In our repeated runs, activation-gated L1 is generally more stable and better preserves attractor structure while keeping spectral behavior competitive.

### 2) Two-phase training (warm-up)
For warm-up experiments, the model is trained with:
- **Epochs 0-499**: MSE only
- **Epochs 500-999**: MSE + L1

This behavior is controlled by `l1_warmup_epochs=500` in `task3_warmup_runs.py`.

### 3) Dropout train/eval bug fix
Evaluation uses `model.eval()` to disable dropout, but training must resume with `model.train()`. A fix is included and validated in `task3_bugfix_verify.py` to ensure dropout remains active during training epochs after intermediate evaluations.

### 4) P-complexity check
Additional runs compare `P=1` and `P=2`. In our logged runs, `P=1` is unstable, while `P=2` consistently supports richer region dynamics.

---

## Repository structure

### Main analysis notebook
- `alrnn_python/AL-RNN_tutorial.ipynb`: end-to-end exploratory workflow and figures.

### Python modules (`alrnn_python/`)
- `dataset.py`: sequence batching and data sampling utilities.
- `metrics.py`: metric implementations.
  - $D_{stsp}$: KL divergence on occupancy histograms in state space.
  - $D_H$: Hellinger distance on smoothed power spectra.
- `linear_region_functions.py`: linear-region/Jacobian utilities.

### Experiment scripts (`alrnn_python/`)
- `task3_statistical_runs.py`: 20-seed baseline statistical comparison.
- `task3_improved_runs.py`: improved training configuration runs.
- `task3_warmup_runs.py`: warm-up (delayed L1) experiments.
- `task3_bugfix_verify.py`: verifies dropout mode restoration after evaluation.

### Experiment outputs
- `alrnn_python/experiments/`: JSON summaries and model artifacts from runs.

---


## Notes

- Results depend on random seed and hardware/runtime settings.
- Reported conclusions in this repo refer to the included experiment outputs and scripts.