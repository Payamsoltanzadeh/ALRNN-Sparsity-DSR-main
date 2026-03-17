#!/usr/bin/env python3
"""
Task 3.5: Improved Runs — M=20, 1000 epochs, gradient clipping
===============================================================
Runs 5 seeds × 2 conditions (D, E) and saves results to
experiments/task3_improved/

Estimated: ~20 min/run × 10 runs = ~3.3 hours
Resume-safe: skips already-completed seeds.
"""

import os, sys, json, time, copy, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from collections import defaultdict

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

from dataset import TimeSeriesDataset
from linear_region_functions import convert_to_bits
from metrics import state_space_divergence_binning, power_spectrum_error

# ══════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════

class AL_RNN(nn.Module):
    def __init__(self, M, P, N, dropout_p=0.0):
        super().__init__()
        self.M, self.P, self.N = M, P, N
        self.dropout_p = dropout_p
        self.A, self.W, self.h = self._init_AWh()
        self.B = self._init_uniform((N, M))
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, z, mask=None):
        z_un = torch.clone(z)
        z_act = z.clone()
        z_act[:, -self.P:] = F.relu(z[:, -self.P:])
        out = self.A * z_un + z_act @ self.W.t() + self.h
        if self.training and self.dropout_p > 0:
            if mask is not None:
                out[:, self.N:] = out[:, self.N:] * mask
            else:
                out[:, self.N:] = self.dropout(out[:, self.N:])
        return out

    def _init_AWh(self):
        R = np.random.randn(self.M, self.M).astype(np.float32)
        K = R.T @ R / self.M + np.eye(self.M)
        lmax = np.max(np.abs(np.linalg.eigvals(K)))
        A = nn.Parameter(torch.diagonal(torch.tensor(K / lmax).float(), 0))
        W = nn.Parameter(torch.randn(self.M, self.M) * 0.01)
        h = nn.Parameter(torch.zeros(self.M))
        return A, W, h

    def _init_uniform(self, shape):
        t = torch.empty(*shape)
        r = 1 / math.sqrt(shape[0])
        nn.init.uniform_(t, -r, r)
        return nn.Parameter(t, requires_grad=True)


@torch.no_grad()
def predict_free(model, x, T):
    model.eval()
    b, N = x.size()
    Z = torch.empty(T, b, model.M, device=x.device)
    z = x @ model.B
    z[:, :N] = x
    for t in range(T):
        z = model(z, mask=None)
        Z[t] = z
    return Z.permute(1, 0, 2)


def teacher_force(z, x, N, alpha):
    z[:, :N] = alpha * x + (1 - alpha) * z[:, :N]
    return z


def predict_gtf(model, x, alpha, n_interleave):
    x_ = x.permute(1, 0, 2)
    T, b, dx = x_.size()
    Z = torch.empty(T, b, model.M, device=x.device)
    z = x_[0] @ model.B
    z = teacher_force(z, x_[0], model.N, alpha=1)
    mask = None
    if model.training and model.dropout_p > 0:
        kp = 1 - model.dropout_p
        mask = torch.bernoulli(torch.full((b, model.M - model.N), kp, device=x.device)) / kp
    for t in range(T):
        if (t % n_interleave == 0) and (t > 0):
            z = teacher_force(z, x_[t], model.N, alpha)
        z = model(z, mask=mask)
        Z[t] = z
    return Z.permute(1, 0, 2)


def train_improved(model, dataset, optimizer, scheduler, loss_fn,
                   num_epochs, alpha, n_interleave,
                   lambda_l1=0.0, l1_mode='activated',
                   batches_per_epoch=50, ssi=25, use_best_model=True,
                   grad_clip=10.0, verbose=True):
    """Training loop with gradient clipping."""
    model.train()
    best_model = copy.deepcopy(model)
    losses, klx, dh = [], [], []

    iterator = trange(num_epochs, desc="Training") if verbose else range(num_epochs)

    for e in iterator:
        epoch_losses = []
        for _ in range(batches_per_epoch):
            optimizer.zero_grad()
            x, y, s = dataset.sample_batch()
            z_hat = predict_gtf(model, x, alpha, n_interleave)

            if l1_mode == 'activated':
                l1_pen = torch.mean(torch.abs(F.relu(z_hat[:, :, -model.P:])))
            elif l1_mode == 'full':
                l1_pen = torch.mean(torch.abs(z_hat))
            else:
                l1_pen = 0.0

            loss = loss_fn(z_hat[:, :, :model.N], y) + lambda_l1 * l1_pen
            loss.backward()

            # Gradient clipping
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        if verbose and hasattr(iterator, 'set_postfix'):
            iterator.set_postfix(loss=avg_loss)

        if e % ssi == 0:
            with torch.no_grad():
                z_test = predict_free(model, dataset.X.clone().detach()[0:1, :], 10000)
                klx.append(state_space_divergence_binning(
                    z_test[0, :, :model.N], dataset.X.clone().detach()))
                dh.append(power_spectrum_error(
                    z_test[0, :, :model.N], dataset.X.clone().detach()[:10000, :]))
                if torch.argmin(torch.tensor(klx)) + 1 == len(klx):
                    best_model = copy.deepcopy(model)
            model.train()

    if use_best_model:
        model.load_state_dict(best_model.state_dict())
    return [losses, klx, dh]


# ══════════════════════════════════════════════════════════════════════
# CONDITIONS
# ══════════════════════════════════════════════════════════════════════

CONDITIONS = {
    'D': {'label': 'L1(activated) M=20', 'M': 20, 'P': 2, 'N': 3,
          'dropout_p': 0.0, 'lambda_l1': 1e-4, 'l1_mode': 'activated'},
    'E': {'label': 'L1(full) M=20',      'M': 20, 'P': 2, 'N': 3,
          'dropout_p': 0.0, 'lambda_l1': 1e-4, 'l1_mode': 'full'},
}

CFG = {
    'num_epochs': 1000, 'batch_size': 16, 'sequence_length': 200,
    'alpha': 1, 'n_interleave': 16, 'lr_start': 1e-3, 'lr_end': 1e-5,
    'batches_per_epoch': 50, 'ssi': 20, 'grad_clip': 10.0,
}

SEEDS = list(range(5))


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    output_dir = os.path.join(script_dir, 'experiments', 'task3_improved')
    os.makedirs(output_dir, exist_ok=True)

    X_train = np.load("lorenz63_train.npy").astype(np.float32)[500:]
    X_test  = np.load("lorenz63_test.npy").astype(np.float32)[500:]

    existing = [f for f in os.listdir(output_dir) if f.startswith('cond_') and f.endswith('.json')]
    print("=" * 70)
    print("  TASK 3.5: Improved Runs — M=20, 1000 epochs, grad_clip=10")
    print("=" * 70)
    print(f"  Conditions  : D (L1 activated, M=20), E (L1 full, M=20)")
    print(f"  Seeds       : {SEEDS}")
    print(f"  Epochs      : {CFG['num_epochs']} per run")
    print(f"  Grad clip   : {CFG['grad_clip']}")
    print(f"  Existing    : {len(existing)} results found")
    print(f"  Data        : train={X_train.shape}, test={X_test.shape}")
    print("=" * 70)

    total = len(SEEDS) * len(CONDITIONS)
    done_count = 0
    skip_count = 0

    for cond_key, cond in CONDITIONS.items():
        for seed in SEEDS:
            result_path = os.path.join(output_dir, f"cond_{cond_key}_seed_{seed:02d}.json")

            if os.path.exists(result_path):
                skip_count += 1
                print(f"  SKIP cond={cond_key} seed={seed} (already done)")
                continue

            done_count += 1
            print(f"\n{'─' * 70}")
            print(f"  [{skip_count + done_count}/{total}]  Condition {cond_key} "
                  f"({cond['label']}), Seed {seed}")
            print(f"{'─' * 70}")

            torch.manual_seed(seed)
            np.random.seed(seed)

            model = AL_RNN(M=cond['M'], P=cond['P'], N=cond['N'],
                           dropout_p=cond['dropout_p'])
            dataset = TimeSeriesDataset(X_train,
                                        sequence_length=CFG['sequence_length'],
                                        batch_size=CFG['batch_size'])
            optimizer = torch.optim.RAdam(model.parameters(), lr=CFG['lr_start'])
            gamma = np.exp(np.log(CFG['lr_end'] / CFG['lr_start']) / CFG['num_epochs'])
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
            loss_fn = nn.MSELoss()

            t0 = time.time()
            hist = train_improved(
                model, dataset, optimizer, scheduler, loss_fn,
                num_epochs=CFG['num_epochs'], alpha=CFG['alpha'],
                n_interleave=CFG['n_interleave'],
                lambda_l1=cond['lambda_l1'], l1_mode=cond['l1_mode'],
                batches_per_epoch=CFG['batches_per_epoch'],
                ssi=CFG['ssi'], use_best_model=True,
                grad_clip=CFG['grad_clip'], verbose=True,
            )
            elapsed = time.time() - t0

            # Evaluate
            model.eval()
            with torch.no_grad():
                x0 = torch.tensor(X_test[:1]).unsqueeze(0)
                orbit = predict_free(model, x0[:, 0, :], 11000)
                orbit = orbit.detach().numpy()[0][1000:, :]

            Dstsp = float(state_space_divergence_binning(
                torch.tensor(orbit[:, :cond['N']]),
                torch.tensor(X_test).unsqueeze(0)[0, :, :]))
            DH = float(power_spectrum_error(
                torch.tensor(orbit[:, :cond['N']]),
                torch.tensor(X_test[:10000])))

            pwl = orbit[:, -cond['P']:]
            bitcodes = convert_to_bits(pwl)
            unique_codes, counts = np.unique(bitcodes, axis=0, return_counts=True)  # axis=0: count unique ROW patterns
            subregions = len(unique_codes)

            pwl_relu = np.maximum(pwl, 0)
            pwl_mean = [float(np.mean(pwl_relu[:, j])) for j in range(cond['P'])]
            pwl_pct_zero = [float(np.mean(pwl_relu[:, j] == 0) * 100) for j in range(cond['P'])]

            result = {
                'condition': cond_key,
                'condition_label': cond['label'],
                'seed': seed,
                'settings': {
                    'M': cond['M'], 'P': cond['P'], 'N': cond['N'],
                    'dropout_p': cond['dropout_p'],
                    'lambda_l1': cond['lambda_l1'], 'l1_mode': cond['l1_mode'],
                    'num_epochs': CFG['num_epochs'], 'grad_clip': CFG['grad_clip'],
                },
                'metrics': {
                    'Dstsp': Dstsp, 'DH': DH, 'subregions': subregions,
                    'frequencies': sorted(counts.tolist(), reverse=True),
                    'final_loss': float(hist[0][-1]),
                },
                'pwl_stats': {'pwl_mean': pwl_mean, 'pwl_pct_zero': pwl_pct_zero},
                'train_time_seconds': round(elapsed, 1),
            }

            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)

            print(f"\n  ✓ Seed {seed}: Dstsp={Dstsp:.3f}, DH={DH:.3f}, "
                  f"Subregions={subregions}, Time={elapsed:.0f}s")

    # Summary
    print(f"\n{'=' * 70}")
    print("  TASK 3.5 FINAL SUMMARY")
    print(f"{'=' * 70}")

    all_done = [f for f in os.listdir(output_dir) if f.startswith('cond_') and f.endswith('.json')]
    results_all = []
    for fp in sorted(all_done):
        with open(os.path.join(output_dir, fp)) as f:
            results_all.append(json.load(f))

    groups = defaultdict(list)
    for r in results_all:
        groups[r['condition']].append(r)

    paper = {'Dstsp': 3.0, 'DH': 0.10, 'sub': 3}

    for ck in sorted(groups.keys()):
        runs = groups[ck]
        dstsp = [r['metrics']['Dstsp'] for r in runs]
        dh_v  = [r['metrics']['DH'] for r in runs]
        sub_v = [r['metrics']['subregions'] for r in runs]
        print(f"\n  {ck} ({runs[0]['condition_label']})  n={len(runs)}")
        print(f"    Dstsp      : {np.mean(dstsp):.3f} +/- {np.std(dstsp):.3f}  "
              f"(paper: {paper['Dstsp']:.1f})")
        print(f"    DH         : {np.mean(dh_v):.3f} +/- {np.std(dh_v):.3f}  "
              f"(paper: {paper['DH']:.2f})")
        print(f"    Subregions : {np.mean(sub_v):.1f} +/- {np.std(sub_v):.1f}  "
              f"(paper: {paper['sub']})")
        print(f"    Per-seed   : {[f'{d:.2f}' for d in dstsp]}")

    print(f"\n  Skipped: {skip_count}, Trained: {done_count}")
    print("=" * 70)
    print("  ALL DONE!")


if __name__ == '__main__':
    main()
