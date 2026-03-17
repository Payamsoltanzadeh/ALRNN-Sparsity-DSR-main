#!/usr/bin/env python3
"""
Task 3: Statistical Validation — Repeated Runs with Multiple Seeds
====================================================

This script trains the AL-RNN model multiple times per condition with different
random seeds and saves all metrics to JSON files for later analysis.

Conditions:
  A: M=40, dropout=10%, NO L1          (baseline)
  B: M=40, dropout=10%, L1(activated)  (targeted sparsity)
  C: M=40, dropout=10%, L1(full)       (global sparsity)

Usage:
  python task3_statistical_runs.py                    # Run all
  python task3_statistical_runs.py --seeds 0 5        # Run seeds 0-4 only
  python task3_statistical_runs.py --conditions B     # Run condition B only
  python task3_statistical_runs.py --resume            # Skip already-done runs

Output:
  experiments/task3_stats/cond_{A,B,C}_seed_{0-19}.json
  experiments/task3_stats/summary.json  (aggregate statistics)
"""

import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

# ── Ensure we can import local modules ──
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

from dataset import TimeSeriesDataset
from linear_region_functions import convert_to_bits
from metrics import state_space_divergence_binning, power_spectrum_error

# ═══════════════════════════════════════════════════════════════════════
# MODEL DEFINITION (identical to notebook)
# ═══════════════════════════════════════════════════════════════════════

class AL_RNN(nn.Module):
    def __init__(self, M, P, N, dropout_p=0.0):
        super(AL_RNN, self).__init__()
        self.M = M
        self.P = P
        self.N = N
        self.dropout_p = dropout_p
        self.A, self.W, self.h = self.initialize_AWh_random()
        self.B = self.init_uniform((self.N, self.M))
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, z, mask=None):
        z_unactivated = torch.clone(z)
        z_activated = z.clone()
        z_activated[:, -self.P:] = F.relu(z[:, -self.P:])
        output = self.A * z_unactivated + z_activated @ self.W.t() + self.h
        if self.training and self.dropout_p > 0:
            if mask is not None:
                output[:, self.N:] = output[:, self.N:] * mask
            else:
                output[:, self.N:] = self.dropout(output[:, self.N:])
        return output

    def initialize_AWh_random(self):
        A = nn.Parameter(torch.diagonal(self.normalized_positive_definite(self.M), 0))
        W = nn.Parameter(torch.randn(self.M, self.M) * 0.01)
        h = nn.Parameter(torch.zeros(self.M))
        return A, W, h

    def normalized_positive_definite(self, M):
        R = np.random.randn(M, M).astype(np.float32)
        K = np.matmul(R.T, R) / M + np.eye(M)
        eigenvalues = np.linalg.eigvals(K)
        lambda_max = np.max(np.abs(eigenvalues))
        return torch.tensor(K / lambda_max).float()

    def init_uniform(self, shape):
        tensor = torch.empty(*shape)
        r = 1 / math.sqrt(shape[0])
        torch.nn.init.uniform_(tensor, -r, r)
        return nn.Parameter(tensor, requires_grad=True)


# ═══════════════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS (identical to notebook)
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_free_sequence(model, x, T):
    model.eval()
    b, N = x.size()
    Z = torch.empty(size=(T, b, model.M), device=x.device)
    z = x @ model.B
    z[:, 0:N] = x
    for t in range(T):
        z = model(z, mask=None)
        Z[t] = z
    return Z.permute(1, 0, 2)


def teacher_force(z, x, N, alpha):
    z[:, :N] = alpha * x + (1 - alpha) * z[:, :N]
    return z


def predict_sequence_using_gtf(model, x, alpha, n_interleave):
    x_ = x.permute(1, 0, 2)
    T, b, dx = x_.size()
    Z = torch.empty(size=(T, b, model.M), device=x.device)
    z = x_[0] @ model.B
    z = teacher_force(z, x_[0], model.N, alpha=1)
    mask = None
    if model.training and model.dropout_p > 0:
        keep_prob = 1 - model.dropout_p
        mask = (torch.bernoulli(
            torch.full((b, model.M - model.N), keep_prob, device=x.device)
        ) / keep_prob)
    for t in range(T):
        if (t % n_interleave == 0) and (t > 0):
            z = teacher_force(z, x_[t], model.N, alpha)
        z = model(z, mask=mask)
        Z[t] = z
    return Z.permute(1, 0, 2)


def train_sh(model, dataset, optimizer, scheduler, loss_fn,
             num_epochs, alpha, n_interleave,
             lambda_l1=0.0, l1_mode='activated',
             batches_per_epoch=50, ssi=25, use_best_model=True,
             verbose=True):
    model.train()
    best_model = copy.deepcopy(model)
    losses, klx, dh = [], [], []

    if verbose:
        iterator = trange(num_epochs, desc="Training")
    else:
        iterator = range(num_epochs)

    for e in iterator:
        epoch_losses = []
        for _ in range(batches_per_epoch):
            optimizer.zero_grad()
            x, y, s = dataset.sample_batch()
            z_hat = predict_sequence_using_gtf(model, x, alpha, n_interleave)

            if l1_mode == 'activated':
                l1_penalty = torch.mean(torch.abs(F.relu(z_hat[:, :, -model.P:])))
            elif l1_mode == 'full':
                l1_penalty = torch.mean(torch.abs(z_hat))
            else:
                l1_penalty = 0.0

            mse_loss = loss_fn(z_hat[:, :, :model.N], y)
            loss = mse_loss + lambda_l1 * l1_penalty
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()
        average_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(average_epoch_loss)

        if verbose and hasattr(iterator, 'set_postfix'):
            iterator.set_postfix(loss=average_epoch_loss)

        if e % ssi == 0:
            with torch.no_grad():
                z_test = predict_free_sequence(
                    model, dataset.X.clone().detach()[0:1, :], 10000
                )
                klx.append(state_space_divergence_binning(
                    z_test[0, :, 0:model.N], dataset.X.clone().detach()
                ))
                dh.append(power_spectrum_error(
                    z_test[0, :, 0:model.N], dataset.X.clone().detach()[0:10000, :]
                ))
                if torch.argmin(torch.tensor(klx)) + 1 == len(klx):
                    best_model = copy.deepcopy(model)
            # BUG FIX: Restore training mode after evaluation
            # predict_free_sequence() calls model.eval(), which disables dropout.
            # Without this line, dropout stays OFF for all subsequent epochs.
            model.train()

    if use_best_model:
        model.load_state_dict(best_model.state_dict())

    return [losses, klx, dh]


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENTAL CONDITIONS
# ═══════════════════════════════════════════════════════════════════════

CONDITIONS = {
    'A': {
        'label': 'No L1 (dropout only)',
        'M': 40, 'P': 2, 'N': 3,
        'dropout_p': 0.10,
        'lambda_l1': 0.0,
        'l1_mode': 'none',
    },
    'B': {
        'label': 'L1(activated)',
        'M': 40, 'P': 2, 'N': 3,
        'dropout_p': 0.10,
        'lambda_l1': 1e-4,
        'l1_mode': 'activated',
    },
    'C': {
        'label': 'L1(full)',
        'M': 40, 'P': 2, 'N': 3,
        'dropout_p': 0.10,
        'lambda_l1': 1e-4,
        'l1_mode': 'full',
    },
}

# Training hyperparameters (fixed across all runs)
TRAIN_CONFIG = {
    'num_epochs': 500,
    'batch_size': 16,
    'sequence_length': 200,
    'alpha': 1,
    'n_interleave': 16,
    'lr_start': 1e-3,
    'lr_end': 1e-5,
    'batches_per_epoch': 50,
    'ssi': 20,  # evaluate every 20 epochs
}

# Evaluation parameters
EVAL_CONFIG = {
    'T_gen': 10000,
    'T_transient': 1000,
}


# ═══════════════════════════════════════════════════════════════════════
# SINGLE RUN
# ═══════════════════════════════════════════════════════════════════════

def run_single(condition_key, seed, output_dir, X_train, X_test, verbose=True):
    """
    Train one model with one seed and one condition. Save results to JSON.
    Returns the result dict.
    """
    cond = CONDITIONS[condition_key]
    cfg = TRAIN_CONFIG

    # Output path
    result_path = os.path.join(output_dir, f"cond_{condition_key}_seed_{seed:02d}.json")

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create model
    model = AL_RNN(M=cond['M'], P=cond['P'], N=cond['N'], dropout_p=cond['dropout_p'])

    # Create dataset and optimizer
    dataset = TimeSeriesDataset(X_train, sequence_length=cfg['sequence_length'],
                                 batch_size=cfg['batch_size'])
    optimizer = torch.optim.RAdam(model.parameters(), lr=cfg['lr_start'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=np.exp(np.log(cfg['lr_end'] / cfg['lr_start']) / cfg['num_epochs'])
    )
    loss_fn = nn.MSELoss()

    # ── Train ──
    t0 = time.time()
    metrics_history = train_sh(
        model, dataset, optimizer, scheduler, loss_fn,
        num_epochs=cfg['num_epochs'],
        alpha=cfg['alpha'],
        n_interleave=cfg['n_interleave'],
        lambda_l1=cond['lambda_l1'],
        l1_mode=cond['l1_mode'],
        batches_per_epoch=cfg['batches_per_epoch'],
        ssi=cfg['ssi'],
        use_best_model=True,
        verbose=verbose,
    )
    train_time = time.time() - t0

    # ── Evaluate ──
    model.eval()
    T_gen = EVAL_CONFIG['T_gen']
    T_r = EVAL_CONFIG['T_transient']

    with torch.no_grad():
        x0 = torch.tensor(X_test[:1]).unsqueeze(0)
        orbit = predict_free_sequence(model, x0[:, 0, :], T_gen + T_r)
        orbit = orbit.detach().numpy()[0][T_r:, :]  # (T_gen, M)

    X_test_torch = torch.tensor(X_test[:]).unsqueeze(0)
    Dstsp = state_space_divergence_binning(
        torch.tensor(orbit[:, :cond['N']]), X_test_torch[0, :, :]
    )
    DH = power_spectrum_error(
        torch.tensor(orbit[:, :cond['N']]), X_test_torch[0, :T_gen, :]
    )

    # Bitcodes & subregions
    pwl = orbit[:, -cond['P']:]
    bitcodes = convert_to_bits(pwl)
    unique_codes, counts = np.unique(bitcodes, axis=0, return_counts=True)  # axis=0: count unique ROW patterns
    subregions = len(unique_codes)
    frequencies = sorted(counts.tolist(), reverse=True)

    # PWL statistics
    pwl_relu = np.maximum(pwl, 0)
    pwl_mean = [float(np.mean(pwl_relu[:, j])) for j in range(cond['P'])]
    pwl_pct_zero = [float(np.mean(pwl_relu[:, j] == 0) * 100) for j in range(cond['P'])]

    # Final loss
    final_loss = float(metrics_history[0][-1])

    # ── Build result dict ──
    result = {
        'condition': condition_key,
        'condition_label': cond['label'],
        'seed': seed,
        'settings': {
            'M': cond['M'], 'P': cond['P'], 'N': cond['N'],
            'dropout_p': cond['dropout_p'],
            'lambda_l1': cond['lambda_l1'],
            'l1_mode': cond['l1_mode'],
            'num_epochs': cfg['num_epochs'],
        },
        'metrics': {
            'Dstsp': float(Dstsp),
            'DH': float(DH),
            'subregions': subregions,
            'frequencies': frequencies,
            'final_loss': final_loss,
        },
        'pwl_stats': {
            'pwl_mean': pwl_mean,
            'pwl_pct_zero': pwl_pct_zero,
        },
        'train_time_seconds': round(train_time, 1),
    }

    # Save
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    return result


# ═══════════════════════════════════════════════════════════════════════
# AGGREGATE SUMMARY
# ═══════════════════════════════════════════════════════════════════════

def aggregate_results(output_dir):
    """Load all result JSONs and compute aggregate statistics."""
    all_results = []
    for fname in sorted(os.listdir(output_dir)):
        if fname.startswith('cond_') and fname.endswith('.json'):
            with open(os.path.join(output_dir, fname), 'r') as f:
                all_results.append(json.load(f))

    if not all_results:
        print("No results found!")
        return

    # Group by condition
    from collections import defaultdict
    groups = defaultdict(list)
    for r in all_results:
        groups[r['condition']].append(r)

    summary = {}
    print("\n" + "=" * 80)
    print("  TASK 3: AGGREGATE STATISTICS")
    print("=" * 80)

    for cond_key in sorted(groups.keys()):
        runs = groups[cond_key]
        n = len(runs)
        label = runs[0]['condition_label']

        dstsp_vals = [r['metrics']['Dstsp'] for r in runs]
        dh_vals = [r['metrics']['DH'] for r in runs]
        sub_vals = [r['metrics']['subregions'] for r in runs]
        pz0_vals = [r['pwl_stats']['pwl_pct_zero'][0] for r in runs]
        pz1_vals = [r['pwl_stats']['pwl_pct_zero'][1] for r in runs]
        loss_vals = [r['metrics']['final_loss'] for r in runs]
        time_vals = [r['train_time_seconds'] for r in runs]

        stats = {
            'condition': cond_key,
            'label': label,
            'n_runs': n,
            'Dstsp': {'mean': np.mean(dstsp_vals), 'std': np.std(dstsp_vals),
                      'min': np.min(dstsp_vals), 'max': np.max(dstsp_vals),
                      'median': np.median(dstsp_vals)},
            'DH': {'mean': np.mean(dh_vals), 'std': np.std(dh_vals),
                   'min': np.min(dh_vals), 'max': np.max(dh_vals),
                   'median': np.median(dh_vals)},
            'subregions': {'mean': np.mean(sub_vals), 'std': np.std(sub_vals),
                           'values': sub_vals},
            'pwl0_pct_zero': {'mean': np.mean(pz0_vals), 'std': np.std(pz0_vals)},
            'pwl1_pct_zero': {'mean': np.mean(pz1_vals), 'std': np.std(pz1_vals)},
            'final_loss': {'mean': np.mean(loss_vals), 'std': np.std(loss_vals)},
            'train_time': {'mean': np.mean(time_vals), 'total': np.sum(time_vals)},
        }
        summary[cond_key] = stats

        print(f"\n  Condition {cond_key}: {label}  (n={n} runs)")
        print(f"  {'─' * 60}")
        print(f"    Dstsp       : {stats['Dstsp']['mean']:.3f} ± {stats['Dstsp']['std']:.3f}  "
              f"(min={stats['Dstsp']['min']:.3f}, max={stats['Dstsp']['max']:.3f})")
        print(f"    DH          : {stats['DH']['mean']:.3f} ± {stats['DH']['std']:.3f}  "
              f"(min={stats['DH']['min']:.3f}, max={stats['DH']['max']:.3f})")
        print(f"    Subregions  : {stats['subregions']['mean']:.1f} ± {stats['subregions']['std']:.1f}  "
              f"values={sub_vals}")
        print(f"    PWL0 % zero : {stats['pwl0_pct_zero']['mean']:.1f}% ± {stats['pwl0_pct_zero']['std']:.1f}%")
        print(f"    PWL1 % zero : {stats['pwl1_pct_zero']['mean']:.1f}% ± {stats['pwl1_pct_zero']['std']:.1f}%")
        print(f"    Avg time    : {stats['train_time']['mean']:.0f}s per run")

    # Convert numpy types for JSON serialization
    def to_serializable(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj

    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(to_serializable(summary), f, indent=2)

    print(f"\n  Summary saved to {summary_path}")
    print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Task 3: Statistical Runs')
    parser.add_argument('--seeds', nargs=2, type=int, default=[0, 20],
                        help='Seed range [start, end). Default: 0 20')
    parser.add_argument('--conditions', nargs='+', default=['A', 'B', 'C'],
                        help='Conditions to run. Default: A B C')
    parser.add_argument('--resume', action='store_true',
                        help='Skip runs whose result JSON already exists')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress per-epoch progress bars')
    parser.add_argument('--aggregate-only', action='store_true',
                        help='Only aggregate existing results, do not train')
    args = parser.parse_args()

    # Output directory
    output_dir = os.path.join(script_dir, 'experiments', 'task3_stats')
    os.makedirs(output_dir, exist_ok=True)

    if args.aggregate_only:
        aggregate_results(output_dir)
        return

    # Load data
    print("Loading Lorenz63 data...")
    X_train = np.load("lorenz63_train.npy").astype(np.float32)[500:]
    X_test = np.load("lorenz63_test.npy").astype(np.float32)[500:]
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # Build run list
    seeds = list(range(args.seeds[0], args.seeds[1]))
    conditions = args.conditions
    runs = [(c, s) for c in conditions for s in seeds]
    total = len(runs)

    # Filter already done
    if args.resume:
        remaining = []
        for c, s in runs:
            result_path = os.path.join(output_dir, f"cond_{c}_seed_{s:02d}.json")
            if os.path.exists(result_path):
                print(f"  SKIP cond={c} seed={s} (already exists)")
            else:
                remaining.append((c, s))
        runs = remaining

    print(f"\n{'=' * 60}")
    print(f"  TASK 3: {len(runs)} runs to execute (of {total} total)")
    print(f"  Conditions: {conditions}")
    print(f"  Seeds: {seeds[0]}–{seeds[-1]}")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 60}\n")

    # Run
    for i, (cond_key, seed) in enumerate(runs):
        print(f"\n{'─' * 60}")
        print(f"  [{i+1}/{len(runs)}]  Condition {cond_key} ({CONDITIONS[cond_key]['label']}), Seed {seed}")
        print(f"{'─' * 60}")

        result = run_single(
            cond_key, seed, output_dir, X_train, X_test,
            verbose=not args.quiet
        )

        print(f"  ✓ Dstsp={result['metrics']['Dstsp']:.3f}, "
              f"DH={result['metrics']['DH']:.3f}, "
              f"Subregions={result['metrics']['subregions']}, "
              f"Time={result['train_time_seconds']:.0f}s")

    # Aggregate
    aggregate_results(output_dir)
    print("\n  ALL DONE!")


if __name__ == '__main__':
    main()
