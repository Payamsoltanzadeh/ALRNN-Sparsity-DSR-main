"""
Task 3 Dropout Bug Fix Verification
====================================
Runs 5 seeds × 2 conditions (B, C) = 10 runs WITH the model.train() fix.
Results saved to experiments/task3_bugfix_verify/ (separate from original results).
Compare these against the original task3_stats results to verify the fix
doesn't change the B vs C comparison significantly.
"""

import sys
import os

# Patch the output directory in task3_statistical_runs before importing
script_dir = os.path.dirname(os.path.abspath(__file__))

# Import everything from the main script
sys.path.insert(0, script_dir)
os.chdir(script_dir)

import numpy as np
import json
import time
from task3_statistical_runs import (
    CONDITIONS, TRAIN_CONFIG, run_single, aggregate_results
)

def main():
    output_dir = os.path.join(script_dir, 'experiments', 'task3_bugfix_verify')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  DROPOUT BUG FIX VERIFICATION")
    print("  10 runs: Conditions B & C × Seeds 20-24")
    print(f"  Output: {output_dir}")
    print("  Fix: model.train() restored after evaluation")
    print("=" * 70)

    # Load data
    print("\nLoading Lorenz63 data...")
    X_train = np.load("lorenz63_train.npy").astype(np.float32)[500:]
    X_test = np.load("lorenz63_test.npy").astype(np.float32)[500:]
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    conditions = ['B', 'C']
    seeds = list(range(20, 25))  # 5 new seeds
    runs = [(c, s) for c in conditions for s in seeds]
    total = len(runs)

    # Skip already completed
    remaining = []
    for c, s in runs:
        result_path = os.path.join(output_dir, f"cond_{c}_seed_{s:02d}.json")
        if os.path.exists(result_path):
            print(f"  SKIP cond={c} seed={s} (already exists)")
        else:
            remaining.append((c, s))

    print(f"\n  {len(remaining)} runs remaining (of {total} total)")
    print(f"  Estimated time: ~{len(remaining) * 40} minutes\n")

    t_start = time.time()
    for i, (cond_key, seed) in enumerate(remaining):
        print(f"\n{'─' * 60}")
        print(f"  [{i+1}/{len(remaining)}]  Condition {cond_key} "
              f"({CONDITIONS[cond_key]['label']}), Seed {seed}")
        print(f"{'─' * 60}")

        result = run_single(cond_key, seed, output_dir, X_train, X_test, verbose=True)

        elapsed = time.time() - t_start
        runs_done = i + 1
        runs_left = len(remaining) - runs_done
        eta = (elapsed / runs_done) * runs_left if runs_done > 0 else 0

        print(f"  Done: Dstsp={result['metrics']['Dstsp']:.3f}, "
              f"DH={result['metrics']['DH']:.3f}, "
              f"Sub={result['metrics']['subregions']}, "
              f"Time={result['train_time_seconds']:.0f}s")
        print(f"  Progress: {runs_done}/{len(remaining)} | "
              f"Elapsed: {elapsed/60:.0f}min | ETA: {eta/60:.0f}min")

    # Aggregate
    aggregate_results(output_dir)

    # Quick comparison with original results
    print("\n" + "=" * 70)
    print("  QUICK COMPARISON: Bug-fixed vs Original")
    print("=" * 70)

    orig_dir = os.path.join(script_dir, 'experiments', 'task3_stats')
    for cond in conditions:
        # Load original (seeds 0-19)
        orig_dstsp = []
        for s in range(20):
            fpath = os.path.join(orig_dir, f"cond_{cond}_seed_{s:02d}.json")
            if os.path.exists(fpath):
                with open(fpath) as f:
                    orig_dstsp.append(json.load(f)['metrics']['Dstsp'])

        # Load fixed (seeds 20-24)
        fixed_dstsp = []
        for s in seeds:
            fpath = os.path.join(output_dir, f"cond_{cond}_seed_{s:02d}.json")
            if os.path.exists(fpath):
                with open(fpath) as f:
                    fixed_dstsp.append(json.load(f)['metrics']['Dstsp'])

        if orig_dstsp and fixed_dstsp:
            print(f"\n  Condition {cond} ({CONDITIONS[cond]['label']}):")
            print(f"    Original (N={len(orig_dstsp)}, no fix):  "
                  f"Dstsp = {np.mean(orig_dstsp):.3f} ± {np.std(orig_dstsp):.3f}  "
                  f"median = {np.median(orig_dstsp):.3f}")
            print(f"    Fixed    (N={len(fixed_dstsp)}, with fix): "
                  f"Dstsp = {np.mean(fixed_dstsp):.3f} ± {np.std(fixed_dstsp):.3f}  "
                  f"median = {np.median(fixed_dstsp):.3f}")

    print("\n  If distributions are similar → original results validated.")
    print("  If very different → full re-run needed.")
    print("=" * 70)
    print("\n  ALL DONE!")


if __name__ == '__main__':
    main()
