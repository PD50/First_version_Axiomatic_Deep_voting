#!/usr/bin/env python3
"""
Run Hyperparameter Grid Search for Set Transformer v2.

Two-phase approach:
  Phase 1: Coarse search over primary hyperparams (36 configs × 10k steps)
  Phase 2: Fine-tune secondary hyperparams around Phase 1 winner (select configs × 15k steps)

Intermediate setting: 20 voters, 4 alternatives, IC sampling.
Designed for Apple M2 (8-core, 16GB RAM).

Usage:
    # Phase 1 only (recommended first):
    python run_grid_search_v2.py --phase 1

    # Phase 2 (after reviewing Phase 1 results):
    python run_grid_search_v2.py --phase 2

    # Both phases sequentially:
    python run_grid_search_v2.py --phase both

    # Quick test (2 configs, 1000 steps):
    python run_grid_search_v2.py --phase test

    # Resume from where you left off (auto-detects latest run):
    python run_grid_search_v2.py --phase both --resume

    # Resume from a specific directory:
    python run_grid_search_v2.py --phase 1 --resume ./results/grid_search/phase1_2026-03-12_15-30-00

Expected runtime:
    Phase 1: ~2-3 hours (36 configs × ~3-5 min each)
    Phase 2: ~1-2 hours (select configs × ~5-7 min each)
"""

import os
import sys
import argparse
import json

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

# MPS fallback for custom ops
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from grid_search_v2 import run_grid_search, _config_to_str, load_completed_results


def run_phase1(resume_dir=None):
    """Phase 1: Coarse search over primary hyperparameters."""
    print("=" * 75)
    print("  PHASE 1: Coarse Hyperparameter Search")
    print("  Setting: 20 voters, 4 alternatives, IC")
    print("  Grid: learning_rate × d_model × n_enc_layers × dropout")
    print("  Configs: 3 × 2 × 3 × 2 = 36 runs @ 10k steps each")
    if resume_dir:
        print(f"  RESUMING from: {resume_dir}")
    print("=" * 75)

    results, location = run_grid_search(
        phase='phase1',
        max_num_voters=20,
        max_num_alternatives=4,
        num_gradient_steps=10000,
        eval_interval=5000,
        random_seed=42,
        save_dir='./results/grid_search',
        resume_dir=resume_dir,
    )
    return results, location


def run_phase2(winner_config, resume_dir=None):
    """Phase 2: Fine-tune secondary hyperparameters around Phase 1 winner."""
    print("\n" + "=" * 75)
    print("  PHASE 2: Fine-Tune Secondary Hyperparameters")
    print("  Setting: 20 voters, 4 alternatives, IC")
    print(f"  Base config: {_config_to_str(winner_config)}")
    print("  Grid: batch_size × n_heads × n_inducing × warmup_steps × weight_decay")
    print("  15k steps per config")
    if resume_dir:
        print(f"  RESUMING from: {resume_dir}")
    print("=" * 75)

    results, location = run_grid_search(
        phase='phase2',
        max_num_voters=20,
        max_num_alternatives=4,
        num_gradient_steps=15000,
        eval_interval=5000,
        random_seed=42,
        winner_config=winner_config,
        save_dir='./results/grid_search',
        resume_dir=resume_dir,
    )
    return results, location


def run_test():
    """Quick test: 2 configs, 1000 steps."""
    print("=" * 75)
    print("  TEST MODE: Quick validation (2 configs × 1000 steps)")
    print("=" * 75)

    # Override Phase 1 grid to just 2 configs
    import grid_search_v2 as gs

    # Monkey-patch the grid to a tiny test
    original_fn = gs.get_phase1_grid
    def test_grid():
        grid = {
            'learning_rate': [5e-4, 1e-3],
            'd_model':       [64],
            'n_enc_layers':  [2],
            'dropout':       [0.1],
        }
        fixed = {
            'n_heads':      4,
            'd_ff_ratio':   2,
            'n_inducing':   8,
            'batch_size':   32,
            'warmup_steps': 300,
            'weight_decay': 0.01,
            'grad_clip':    1.0,
        }
        return grid, fixed

    gs.get_phase1_grid = test_grid

    results, location = run_grid_search(
        phase='phase1',
        max_num_voters=20,
        max_num_alternatives=4,
        num_gradient_steps=1000,
        eval_interval=500,
        random_seed=42,
        save_dir='./results/grid_search_test',
    )

    # Restore
    gs.get_phase1_grid = original_fn
    return results, location


def find_latest_run(save_dir, phase_prefix):
    """Find the most recent run directory for a given phase."""
    if not os.path.exists(save_dir):
        return None
    dirs = [
        d for d in os.listdir(save_dir)
        if d.startswith(phase_prefix) and os.path.isdir(os.path.join(save_dir, d))
    ]
    if not dirs:
        return None
    dirs.sort(reverse=True)  # Timestamp-based names sort correctly
    latest = os.path.join(save_dir, dirs[0])
    # Check it actually has results
    if os.path.exists(os.path.join(latest, "results_partial.json")) or \
       os.path.exists(os.path.join(latest, "results_final.json")):
        return latest
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Hyperparameter Grid Search for Set Transformer v2'
    )
    parser.add_argument(
        '--phase', type=str, default='1',
        choices=['1', '2', 'both', 'test'],
        help='Which phase to run: 1, 2, both, or test'
    )
    parser.add_argument(
        '--phase1-results', type=str, default=None,
        help='Path to Phase 1 results_final.json (required for --phase 2)'
    )
    parser.add_argument(
        '--resume', type=str, nargs='?', const='auto', default=None,
        help='Resume from a previous run. Pass a directory path, or just '
             '--resume to auto-detect the latest run for the current phase.'
    )
    args = parser.parse_args()

    if args.phase == 'test':
        results, location = run_test()
        print(f"\n✓ Test complete. Results at: {location}")
        return

    # Resolve resume directory
    resume_dir = None
    if args.resume is not None:
        if args.resume == 'auto':
            # Auto-detect latest run for the relevant phase
            phase_prefix = 'phase1' if args.phase in ['1', 'both'] else 'phase2'
            resume_dir = find_latest_run('./results/grid_search', phase_prefix)
            if resume_dir:
                print(f"\n  Auto-detected previous run: {resume_dir}")
            else:
                print(f"\n  No previous {phase_prefix} run found to resume. Starting fresh.")
        else:
            resume_dir = args.resume
            if not os.path.isdir(resume_dir):
                print(f"Error: Resume directory not found: {resume_dir}")
                return

    if args.phase in ['1', 'both']:
        p1_results, p1_location = run_phase1(resume_dir=resume_dir)

        if not p1_results:
            print("Phase 1 produced no results. Exiting.")
            return

        winner = p1_results[0]
        print(f"\n{'='*75}")
        print(f"  ★ PHASE 1 WINNER")
        print(f"  Config: {_config_to_str(winner['config'])}")
        print(f"  Average axiom satisfaction: {winner['final_axioms']['Average']:.2f}%")
        print(f"  Results: {p1_location}/results_final.json")
        print(f"{'='*75}")

        if args.phase == 'both':
            p2_results, p2_location = run_phase2(winner['config'])
            if p2_results:
                final_winner = p2_results[0]
                print(f"\n{'='*75}")
                print(f"  ★★ OVERALL WINNER (Phase 2)")
                print(f"  Config: {_config_to_str(final_winner['config'])}")
                print(f"  Average axiom satisfaction: {final_winner['final_axioms']['Average']:.2f}%")
                print(f"  Results: {p2_location}/results_final.json")
                print(f"{'='*75}")
                print(f"\n  Next step: Run this config on the large setting")
                print(f"  (55 voters, 5 alternatives) with 20k gradient steps.")

    elif args.phase == '2':
        if args.phase1_results is None:
            print("Error: --phase1-results required for Phase 2.")
            print("Usage: python run_grid_search_v2.py --phase 2 --phase1-results path/to/results_final.json")
            return

        with open(args.phase1_results) as f:
            p1_data = json.load(f)

        winner_config = p1_data['winner']['config']
        print(f"Phase 1 winner: {_config_to_str(winner_config)}")

        p2_results, p2_location = run_phase2(winner_config, resume_dir=resume_dir)
        if p2_results:
            final_winner = p2_results[0]
            print(f"\n★ Phase 2 Winner: {_config_to_str(final_winner['config'])}")
            print(f"  Average: {final_winner['final_axioms']['Average']:.2f}%")


if __name__ == "__main__":
    main()