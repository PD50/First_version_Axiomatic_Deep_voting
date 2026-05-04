#!/usr/bin/env python3
"""
Run Explainability Analysis for Set Transformer v3.

Computes how often the v3 model agrees with classical voting rules,
producing tables similar to Table 2 in Hornischer & Terzopoulou (JAIR 2025).

Usage:
    python run_explainability_v3.py

    # With custom checkpoint path:
    python run_explainability_v3.py --checkpoint ./results/exp3/SetTransformerV3/.../model.pth

    # Quick test (fewer profiles):
    python run_explainability_v3.py --eval-size 1000

    # Full analysis with all rules (including slow ones like Slater):
    python run_explainability_v3.py --all-rules

Designed for Apple M2 (8-core, 16GB RAM).
Expected runtime:
  - 1,000 profiles, fast rules: ~5–10 minutes
  - 10,000 profiles, fast rules: ~45–90 minutes
  - 10,000 profiles, all rules:  ~2–4 hours
"""

import os
import sys
import glob
import argparse

# Ensure correct directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from explainability_v3 import run_explainability_analysis


def find_latest_checkpoint(base_dir="./results/exp3/SetTransformerV3"):
    """Find the most recent v3 model checkpoint."""
    pattern = os.path.join(base_dir, "*/model.pth")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None
    # Sort by modification time (most recent first)
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    return checkpoints[0]


def main():
    parser = argparse.ArgumentParser(
        description="Explainability Analysis: v3 Set Transformer vs Classical Voting Rules"
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to model.pth checkpoint. If not provided, uses the latest.'
    )
    parser.add_argument(
        '--eval-size', type=int, default=10000,
        help='Number of profiles to evaluate on (default: 10000).'
    )
    parser.add_argument(
        '--n-best', type=int, default=6,
        help='Number of top rules for cross-comparison table (default: 6).'
    )
    parser.add_argument(
        '--all-rules', action='store_true',
        help='Use all rules including slow ones (Slater, etc).'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory for results. Defaults to checkpoint directory.'
    )
    args = parser.parse_args()

    # Find checkpoint
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = find_latest_checkpoint()
        if checkpoint is None:
            print("ERROR: No v3 model checkpoint found.")
            print("  Train a model first with: python run_exp3_settransformer_v3.py")
            print("  Or specify a checkpoint:  python run_explainability_v3.py "
                  "--checkpoint /path/to/model.pth")
            sys.exit(1)
        print(f"Auto-detected latest checkpoint: {checkpoint}")

    if not os.path.exists(checkpoint):
        print(f"ERROR: Checkpoint not found: {checkpoint}")
        sys.exit(1)

    # Run analysis
    results, results_path = run_explainability_analysis(
        checkpoint_path=checkpoint,
        eval_dataset_size=args.eval_size,
        n_best_rules=args.n_best,
        use_all_rules=args.all_rules,
        output_dir=args.output_dir,
    )

    print(f"\n{'='*70}")
    print(f"  ANALYSIS COMPLETE")
    print(f"  Results saved to: {results_path}")
    print(f"{'='*70}")
    print(f"\n  Top-{args.n_best} most similar rules (by identity accuracy):")
    for i, rule in enumerate(results['top_rules'], 1):
        accu = results['model_vs_rules'][rule]['identity_accu']
        print(f"    {i}. {rule}: {accu:.1f}%")


if __name__ == "__main__":
    main()