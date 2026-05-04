#!/usr/bin/env python3
"""
Run Experiment 3 with Enhanced Set Transformer v2.

Usage:
    python run_exp3_settransformer_v2.py

Designed for Apple M2 (8-core, 16GB RAM).
Expected runtime: ~25-40 minutes for 20,000 gradient steps at 55 voters, 5 alternatives.
"""

import os
import sys

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

# MPS fallback for custom ops
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from exp3_settransformer_v2 import run_experiment3_settransformer_v2


def main():
    print("=" * 65)
    print("  Enhanced Set Transformer v2 — Experiment 3")
    print("  Dual-Pathway + ISAB + Multi-Seed PMA")
    print("  55 voters, 5 alternatives, IC sampling")
    print("=" * 65)
    print()

    location = run_experiment3_settransformer_v2(
        # Election settings (matching n-WEC benchmark)
        max_num_voters=55,
        max_num_alternatives=5,
        election_sampling={'probmodel': 'IC'},

        # Training config
        num_gradient_steps=5000,
        report_interval=1000,
        eval_dataset_size=500,
        sample_size_applicable=100,
        sample_size_maximal=int(1e5),
        batch_size=64,
        learning_rate=5e-4,
        random_seed=42,

        # Set Transformer v2 hyperparameters
        d_model=128,
        n_heads=8,
        d_ff=256,
        n_enc_layers=4,
        n_inducing=16,
        dropout=0.12,

        # Training improvements
        warmup_steps=1000,
        grad_clip=1.0,
        weight_decay=0.01,

        # Axiom optimization (same as WEC paper)
        axiom_opt={
            'No_winner':    {'weight': 10, 'period': 'always'},
            'All_winners':  None,
            'Inadmissible': None,
            'Resoluteness': None,
            'Parity':       None,
            'Anonymity':    None,
            'Neutrality':   None,
            'Condorcet1':   {'weight': 2, 'period': 'always'},
            'Condorcet2':   None,
            'Pareto1':      None,
            'Pareto2':      {'weight': 1, 'period': 'always'},
            'Independence': None,
        },
        distance='KLD',
    )

    print(f"\n{'='*65}")
    print(f"  ✓ Experiment complete. Results at: {location}")
    print(f"    • training_progress.png  — loss + axiom satisfaction + LR")
    print(f"    • final_comparison.png   — bar chart vs n-WEC")
    print(f"    • results.json           — full numerical results")
    print(f"    • model.pth              — saved model weights")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()