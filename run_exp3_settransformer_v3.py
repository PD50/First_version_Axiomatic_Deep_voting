#!/usr/bin/env python3
"""
Run Experiment 3 with Set Transformer v3.

Usage:
    python run_exp3_settransformer_v3.py

Designed for Apple M2 (8-core, 16GB RAM).
Expected runtime: ~45-70 minutes for 12,000 gradient steps at 55 voters, 5 alternatives.

v3 Key Innovations over v2:
  - Pairwise margin injection (explicit aggregate pairwise majority structure)
  - Deeper encoder (6 layers: SAB + ISAB(32)x4 + SAB)
  - Wider feedforward (d_ff=512)
  - Extended training (12,000 steps) with tuned hyperparameters
"""

import os
import sys

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

# MPS fallback for custom ops in pref_voting
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from exp3_settransformer_v3 import run_experiment3_settransformer_v3


def main():
    print("=" * 70)
    print("  Set Transformer v3 — Experiment 3")
    print("  Margin Injection + Deep ISAB(32) Encoder + d_ff=512")
    print("  55 voters, 5 alternatives, IC sampling")
    print("=" * 70)
    print()

    location = run_experiment3_settransformer_v3(
        # Election settings (matching n-WEC benchmark from JAIR 2025)
        max_num_voters=55,
        max_num_alternatives=5,
        election_sampling={'probmodel': 'IC'},

        # Training config — extended from v2
        num_gradient_steps=12000,
        report_interval=12000,
        eval_dataset_size=500,
        sample_size_applicable=100,
        sample_size_maximal=int(1e5),
        batch_size=48,            # Reduced from 64 for M2 memory headroom
        learning_rate=8e-4,       # Higher peak LR for deeper model
        random_seed=42,

        # Set Transformer v3 hyperparameters
        d_model=128,
        n_heads=8,
        d_ff=512,                 # 2x v2 — richer nonlinear capacity
        n_enc_layers=6,           # +2 over v2 — deeper encoder
        n_inducing=32,            # 2x v2 — captures 10 pairwise margins
        dropout=0.10,             # Slightly reduced for deeper net

        # Training stability
        warmup_steps=800,         # Faster ramp-up
        grad_clip=0.5,            # Tighter clipping for stability
        weight_decay=0.02,        # Stronger regularization

        # Axiom optimization (same as WEC paper, Section 6.3)
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

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"  Results at: {location}")
    print(f"{'='*70}")
    print(f"  Output files:")
    print(f"    training_progress.png  — loss + axiom satisfaction + LR")
    print(f"    final_comparison.png   — bar chart vs v2 vs n-WEC")
    print(f"    results.json           — full numerical results")
    print(f"    model.pth              — saved model weights")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()