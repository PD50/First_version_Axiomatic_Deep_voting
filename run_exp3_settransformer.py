#!/usr/bin/env python3
"""
Run Experiment 3 with Set Transformer — Majority Tournament Encoding (v2).

Changes from v1:
  1. Tournament Encoding: (n, m²) → (m, m) pairwise majority margins
  2. CosineAnnealingLR scheduler (T_max=15000, eta_min=1e-6)
  3. Independence axiom loss (weight=0.5) added to training
  4. Encoder reduced to 2 SAB layers (input is pre-aggregated)

Usage:
    python run_exp3_settransformer.py

Designed for Apple M2 (8-core, 16GB RAM).
Expected runtime: ~12-20 minutes for 15,000 gradient steps.
"""

import os
import sys

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

# Enable MPS fallback for any ops not yet supported on Apple Silicon
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from exp3_settransformer import run_experiment3_settransformer


def main():
    print("=" * 65)
    print("  Set Transformer — Experiment 3 (v2: Tournament Encoding)")
    print("  Unsupervised Axiom-Guided Voting Rule Synthesis")
    print("  Encoding: Pairwise Majority Margins (m × m)")
    print("  Axioms: No Winner + Condorcet + Pareto + Independence")
    print("  Scheduler: CosineAnnealingLR")
    print("=" * 65)
    print()

    # Run with JAIR-benchmark settings + tournament encoding upgrades
    location = run_experiment3_settransformer(
        max_num_voters=25,
        max_num_alternatives=3,
        election_sampling={'probmodel': 'IC'},
        num_gradient_steps=5000,
        report_interval=1000,
        eval_dataset_size=500,
        sample_size_applicable=100,
        sample_size_maximal=int(1e5),
        batch_size=100,
        learning_rate=1e-3,
        random_seed=42,
        # Set Transformer hyperparameters (simplified for tournament input)
        d_model=128,
        n_heads=8,
        d_ff=256,
        n_enc_layers=2,     # Reduced from 4: tournament input is pre-aggregated
        dropout=0.1,
        # Axiom optimization — now includes Independence
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
            'Independence': {'weight': 0.5, 'period': 'always'},   # NEW
        },
        distance='KLD',
    )

    print(f"\n✓ Experiment complete. Results at: {location}")
    print(f"  • training_progress.png  — loss curves + LR schedule + axiom satisfaction")
    print(f"  • final_comparison.png   — bar chart vs n-WEC and classical rules")
    print(f"  • results.json           — full numerical results")
    print(f"  • model.pth              — saved model + optimizer + scheduler state")


if __name__ == "__main__":
    main()