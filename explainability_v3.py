"""
Explainability Analysis for Set Transformer v3.

Computes how often the trained v3 model agrees with classical voting rules
from the literature, reproducing the style of Table 2 in Hornischer &
Terzopoulou (JAIR 2025).

Two tables are produced:
  1. Identity accuracy:  % of profiles where winning sets are identical
  2. Subset accuracy:    % of profiles where one winning set ⊆ the other

Additionally, a summary "agreement rate" table is printed and saved.

Optimized for Apple M2 (8-core, 16GB RAM).
"""

import os
import sys
import time
import json
import itertools
from datetime import datetime

import numpy as np
import torch
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

# Add project root
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import utils
from generate_data import generate_profile_data
import train_and_eval
from set_transformer_models_v3 import (
    SetTransformerV3,
    SetTransformerV3_2rule_n,
)


# ============================================================
# Classical voting rules to compare against
# ============================================================

# Full set of rules available in utils.py
RULES_ALL = utils.dict_rules_all

# Subset of "fast" rules for quicker evaluation
RULES_FAST = utils.dict_rules_all_fast

# n-WEC benchmark results (from JAIR 2025, Table 5)
NWEC_BENCHMARKS = {
    'WEC n (NW,C,P)': {
        'Anon': 100, 'Neut': 100, 'Condorcet': 96.78,
        'Pareto': 100, 'Indep': 45.9, 'Avg': 88.54,
    },
}

# v2 results
V2_RESULTS = {
    'SetTransV2 n (NW,C,P)': {
        'Anon': 100, 'Neut': 100, 'Condorcet': 100,
        'Pareto': 100, 'Indep': 76.0, 'Avg': 95.2,
    },
}


# ============================================================
# Load trained v3 model
# ============================================================

def load_v3_model(checkpoint_path):
    """
    Load a trained SetTransformerV3 from a checkpoint file.

    Args:
        checkpoint_path: path to model.pth

    Returns:
        model: loaded SetTransformerV3 in eval mode
        config: training config dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']

    model = SetTransformerV3(
        max_num_voters=config['max_num_voters'],
        max_num_alternatives=config['max_num_alternatives'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        d_ff=config['d_ff'],
        n_enc_layers=config['n_enc_layers'],
        n_inducing=config['n_inducing'],
        dropout=config['dropout'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, config


# ============================================================
# Core explainability computation
# ============================================================

def compute_rule_agreement(
    model,
    max_num_voters,
    max_num_alternatives,
    election_sampling,
    eval_dataset_size=10000,
    rules_dict=None,
    n_best_rules=6,
    verbose=True,
):
    """
    Compute agreement rates between the v3 model and classical voting rules.

    Mirrors the methodology of Table 2 in Hornischer & Terzopoulou (JAIR 2025):
    - Identity accuracy: % of profiles where winning sets are identical
    - Subset accuracy: % of profiles where model ⊆ rule
    - Superset accuracy: % of profiles where model ⊇ rule
    - Overlap accuracy: % of profiles where winning sets share ≥1 winner
    - Hamming distance: avg disagreement per alternative

    Args:
        model: trained SetTransformerV3
        max_num_voters: number of voters
        max_num_alternatives: number of alternatives
        election_sampling: sampling config dict
        eval_dataset_size: number of profiles to evaluate on
        rules_dict: dict of {rule_name: rule_function} to compare against.
                    If None, uses all rules in utils.dict_rules_all_fast.
        n_best_rules: number of top rules to include in the cross-comparison
                      table (ranked by identity accuracy with model).
        verbose: whether to print progress

    Returns:
        results: dictionary with all computed similarities
    """
    if rules_dict is None:
        rules_dict = RULES_FAST

    # Create neutrality-averaged rule from model
    model_rule_n = SetTransformerV3_2rule_n(model, None)

    # Generate evaluation profiles
    if verbose:
        print(f"Generating {eval_dataset_size} IC-sampled profiles "
              f"({max_num_voters}v, {max_num_alternatives}a)...")
    test_profs, _, _ = generate_profile_data(
        max_num_voters, max_num_alternatives, eval_dataset_size,
        election_sampling, [], merge='empty',
    )

    # ============================================================
    # Step 1: Model vs all rules
    # ============================================================
    if verbose:
        print(f"\nComputing model agreement with {len(rules_dict)} classical rules...")

    model_vs_rules = train_and_eval.rule_similarity(
        model_rule_n,
        rules_dict.keys(),
        test_profs,
        verbose=verbose,
    )

    # Sort rules by identity accuracy (descending)
    sorted_rules = sorted(
        model_vs_rules.keys(),
        key=lambda r: model_vs_rules[r]['identity_accu'],
        reverse=True,
    )

    # ============================================================
    # Step 2: Print full agreement summary
    # ============================================================
    if verbose:
        print("\n" + "=" * 90)
        print("MODEL AGREEMENT WITH CLASSICAL VOTING RULES")
        print(f"(SetTransformer v3, neutrality-averaged, {eval_dataset_size} "
              f"IC-sampled profiles, {max_num_voters}v/{max_num_alternatives}a)")
        print("=" * 90)
        header = (f"{'Rule':<28} {'Identity%':>10} {'Subset%':>9} "
                  f"{'Superset%':>10} {'Overlap%':>9} {'Hamming':>9}")
        print(header)
        print("-" * 90)

        for rule_name in sorted_rules:
            s = model_vs_rules[rule_name]
            print(f"{rule_name:<28} "
                  f"{100*s['identity_accu']:>10.1f} "
                  f"{100*s['subset_accu']:>9.1f} "
                  f"{100*s['superset_accu']:>10.1f} "
                  f"{100*s['overlap_accu']:>9.1f} "
                  f"{s['hamming']:>9.4f}")
        print("=" * 90)

    # ============================================================
    # Step 3: Cross-comparison table (top-N rules)
    # ============================================================
    top_rules = sorted_rules[:n_best_rules]
    considered_rules = {k: rules_dict[k] for k in top_rules}

    if verbose:
        print(f"\nComputing cross-comparison table for top-{n_best_rules} rules...")

    # Model similarities (already computed)
    data_identity = []
    data_subset = []

    data_identity.append(
        [round(100 * model_vs_rules[r]['identity_accu'], 1)
         for r in top_rules]
    )
    data_subset.append(
        [round(100 * model_vs_rules[r]['subset_accu'], 1)
         for r in top_rules]
    )
    model_superset = [
        round(100 * model_vs_rules[r]['superset_accu'], 1)
        for r in top_rules
    ]

    # Rule vs rule similarities
    for name in tqdm(top_rules, desc="Rule×Rule", disable=not verbose):
        rule = considered_rules[name]
        sims = train_and_eval.rule_similarity(
            rule,
            considered_rules.keys(),
            test_profs,
            verbose=False,
        )
        data_identity.append(
            [round(100 * sims[r]['identity_accu'], 1) for r in top_rules]
        )
        data_subset.append(
            [round(100 * sims[r]['subset_accu'], 1) for r in top_rules]
        )

    # Build DataFrames
    row_labels = ['SetTransV3 n'] + list(top_rules)

    df_identity = pd.DataFrame(
        data_identity,
        index=row_labels,
        columns=top_rules,
    )
    df_subset = pd.DataFrame(
        data_subset,
        index=row_labels,
        columns=top_rules,
    )

    # Add the model self-comparison column
    df_identity['SetTransV3 n'] = [100] + data_identity[0]
    df_identity = df_identity[['SetTransV3 n'] + list(top_rules)]

    df_subset['SetTransV3 n'] = [100] + model_superset
    df_subset = df_subset[['SetTransV3 n'] + list(top_rules)]

    # ============================================================
    # Step 4: Print cross-comparison tables
    # ============================================================
    if verbose:
        print("\n" + "=" * 90)
        print("IDENTITY ACCURACY (Table 2 style)")
        print("=" * 90)
        print(tabulate(
            df_identity,
            headers=df_identity.columns,
            tablefmt='grid',
            numalign='center',
            floatfmt='.1f',
        ))

        print("\n" + "=" * 90)
        print("SUBSET ACCURACY (Table 2 style)")
        print("=" * 90)
        print(tabulate(
            df_subset,
            headers=df_subset.columns,
            tablefmt='grid',
            numalign='center',
            floatfmt='.1f',
        ))

        # LaTeX versions
        print("\n--- LaTeX: IDENTITY ACCURACY ---")
        print(tabulate(
            df_identity,
            headers=df_identity.columns,
            tablefmt='latex_raw',
            numalign='center',
        ))

        print("\n--- LaTeX: SUBSET ACCURACY ---")
        print(tabulate(
            df_subset,
            headers=df_subset.columns,
            tablefmt='latex_raw',
            numalign='center',
        ))

    return {
        'model_vs_rules': {
            rule: {
                'identity_accu': round(100 * model_vs_rules[rule]['identity_accu'], 2),
                'subset_accu': round(100 * model_vs_rules[rule]['subset_accu'], 2),
                'superset_accu': round(100 * model_vs_rules[rule]['superset_accu'], 2),
                'overlap_accu': round(100 * model_vs_rules[rule]['overlap_accu'], 2),
                'hamming': round(model_vs_rules[rule]['hamming'], 5),
            }
            for rule in sorted_rules
        },
        'top_rules': top_rules,
        'identity_table': df_identity.to_dict(),
        'subset_table': df_subset.to_dict(),
        'eval_dataset_size': eval_dataset_size,
        'max_num_voters': max_num_voters,
        'max_num_alternatives': max_num_alternatives,
    }


# ============================================================
# Full pipeline: load model → compute → save
# ============================================================

def run_explainability_analysis(
    checkpoint_path,
    eval_dataset_size=10000,
    n_best_rules=6,
    use_all_rules=False,
    output_dir=None,
):
    """
    Full explainability pipeline.

    Args:
        checkpoint_path: path to model.pth from a v3 training run
        eval_dataset_size: number of profiles to evaluate on
        n_best_rules: number of top rules for the cross-comparison table
        use_all_rules: if True, uses all rules (including slow ones like Slater)
        output_dir: where to save results. If None, saves alongside the model.
    """
    start_time = time.time()

    # Load model
    print("=" * 70)
    print("  EXPLAINABILITY ANALYSIS — Set Transformer v3")
    print("  Agreement with Classical Voting Rules")
    print("=" * 70)

    print(f"\nLoading model from: {checkpoint_path}")
    model, config = load_v3_model(checkpoint_path)
    print(f"  Architecture: SetTransformerV3")
    print(f"  Settings: {config['max_num_voters']}v, "
          f"{config['max_num_alternatives']}a")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    # Decide which rules to use
    rules_dict = RULES_ALL if use_all_rules else RULES_FAST
    print(f"  Comparing against: {len(rules_dict)} voting rules "
          f"({'all' if use_all_rules else 'fast'})")

    election_sampling = config.get('election_sampling', {'probmodel': 'IC'})

    # Run analysis
    results = compute_rule_agreement(
        model=model,
        max_num_voters=config['max_num_voters'],
        max_num_alternatives=config['max_num_alternatives'],
        election_sampling=election_sampling,
        eval_dataset_size=eval_dataset_size,
        rules_dict=rules_dict,
        n_best_rules=n_best_rules,
        verbose=True,
    )

    # Add benchmarks
    results['nwec_benchmarks'] = NWEC_BENCHMARKS
    results['v2_results'] = V2_RESULTS
    results['checkpoint_path'] = checkpoint_path
    results['config'] = config

    # Save results
    if output_dir is None:
        output_dir = os.path.dirname(checkpoint_path)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_path = os.path.join(output_dir, f"explainability_{timestamp}.json")

    # Convert non-serializable items
    results_save = json.loads(json.dumps(results, default=str))
    with open(results_path, 'w') as f:
        json.dump(results_save, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\nRuntime: {elapsed / 60:.1f} minutes")
    print(f"Results saved to: {results_path}")

    return results, results_path