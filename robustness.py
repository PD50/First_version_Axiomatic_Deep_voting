#!/usr/bin/env python3
"""
Distribution Robustness Test for Set Transformer v3.

Tests the trained v3 model (originally trained on IC) across all four
distributions from Hornischer & Terzopoulou (JAIR 2025, Section 5.1):

  1. IC              — Impartial Culture (uniform random rankings)
  2. MALLOWS-RELPHI  — Mallows with rel-phi parameterization
  3. URN-R           — Polya-Eggenberger Urn with Gamma-distributed alpha
  4. euclidean       — 2D Euclidean spatial model

Additionally tests on smaller settings to address Levin's scaling concern:
  - Small:   10 voters, 3 alternatives  (original v1 toy setting)
  - Medium:  20 voters, 4 alternatives  (intermediate)
  - Full:    55 voters, 5 alternatives  (paper benchmark)

For each (distribution × setting) combination, computes:
  - Axiom satisfaction (Anonymity, Neutrality, Condorcet, Pareto, Independence)
  - Agreement with top classical voting rules (identity + subset accuracy)
  - n-WEC benchmark comparison

Outputs: consolidated tables, per-distribution JSON, and summary bar chart.

Usage:
    # Full analysis (all distributions, all settings):
    python run_robustness_v3.py

    # Quick test:
    python run_robustness_v3.py --eval-size 500 --sample-size 50

    # Specific checkpoint:
    python run_robustness_v3.py --checkpoint ./results/.../model.pth

    # Only distributions (skip smaller settings):
    python run_robustness_v3.py --skip-scaling

    # Only scaling (skip non-IC distributions):
    python run_robustness_v3.py --skip-distributions

Designed for Apple M2 (8-core, 16GB RAM).
Expected runtime: ~2-4 hours for full analysis.
"""

import os
import sys
import time
import json
import glob
import argparse
from datetime import datetime

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# Project root
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
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
# Distribution configs — matching JAIR 2025, Section 5.1
# ============================================================

DISTRIBUTIONS = {
    'IC': {
        'probmodel': 'IC',
    },
    'Mallows': {
        'probmodel': 'MALLOWS-RELPHI',
        # rel-phi is randomly generated internally by pref_voting,
        # matching the methodology of Boehmer et al. (2021) as
        # described in Section 5.1 of the paper.
    },
    'Urn': {
        'probmodel': 'URN-R',
        # alpha drawn from Gamma(k=0.8, theta=1) per profile,
        # handled internally by pref_voting's URN-R sampler.
    },
    'Euclidean': {
        'probmodel': 'euclidean',
    },
}

# Scaling settings — addressing Levin's concern about smaller settings
SCALING_SETTINGS = {
    'small':  {'voters': 10, 'alternatives': 3, 'label': '10v/3a'},
    'medium': {'voters': 20, 'alternatives': 4, 'label': '20v/4a'},
    'full':   {'voters': 55, 'alternatives': 5, 'label': '55v/5a'},
}

# n-WEC benchmarks per distribution (from JAIR 2025 Tables 1, 6, 9, 12)
NWEC_BENCHMARKS = {
    'IC': {
        'WEC n (NW,C,P)': {
            'Anon': 100, 'Neut': 100, 'Condorcet': 96.78,
            'Pareto': 100, 'Indep': 45.9, 'Avg': 88.54,
        },
    },
    'Mallows': {
        'WEC n (NW,C,P)': {
            'Anon': 100, 'Neut': 100, 'Condorcet': 97.0,
            'Pareto': 100, 'Indep': 44.0, 'Avg': 88.2,
        },
    },
    'Urn': {
        'WEC n (NW,C,P)': {
            'Anon': 100, 'Neut': 100, 'Condorcet': 93.5,
            'Pareto': 100, 'Indep': 39.0, 'Avg': 86.5,
        },
    },
    'Euclidean': {
        'WEC n (NW,C,P)': {
            'Anon': 100, 'Neut': 100, 'Condorcet': 97.8,
            'Pareto': 100, 'Indep': 42.2, 'Avg': 88.0,
        },
    },
}

# Top classical rules to check agreement with (same set across distributions)
COMPARISON_RULES = [
    'Stable Voting', 'Blacks', 'Weak Nanson', 'Baldwin',
    'Kemeny-Young', 'Borda', 'Copeland',
]


# ============================================================
# Load model
# ============================================================

def load_v3_model(checkpoint_path):
    """Load a trained SetTransformerV3."""
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
# Core evaluation for one (distribution, setting) pair
# ============================================================

def evaluate_single_config(
    model,
    max_num_voters,
    max_num_alternatives,
    election_sampling,
    eval_dataset_size,
    sample_size_applicable,
    sample_size_maximal,
    compute_agreement=True,
    agreement_rules=None,
    agreement_profiles=None,
    verbose=True,
):
    """
    Evaluate v3 model on a specific distribution and setting.

    Returns dict with axiom satisfaction and (optionally) rule agreement.
    """
    model_rule_n = SetTransformerV3_2rule_n(model, None)

    axioms_to_check = ['Anonymity', 'Neutrality', 'Condorcet', 'Pareto', 'Independence']
    axiom_results = {}

    for ax_name in axioms_to_check:
        sat = train_and_eval.axiom_satisfaction(
            model_rule_n,
            utils.dict_axioms[ax_name],
            max_num_voters,
            max_num_alternatives,
            election_sampling,
            sample_size_applicable,
            sample_size_maximal,
            utils.dict_axioms_sample[ax_name],
            full_profile=False,
            comparison_rule=None,
        )
        axiom_results[ax_name] = round(100 * sat['cond_satisfaction'], 2)
        if verbose:
            print(f"      {ax_name}: {axiom_results[ax_name]}%")

    avg = round(sum(axiom_results.values()) / len(axiom_results), 2)
    axiom_results['Average'] = avg
    if verbose:
        print(f"      Average: {avg}%")

    result = {'axiom_satisfaction': axiom_results}

    # Rule agreement (optional, for full-size setting)
    if compute_agreement and agreement_rules is not None:
        if verbose:
            print(f"    Computing agreement with {len(agreement_rules)} rules...")

        # Generate profiles for agreement comparison
        if agreement_profiles is None:
            agreement_profiles, _, _ = generate_profile_data(
                max_num_voters, max_num_alternatives,
                min(eval_dataset_size, 5000),  # cap at 5k for agreement
                election_sampling, [], merge='empty',
            )

        rule_funcs = {
            name: utils.dict_rules_all_fast[name]
            for name in agreement_rules
            if name in utils.dict_rules_all_fast
        }

        similarities = train_and_eval.rule_similarity(
            model_rule_n,
            rule_funcs.keys(),
            agreement_profiles,
            verbose=False,
        )

        agreement = {}
        for rule_name in rule_funcs.keys():
            agreement[rule_name] = {
                'identity_accu': round(100 * similarities[rule_name]['identity_accu'], 2),
                'subset_accu': round(100 * similarities[rule_name]['subset_accu'], 2),
                'superset_accu': round(100 * similarities[rule_name]['superset_accu'], 2),
            }
            if verbose:
                print(f"      vs {rule_name}: {agreement[rule_name]['identity_accu']}% identity")

        result['rule_agreement'] = agreement

    return result


# ============================================================
# Distribution robustness sweep
# ============================================================

def run_distribution_sweep(
    model, config, eval_dataset_size, sample_size_applicable,
    sample_size_maximal, distributions=None, verbose=True,
):
    """Test model across all four paper distributions at full scale."""
    if distributions is None:
        distributions = DISTRIBUTIONS

    max_v = config['max_num_voters']
    max_a = config['max_num_alternatives']
    results = {}

    for dist_name, sampling_dict in distributions.items():
        if verbose:
            print(f"\n{'='*70}")
            print(f"  Distribution: {dist_name}")
            print(f"  Setting: {max_v} voters, {max_a} alternatives")
            print(f"{'='*70}")

        result = evaluate_single_config(
            model=model,
            max_num_voters=max_v,
            max_num_alternatives=max_a,
            election_sampling=sampling_dict,
            eval_dataset_size=eval_dataset_size,
            sample_size_applicable=sample_size_applicable,
            sample_size_maximal=sample_size_maximal,
            compute_agreement=True,
            agreement_rules=COMPARISON_RULES,
            verbose=verbose,
        )

        # Add n-WEC benchmark for this distribution
        if dist_name in NWEC_BENCHMARKS:
            result['nwec_benchmark'] = NWEC_BENCHMARKS[dist_name]

        results[dist_name] = result

    return results


# ============================================================
# Scaling sweep (smaller settings, IC only)
# ============================================================

def run_scaling_sweep(
    model, config, eval_dataset_size, sample_size_applicable,
    sample_size_maximal, settings=None, verbose=True,
):
    """Test model at different sizes to check scaling behavior."""
    if settings is None:
        settings = SCALING_SETTINGS

    results = {}

    for setting_name, setting in settings.items():
        v = setting['voters']
        a = setting['alternatives']

        # Skip if model was not trained for this size
        if v > config['max_num_voters'] or a > config['max_num_alternatives']:
            if verbose:
                print(f"\n  Skipping {setting['label']}: exceeds model capacity "
                      f"({config['max_num_voters']}v, {config['max_num_alternatives']}a)")
            continue

        if verbose:
            print(f"\n{'='*70}")
            print(f"  Scaling Test: {setting['label']} (IC sampling)")
            print(f"{'='*70}")

        result = evaluate_single_config(
            model=model,
            max_num_voters=v,
            max_num_alternatives=a,
            election_sampling={'probmodel': 'IC'},
            eval_dataset_size=eval_dataset_size,
            sample_size_applicable=sample_size_applicable,
            sample_size_maximal=sample_size_maximal,
            compute_agreement=False,  # axioms only for scaling
            verbose=verbose,
        )

        result['setting'] = setting
        results[setting_name] = result

    return results


# ============================================================
# Printing and plotting
# ============================================================

def print_distribution_table(dist_results):
    """Print consolidated axiom satisfaction table across distributions."""
    print("\n" + "=" * 105)
    print("DISTRIBUTION ROBUSTNESS — Axiom Satisfaction (SetTransformer v3, neutrality-averaged)")
    print("=" * 105)

    header = (f"{'Distribution':<15} {'Anon.':>7} {'Neut.':>7} {'Condorcet':>10} "
              f"{'Pareto':>8} {'Indep.':>8} {'Avg.':>8}  {'n-WEC Avg':>10}")
    print(header)
    print("-" * 105)

    for dist_name, result in dist_results.items():
        ax = result['axiom_satisfaction']
        nwec_avg = '--'
        if 'nwec_benchmark' in result:
            nwec_avg = f"{result['nwec_benchmark']['WEC n (NW,C,P)']['Avg']:.2f}"

        print(f"{dist_name:<15} "
              f"{ax['Anonymity']:>7.1f} "
              f"{ax['Neutrality']:>7.1f} "
              f"{ax['Condorcet']:>10.2f} "
              f"{ax['Pareto']:>8.1f} "
              f"{ax['Independence']:>8.2f} "
              f"{ax['Average']:>8.2f}  "
              f"{nwec_avg:>10}")

    print("=" * 105)

    # Deltas vs n-WEC
    print("\n  Delta vs n-WEC (positive = v3 outperforms):")
    for dist_name, result in dist_results.items():
        if 'nwec_benchmark' in result:
            our_avg = result['axiom_satisfaction']['Average']
            nwec_avg = result['nwec_benchmark']['WEC n (NW,C,P)']['Avg']
            delta = our_avg - nwec_avg
            sym = "\u25B2" if delta > 0 else "\u25BC"
            print(f"    {dist_name:<12}: {sym} {abs(delta):.2f}%")


def print_scaling_table(scaling_results):
    """Print axiom satisfaction across different voter/alternative counts."""
    print("\n" + "=" * 95)
    print("SCALING ANALYSIS — Axiom Satisfaction (SetTransformer v3, IC sampling)")
    print("=" * 95)

    header = (f"{'Setting':<15} {'Anon.':>7} {'Neut.':>7} {'Condorcet':>10} "
              f"{'Pareto':>8} {'Indep.':>8} {'Avg.':>8}")
    print(header)
    print("-" * 95)

    for setting_name, result in scaling_results.items():
        ax = result['axiom_satisfaction']
        label = result['setting']['label']
        print(f"{label:<15} "
              f"{ax['Anonymity']:>7.1f} "
              f"{ax['Neutrality']:>7.1f} "
              f"{ax['Condorcet']:>10.2f} "
              f"{ax['Pareto']:>8.1f} "
              f"{ax['Independence']:>8.2f} "
              f"{ax['Average']:>8.2f}")

    print("=" * 95)


def print_agreement_table(dist_results):
    """Print rule agreement across distributions."""
    print("\n" + "=" * 105)
    print("RULE AGREEMENT — Identity Accuracy % (SetTransformer v3 vs classical rules)")
    print("=" * 105)

    # Collect all rules present
    all_rules = []
    for result in dist_results.values():
        if 'rule_agreement' in result:
            for r in result['rule_agreement'].keys():
                if r not in all_rules:
                    all_rules.append(r)

    # Header
    rule_header = "".join(f"{r[:12]:>14}" for r in all_rules)
    print(f"{'Distribution':<15}{rule_header}")
    print("-" * 105)

    for dist_name, result in dist_results.items():
        if 'rule_agreement' not in result:
            continue
        values = "".join(
            f"{result['rule_agreement'].get(r, {}).get('identity_accu', 0):>14.1f}"
            for r in all_rules
        )
        print(f"{dist_name:<15}{values}")

    print("=" * 105)


def plot_robustness_summary(dist_results, scaling_results, output_dir):
    """Generate summary bar charts."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # ─── Left panel: Distribution robustness ───
    ax1 = axes[0]
    dist_names = list(dist_results.keys())
    axiom_names = ['Anonymity', 'Neutrality', 'Condorcet', 'Pareto', 'Independence', 'Average']
    short_names = ['Anon', 'Neut', 'Cond', 'Pareto', 'Indep', 'Avg']

    x = np.arange(len(axiom_names))
    width = 0.18
    colors = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6']

    for i, dist_name in enumerate(dist_names):
        ax = dist_results[dist_name]['axiom_satisfaction']
        values = [ax[a] for a in axiom_names]
        offset = (i - len(dist_names) / 2 + 0.5) * width
        bars = ax1.bar(x + offset, values, width, label=dist_name,
                       color=colors[i % len(colors)], edgecolor='white', linewidth=0.5)

    # Add n-WEC reference line
    nwec_ic = NWEC_BENCHMARKS.get('IC', {}).get('WEC n (NW,C,P)', {}).get('Avg', 0)
    if nwec_ic > 0:
        ax1.axhline(y=nwec_ic, color='red', linestyle=':', linewidth=1.5,
                     label=f'n-WEC Avg IC ({nwec_ic}%)')

    ax1.set_ylabel('Satisfaction (%)')
    ax1.set_title('Distribution Robustness (55v, 5a)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names)
    ax1.legend(fontsize=8, loc='lower left')
    ax1.set_ylim([0, 110])
    ax1.grid(True, alpha=0.2, axis='y')

    # ─── Right panel: Scaling analysis ───
    ax2 = axes[1]
    if scaling_results:
        settings = list(scaling_results.keys())
        labels = [scaling_results[s]['setting']['label'] for s in settings]

        for i, ax_name in enumerate(axiom_names):
            values = [scaling_results[s]['axiom_satisfaction'][ax_name] for s in settings]
            ax2.plot(labels, values, marker='o', linewidth=2, label=ax_name)

        ax2.set_ylabel('Satisfaction (%)')
        ax2.set_title('Scaling Analysis (IC sampling)')
        ax2.set_xlabel('Setting (voters/alternatives)')
        ax2.legend(fontsize=8, loc='lower left')
        ax2.set_ylim([0, 110])
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Scaling analysis skipped',
                 ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_axis_off()

    plt.tight_layout()
    path = os.path.join(output_dir, 'robustness_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {path}")


# ============================================================
# Main pipeline
# ============================================================

def find_latest_checkpoint(base_dir="./results/exp3/SetTransformerV3"):
    """Find the most recent v3 model checkpoint."""
    pattern = os.path.join(base_dir, "*/model.pth")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    return checkpoints[0]


def main():
    parser = argparse.ArgumentParser(
        description="Distribution Robustness & Scaling Test for Set Transformer v3"
    )
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model.pth')
    parser.add_argument('--eval-size', type=int, default=500,
                        help='Profiles for axiom evaluation (default: 500)')
    parser.add_argument('--sample-size', type=int, default=100,
                        help='Applicable sample size for axiom checks (default: 100)')
    parser.add_argument('--agreement-profiles', type=int, default=5000,
                        help='Profiles for rule agreement computation (default: 5000)')
    parser.add_argument('--skip-distributions', action='store_true',
                        help='Skip distribution robustness (non-IC) tests')
    parser.add_argument('--skip-scaling', action='store_true',
                        help='Skip scaling (smaller settings) tests')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')
    args = parser.parse_args()

    # Find checkpoint
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = find_latest_checkpoint()
        if checkpoint is None:
            print("ERROR: No v3 checkpoint found.")
            print("  Train first: python run_exp3_settransformer_v3.py")
            sys.exit(1)
        print(f"Auto-detected checkpoint: {checkpoint}")

    if not os.path.exists(checkpoint):
        print(f"ERROR: Checkpoint not found: {checkpoint}")
        sys.exit(1)

    # Load model
    print("=" * 70)
    print("  ROBUSTNESS ANALYSIS — Set Transformer v3")
    print("  Distribution Robustness + Scaling Tests")
    print("=" * 70)

    print(f"\nLoading model from: {checkpoint}")
    model, config = load_v3_model(checkpoint)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    print(f"  Trained on: {config['max_num_voters']}v, "
          f"{config['max_num_alternatives']}a, "
          f"{config.get('election_sampling', {}).get('probmodel', 'IC')}")

    # Setup output
    if args.output_dir is None:
        args.output_dir = os.path.dirname(checkpoint)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    start_time = time.time()
    sample_size_maximal = int(1e5)

    # ============================================================
    # Part 1: Distribution robustness
    # ============================================================
    dist_results = {}
    if not args.skip_distributions:
        print("\n" + "#" * 70)
        print("#  PART 1: DISTRIBUTION ROBUSTNESS")
        print("#" * 70)

        dist_results = run_distribution_sweep(
            model=model,
            config=config,
            eval_dataset_size=args.eval_size,
            sample_size_applicable=args.sample_size,
            sample_size_maximal=sample_size_maximal,
            verbose=True,
        )

        print_distribution_table(dist_results)
        print_agreement_table(dist_results)

    # ============================================================
    # Part 2: Scaling analysis
    # ============================================================
    scaling_results = {}
    if not args.skip_scaling:
        print("\n" + "#" * 70)
        print("#  PART 2: SCALING ANALYSIS (IC)")
        print("#" * 70)

        scaling_results = run_scaling_sweep(
            model=model,
            config=config,
            eval_dataset_size=args.eval_size,
            sample_size_applicable=args.sample_size,
            sample_size_maximal=sample_size_maximal,
            verbose=True,
        )

        print_scaling_table(scaling_results)

    # ============================================================
    # Save results + plot
    # ============================================================
    results_path = os.path.join(args.output_dir, f"robustness_{timestamp}.json")
    all_results = {
        'distribution_robustness': {
            k: {
                'axiom_satisfaction': v['axiom_satisfaction'],
                'rule_agreement': v.get('rule_agreement', {}),
                'nwec_benchmark': v.get('nwec_benchmark', {}),
            }
            for k, v in dist_results.items()
        },
        'scaling_analysis': {
            k: {
                'axiom_satisfaction': v['axiom_satisfaction'],
                'setting': v['setting'],
            }
            for k, v in scaling_results.items()
        },
        'config': config,
        'checkpoint_path': checkpoint,
        'eval_dataset_size': args.eval_size,
        'sample_size_applicable': args.sample_size,
        'timestamp': timestamp,
    }

    results_save = json.loads(json.dumps(all_results, default=str))
    with open(results_path, 'w') as f:
        json.dump(results_save, f, indent=2)

    # Plot
    plot_robustness_summary(dist_results, scaling_results, args.output_dir)

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  ROBUSTNESS ANALYSIS COMPLETE")
    print(f"  Runtime: {elapsed / 60:.1f} minutes")
    print(f"  Results: {results_path}")
    print(f"  Plot:    {os.path.join(args.output_dir, 'robustness_summary.png')}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()