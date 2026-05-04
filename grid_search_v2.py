"""
Hyperparameter Grid Search for Set Transformer v2.

Two-phase grid search in an intermediate setting (20 voters, 4 alternatives).
Phase 1: Coarse search over primary hyperparameters (LR, d_model, n_layers, dropout)
Phase 2: Fine-tune secondary hyperparameters around Phase 1 winner

Optimized for Apple M2 (8-core, 16GB RAM).
"""

import os
import sys
import time
import json
import math
import random
import itertools
from datetime import datetime
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils
from generate_data import generate_profile_data
import train_and_eval
import axioms_continuous
from set_transformer_models_v2 import (
    SetTransformerV2,
    SetTransformerV2_2logits,
    SetTransformerV2_2rule,
    SetTransformerV2_2rule_n,
)


# ============================================================
# n-WEC BENCHMARK RESULTS (from Table 5, Hornischer & Terzopoulou JAIR 2025)
# IC sampling, neutrality-averaged, averaged over 5 runs
# Settings: 55 voters, 5 alternatives
# NOTE: These are for the large setting; we include them for reference.
# The intermediate setting (20 voters, 4 alt) won't have exact n-WEC
# benchmarks, but we still include them to track where the model stands.
# ============================================================
NWEC_BENCHMARKS_55v5a = {
    'Blacks':        {'Anon': 100, 'Neut': 100, 'Condorcet': 100,   'Pareto': 100, 'Indep': 36.04, 'Avg': 87.21},
    'Stable Voting': {'Anon': 100, 'Neut': 100, 'Condorcet': 100,   'Pareto': 100, 'Indep': 40.48, 'Avg': 88.10},
    'Borda':         {'Anon': 100, 'Neut': 100, 'Condorcet': 93.82, 'Pareto': 100, 'Indep': 37.72, 'Avg': 86.31},
    'Weak Nanson':   {'Anon': 100, 'Neut': 100, 'Condorcet': 100,   'Pareto': 100, 'Indep': 38.28, 'Avg': 87.66},
    'Copeland':      {'Anon': 100, 'Neut': 100, 'Condorcet': 100,   'Pareto': 100, 'Indep': 28.54, 'Avg': 85.71},
    'WEC n (NW,C,P)':{'Anon': 100, 'Neut': 100, 'Condorcet': 96.78,'Pareto': 100, 'Indep': 45.90, 'Avg': 88.54},
}


# ============================================================
# Cosine Annealing with Linear Warmup (same as exp3_settransformer_v2.py)
# ============================================================

class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr = self.base_lr * (self.step_count / self.warmup_steps)
        else:
            progress = (self.step_count - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr


# ============================================================
# Grid Search Configuration
# ============================================================

def get_phase1_grid():
    """
    Phase 1: Coarse search over primary hyperparameters.
    4 hyperparameters × 2-3 values each = 36 total configurations.
    """
    grid = {
        'learning_rate': [1e-4, 5e-4, 1e-3],
        'd_model':       [64, 128],
        'n_enc_layers':  [2, 4, 6],
        'dropout':       [0.05, 0.15],
    }
    # Fixed secondary hyperparameters during Phase 1
    fixed = {
        'n_heads':      8,      # Will be clamped to d_model // 16 if needed
        'd_ff_ratio':   2,      # d_ff = d_ff_ratio * d_model
        'n_inducing':   16,
        'batch_size':   64,
        'warmup_steps': 700,
        'weight_decay': 0.01,
        'grad_clip':    1.0,
    }
    return grid, fixed


def get_phase2_grid(winner_config):
    """
    Phase 2: Fine-tune secondary hyperparameters around Phase 1 winner.
    """
    grid = {
        'batch_size':   [32, 64, 128],
        'n_heads':      [4, 8],
        'n_inducing':   [8, 16, 32],
        'warmup_steps': [300, 700, 1000],
        'weight_decay': [0.0, 0.01, 0.05],
    }
    # Fixed primary hyperparameters from Phase 1 winner
    fixed = {
        'learning_rate': winner_config['learning_rate'],
        'd_model':       winner_config['d_model'],
        'n_enc_layers':  winner_config['n_enc_layers'],
        'dropout':       winner_config['dropout'],
        'd_ff_ratio':    2,
        'grad_clip':     1.0,
    }
    return grid, fixed


def expand_grid(grid_dict):
    """Expand a grid dictionary into a list of config dicts."""
    keys = list(grid_dict.keys())
    values = list(grid_dict.values())
    configs = []
    for combo in itertools.product(*values):
        configs.append(dict(zip(keys, combo)))
    return configs


def config_fingerprint(config):
    """
    Create a hashable fingerprint from a config dict for resume matching.
    Converts floats to rounded strings to avoid floating-point comparison issues.
    """
    parts = []
    for k in sorted(config.keys()):
        v = config[k]
        if isinstance(v, float):
            parts.append(f"{k}={v:.8g}")
        else:
            parts.append(f"{k}={v}")
    return "|".join(parts)


def load_completed_results(resume_dir):
    """
    Load previously completed results from a resume directory.
    Returns (list_of_results, set_of_fingerprints).
    """
    partial_path = os.path.join(resume_dir, "results_partial.json")
    final_path = os.path.join(resume_dir, "results_final.json")

    # Prefer partial (has all runs in order), fall back to final
    load_path = None
    if os.path.exists(partial_path):
        load_path = partial_path
    elif os.path.exists(final_path):
        load_path = final_path

    if load_path is None:
        print(f"  ⚠ No results found in {resume_dir}")
        return [], set()

    with open(load_path) as f:
        data = json.load(f)

    # results_final.json nests results under a 'results' key
    if isinstance(data, dict) and 'results' in data:
        results = data['results']
    elif isinstance(data, list):
        results = data
    else:
        print(f"  ⚠ Unexpected format in {load_path}")
        return [], set()

    fingerprints = set()
    for r in results:
        fp = config_fingerprint(r['config'])
        fingerprints.add(fp)

    return results, fingerprints


# ============================================================
# Single Run
# ============================================================

def run_single_config(
    config,
    max_num_voters=20,
    max_num_alternatives=4,
    election_sampling=None,
    num_gradient_steps=10000,
    eval_interval=5000,
    eval_dataset_size=300,
    sample_size_applicable=100,
    sample_size_maximal=int(1e5),
    random_seed=42,
    axiom_opt=None,
    distance='KLD',
    verbose=True,
):
    """
    Train a single Set Transformer v2 configuration and return axiom results.

    Returns dict with: config, final_axioms, final_loss, runtime_sec, param_count
    """
    if election_sampling is None:
        election_sampling = {'probmodel': 'IC'}

    if axiom_opt is None:
        axiom_opt = {
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
        }

    start_time = time.time()

    # Set seeds
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    # Extract hyperparameters from config
    learning_rate = config['learning_rate']
    d_model = config['d_model']
    n_enc_layers = config['n_enc_layers']
    dropout = config['dropout']
    batch_size = config.get('batch_size', 64)
    n_heads = config.get('n_heads', 8)
    n_inducing = config.get('n_inducing', 16)
    warmup_steps = config.get('warmup_steps', 700)
    weight_decay = config.get('weight_decay', 0.01)
    grad_clip = config.get('grad_clip', 1.0)
    d_ff_ratio = config.get('d_ff_ratio', 2)
    d_ff = d_model * d_ff_ratio

    # Clamp n_heads to ensure d_model is divisible
    while d_model % n_heads != 0 and n_heads > 1:
        n_heads -= 1

    # Distance function
    KLD = lambda x, y: nn.KLDivLoss(log_target=True, reduction='batchmean')(
        x.log_softmax(dim=1), y.log_softmax(dim=1)
    )
    distance_fn = KLD

    # Initialize model
    model = SetTransformerV2(
        max_num_voters=max_num_voters,
        max_num_alternatives=max_num_alternatives,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_enc_layers=n_enc_layers,
        n_inducing=n_inducing,
        dropout=dropout,
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_steps, num_gradient_steps, min_lr=1e-6
    )

    model_on_profiles = lambda X: SetTransformerV2_2logits(model, X)

    # Training loop
    final_loss = float('inf')
    axiom_history = {'step': [], 'Average': []}

    for step in range(num_gradient_steps):
        model.train()

        # Generate batch
        X_batch, _, _ = generate_profile_data(
            max_num_voters, max_num_alternatives, batch_size,
            election_sampling, [], merge='empty',
        )

        # Compute axiom losses
        loss_nw = torch.tensor([0.0])
        loss_cond = torch.tensor([0.0])
        loss_pareto = torch.tensor([0.0])

        if axiom_opt['No_winner'] is not None:
            nw_cfg = axiom_opt['No_winner']
            if nw_cfg['period'] == 'always':
                loss_nw = nw_cfg['weight'] * axioms_continuous.ax_no_winners_cont(
                    model_on_profiles, X_batch
                )

        if axiom_opt['Condorcet1'] is not None:
            c_cfg = axiom_opt['Condorcet1']
            if c_cfg['period'] == 'always':
                loss_cond = c_cfg['weight'] * axioms_continuous.ax_condorcet1_cont(
                    model_on_profiles, X_batch, distance_fn
                )

        if axiom_opt['Pareto2'] is not None:
            p_cfg = axiom_opt['Pareto2']
            if p_cfg['period'] == 'always':
                loss_pareto = p_cfg['weight'] * axioms_continuous.ax_pareto2_cont(
                    model_on_profiles, X_batch, distance_fn
                )

        total_loss = loss_nw + loss_cond + loss_pareto

        optimizer.zero_grad()
        total_loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        final_loss = total_loss.item()

        # Mid-training evaluation (optional, for progress tracking)
        if eval_interval and (step + 1) % eval_interval == 0 and verbose:
            print(f"    Step {step+1}/{num_gradient_steps} | Loss: {final_loss:.4f}")

    # Final evaluation — neutrality-averaged
    model.eval()
    axioms_to_check = ['Anonymity', 'Neutrality', 'Condorcet', 'Pareto', 'Independence']
    model_rule_n = SetTransformerV2_2rule_n(model, None)

    final_axioms = {}
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
        final_axioms[ax_name] = round(100 * sat['cond_satisfaction'], 2)

    avg = round(sum(final_axioms.values()) / len(final_axioms), 2)
    final_axioms['Average'] = avg

    runtime = time.time() - start_time

    return {
        'config': config,
        'final_axioms': final_axioms,
        'final_loss': final_loss,
        'runtime_sec': runtime,
        'param_count': num_params,
        'model_state': model.state_dict(),  # Keep for potential Phase 2
    }


# ============================================================
# Grid Search Runner
# ============================================================

def run_grid_search(
    phase='phase1',
    max_num_voters=20,
    max_num_alternatives=4,
    num_gradient_steps=10000,
    eval_interval=5000,
    random_seed=42,
    winner_config=None,  # Required for phase2
    save_dir='./results/grid_search',
    resume_dir=None,  # Path to previous run directory to resume from
):
    """
    Run the full grid search for the specified phase.

    If resume_dir is provided, loads completed results from that directory,
    skips configs that already finished, and appends new results.
    The resumed run saves to a NEW directory (not the old one).
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    location = f"{save_dir}/{phase}_{timestamp}"
    os.makedirs(location, exist_ok=True)

    if phase == 'phase1':
        grid, fixed = get_phase1_grid()
    elif phase == 'phase2':
        assert winner_config is not None, "Phase 2 requires winner_config from Phase 1"
        grid, fixed = get_phase2_grid(winner_config)
    else:
        raise ValueError(f"Unknown phase: {phase}")

    # Expand grid into list of configs
    variable_configs = expand_grid(grid)
    total_runs = len(variable_configs)

    # ============================================================
    # Resume: load completed results and build skip set
    # ============================================================
    completed_results = []
    completed_fps = set()
    skipped_count = 0

    if resume_dir is not None:
        print(f"\n  Resuming from: {resume_dir}")
        completed_results, completed_fps = load_completed_results(resume_dir)
        print(f"  Found {len(completed_results)} completed runs to skip.")

    print(f"\n{'='*75}")
    print(f"  GRID SEARCH — {phase.upper()}")
    if resume_dir:
        print(f"  *** RESUME MODE — skipping {len(completed_fps)} completed configs ***")
    print(f"  Setting: {max_num_voters} voters, {max_num_alternatives} alternatives, IC")
    print(f"  Total configurations: {total_runs}")
    print(f"  Remaining to run: {total_runs - len(completed_fps)}")
    print(f"  Gradient steps per run: {num_gradient_steps}")
    print(f"  Variable params: {list(grid.keys())}")
    print(f"  Fixed params: {fixed}")
    print(f"{'='*75}\n")

    all_results = list(completed_results)  # Start with previously completed runs

    for run_idx, var_config in enumerate(variable_configs):
        # Merge variable + fixed params into full config
        full_config = {**fixed, **var_config}

        # Skip if already completed in a previous run
        fp = config_fingerprint(full_config)
        if fp in completed_fps:
            skipped_count += 1
            config_str = " | ".join(f"{k}={v}" for k, v in var_config.items())
            print(f"[Run {run_idx+1}/{total_runs}] {config_str}  ← SKIPPED (already done)")
            continue

        config_str = " | ".join(f"{k}={v}" for k, v in var_config.items())
        remaining = total_runs - run_idx - skipped_count
        print(f"\n[Run {run_idx+1}/{total_runs}] {config_str}"
              f"  ({len(all_results) - len(completed_results) + 1} new / ~{remaining} left)")

        result = run_single_config(
            config=full_config,
            max_num_voters=max_num_voters,
            max_num_alternatives=max_num_alternatives,
            num_gradient_steps=num_gradient_steps,
            eval_interval=eval_interval,
            random_seed=random_seed,
            verbose=True,
        )

        # Remove model_state from serializable results (too large for JSON)
        result_serializable = {
            k: v for k, v in result.items() if k != 'model_state'
        }
        all_results.append(result_serializable)

        # Print result summary
        axioms = result['final_axioms']
        print(f"  → Avg: {axioms['Average']:.1f}% | "
              f"Anon: {axioms['Anonymity']:.1f} | "
              f"Neut: {axioms['Neutrality']:.1f} | "
              f"Cond: {axioms['Condorcet']:.1f} | "
              f"Par: {axioms['Pareto']:.1f} | "
              f"Ind: {axioms['Independence']:.1f} | "
              f"Loss: {result['final_loss']:.4f} | "
              f"Params: {result['param_count']:,} | "
              f"Time: {result['runtime_sec']:.0f}s")

        # Save intermediate results (crash recovery)
        with open(f"{location}/results_partial.json", "w") as f:
            json.dump(all_results, f, indent=2)

    # ============================================================
    # Sort by Average axiom satisfaction (tiebreak: Condorcet)
    # ============================================================
    all_results.sort(
        key=lambda r: (r['final_axioms']['Average'], r['final_axioms']['Condorcet']),
        reverse=True,
    )

    # ============================================================
    # Print final comparison table with n-WEC benchmarks
    # ============================================================
    _print_full_table(all_results, max_num_voters, max_num_alternatives)

    # ============================================================
    # Generate plots
    # ============================================================
    _plot_grid_results(all_results, location, phase)

    # ============================================================
    # Save final results
    # ============================================================
    output = {
        'phase': phase,
        'setting': {
            'max_num_voters': max_num_voters,
            'max_num_alternatives': max_num_alternatives,
            'num_gradient_steps': num_gradient_steps,
        },
        'grid': {k: [str(v) for v in vals] for k, vals in grid.items()},
        'fixed': {k: str(v) for k, v in fixed.items()},
        'results': all_results,
        'winner': all_results[0] if all_results else None,
        'nwec_benchmarks_55v5a': NWEC_BENCHMARKS_55v5a,
    }

    with open(f"{location}/results_final.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {location}/results_final.json")

    return all_results, location


# ============================================================
# Comparison Table (includes n-WEC)
# ============================================================

def _print_full_table(results, n_voters, n_alts):
    """Print the full comparison table with n-WEC benchmarks."""
    print("\n" + "=" * 120)
    print(f"GRID SEARCH RESULTS — {n_voters} voters, {n_alts} alternatives, IC sampling")
    print("=" * 120)

    header = (
        f"{'Rank':<5} {'Config':<45} "
        f"{'Anon':>6} {'Neut':>6} {'Cond':>6} {'Par':>6} {'Ind':>6} "
        f"{'Avg':>7} {'Loss':>8} {'Params':>9} {'Time':>6}"
    )
    print(header)
    print("-" * 120)

    # Print n-WEC benchmarks first (reference, 55v/5a setting)
    print("  n-WEC BENCHMARKS (55 voters, 5 alternatives — reference only):")
    for method, vals in NWEC_BENCHMARKS_55v5a.items():
        print(f"  {'ref':<3} {method:<45} "
              f"{vals['Anon']:>6.1f} {vals['Neut']:>6.1f} "
              f"{vals['Condorcet']:>6.2f} {vals['Pareto']:>6.1f} "
              f"{vals['Indep']:>6.2f} {vals['Avg']:>7.2f} "
              f"{'—':>8} {'—':>9} {'—':>6}")
    print("-" * 120)

    # Print grid search results
    print("  SET TRANSFORMER v2 GRID SEARCH RESULTS:")
    for rank, res in enumerate(results):
        cfg = res['config']
        ax = res['final_axioms']
        # Build compact config string
        config_str = (
            f"lr={cfg['learning_rate']:.0e} "
            f"d={cfg['d_model']} "
            f"L={cfg['n_enc_layers']} "
            f"do={cfg['dropout']}"
        )
        # Add secondary params if they differ from defaults
        if cfg.get('batch_size', 64) != 64:
            config_str += f" bs={cfg['batch_size']}"
        if cfg.get('n_heads', 8) != 8:
            config_str += f" h={cfg['n_heads']}"
        if cfg.get('n_inducing', 16) != 16:
            config_str += f" ind={cfg['n_inducing']}"
        if cfg.get('warmup_steps', 700) != 700:
            config_str += f" wu={cfg['warmup_steps']}"
        if cfg.get('weight_decay', 0.01) != 0.01:
            config_str += f" wd={cfg['weight_decay']}"

        marker = " ★" if rank == 0 else ""
        print(f"  {rank+1:<3}{marker} {config_str:<45} "
              f"{ax['Anonymity']:>6.1f} {ax['Neutrality']:>6.1f} "
              f"{ax['Condorcet']:>6.2f} {ax['Pareto']:>6.1f} "
              f"{ax['Independence']:>6.2f} {ax['Average']:>7.2f} "
              f"{res['final_loss']:>8.4f} {res['param_count']:>9,} "
              f"{res['runtime_sec']:>5.0f}s")

    print("=" * 120)

    # Winner summary
    if results:
        winner = results[0]
        print(f"\n  ★ WINNER: {_config_to_str(winner['config'])}")
        print(f"    Average axiom satisfaction: {winner['final_axioms']['Average']:.2f}%")
        wec_avg = NWEC_BENCHMARKS_55v5a['WEC n (NW,C,P)']['Avg']
        diff = winner['final_axioms']['Average'] - wec_avg
        symbol = "▲" if diff > 0 else "▼"
        print(f"    vs n-WEC (55v/5a reference): {symbol} {abs(diff):.2f}%")


def _config_to_str(cfg):
    """Compact config string."""
    parts = []
    for k in ['learning_rate', 'd_model', 'n_enc_layers', 'dropout',
              'batch_size', 'n_heads', 'n_inducing', 'warmup_steps', 'weight_decay']:
        if k in cfg:
            if k == 'learning_rate':
                parts.append(f"lr={cfg[k]:.0e}")
            else:
                parts.append(f"{k}={cfg[k]}")
    return " | ".join(parts)


# ============================================================
# Plotting
# ============================================================

def _plot_grid_results(results, location, phase):
    """Generate summary plots for the grid search."""
    if not results:
        return

    # Plot 1: Average axiom satisfaction for all configs (horizontal bar chart)
    fig, ax = plt.subplots(figsize=(14, max(8, len(results) * 0.4)))

    labels = []
    avgs = []
    colors = []

    for i, res in enumerate(results):
        cfg = res['config']
        label = (f"lr={cfg['learning_rate']:.0e} d={cfg['d_model']} "
                 f"L={cfg['n_enc_layers']} do={cfg['dropout']}")
        labels.append(label)
        avgs.append(res['final_axioms']['Average'])
        colors.append('#1abc9c' if i == 0 else '#3498db' if i < 5 else '#95a5a6')

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, avgs, color=colors, edgecolor='white', linewidth=0.5)

    # Add n-WEC reference line
    wec_avg = NWEC_BENCHMARKS_55v5a['WEC n (NW,C,P)']['Avg']
    ax.axvline(x=wec_avg, color='red', linestyle='--', linewidth=1.5,
               label=f'n-WEC Avg (55v/5a): {wec_avg}%')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Average Axiom Satisfaction (%)')
    ax.set_title(f'Grid Search {phase.upper()} — Average Axiom Satisfaction\n'
                 f'(20 voters, 4 alternatives, IC)')
    ax.legend(loc='lower right')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.2, axis='x')
    plt.tight_layout()
    plt.savefig(f"{location}/grid_results_avg.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Per-axiom breakdown for top 5 configs
    top_n = min(5, len(results))
    axiom_names = ['Anonymity', 'Neutrality', 'Condorcet', 'Pareto', 'Independence']

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(axiom_names))
    width = 0.15

    for i in range(top_n):
        res = results[i]
        cfg = res['config']
        vals = [res['final_axioms'][a] for a in axiom_names]
        label = f"#{i+1} lr={cfg['learning_rate']:.0e} d={cfg['d_model']} L={cfg['n_enc_layers']}"
        offset = (i - top_n / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=label,
                       edgecolor='white', linewidth=0.5)
        if i == 0:
            for bar in bars:
                bar.set_edgecolor('black')
                bar.set_linewidth(2)

    # Add n-WEC reference bars
    nwec = NWEC_BENCHMARKS_55v5a['WEC n (NW,C,P)']
    nwec_vals = [nwec['Anon'], nwec['Neut'], nwec['Condorcet'],
                 nwec['Pareto'], nwec['Indep']]
    offset = (top_n - top_n / 2 + 0.5) * width
    ax.bar(x + offset, nwec_vals, width, label='n-WEC (55v/5a ref)',
            color='#e74c3c', alpha=0.5, edgecolor='black', linewidth=1)

    ax.set_ylabel('Satisfaction (%)')
    ax.set_title(f'Top {top_n} Configs — Per-Axiom Breakdown (20v/4a, IC)')
    ax.set_xticks(x)
    ax.set_xticklabels(axiom_names)
    ax.legend(loc='lower left', fontsize=8)
    ax.set_ylim([0, 110])
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(f"{location}/grid_results_top5.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Hyperparameter sensitivity heatmaps (Phase 1 only)
    if 'learning_rate' in results[0]['config'] and 'd_model' in results[0]['config']:
        _plot_sensitivity(results, location)

    print(f"Saved plots to: {location}/")


def _plot_sensitivity(results, location):
    """Plot hyperparameter sensitivity heatmaps."""
    # Extract unique values
    lrs = sorted(set(r['config']['learning_rate'] for r in results))
    d_models = sorted(set(r['config']['d_model'] for r in results))
    layers = sorted(set(r['config']['n_enc_layers'] for r in results))
    dropouts = sorted(set(r['config']['dropout'] for r in results))

    # Build lookup
    lookup = {}
    for r in results:
        cfg = r['config']
        key = (cfg['learning_rate'], cfg['d_model'], cfg['n_enc_layers'], cfg['dropout'])
        lookup[key] = r['final_axioms']['Average']

    # Plot: LR × d_model (averaged over layers and dropout)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Heatmap 1: LR × d_model
    data = np.zeros((len(lrs), len(d_models)))
    for i, lr in enumerate(lrs):
        for j, dm in enumerate(d_models):
            vals = [lookup.get((lr, dm, l, d), 0)
                    for l in layers for d in dropouts]
            data[i, j] = np.mean([v for v in vals if v > 0]) if any(v > 0 for v in vals) else 0

    im = axes[0].imshow(data, aspect='auto', cmap='YlOrRd')
    axes[0].set_xticks(range(len(d_models)))
    axes[0].set_xticklabels(d_models)
    axes[0].set_yticks(range(len(lrs)))
    axes[0].set_yticklabels([f"{lr:.0e}" for lr in lrs])
    axes[0].set_xlabel('d_model')
    axes[0].set_ylabel('Learning Rate')
    axes[0].set_title('Avg Axiom Sat: LR × d_model')
    for i in range(len(lrs)):
        for j in range(len(d_models)):
            axes[0].text(j, i, f"{data[i,j]:.1f}", ha='center', va='center', fontsize=10)
    plt.colorbar(im, ax=axes[0])

    # Heatmap 2: LR × n_layers
    data = np.zeros((len(lrs), len(layers)))
    for i, lr in enumerate(lrs):
        for j, nl in enumerate(layers):
            vals = [lookup.get((lr, dm, nl, d), 0)
                    for dm in d_models for d in dropouts]
            data[i, j] = np.mean([v for v in vals if v > 0]) if any(v > 0 for v in vals) else 0

    im = axes[1].imshow(data, aspect='auto', cmap='YlOrRd')
    axes[1].set_xticks(range(len(layers)))
    axes[1].set_xticklabels(layers)
    axes[1].set_yticks(range(len(lrs)))
    axes[1].set_yticklabels([f"{lr:.0e}" for lr in lrs])
    axes[1].set_xlabel('n_enc_layers')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Avg Axiom Sat: LR × Depth')
    for i in range(len(lrs)):
        for j in range(len(layers)):
            axes[1].text(j, i, f"{data[i,j]:.1f}", ha='center', va='center', fontsize=10)
    plt.colorbar(im, ax=axes[1])

    # Heatmap 3: n_layers × dropout
    data = np.zeros((len(layers), len(dropouts)))
    for i, nl in enumerate(layers):
        for j, d in enumerate(dropouts):
            vals = [lookup.get((lr, dm, nl, d), 0)
                    for lr in lrs for dm in d_models]
            data[i, j] = np.mean([v for v in vals if v > 0]) if any(v > 0 for v in vals) else 0

    im = axes[2].imshow(data, aspect='auto', cmap='YlOrRd')
    axes[2].set_xticks(range(len(dropouts)))
    axes[2].set_xticklabels(dropouts)
    axes[2].set_yticks(range(len(layers)))
    axes[2].set_yticklabels(layers)
    axes[2].set_xlabel('Dropout')
    axes[2].set_ylabel('n_enc_layers')
    axes[2].set_title('Avg Axiom Sat: Depth × Dropout')
    for i in range(len(layers)):
        for j in range(len(dropouts)):
            axes[2].text(j, i, f"{data[i,j]:.1f}", ha='center', va='center', fontsize=10)
    plt.colorbar(im, ax=axes[2])

    plt.suptitle('Hyperparameter Sensitivity (20 voters, 4 alternatives, IC)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{location}/sensitivity_heatmaps.png", dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Phase 1
    results, location = run_grid_search(
        phase='phase1',
        max_num_voters=20,
        max_num_alternatives=4,
        num_gradient_steps=10000,
        eval_interval=5000,
        random_seed=42,
    )

    if results:
        winner = results[0]
        print(f"\nPhase 1 Winner: {_config_to_str(winner['config'])}")
        print(f"Average: {winner['final_axioms']['Average']:.2f}%")
        print("\nTo run Phase 2, pass the winner config to run_grid_search(phase='phase2', ...)")