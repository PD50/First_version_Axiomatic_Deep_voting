"""
Experiment 3 with Set Transformer — Majority Tournament Encoding (v2).

Unsupervised axiom-guided voting rule synthesis using a Set Transformer
with Majority Tournament Encoding. Changes from v1:
  1. Tournament Encoding: (n, m²) one-hot → (m, m) pairwise majority margins
  2. CosineAnnealingLR scheduler for smoother convergence
  3. Independence axiom loss (weight 0.5) added to training
  4. 15,000 gradient steps (JAIR benchmark)

Includes:
- Training with axiom losses (No Winner, Condorcet, Pareto, Independence)
- Periodic evaluation every 1000 steps with axiom satisfaction tracking
- Loss evolution plots (matplotlib)
- Final comparison table including n-WEC benchmark results

Optimized for Apple M2 (8-core, 16GB RAM).
"""

import os
import sys
import time
import json
import math
import random
from datetime import datetime

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils
from generate_data import generate_profile_data
import train_and_eval
import axioms_continuous
from set_transformer_models import (
    SetTransformer,
    SetTransformer2logits,
    SetTransformer2rule,
    SetTransformer2rule_n,
)


# ============================================================
# n-WEC BENCHMARK RESULTS (from Table 5, Hornischer & Terzopoulou JAIR 2025)
# IC sampling, neutrality-averaged, averaged over 5 runs
# ============================================================
NWEC_BENCHMARKS = {
    'Blacks':        {'Anon': 100, 'Neut': 100, 'Condorcet': 100,   'Pareto': 100, 'Indep': 36.04, 'Avg': 87.2},
    'Stable Voting': {'Anon': 100, 'Neut': 100, 'Condorcet': 100,   'Pareto': 100, 'Indep': 40.48, 'Avg': 88.1},
    'Borda':         {'Anon': 100, 'Neut': 100, 'Condorcet': 93.82, 'Pareto': 100, 'Indep': 37.72, 'Avg': 86.32},
    'Weak Nanson':   {'Anon': 100, 'Neut': 100, 'Condorcet': 100,   'Pareto': 100, 'Indep': 38.28, 'Avg': 87.68},
    'Copeland':      {'Anon': 100, 'Neut': 100, 'Condorcet': 100,   'Pareto': 100, 'Indep': 28.54, 'Avg': 85.72},
    'WEC n (NW,C,P)':{'Anon': 100, 'Neut': 100, 'Condorcet': 96.78,'Pareto': 100, 'Indep': 45.9,  'Avg': 88.54},
}


def run_experiment3_settransformer(
    max_num_voters=10,
    max_num_alternatives=3,
    election_sampling=None,
    num_gradient_steps=15000,
    report_interval=1000,
    eval_dataset_size=500,
    sample_size_applicable=100,
    sample_size_maximal=int(1e5),
    batch_size=100,
    learning_rate=1e-3,
    random_seed=42,
    # Set Transformer hyperparameters
    d_model=128,
    n_heads=8,
    d_ff=256,
    n_enc_layers=2,       # Reduced: tournament input is pre-aggregated
    dropout=0.1,
    # Axiom optimization config — now includes Independence
    axiom_opt=None,
    distance='KLD',
):
    """
    Run Experiment 3 with Set Transformer + Majority Tournament Encoding.

    Returns the path to the results directory.
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
            'Independence': {'weight': 0.5, 'period': 'always'},  # NEW
        }

    start_time = time.time()

    # Set seeds
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    # Distance function
    KLD = lambda x, y: nn.KLDivLoss(log_target=True, reduction='batchmean')(
        x.log_softmax(dim=1), y.log_softmax(dim=1)
    )
    L2 = lambda x, y: (1 / len(x)) * sum(nn.PairwiseDistance(p=2)(x, y))

    if distance == 'KLD':
        distance_fn = KLD
    elif distance == 'L2':
        distance_fn = L2
    else:
        distance_fn = KLD

    # Setup results directory
    prob_model = election_sampling['probmodel']
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    location = f"./results/exp3/SetTransformer_Tournament/exp3_{current_time}_{prob_model}"
    os.makedirs(location, exist_ok=True)
    print(f"Saving location: {location}")

    # Save config
    config = {
        'architecture': 'SetTransformer_MajorityTournament',
        'encoding': 'pairwise_majority_margins',
        'input_shape': f'(batch, {max_num_alternatives}, {max_num_alternatives})',
        'max_num_voters': max_num_voters,
        'max_num_alternatives': max_num_alternatives,
        'election_sampling': election_sampling,
        'num_gradient_steps': num_gradient_steps,
        'd_model': d_model,
        'n_heads': n_heads,
        'd_ff': d_ff,
        'n_enc_layers': n_enc_layers,
        'dropout': dropout,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'lr_scheduler': 'CosineAnnealingLR',
        'axiom_opt': {k: str(v) for k, v in axiom_opt.items()},
        'distance': distance,
        'random_seed': random_seed,
    }
    with open(f"{location}/results.json", "w") as f:
        json.dump(config, f, indent=2)

    # ============================================================
    # Generate dev/test data
    # ============================================================
    print("Now generate dev and test profiles")
    X_dev_profs, _, _ = generate_profile_data(
        max_num_voters, max_num_alternatives, eval_dataset_size,
        election_sampling, [], merge='empty',
    )
    X_test_profs, _, _ = generate_profile_data(
        max_num_voters, max_num_alternatives, eval_dataset_size,
        election_sampling, [], merge='empty',
    )

    # ============================================================
    # Initialize model
    # ============================================================
    print("Initializing Set Transformer (Tournament Encoding)...")
    model = SetTransformer(
        max_num_voters=max_num_voters,
        max_num_alternatives=max_num_alternatives,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_enc_layers=n_enc_layers,
        dropout=dropout,
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {num_params:,}")
    print(f"  Input shape: (batch, {max_num_alternatives}, {max_num_alternatives})")
    print(f"  Encoding: Pairwise Majority Margins")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # ============================================================
    # CosineAnnealingLR Scheduler
    # T_max = total gradient steps → LR decays from lr to ~0 over training
    # ============================================================
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_gradient_steps, eta_min=1e-6
    )
    print(f"  LR Scheduler: CosineAnnealingLR (T_max={num_gradient_steps}, eta_min=1e-6)")

    model_on_profiles = lambda X: SetTransformer2logits(model, X)

    # ============================================================
    # Training loop with tracking
    # ============================================================
    print("Now starting to train")

    loss_history = {
        'step': [],
        'total_loss': [],
        'loss_nw': [],
        'loss_cond': [],
        'loss_pareto': [],
        'loss_indep': [],     # NEW: track Independence loss
        'lr': [],             # NEW: track learning rate
    }

    axiom_history = {
        'step': [],
        'Anonymity': [],
        'Neutrality': [],
        'Condorcet': [],
        'Pareto': [],
        'Independence': [],
        'Average': [],
    }

    axioms_to_check = ['Anonymity', 'Neutrality', 'Condorcet', 'Pareto', 'Independence']

    for step in tqdm(range(num_gradient_steps), desc="Training"):
        model.train()

        # Generate batch of profiles
        X_batch, _, _ = generate_profile_data(
            max_num_voters, max_num_alternatives, batch_size,
            election_sampling, [], merge='empty',
        )

        # Compute axiom losses
        loss_nw = torch.tensor([0.0])
        loss_cond = torch.tensor([0.0])
        loss_pareto = torch.tensor([0.0])
        loss_indep = torch.tensor([0.0])

        # --- No Winner loss ---
        if axiom_opt['No_winner'] is not None:
            nw_cfg = axiom_opt['No_winner']
            if nw_cfg['period'] == 'always' or step in range(
                nw_cfg['period'].get('from', 0), nw_cfg['period'].get('until', num_gradient_steps)
            ):
                loss_nw = nw_cfg['weight'] * axioms_continuous.ax_no_winners_cont(
                    model_on_profiles, X_batch
                )

        # --- Condorcet loss ---
        if axiom_opt['Condorcet1'] is not None:
            c_cfg = axiom_opt['Condorcet1']
            if c_cfg['period'] == 'always' or step in range(
                c_cfg['period'].get('from', 0), c_cfg['period'].get('until', num_gradient_steps)
            ):
                loss_cond = c_cfg['weight'] * axioms_continuous.ax_condorcet1_cont(
                    model_on_profiles, X_batch, distance_fn
                )

        # --- Pareto loss ---
        if axiom_opt['Pareto2'] is not None:
            p_cfg = axiom_opt['Pareto2']
            if p_cfg['period'] == 'always' or step in range(
                p_cfg['period'].get('from', 0), p_cfg['period'].get('until', num_gradient_steps)
            ):
                loss_pareto = p_cfg['weight'] * axioms_continuous.ax_pareto2_cont(
                    model_on_profiles, X_batch, distance_fn
                )

        # --- Independence loss (NEW) ---
        if axiom_opt['Independence'] is not None:
            i_cfg = axiom_opt['Independence']
            if i_cfg['period'] == 'always' or step in range(
                i_cfg['period'].get('from', 0), i_cfg['period'].get('until', num_gradient_steps)
            ):
                loss_indep = i_cfg['weight'] * axioms_continuous.ax_independence_cont(
                    model_on_profiles, X_batch, 1, distance_fn
                )

        total_loss = loss_nw + loss_cond + loss_pareto + loss_indep

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()   # Step the cosine annealing scheduler

        # Record losses every 100 steps
        if step % 100 == 0:
            loss_history['step'].append(step)
            loss_history['total_loss'].append(total_loss.item())
            loss_history['loss_nw'].append(loss_nw.item())
            loss_history['loss_cond'].append(loss_cond.item())
            loss_history['loss_pareto'].append(loss_pareto.item())
            loss_history['loss_indep'].append(loss_indep.item())
            loss_history['lr'].append(optimizer.param_groups[0]['lr'])

        # Print progress every 1000 steps
        if (step + 1) % 1000 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\nStep {step+1}/{num_gradient_steps} | "
                  f"Loss: {total_loss.item():.4f} | "
                  f"NW: {loss_nw.item():.3f} | "
                  f"Cond: {loss_cond.item():.3f} | "
                  f"Pareto: {loss_pareto.item():.3f} | "
                  f"Indep: {loss_indep.item():.3f} | "
                  f"LR: {current_lr:.2e}")

        # ============================================================
        # Periodic axiom evaluation every report_interval steps
        # ============================================================
        if (step + 1) % report_interval == 0:
            print(f"\n--- Evaluating at step {step+1} ---")
            model_rule_n = SetTransformer2rule_n(model, None)

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
                print(f"    {ax_name}: {axiom_results[ax_name]}%")

            avg = round(sum(axiom_results.values()) / len(axiom_results), 2)
            axiom_results['Average'] = avg
            print(f"    Average: {avg}%")

            axiom_history['step'].append(step + 1)
            for ax_name in axioms_to_check:
                axiom_history[ax_name].append(axiom_results[ax_name])
            axiom_history['Average'].append(avg)

    # ============================================================
    # Final evaluation
    # ============================================================
    print("\n=== FINAL EVALUATION ===")
    model_rule_n = SetTransformer2rule_n(model, None)

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
        print(f"    {ax_name}: {final_axioms[ax_name]}%")

    avg = round(sum(final_axioms.values()) / len(final_axioms), 2)
    final_axioms['Average'] = avg
    print(f"    Average: {avg}%")

    # ============================================================
    # Save model
    # ============================================================
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config,
    }, f"{location}/model.pth")

    # ============================================================
    # Generate plots
    # ============================================================
    _plot_training_progress(loss_history, axiom_history, location)
    _plot_final_comparison(final_axioms, location)

    # ============================================================
    # Print final comparison table
    # ============================================================
    _print_comparison_table(final_axioms)

    # Save all results
    end_time = time.time()
    with open(f"{location}/results.json") as f:
        data = json.load(f)
    data.update({
        'final_axiom_satisfaction': final_axioms,
        'axiom_history': axiom_history,
        'loss_history': loss_history,
        'runtime_sec': end_time - start_time,
        'nwec_benchmarks': NWEC_BENCHMARKS,
    })
    with open(f"{location}/results.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nRuntime: {round((end_time - start_time) / 60, 1)} minutes")
    print(f"Results saved to: {location}")

    return location


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def _plot_training_progress(loss_history, axiom_history, location):
    """
    Plot 1: Loss evolution (top), LR schedule (middle),
    and axiom satisfaction evolution (bottom).
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))

    # --- Top: Loss curves ---
    ax1 = axes[0]
    ax1.plot(loss_history['step'], loss_history['total_loss'],
             label='Total Loss', color='black', linewidth=2)
    ax1.plot(loss_history['step'], loss_history['loss_nw'],
             label='No Winner (×10)', color='red', linewidth=1, alpha=0.7)
    ax1.plot(loss_history['step'], loss_history['loss_cond'],
             label='Condorcet (×2)', color='blue', linewidth=1, alpha=0.7)
    ax1.plot(loss_history['step'], loss_history['loss_pareto'],
             label='Pareto (×1)', color='green', linewidth=1, alpha=0.7)
    ax1.plot(loss_history['step'], loss_history['loss_indep'],
             label='Independence (×0.5)', color='purple', linewidth=1, alpha=0.7)
    ax1.set_xlabel('Gradient Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Set Transformer (Tournament) — Loss Evolution During Training')
    ax1.legend(loc='upper right')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # --- Middle: Learning rate schedule ---
    ax_lr = axes[1]
    ax_lr.plot(loss_history['step'], loss_history['lr'],
               color='darkorange', linewidth=2)
    ax_lr.set_xlabel('Gradient Steps')
    ax_lr.set_ylabel('Learning Rate')
    ax_lr.set_title('CosineAnnealingLR Schedule')
    ax_lr.set_yscale('log')
    ax_lr.grid(True, alpha=0.3)

    # --- Bottom: Axiom satisfaction over time ---
    ax2 = axes[2]
    colors = {
        'Anonymity': '#2ecc71',
        'Neutrality': '#e74c3c',
        'Condorcet': '#3498db',
        'Pareto': '#f39c12',
        'Independence': '#9b59b6',
        'Average': '#2c3e50',
    }
    for ax_name in ['Anonymity', 'Neutrality', 'Condorcet', 'Pareto', 'Independence', 'Average']:
        style = '-' if ax_name != 'Average' else '--'
        lw = 1.5 if ax_name != 'Average' else 2.5
        ax2.plot(
            axiom_history['step'],
            axiom_history[ax_name],
            label=ax_name,
            color=colors[ax_name],
            linestyle=style,
            linewidth=lw,
            marker='o',
            markersize=4,
        )

    # Add n-WEC average as horizontal reference line
    ax2.axhline(y=88.54, color='gray', linestyle=':', linewidth=1.5,
                label='n-WEC Avg (88.54%)')

    ax2.set_xlabel('Gradient Steps')
    ax2.set_ylabel('Axiom Satisfaction (%)')
    ax2.set_title('Set Transformer (Tournament) — Axiom Satisfaction Every 1000 Steps')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.set_ylim([0, 105])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"{location}/training_progress.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {path}")


def _plot_final_comparison(final_axioms, location):
    """
    Plot 2: Bar chart comparing Set Transformer (Tournament) vs n-WEC and classical rules.
    """
    methods = list(NWEC_BENCHMARKS.keys()) + ['SetTrans-T n (NW,C,P,I)']
    axiom_names = ['Anon', 'Neut', 'Condorcet', 'Pareto', 'Indep', 'Avg']

    # Build data matrix
    data = []
    for method in methods[:-1]:
        row = NWEC_BENCHMARKS[method]
        data.append([row['Anon'], row['Neut'], row['Condorcet'],
                      row['Pareto'], row['Indep'], row['Avg']])
    # Add our model
    data.append([
        final_axioms['Anonymity'],
        final_axioms['Neutrality'],
        final_axioms['Condorcet'],
        final_axioms['Pareto'],
        final_axioms['Independence'],
        final_axioms['Average'],
    ])

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(axiom_names))
    width = 0.12
    n_methods = len(methods)

    colors_bar = ['#95a5a6', '#3498db', '#e67e22', '#2ecc71', '#9b59b6', '#e74c3c', '#1abc9c']

    for i, (method, values) in enumerate(zip(methods, data)):
        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=method, color=colors_bar[i % len(colors_bar)],
                       edgecolor='white', linewidth=0.5)
        # Bold outline for our model
        if method == 'SetTrans-T n (NW,C,P,I)':
            for bar in bars:
                bar.set_edgecolor('black')
                bar.set_linewidth(2)

    ax.set_ylabel('Satisfaction (%)')
    ax.set_title('Experiment 3: Axiom Satisfaction — Tournament Encoding vs n-WEC (IC, n-averaged)')
    ax.set_xticks(x)
    ax.set_xticklabels(axiom_names)
    ax.legend(loc='lower left', fontsize=8)
    ax.set_ylim([0, 110])
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    path = f"{location}/final_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {path}")


def _print_comparison_table(final_axioms):
    """Print the final comparison table in a clean format with n-WEC benchmarks."""
    print("\n" + "=" * 85)
    print("FINAL COMPARISON TABLE")
    print("=" * 85)
    header = f"{'Method':<30} {'Anon.':>6} {'Neut.':>6} {'Condorcet':>10} {'Pareto':>7} {'Indep.':>7} {'Avg.':>7}"
    print(header)
    print("-" * 85)

    for method, vals in NWEC_BENCHMARKS.items():
        print(f"{method:<30} {vals['Anon']:>6.1f} {vals['Neut']:>6.1f} "
              f"{vals['Condorcet']:>10.2f} {vals['Pareto']:>7.1f} "
              f"{vals['Indep']:>7.2f} {vals['Avg']:>7.2f}")

    print("-" * 85)
    print(f"{'SetTrans-T n (NW,C,P,I)':<30} "
          f"{final_axioms['Anonymity']:>6.1f} "
          f"{final_axioms['Neutrality']:>6.1f} "
          f"{final_axioms['Condorcet']:>10.2f} "
          f"{final_axioms['Pareto']:>7.1f} "
          f"{final_axioms['Independence']:>7.2f} "
          f"{final_axioms['Average']:>7.2f}")
    print("=" * 85)

    # Highlight comparison
    wec_avg = NWEC_BENCHMARKS['WEC n (NW,C,P)']['Avg']
    our_avg = final_axioms['Average']
    diff = our_avg - wec_avg
    symbol = "▲" if diff > 0 else "▼" if diff < 0 else "="
    print(f"\n  vs n-WEC:  {symbol} {abs(diff):.2f}% {'improvement' if diff > 0 else 'difference'} in average axiom satisfaction")

    # Independence-specific comparison
    wec_indep = NWEC_BENCHMARKS['WEC n (NW,C,P)']['Indep']
    our_indep = final_axioms['Independence']
    diff_indep = our_indep - wec_indep
    symbol_indep = "▲" if diff_indep > 0 else "▼" if diff_indep < 0 else "="
    print(f"  vs n-WEC (Indep): {symbol_indep} {abs(diff_indep):.2f}% {'improvement' if diff_indep > 0 else 'difference'}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_experiment3_settransformer()