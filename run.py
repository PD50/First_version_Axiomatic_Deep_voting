"""
run.py — Robustness Experiments for Axiomatic Deep Voting
=========================================================

Runs the Set-Transformer (with PMA) on Experiment 3 from Hornischer &
Terzopoulou (JAIR 2025), using the architecture from Anil & Bao (2021).

Implements four supervisor-directed robustness checks:
  1. Distribution check: IC vs Mallows (MALLOWS-RELPHI)
  2. Axiom ablation: NW+C+P baseline vs NW+C+P+I
  3. n-WEC baseline comparison in every output table
  4. Extended training: axiom satisfaction every 2,500 steps up to 15,000

The Set-Transformer is anonymous by design (permutation-invariant over
voters via PMA pooling).  Neutrality is achieved via neutrality-averaged
decoding, identical to the WEC's approach.

Apple M2 compatible — runs on CPU (same as the JAIR paper's setup).
"""

import os
import json
import time
from datetime import datetime

import torch
import numpy as np
import random

from exp3 import experiment3


# ═══════════════════════════════════════════════════════════════════════
#  SHARED PARAMETERS  (Section 5 of Hornischer & Terzopoulou)
# ═══════════════════════════════════════════════════════════════════════

MAX_NUM_VOTERS = 55
MAX_NUM_ALTERNATIVES = 5
BATCH_SIZE = 200
LEARNING_RATE = 1e-3
NUM_GRADIENT_STEPS = 15_000
REPORT_INTERVAL = 2_500          # axiom check every 2,500 steps
EVAL_DATASET_SIZE = 500
LOSS_REPORT_INTERVAL = 500
RANDOM_SEED = 42
LEARNING_SCHEDULER = 1_000      # T_0 for CosineAnnealingWarmRestarts
WEIGHT_DECAY = 0

SAMPLE_SIZE_APPLICABLE = 400    # Section 5.3
SAMPLE_SIZE_MAXIMAL = 10_000

AXIOMS_CHECK_MODEL = [
    'Anonymity', 'Neutrality', 'Condorcet', 'Pareto', 'Independence'
]
AXIOMS_CHECK_RULE = ['Condorcet', 'Independence']

COMP_RULES_AXIOMS = [
    'Stable Voting', 'Blacks', 'Borda', 'Weak Nanson',
    'Copeland', 'Baldwin', 'Kemeny-Young'
]
COMP_RULES_SIMILARITY = [
    'Stable Voting', 'Blacks', 'Borda', 'Weak Nanson', 'Copeland'
]

# Set-Transformer architecture (Anil & Bao, Appendix D.1)
# Scaled to be comparable in expressiveness to the WEC while fitting
# on M2 memory. 4 SAB encoder layers + PMA decoder, pre-LN variant.
SET_TRANSFORMER_PARAMS = {
    'd_model': 128,
    'n_heads': 4,
    'd_head': 32,
    'n_encoder_layers': 4,
    'pma_outputs': 1,
}

MODEL_TO_RULE = {
    'plain': True,
    'neut-averaged': None,       # None = exact (all m! permutations)
    'neut-anon-averaged': False,
}


# ═══════════════════════════════════════════════════════════════════════
#  DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════

DIST_IC = {'probmodel': 'IC'}
DIST_MALLOWS = {'probmodel': 'MALLOWS-RELPHI'}


# ═══════════════════════════════════════════════════════════════════════
#  AXIOM LOSS CONFIGS
# ═══════════════════════════════════════════════════════════════════════

def axiom_opt_NW_C_P():
    """Baseline: No-winner + Condorcet + Pareto"""
    return {
        'No_winner':     {'weight': 10, 'period': 'always'},
        'All_winners':   None,
        'Inadmissible':  None,
        'Resoluteness':  None,
        'Parity':        None,
        'Anonymity':     None,
        'Neutrality':    None,
        'Condorcet1':    {'weight': 1, 'period': 'always'},
        'Condorcet2':    None,
        'Pareto1':       {'weight': 1, 'period': 'always'},
        'Pareto2':       None,
        'Independence':  None,
    }

def axiom_opt_NW_C_P_I():
    """Ablation: NW + C + P + Independence"""
    return {
        'No_winner':     {'weight': 10, 'period': 'always'},
        'All_winners':   None,
        'Inadmissible':  None,
        'Resoluteness':  None,
        'Parity':        None,
        'Anonymity':     None,
        'Neutrality':    None,
        'Condorcet1':    {'weight': 1, 'period': 'always'},
        'Condorcet2':    None,
        'Pareto1':       {'weight': 1, 'period': 'always'},
        'Pareto2':       None,
        'Independence':  {'weight': 1, 'period': 'always', 'sample': 50},
    }


# ═══════════════════════════════════════════════════════════════════════
#  RUN DEFINITIONS  (2 distributions × 2 axiom sets = 4 runs)
# ═══════════════════════════════════════════════════════════════════════

RUNS = {
    'A_IC_NW_C_P': {
        'dist': DIST_IC,
        'axiom_fn': axiom_opt_NW_C_P,
        'tag': 'SetTrans n | IC      | NW+C+P',
    },
    'B_MALLOWS_NW_C_P': {
        'dist': DIST_MALLOWS,
        'axiom_fn': axiom_opt_NW_C_P,
        'tag': 'SetTrans n | Mallows | NW+C+P',
    },
    'C_IC_NW_C_P_I': {
        'dist': DIST_IC,
        'axiom_fn': axiom_opt_NW_C_P_I,
        'tag': 'SetTrans n | IC      | NW+C+P+I',
    },
    'D_MALLOWS_NW_C_P_I': {
        'dist': DIST_MALLOWS,
        'axiom_fn': axiom_opt_NW_C_P_I,
        'tag': 'SetTrans n | Mallows | NW+C+P+I',
    },
}


# ═══════════════════════════════════════════════════════════════════════
#  n-WEC REFERENCE  (Table 1 & Table 6 from JAIR paper)
# ═══════════════════════════════════════════════════════════════════════

NWEC_REF = {
    'n-WEC (IC)': {
        'Anonymity': 100.0, 'Neutrality': 100.0, 'Condorcet': 96.8,
        'Pareto': 100.0, 'Independence': 41.2, 'Avg': 87.6,
    },
    'n-WEC (Mallows)': {
        'Anonymity': 100.0, 'Neutrality': 100.0, 'Condorcet': 97.0,
        'Pareto': 100.0, 'Independence': 44.0, 'Avg': 88.2,
    },
}


# ═══════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════

def print_table(header, rows, col_width=8):
    fmt = f'{{:<38}}' + f'{{:>{col_width}}}' * (len(header) - 1)
    print(fmt.format(*header))
    print('-' * (38 + col_width * (len(header) - 1)))
    for row in rows:
        print(fmt.format(*row))


def extract_scores(ax_sat_dict):
    scores = {}
    for ax in AXIOMS_CHECK_MODEL:
        e = ax_sat_dict.get(ax, {})
        scores[ax] = round(100.0 * e.get('cond_satisfaction', 0.0), 1)
    scores['Avg'] = round(sum(scores.values()) / len(scores), 1)
    return scores


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    os.makedirs('./results/exp3/SetTransformer', exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    summary_dir = f'./results/robustness_{timestamp}'
    os.makedirs(summary_dir, exist_ok=True)

    all_results = {}

    for run_name, cfg in RUNS.items():
        print('\n' + '=' * 72)
        print(f'  RUN: {run_name}  ({cfg["tag"]})')
        print('=' * 72 + '\n')

        t0 = time.time()

        location = experiment3(
            architecture='SetTransformer',
            max_num_voters=MAX_NUM_VOTERS,
            max_num_alternatives=MAX_NUM_ALTERNATIVES,
            election_sampling=cfg['dist'],
            num_gradient_steps=NUM_GRADIENT_STEPS,
            report_intervals=REPORT_INTERVAL,
            eval_dataset_size=EVAL_DATASET_SIZE,
            model_to_rule=MODEL_TO_RULE,
            sample_size_applicable=SAMPLE_SIZE_APPLICABLE,
            sample_size_maximal=SAMPLE_SIZE_MAXIMAL,
            architecture_parameters=SET_TRANSFORMER_PARAMS,
            axioms_check_model=AXIOMS_CHECK_MODEL,
            axioms_check_rule=AXIOMS_CHECK_RULE,
            axiom_opt=cfg['axiom_fn'](),
            comp_rules_axioms=COMP_RULES_AXIOMS,
            comp_rules_similarity=COMP_RULES_SIMILARITY,
            distance='L2',
            random_seed=RANDOM_SEED,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            learning_scheduler=LEARNING_SCHEDULER,
            weight_decay=WEIGHT_DECAY,
            loss_report_intervals=LOSS_REPORT_INTERVAL,
            save_model=True,
        )

        elapsed = time.time() - t0
        print(f'\n  {run_name} done in {elapsed/60:.1f} min → {location}\n')

        with open(f'{location}/results.json') as f:
            run_data = json.load(f)

        all_results[run_name] = {
            'tag': cfg['tag'],
            'location': location,
            'elapsed_min': round(elapsed / 60, 1),
            'results': run_data,
        }

    # ─── UNIFIED COMPARISON TABLE ───────────────────────────────────
    print('\n\n' + '=' * 72)
    print('  FINAL COMPARISON TABLE  (Set-Transformer + n-WEC baseline)')
    print('=' * 72 + '\n')

    header = ['Rule / Model', 'Anon.', 'Neut.', 'Cond.', 'Pareto', 'Indep.', 'Avg.']
    rows = []

    for label, sc in NWEC_REF.items():
        rows.append([label] + [f'{sc[k]:.1f}' for k in
            ['Anonymity','Neutrality','Condorcet','Pareto','Independence','Avg']])

    rows.append(['─' * 38] + ['─' * 8] * 6)

    for run_name, data in all_results.items():
        ax_sat = data['results'].get('axiom_satisfaction', {})
        m_sat = ax_sat.get('model_neut') or ax_sat.get('model_plain')
        if m_sat is None:
            rows.append([data['tag']] + ['N/A'] * 6)
            continue
        sc = extract_scores(m_sat)
        rows.append([data['tag']] + [f'{sc[k]:.1f}' for k in
            ['Anonymity','Neutrality','Condorcet','Pareto','Independence','Avg']])

    print_table(header, rows)

    # ─── LEARNING CURVES ────────────────────────────────────────────
    print('\n\n' + '=' * 72)
    print('  LEARNING CURVES (axiom sat. at each 2,500-step checkpoint)')
    print('=' * 72)

    for run_name, data in all_results.items():
        lc = data['results'].get('learning curve', {})
        if not lc:
            continue
        print(f'\n  {data["tag"]}:')
        print(f'  {"Step":>8}  {"Anon":>7}  {"Neut":>7}  {"Cond":>7}  {"Par":>7}  {"Indep":>7}')
        for step_key in sorted(lc.keys(), key=lambda x: int(x)):
            ax_s = lc[step_key].get('axiom_satisfaction', {})
            m_s = ax_s.get('neut') or ax_s.get('plain')
            if m_s is None:
                continue
            vals = []
            for ax in AXIOMS_CHECK_MODEL:
                e = m_s.get(ax, {})
                vals.append(100.0 * e.get('cond_satisfaction', 0.0))
            print(f'  {int(step_key)+1:>8}' + ''.join(f'  {v:>7.1f}' for v in vals))

    # ─── SAVE ───────────────────────────────────────────────────────
    p = f'{summary_dir}/summary.json'
    with open(p, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\n\nSummary → {p}\nDone.')


if __name__ == '__main__':
    main()