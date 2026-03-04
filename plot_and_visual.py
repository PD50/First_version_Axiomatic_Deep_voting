"""
Different functions to plot and otherwise visualize the data obtained from
the experiments
"""

import os 
from datetime import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

import torch
from gensim.models import Word2Vec

import utils
import models
from models import MLP, CNN, WEC
import train_and_eval
from generate_data import generate_profile_data


def plot_exp1(file_plurality, file_borda, file_copeland, ylim = None):
    """
    Creates barplot of data in the three specified files

    If ylim is set to, e.g., to [-15, 40], then the y-axis with the error 
    values ranges from -15 to 40. If ylim is left to default None, the range
    is determined automatically based on the values in the datafiles.
    """
    
    # (1) Load result json files
    with open(file_plurality +'/results.json', 'r') as file:
        results_plurality = json.load(file)
    with open(file_borda +'/results.json', 'r') as file:
        results_borda = json.load(file)
    with open(file_copeland +'/results.json', 'r') as file:
        results_copeland = json.load(file)
    # and combine into a single dictionary
    results_rules = {
        'Plurality': results_plurality,
        'Borda': results_borda,
        'Copeland': results_copeland,
    }


    # (2) Check that the files are compatible and load basic data
    assert (results_plurality['rule_names'] == ["Plurality"]
        ), f'The file_plurality should use Plurality as only rule'
    assert (results_borda['rule_names'] == ["Borda"]
        ), f'The file_borda should use Borda as only rule' 
    assert (results_copeland['rule_names'] == ["Copeland"]
        ), f'The file_copeland should use Copeland as only rule'
    
    architectures = set([result['architecture'] 
                    for result in results_rules.values()])
    max_nums_voters = set([result['max_num_voters'] 
                    for result in results_rules.values()])
    max_nums_alternatives = set([result['max_num_alternatives'] 
                            for result in results_rules.values()])
    election_samplings = set([result['election_sampling']['probmodel'] 
                            for result in results_rules.values()])

    assert (len(architectures) == 1
        ), 'The provided files do not agree in their architecture'
    assert (len(max_nums_voters) == 1
        ), 'The provided files do not agree in their max_num_voters'
    assert (len(max_nums_alternatives) == 1
        ), 'The provided files do not agree in their max_num_alternatives'
    assert (len(election_samplings) == 1
        ), 'The provided files do not agree in their election_sampling'

    architecture = list(architectures)[0]
    max_num_voters = list(max_nums_voters)[0]
    max_num_alternatives = list(max_nums_alternatives)[0]
    election_sampling = list(election_samplings)[0]
    # rename election_sampling into more colloquial name
    election_sampling = utils.dict_sampling_methods[election_sampling]


    # (3) Gather the data for the plot
    categories = [
        'Accu. (identity)', #'Accuracy (identity)'
        'Accu. (subset)', #'Accuracy (subset)'
        'Anonymity',
        'Neutrality',
        'Pareto',
        'Condorcet',
        'Independ.' #'Independence' 
    ]

    values = {}

    for rule, results in results_rules.items():
        errors = [
        # Accuracy (identity) error
        1
        -
        results['rule_comparison'][rule]
            ['identity_accu'],
        # Accuracy (subset) error
        1
        -
        results['rule_comparison'][rule]
            ['subset_accu'],
        # Anonymity error (rule minus model)
        1
        -
        results['axiom_satisfaction']['learned_rule']
            ['Anonymity']['cond_satisfaction'],
        # Neutrality error (rule minus model)
        1
        -
        results['axiom_satisfaction']['learned_rule']
            ['Neutrality']['cond_satisfaction'],
        # Pareto error (rule minus model)
        1
        -
        results['axiom_satisfaction']['learned_rule']
            ['Pareto']['cond_satisfaction'],
        # Condorcet error (rule minus model)
        results['axiom_satisfaction'][rule]
            ['Condorcet']['cond_satisfaction']
        -
        results['axiom_satisfaction']['learned_rule']
            ['Condorcet']['cond_satisfaction'],
        # Independence error (rule minus model)
        results['axiom_satisfaction'][rule]
            ['Independence']['cond_satisfaction']
        -
        results['axiom_satisfaction']['learned_rule']
            ['Independence']['cond_satisfaction'],
        ]
        values[rule] = [round(100*e,1) for e in errors]

    # (4) Build the plot
    x = np.arange(len(categories))  # the label locations
    width = 0.3  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in values.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3, size=8)
        multiplier += 1

    ax.set_ylabel('Error in %') #, fontsize=12)
    ax.set_title(f'Exp. 1: {architecture} up to {max_num_voters} voters and {max_num_alternatives} alternatives ({election_sampling} sampling)')#, fontsize=14)
    ax.set_xticks(x + width, categories, rotation=45) #, labelsize=12)
    if ylim is not None:
        ax.set_ylim(ylim[0],ylim[1])
    ax.legend() 

    # (5) Save the plot
    # time stamp
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 
    # metadata (to save png images this requires backend 'agg',
    # which is default anyways)
    # To access metadata later, use imagemagick and run
    # $ identify -verbose image.png
    metadata = {
        'plotted_plurality_data_from': file_plurality,
        'plotted_borda_data_from': file_borda,
        'plotted_copeland_data_from': file_copeland,    
    }

    plt.savefig(f'./results/exp1/plots/plot_{current_time}_{architecture}_{max_num_voters}_{max_num_alternatives}_{election_sampling}.png',
        metadata = metadata,
        backend = 'agg', # default
        dpi=300, 
        bbox_inches = 'tight'
    )

    return







def plot_exp1_learning_curve(
        file_location_MLP_small,
        file_location_MLP_standard,
        file_location_MLP_large,
    ):
    list_of_file_locations = [
        file_location_MLP_small,
        file_location_MLP_standard,
        file_location_MLP_large,
    ]

    data = {}

    for file_location in list_of_file_locations:
        # (1) Load result json files
        with open(file_location +'/results.json', 'r') as file:
            results = json.load(file)

        # (2) Load basic data
        architecture = results['architecture']
        rule = results['rule_names'][0]
        max_num_voters = results['max_num_voters']
        max_num_alternatives = results['max_num_alternatives']
        election_sampling = results['election_sampling']['probmodel']

        # (3) Gather the data for the plot
        learning_curve = results['learning curve']
        gradient_steps = []
        accuracy = []
        loss = []
        for k,v in learning_curve.items():
            gradient_steps.append(float(k))
            accuracy.append(float(v['dev_accuracy']))
            loss.append(float(v['dev_loss']))

        data[architecture] = [gradient_steps, accuracy, loss]

    # (4) Plot
    fig = plt.figure()
    colors = plt.cm.tab10.colors
    ax = fig.add_subplot(111)
    ln0 = ax.plot(
        data['MLP_small'][0],
        data['MLP_small'][1],
        '-', 
        alpha=.7,
        color = colors[0],
        label = 'Accuracy MLP (small)'
    )
    ln1 = ax.plot(
        data['MLP'][0],
        data['MLP'][1],
        '-', 
        alpha=.7,
        color = colors[1],
        label = 'Accuracy MLP (standard)'
    )
    ln2 = ax.plot(
        data['MLP_large'][0],
        data['MLP_large'][1],
        '-', 
        alpha=.7,
        color = colors[2],
        label = 'Accuracy MLP (large)'
    )

    ax2 = ax.twinx()
    ln3 = ax.plot(
        data['MLP_small'][0],
        data['MLP_small'][2],
        '--', 
        alpha=.7,
        color = colors[0],
        label = 'Loss MLP (small)'
    )
    ln4 = ax.plot(
        data['MLP'][0],
        data['MLP'][2],
        '--', 
        alpha=.7,
        color = colors[1],
        label = 'Loss MLP (standard)'
    )    
    ln5 = ax.plot(
        data['MLP_large'][0],
        data['MLP_large'][2],
        '--', 
        alpha=.7,
        color = colors[2],
        label = 'Loss MLP (large)'
    )
       
    lns = ln0+ln1+ln2+ln3+ln4+ln5
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)
    
    ax.grid()
    
    ax.set_xlabel('Gradient steps')
    ax.set_ylabel('Accuracy (on dev set)')
    ax2.set_ylabel('Loss (on dev set)')
    ax.set_title(f'MLPs learning {rule} ({election_sampling}, up to {max_num_voters} vot. & {max_num_alternatives} alt.)')


    # (5) Save the plot
    # time stamp
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # metadata (to save png images this requires backend 'agg',
    # which is default anyways)
    # To access metadata later, use imagemagick and run
    # $ identify -verbose image.png
    metadata = {'plotted_data_from':file_location}

    plt.savefig(f'./results/exp1/plots/plot_{current_time}_learning_curve_{max_num_voters}_{max_num_alternatives}_{election_sampling}.png',
        metadata = metadata,
        backend = 'agg', # default
        dpi=300, 
        bbox_inches = 'tight'
    )

    return



def plot_exp1_fixed_learning_curve(file_location):
    # (1) Load result json files
    with open(file_location +'/results.json', 'r') as file:
        results = json.load(file)

    # (2) Load basic data
    list_of_architectures = results['list_of_architectures']
    rule = results['rule_names'][0]
    max_num_voters = results['max_num_voters']
    max_num_alternatives = results['max_num_alternatives']
    election_sampling = results['election_sampling']['probmodel']

    # (3) Gather the data for the plot
    data = {}
    for i in range(len(list_of_architectures)):
        architecture = list_of_architectures[i]
        color = plt.cm.tab10.colors[i]
        learning_curve = results[architecture + '_training']['learning curve']
        epochs = []
        train_accuracy = []
        dev_accuracy = []
        for k,v in learning_curve.items():
            epochs.append(float(k))
            train_accuracy.append(float(v['train_accuracy']))
            dev_accuracy.append(float(v['dev_accuracy']))
        data[architecture] = {
            'epochs':epochs,
            'train_accu':train_accuracy,
            'dev_accu':dev_accuracy,
            'color':color
        }
    # (4) Plot

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for architecture in list_of_architectures:
        ax.plot(
            data[architecture]['epochs'],
            data[architecture]['train_accu'],
            '--', 
            color = data[architecture]['color'],
            label = f'{architecture} train accu'
        )
        ax.plot(
            data[architecture]['epochs'],
            data[architecture]['dev_accu'],
            '-', 
            color = data[architecture]['color'],
            label = f'{architecture} dev accu')

    ax.legend(loc=0)
    ax.grid()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Learning {rule} with fixed dataset ({election_sampling}, up to {max_num_voters} vot. & {max_num_alternatives} alt.)')
    # ax.set_title(f'Learning {rule} up to {max_num_voters} voters and {max_num_alternatives} alternatives ({election_sampling} sampling, fixed dataset)')

    # (5) Save the plot
    # time stamp
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # metadata (to save png images this requires backend 'agg',
    # which is default anyways)
    # To access metadata later, use imagemagick and run
    # $ identify -verbose image.png
    metadata = {'plotted_data_from':file_location}

    plt.savefig(f'./results/exp1/plots/plot_{current_time}_fixed_learning_curve.png',
        metadata = metadata,
        backend = 'agg', # default
        dpi=300, 
        bbox_inches = 'tight'
    )
    return




def exp1_table_crossval(file_location):
    """
    Table of results on different folds with average
    """


    # Get results:
    with open(f"{file_location}/results.json") as json_file:
        data = json.load(json_file)

    num_folds = data['num_folds']
    list_of_architectures = data['list_of_architectures']

    for architecture in list_of_architectures:
        fold_data = []

        for fold in range(num_folds):
            # Get relevant data from experiment data
            fold_data_row = [
                fold,
                data[f'{architecture}_fold_{fold}']['result']['train_loss'],
                100*data[f'{architecture}_fold_{fold}']['result']['train_accuracy'],
                data[f'{architecture}_fold_{fold}']['result']['test_loss'],
                100*data[f'{architecture}_fold_{fold}']['result']['test_accuracy'],
            ]
            fold_data.append(fold_data_row)

        # Add averages and mean
        avg_row = ['Avg.']
        std_row = ['Std. dev.']
        for column in [1,2,3,4]: #column 0 contains number of fold
            values = [fold_data[row][column] for row in range(num_folds)]
            mean = np.mean(np.array(values))
            avg_row.append(mean)
            std_dev = np.std(np.array(values))
            std_row.append(std_dev)
        fold_data += [avg_row, std_row]


        # Columns of dataframe
        columns = [
            'Testing fold number',
            'Train loss',
            'Train accuracy (in %)',
            'Test loss',
            'Test accuracy (in %)',
        ]    

        # Generate dataframe
        df = pd.DataFrame(
            data=fold_data,
            # index=[fold for fold in range(num_folds+1)],
            columns=columns
        )
        df = df.round({
            'Train loss':3,
            'Train accuracy (in %)':1,
            'Test loss':3,
            'Test accuracy (in %)':1,
        })

        # Print dataframe in latex table format
        # As headers we abbreviate the content of df.columns
        print(f'{architecture}')
        print(tabulate(
            df,
            headers=columns,
            tablefmt='latex_raw',
            numalign='center',
            showindex = False,
        ))

    return




def plot_exp1__accu_and_axioms_along_training(file_location):
    # (1) Load result json files
    with open(file_location +'/results.json', 'r') as file:
        results = json.load(file)

    # (2) Load basic data
    architecture = results['architecture']
    rule = results['rule_names'][0]
    max_num_voters = results['max_num_voters']
    max_num_alternatives = results['max_num_alternatives']
    election_sampling = results['election_sampling']['probmodel']

    # (3) Gather the data for the plot
    evolution = results['evolution']
    gradient_steps = []
    accuracy = []
    axiom_satisfaction = {axiom:[] for axiom in results['axioms_check_model']}
    for step, v in evolution.items():
        if int(step) > 0: # Ignore 0-th epoch where model is random 
            gradient_steps.append(float(step))
            accuracy.append(float(100 * v['dev_accuracy']))
            for axiom in results['axioms_check_model']:
                a = 100 * v['dev_axiom_satisfaction'][axiom]['cond_satisfaction']
                axiom_satisfaction[axiom].append(float(a))

    # (4) Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(
        gradient_steps,
        accuracy,
        label = f'Accuracy'
    )
    for axiom in results['axioms_check_model']:
        ax.plot(
            gradient_steps,
            axiom_satisfaction[axiom],
            label = f'{axiom}')

    ax.legend(loc=0)
    ax.grid()
    ax.set_xlabel('Gradient steps')
    ax.set_ylabel('Satisfaction (in %)')
    ax.set_title(f'Evolution of evaluation ({architecture}, {rule}, {election_sampling}, up to {max_num_voters} vot. & {max_num_alternatives} alt.)')
    
    
    # (5) Save the plot
    # time stamp
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # metadata (to save png images this requires backend 'agg',
    # which is default anyways)
    # To access metadata later, use imagemagick and run
    # $ identify -verbose image.png
    metadata = {'plotted_data_from':file_location}

    plt.savefig(f'./results/exp1/plots/plot_{current_time}_evolution_{architecture}.png',
        metadata = metadata,
        backend = 'agg', # default
        dpi=300, 
        bbox_inches = 'tight'
    )

    return




def plot_exp2(
        architecture,
        max_num_voters,
        max_num_alternatives,
        election_sampling,
        percentages,
        files_plurality,
        files_borda,
        files_copeland,
        axioms_to_plot,
        augmentation,
        ylim = None
    ):
    """
    Creates lineplot of data in the three specified dictionaries of files

    `architecture` is the network architecture of the experiments for which 
    the plot is created.
    `max_num_voters` is the maximal number of voters of the experiments for 
    which the plot is created.
    `max_num_alternatives` is the maximal number of voters of the experiments 
    for which the plot is created. 

    `percentages` is a list of integers between 0 and 100

    `files_plurality` is a dictionary whose keys are exactly the elements 
    of `percentages` (cast as strings) and whose corresponding value is the 
    path to the folder containing the results of the corresponding experiment 
    using the plurality rule.

    `files_borda` is the same but for the Bora rule

    `files_copeland` is the same but for the Copeland rule

    `axioms_to_plot` is a list of names of axioms which should be plotted

    `augmentation` is string, e.g., 'neutrality' or 'anonymity' describing how
    the augmented data was generated.

    If ylim is set to, e.g., 30, then the y-axis with the error 
    values ranges from 0 to 30. If ylim is left to default None, the range
    is determined automatically based on the values in the datafiles.
    """


    # (1) Load the data to be plotted
    # Initialize the list of values and then loop over the percentages
    y_plurality_accu = []
    y_plurality_subaccu = []
    y_plurality_neut = []
    y_plurality_anon = []
    y_borda_accu = []
    y_borda_subaccu = []
    y_borda_neut = []
    y_borda_anon = []
    y_copeland_accu = []
    y_copeland_subaccu = []
    y_copeland_neut = []
    y_copeland_anon = []

    for p in percentages:

        # (a) Load result json files
        with open(files_plurality[str(p)] +'/results.json', 'r') as file:
            results_plurality = json.load(file)
        with open(files_borda[str(p)] +'/results.json', 'r') as file:
            results_borda = json.load(file)
        with open(files_copeland[str(p)] +'/results.json', 'r') as file:
            results_copeland = json.load(file)
        # and combine into a single dictionary
        results_rules = {
            'Plurality': results_plurality,
            'Borda': results_borda,
            'Copeland': results_copeland,
        }

        # (b) Check that the files have the right kind of data
        assert (results_plurality['rule_names'] == ["Plurality"]
            ), f'The file_plurality should use Plurality as only rule'
        assert (results_borda['rule_names'] == ["Borda"]
            ), f'The file_borda should use Borda as only rule'
        assert (results_copeland['rule_names'] == ["Copeland"]
            ), f'The file_copeland should use Copeland as only rule'

        for rule in results_rules.keys():
            assert (results_rules[rule]['architecture'] == architecture
                ), f'The file for {rule} does not have architecture {architecture}'
            assert (results_rules[rule]['max_num_voters'] == max_num_voters
                ), f'The file for {rule} does not have max_num_voters {max_num_voters}'
            assert (results_rules[rule]['max_num_alternatives'] == max_num_alternatives
                ), f'The file for {rule} does not have max_num_alternatives {max_num_alternatives}'
            assert (results_rules[rule]['election_sampling']['probmodel'] == election_sampling
                ), f'The file for {rule} does not have election_sampling {election_sampling}'        
        
        # (c) Add accuracy and neutrality values
        
        # Plurality Accuracy
        error = 1 - results_rules['Plurality']['rule_similarity']['Plurality']['identity_accu']
        y_plurality_accu.append(round(100*error,1))
        # Plurality Subset Accuracy
        error = 1 - results_rules['Plurality']['rule_similarity']['Plurality']['subset_accu']
        y_plurality_subaccu.append(round(100*error,1))
        # Plurality Neutrality
        if 'Neutrality' in results_rules['Plurality']['axioms_check_model']:
            error = 1 - results_rules['Plurality']['axiom_satisfaction']['learned_rule']['Neutrality']['cond_satisfaction']
            y_plurality_neut.append(round(100*error,1))
        # Plurality Anonymity
        if 'Anonymity' in results_rules['Plurality']['axioms_check_model']:
            error = 1 - results_rules['Plurality']['axiom_satisfaction']['learned_rule']['Anonymity']['cond_satisfaction']
            y_plurality_anon.append(round(100*error,1))

        # Borda Accuracy
        error = 1 - results_rules['Borda']['rule_similarity']['Borda']['identity_accu']
        y_borda_accu.append(round(100*error,1))
        # Borda Subset Accuracy
        error = 1 - results_rules['Borda']['rule_similarity']['Borda']['subset_accu']
        y_borda_subaccu.append(round(100*error,1))
        # Borda Neutrality
        if 'Neutrality' in results_rules['Borda']['axioms_check_model']:
            error = 1 - results_rules['Borda']['axiom_satisfaction']['learned_rule']['Neutrality']['cond_satisfaction']
            y_borda_neut.append(round(100*error,1))
        # Borda Anonymity
        if 'Anonymity' in results_rules['Borda']['axioms_check_model']:
            error = 1 - results_rules['Borda']['axiom_satisfaction']['learned_rule']['Anonymity']['cond_satisfaction']
            y_borda_anon.append(round(100*error,1))

        # Copeland Accuracy
        error = 1 - results_rules['Copeland']['rule_similarity']['Copeland']['identity_accu']
        y_copeland_accu.append(round(100*error,1))
        # Copeland Subset Accuracy
        error = 1 - results_rules['Copeland']['rule_similarity']['Copeland']['subset_accu']
        y_copeland_subaccu.append(round(100*error,1))
        # Copeland Neutrality
        if 'Neutrality' in results_rules['Copeland']['axioms_check_model']:
            error = 1 - results_rules['Copeland']['axiom_satisfaction']['learned_rule']['Neutrality']['cond_satisfaction']
            y_copeland_neut.append(round(100*error,1))
        # Copeland Anonymity
        if 'Anonymity' in results_rules['Copeland']['axioms_check_model']:
            error = 1 - results_rules['Copeland']['axiom_satisfaction']['learned_rule']['Anonymity']['cond_satisfaction']
            y_copeland_anon.append(round(100*error,1))


    # (2) Create plot
    x = percentages # x-axis values

    fig, ax = plt.subplots(layout='constrained')
    fig_legend, ax_legend = plt.subplots(figsize=(4,1))
    
    
    ln1a = ax.plot(x, y_plurality_accu, label='Plurality accu. (identity)', marker='o', color='tab:blue', linestyle='--')
    ln1b = ax.plot(x, y_plurality_subaccu, label='Plurality accu. (subset)', marker='o', color='tab:blue', linestyle=':')
    
    ln2a = ax.plot(x, y_borda_accu, label='Borda accu. (identity)', marker='s', color='tab:orange', linestyle='--')
    ln2b = ax.plot(x, y_borda_subaccu, label='Borda accu. (subset)', marker='s', color='tab:orange', linestyle=':')

    ln3a = ax.plot(x, y_copeland_accu, label='Copeland accu. (identity)', marker='^', color='tab:green', linestyle='--')
    ln3b = ax.plot(x, y_copeland_subaccu, label='Copeland accu. (subset)', marker='^', color='tab:green', linestyle=':')
    
    lns_dict = {
        'Plurality': ln1a + ln1b, 
        'Borda' : ln2a + ln2b, 
        'Copeland' : ln3a + ln3b  
    }

    if 'Neutrality' in axioms_to_plot:
        ln_n = ax.plot(x, y_plurality_neut, label='Plurality neutrality', marker='o', color='tab:blue', linestyle='-')
        lns_dict['Plurality'] += ln_n
        ln_n = ax.plot(x, y_borda_neut, label='Borda neutrality', marker='s', color='tab:orange', linestyle='-')
        lns_dict['Borda'] += ln_n
        ln_n = ax.plot(x, y_copeland_neut, label='Copeland neutrality', marker='^', color='tab:green', linestyle='-')
        lns_dict['Copeland'] += ln_n

    if 'Anonymity' in axioms_to_plot:
        ln_a = ax.plot(x, y_plurality_anon, label='Plurality anonymity', marker='o', color='tab:blue', linestyle='-')
        lns_dict['Plurality'] += ln_a
        ln_a = ax.plot(x, y_borda_anon, label='Borda anonymity', marker='s', color='tab:orange', linestyle='-')
        lns_dict['Borda'] += ln_a
        ln_a = ax.plot(x, y_copeland_anon, label='Copeland anonymity', marker='^', color='tab:green', linestyle='-')
        lns_dict['Copeland'] += ln_a 
    
    lns = lns_dict['Plurality'] + lns_dict['Borda'] + lns_dict['Copeland']
    labs = [l.get_label() for l in lns]
    ax_legend.axis('off')
    ax_legend.legend(lns, labs, ncol=3, loc='center')

    ax.set_xticks(x)
    ax.set_xticklabels([str(num) for num in x])
    if ylim is not None:
        ax.set_ylim(0,ylim)
    else:
        ax.set_ylim([0,None])

    ax.set_xlabel(f'Amount of sampled (as opposed to {augmentation}-augmented) data in %')
    ax.set_ylabel('Error in %') 
    if election_sampling == 'IC':
        election_sampling_short = 'IC'
    if election_sampling == 'MALLOWS-RELPHI':
        election_sampling_short = 'Mallows'
    if election_sampling == 'URN-R':
        election_sampling_short = 'Urn'
    if election_sampling == 'euclidean':
        election_sampling_short = 'Euclidean'        
    ax.set_title(f'Exp. 2: {architecture} up to {max_num_voters} voters and {max_num_alternatives} alternatives ({election_sampling_short})') 
    #ax.set_title(f'Augmentation ratio ({architecture}, {election_sampling_short}, up to {max_num_voters} vot. & {max_num_alternatives} alt.)')


    # (5) Save the plot
    # Time stamp
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 

    # Path
    path = f'./results/exp2/plots/plot_{current_time}_{architecture}_{election_sampling}_{augmentation}'

    # Save metadata in separate file
    metadata = {
        'architecture':architecture,
        'max_num_voters':max_num_voters,
        'max_num_alternatives':max_num_alternatives,
        'election_sampling':election_sampling,
        'percentages':percentages, 
        'files_plurality':files_plurality, 
        'files_borda':files_borda, 
        'files_copeland':files_copeland, 
        'ylim':ylim,
    }

    with open(f"{path}.json", "w") as json_file:
        json.dump(metadata, json_file)

    fig.savefig(f'{path}.png', dpi=300, bbox_inches = 'tight')
    legend_path = path + '_legend'
    fig_legend.savefig(f'{legend_path}.png', dpi=300, bbox_inches = 'tight')


    return current_time




def plot_exp2_straightforward(file_location):
    # (1) Load result json files
    with open(file_location +'/results.json', 'r') as file:
        results = json.load(file)

    # (2) Load basic data
    architecture = results['architecture']
    rule = results['rule_names'][0]
    max_num_voters = results['max_num_voters']
    max_num_alternatives = results['max_num_alternatives']
    election_sampling = results['election_sampling']['probmodel']
    ax_list = results['axioms_check_model']
    initial_gradient_steps = results['initial_gradient_steps']

    # (3) Gather the data for the plot
    learning_curve = results['learning curve']
    gradient_steps = []
    accuracy_orig = []
    accuracy_copy = []
    axiom_satisfaction_orig = {axiom:[] for axiom in ax_list}
    axiom_satisfaction_copy = {axiom:[] for axiom in ax_list}    
    for step, v in learning_curve.items():
        gradient_steps.append(float(step))
        accuracy_orig.append(float(100 * v['dev_accuracy_orig']))
        accuracy_copy.append(float(100 * v['dev_accuracy_copy']))
        for axiom in ax_list:
            a_orig = 100 * v['dev_axiom_satisfaction_orig'][axiom]['cond_satisfaction']
            axiom_satisfaction_orig[axiom].append(float(a_orig))
            a_copy = 100 * v['dev_axiom_satisfaction_copy'][axiom]['cond_satisfaction']
            axiom_satisfaction_copy[axiom].append(float(a_copy))

    # (4) Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = plt.cm.tab10.colors
    color_idx = 0
    ax.plot(
        gradient_steps,
        accuracy_orig,
        label = f'Accuracy augmented',
        linestyle='-',
        color = colors[color_idx]
    )
    ax.plot(
        gradient_steps,
        accuracy_copy,
        label = f'Accuracy sampled',
        linestyle='--',
        color = colors[color_idx]
    )    
    for axiom in ax_list:
        color_idx += 1
        ax.plot(
            gradient_steps, 
            axiom_satisfaction_orig[axiom],
            label = f'{axiom} augmented',
            linestyle='-',
            color = colors[color_idx]
        )
        ax.plot(
            gradient_steps, 
            axiom_satisfaction_copy[axiom],
            label = f'{axiom} sampled',
            linestyle='--',
            color = colors[color_idx]
        )

    ax.legend(loc=0)
    ax.grid()
    ax.set_xlabel(f'Gradient steps (after {initial_gradient_steps} steps of pretraining)')
    ax.set_ylabel('Satisfaction (in %)')
    if election_sampling == 'IC':
        election_sampling_short = 'IC'
    if election_sampling == 'MALLOWS-RELPHI':
        election_sampling_short = 'Mall.'
    if election_sampling == 'URN-R':
        election_sampling_short = 'Urn'
    if election_sampling == 'euclidean':
        election_sampling_short = 'Eucl.'        
    ax.set_title(f'Augmentation ({architecture}, {rule}, {election_sampling_short}, up to {max_num_voters} vot. & {max_num_alternatives} alt.)')
    
    
    # (5) Save the plot
    # time stamp
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # metadata (to save png images this requires backend 'agg',
    # which is default anyways)
    # To access metadata later, use imagemagick and run
    # $ identify -verbose image.png
    metadata = {'plotted_data_from':file_location}

    plt.savefig(f'./results/exp2/plots/plot_{current_time}_aug_{architecture}_{rule}_{election_sampling_short}.png',
        metadata = metadata,
        backend = 'agg', # default
        dpi=300,
        bbox_inches = 'tight'
    )

    return






def exp3_table(
        file_rules=None,
        file_MLP=None,
        file_CNN=None,
        file_WEC=None,
        file_Transformer=None):
    """
    Plots as table the axiom satisfaction of th provided rules and models.

    Each argument can either be None (hence not plotted) or a string (a path 
    to a folder with the file containing the results). Moreover, the 
    `file_rules` and the `file_WEC` can be list of such strings. If so, then 
    the average of the satisfaction in these files is taken. 
    """

    # Preliminaries
    columns_axioms = [
        'Anonymity',
        'Neutrality',
        'Condorcet',
        'Pareto',
        'Independence',
        'Average'
    ]    
    optimization = {
        'No_winner':'NW',
        'All_winners':'AW',
        'Inadmissible':'IA',
        'Resoluteness':'R',
        'Anonymity':'A', 
        'Neutrality':'N', 
        'Condorcet1':'C', 
        'Condorcet2':'C',         
        'Pareto1':'P', 
        'Pareto2':'P',         
        'Independence':'I'
    }

    # Dataframe of axiom satisfaction of rules
    if file_rules is None:
        df_rules = None
    else:    
        if isinstance(file_rules, str):
            list_of_files = [file_rules]
        if isinstance(file_rules, list):
            list_of_files = file_rules

        # Compute dataframe of results for each result in list_of_files:
        dfs_rules = []
        for file in list_of_files:
            # load results
            with open(f"{file}/results.json") as json_file:
                exp_data = json.load(json_file)

            # Row names of dataframe
            row_names = exp_data["comp_rules_axioms"]

            # Gather data for dataframe
            df_data = []
            for rule in row_names:
                # Axiom satisfaction of rule
                axiom_sat = [
                    1.0, # Anonymity
                    1.0, # Neutrality
                    exp_data["axiom_satisfaction"][rule]["Condorcet"]["cond_satisfaction"],
                    1.0, # Pareto
                    exp_data["axiom_satisfaction"][rule]["Independence"]["cond_satisfaction"],
                ]
                # Add average axiom satisfaction
                axiom_sat.append(sum(axiom_sat)/len(axiom_sat))
                # Round and turn into percent 
                axiom_sat = [round(100*itm,1) for itm in axiom_sat]
                # Add to dataframe
                df_data.append(axiom_sat)

            # Generate and print dataframe
            df_rules = pd.DataFrame(
                data=df_data,
                index=row_names,
                columns=columns_axioms
            )
            # Add to list
            dfs_rules.append(df_rules)

        # Take average of the values in the dataframes
        df_rules = sum(dfs_rules)/len(dfs_rules)



    # Dataframe with results from MLP

    if file_MLP is None:
        df_MLP = None
    else:    
        # load results 
        with open(f"{file_MLP}/results.json") as json_file:
            exp_data = json.load(json_file)

        # Plain axiom satisfaction
        axiom_sat_plain = [
            exp_data["axiom_satisfaction"]["model_plain"]["Anonymity"]["cond_satisfaction"],
            exp_data["axiom_satisfaction"]["model_plain"]["Neutrality"]["cond_satisfaction"],
            exp_data["axiom_satisfaction"]["model_plain"]["Condorcet"]["cond_satisfaction"],
            exp_data["axiom_satisfaction"]["model_plain"]["Pareto"]["cond_satisfaction"],
            exp_data["axiom_satisfaction"]["model_plain"]["Independence"]["cond_satisfaction"],
        ]
        axiom_sat_plain.append(sum(axiom_sat_plain)/len(axiom_sat_plain))
        axiom_sat_plain = [round(100*itm, 1) for itm in axiom_sat_plain]

        # Neut-averaged axiom satisfaction
        axiom_sat_neut = [
            exp_data["axiom_satisfaction"]["model_neut"]["Anonymity"]["cond_satisfaction"],
            exp_data["axiom_satisfaction"]["model_neut"]["Neutrality"]["cond_satisfaction"],
            exp_data["axiom_satisfaction"]["model_neut"]["Condorcet"]["cond_satisfaction"],
            exp_data["axiom_satisfaction"]["model_neut"]["Pareto"]["cond_satisfaction"],
            exp_data["axiom_satisfaction"]["model_neut"]["Independence"]["cond_satisfaction"],
        ]
        axiom_sat_neut.append(sum(axiom_sat_neut)/len(axiom_sat_neut))
        axiom_sat_neut = [round(100*itm, 1) for itm in axiom_sat_neut]

        # Neut-anon-averaged axiom satisfaction
        axiom_sat_neut_anon = [
            exp_data["axiom_satisfaction"]["model_neut_anon"]["Anonymity"]["cond_satisfaction"],
            exp_data["axiom_satisfaction"]["model_neut_anon"]["Neutrality"]["cond_satisfaction"],
            exp_data["axiom_satisfaction"]["model_neut_anon"]["Condorcet"]["cond_satisfaction"],
            exp_data["axiom_satisfaction"]["model_neut_anon"]["Pareto"]["cond_satisfaction"],
            exp_data["axiom_satisfaction"]["model_neut_anon"]["Independence"]["cond_satisfaction"],
        ]
        axiom_sat_neut_anon.append(sum(axiom_sat_neut_anon)/len(axiom_sat_neut_anon))
        axiom_sat_neut_anon = [round(100*itm, 1) for itm in axiom_sat_neut_anon]

        # Collect the axioms the model optimized for
        optimized_for = []
        for k,v in exp_data["axiom_opt"].items():
            if v is not None:
                optimized_for.append(optimization[k])
        optimized_for = ', '.join(optimized_for)

        # Generate dataframe
        df_MLP = pd.DataFrame(
            data=[axiom_sat_plain, axiom_sat_neut, axiom_sat_neut_anon], 
            index=[f'MLP p ({optimized_for})', f'MLP n ({optimized_for})', f'MLP na ({optimized_for})'],
            columns=columns_axioms
        )



    # Dataframe with results from CNN

    if file_CNN is None:
        df_CNN = None
    else:    
        # load results
        with open(f"{file_CNN}/results.json") as json_file:
            exp_data = json.load(json_file)

        # Plain axiom satisfaction
        axiom_sat_plain = [
            exp_data["axiom_satisfaction"]["model_plain"]["Anonymity"]["cond_satisfaction"],
            exp_data["axiom_satisfaction"]["model_plain"]["Neutrality"]["cond_satisfaction"],
            exp_data["axiom_satisfaction"]["model_plain"]["Condorcet"]["cond_satisfaction"],
            exp_data["axiom_satisfaction"]["model_plain"]["Pareto"]["cond_satisfaction"],
            exp_data["axiom_satisfaction"]["model_plain"]["Independence"]["cond_satisfaction"],
        ]
        axiom_sat_plain.append(sum(axiom_sat_plain)/len(axiom_sat_plain))
        axiom_sat_plain = [round(100*itm, 1) for itm in axiom_sat_plain]

        # Neut-averaged axiom satisfaction
        axiom_sat_neut = [
            exp_data["axiom_satisfaction"]["model_neut"]["Anonymity"]["cond_satisfaction"],
            exp_data["axiom_satisfaction"]["model_neut"]["Neutrality"]["cond_satisfaction"],
            exp_data["axiom_satisfaction"]["model_neut"]["Condorcet"]["cond_satisfaction"],
            exp_data["axiom_satisfaction"]["model_neut"]["Pareto"]["cond_satisfaction"],
            exp_data["axiom_satisfaction"]["model_neut"]["Independence"]["cond_satisfaction"],
        ]
        axiom_sat_neut.append(sum(axiom_sat_neut)/len(axiom_sat_neut))
        axiom_sat_neut = [round(100*itm, 1) for itm in axiom_sat_neut]

        # Neut-anon-averaged axiom satisfaction
        axiom_sat_neut_anon = [
            exp_data["axiom_satisfaction"]["model_neut_anon"]["Anonymity"]["cond_satisfaction"],
            exp_data["axiom_satisfaction"]["model_neut_anon"]["Neutrality"]["cond_satisfaction"],
            exp_data["axiom_satisfaction"]["model_neut_anon"]["Condorcet"]["cond_satisfaction"],
            exp_data["axiom_satisfaction"]["model_neut_anon"]["Pareto"]["cond_satisfaction"],
            exp_data["axiom_satisfaction"]["model_neut_anon"]["Independence"]["cond_satisfaction"],
        ]
        axiom_sat_neut_anon.append(sum(axiom_sat_neut_anon)/len(axiom_sat_neut_anon))
        axiom_sat_neut_anon = [round(100*itm, 1) for itm in axiom_sat_neut_anon]

        # Collect the axioms the model optimized for
        optimized_for = []
        for k,v in exp_data["axiom_opt"].items():
            if v is not None:
                optimized_for.append(optimization[k])
        optimized_for = ', '.join(optimized_for)

        # Generate dataframe
        df_CNN = pd.DataFrame(
            data=[axiom_sat_plain, axiom_sat_neut, axiom_sat_neut_anon],
            index=[f'CNN p ({optimized_for})', f'CNN n ({optimized_for})', f'CNN na ({optimized_for})'],
            columns=columns_axioms
        )






    # Dataframe with results from WEC

    if file_WEC is None:
        df_WEC = None
    else:
        if isinstance(file_WEC, str):
            list_of_files = [file_WEC]
        if isinstance(file_WEC, list):
            list_of_files = file_WEC
        
        # Compute dataframe of results for each model in list_of_files:
        dfs_WEC = []
        for file in list_of_files:
            with open(f"{file}/results.json") as json_file:
                exp_data = json.load(json_file)

            # Plain axiom satisfaction
            if exp_data["axiom_satisfaction"]["model_plain"] is not None:
                axiom_sat_plain = [
                    exp_data["axiom_satisfaction"]["model_plain"]["Anonymity"]["cond_satisfaction"],
                    exp_data["axiom_satisfaction"]["model_plain"]["Neutrality"]["cond_satisfaction"],
                    exp_data["axiom_satisfaction"]["model_plain"]["Condorcet"]["cond_satisfaction"],
                    exp_data["axiom_satisfaction"]["model_plain"]["Pareto"]["cond_satisfaction"],
                    exp_data["axiom_satisfaction"]["model_plain"]["Independence"]["cond_satisfaction"],
                ]
                axiom_sat_plain.append(sum(axiom_sat_plain)/len(axiom_sat_plain))
                axiom_sat_plain = [round(100*itm, 1) for itm in axiom_sat_plain]
            else: 
                axiom_sat_plain = None

            # Neut-averaged axiom satisfaction
            if exp_data["axiom_satisfaction"]["model_neut"] is not None:
                axiom_sat_neut = []
                if "Anonymity" in exp_data["axiom_satisfaction"]["model_neut"]:
                    axiom_sat_neut.append(exp_data["axiom_satisfaction"]["model_neut"]["Anonymity"]["cond_satisfaction"])
                else:
                    axiom_sat_neut.append(1.0)
                if "Neutrality" in exp_data["axiom_satisfaction"]["model_neut"]:
                    axiom_sat_neut.append(exp_data["axiom_satisfaction"]["model_neut"]["Neutrality"]["cond_satisfaction"])
                else:
                    axiom_sat_neut.append(1.0)
                axiom_sat_neut += [
                    exp_data["axiom_satisfaction"]["model_neut"]["Condorcet"]["cond_satisfaction"],
                    exp_data["axiom_satisfaction"]["model_neut"]["Pareto"]["cond_satisfaction"],
                    exp_data["axiom_satisfaction"]["model_neut"]["Independence"]["cond_satisfaction"],
                ]
                axiom_sat_neut.append(sum(axiom_sat_neut)/len(axiom_sat_neut))
                axiom_sat_neut = [round(100*itm, 1) for itm in axiom_sat_neut]
            else:
                axiom_sat_neut = None

            # Collect the axioms the model optimized for
            optimized_for = []
            for k,v in exp_data["axiom_opt"].items():
                if v is not None:
                    optimized_for.append(optimization[k])
            optimized_for = ', '.join(optimized_for)

            # Generate dataframe

            # Specify index: mention number of run if there are several WECs
            index = []
            if axiom_sat_plain is not None:
                index.append(f'WEC p ({optimized_for})')
            if axiom_sat_neut is not None:
                index.append(f'WEC n ({optimized_for})')

            df_WEC = pd.DataFrame(
                data=[i for i in [axiom_sat_plain, axiom_sat_neut]
                      if i is not None],
                index=index,
                columns=columns_axioms
            )
            dfs_WEC.append(df_WEC)


        # Take average of the values in the dataframes
        df_WEC = sum(dfs_WEC)/len(dfs_WEC)



    # Dataframe with results from Transformer

    if file_Transformer is None:
        df_Transformer = None
    else:
        if isinstance(file_Transformer, str):
            list_of_files = [file_Transformer]
        if isinstance(file_Transformer, list):
            list_of_files = file_Transformer

        dfs_Transformer = []
        for file in list_of_files:
            with open(f"{file}/results.json") as json_file:
                exp_data = json.load(json_file)

            # Plain axiom satisfaction
            if exp_data["axiom_satisfaction"]["model_plain"] is not None:
                axiom_sat_plain = [
                    exp_data["axiom_satisfaction"]["model_plain"]["Anonymity"]["cond_satisfaction"],
                    exp_data["axiom_satisfaction"]["model_plain"]["Neutrality"]["cond_satisfaction"],
                    exp_data["axiom_satisfaction"]["model_plain"]["Condorcet"]["cond_satisfaction"],
                    exp_data["axiom_satisfaction"]["model_plain"]["Pareto"]["cond_satisfaction"],
                    exp_data["axiom_satisfaction"]["model_plain"]["Independence"]["cond_satisfaction"],
                ]
                axiom_sat_plain.append(sum(axiom_sat_plain)/len(axiom_sat_plain))
                axiom_sat_plain = [round(100*itm, 1) for itm in axiom_sat_plain]
            else:
                axiom_sat_plain = None

            # Neut-averaged axiom satisfaction
            if exp_data["axiom_satisfaction"]["model_neut"] is not None:
                axiom_sat_neut = []
                if "Anonymity" in exp_data["axiom_satisfaction"]["model_neut"]:
                    axiom_sat_neut.append(exp_data["axiom_satisfaction"]["model_neut"]["Anonymity"]["cond_satisfaction"])
                else:
                    axiom_sat_neut.append(1.0)
                if "Neutrality" in exp_data["axiom_satisfaction"]["model_neut"]:
                    axiom_sat_neut.append(exp_data["axiom_satisfaction"]["model_neut"]["Neutrality"]["cond_satisfaction"])
                else:
                    axiom_sat_neut.append(1.0)
                axiom_sat_neut += [
                    exp_data["axiom_satisfaction"]["model_neut"]["Condorcet"]["cond_satisfaction"],
                    exp_data["axiom_satisfaction"]["model_neut"]["Pareto"]["cond_satisfaction"],
                    exp_data["axiom_satisfaction"]["model_neut"]["Independence"]["cond_satisfaction"],
                ]
                axiom_sat_neut.append(sum(axiom_sat_neut)/len(axiom_sat_neut))
                axiom_sat_neut = [round(100*itm, 1) for itm in axiom_sat_neut]
            else:
                axiom_sat_neut = None

            # Collect the axioms the model optimized for
            optimized_for = []
            for k,v in exp_data["axiom_opt"].items():
                if v is not None:
                    optimized_for.append(optimization[k])
            optimized_for = ', '.join(optimized_for)

            index = []
            if axiom_sat_plain is not None:
                index.append(f'Transformer p ({optimized_for})')
            if axiom_sat_neut is not None:
                index.append(f'Transformer n ({optimized_for})')

            df_Transformer = pd.DataFrame(
                data=[i for i in [axiom_sat_plain, axiom_sat_neut]
                      if i is not None],
                index=index,
                columns=columns_axioms
            )
            dfs_Transformer.append(df_Transformer)

        df_Transformer = sum(dfs_Transformer)/len(dfs_Transformer)



    # Marge all dataframes

    # Collect all dfs, if not None
    list_of_dfs = [i for i in [df_rules, df_MLP, df_CNN, df_WEC, df_Transformer] 
                   if isinstance(i, pd.DataFrame)]

    df = pd.concat(list_of_dfs)



    # Print dataframe in latex table format
    # As headers we abbreviate the content of df.columns
    print(tabulate(
        df,
        headers=['Anon.', ' Neut.', 'Condorcet', 'Pareto', 'Indep.', 'Avg.'],
        tablefmt='latex_raw',
        numalign='center',
    ))

    return







def exp3_table_WECs(files_WEC):
    """
    Plot each results in the list of files of neut-averaged WEC models
    """
    # Preliminaries
    columns_axioms = [
        'Anonymity',
        'Neutrality',
        'Condorcet',
        'Pareto',
        'Independence',
        'Average'
    ]    
    optimization = {
        'No_winner':'NW',
        'All_winners':'AW',
        'Inadmissible':'IA',
        'Resoluteness':'R',
        'Anonymity':'A', 
        'Neutrality':'N', 
        'Condorcet1':'C', 
        'Condorcet2':'C',         
        'Pareto1':'P', 
        'Pareto2':'P',         
        'Independence':'I'
    }

    # Compute dataframe of results for each model in files_WEC:
    dfs_WEC = []        
    for run, file in enumerate(files_WEC):
        with open(f"{file}/results.json") as json_file:
            exp_data = json.load(json_file)

        # Neut-averaged axiom satisfaction
        axiom_sat_neut = [
            exp_data["axiom_satisfaction"]["model_neut"]["Anonymity"]["cond_satisfaction"],
            exp_data["axiom_satisfaction"]["model_neut"]["Neutrality"]["cond_satisfaction"],
            exp_data["axiom_satisfaction"]["model_neut"]["Condorcet"]["cond_satisfaction"],
            exp_data["axiom_satisfaction"]["model_neut"]["Pareto"]["cond_satisfaction"],
            exp_data["axiom_satisfaction"]["model_neut"]["Independence"]["cond_satisfaction"],
        ]
        axiom_sat_neut.append(sum(axiom_sat_neut)/len(axiom_sat_neut))
        axiom_sat_neut = [round(100*itm, 1) for itm in axiom_sat_neut]

        # Collect the axioms the model optimized for
        optimized_for = []
        for k,v in exp_data["axiom_opt"].items():
            if v is not None:
                optimized_for.append(optimization[k])
        optimized_for = ', '.join(optimized_for)

        # Generate dataframe
        df_WEC = pd.DataFrame(
            data=[axiom_sat_neut],
            index=[f'WEC n ({optimized_for}, run {run})'],
            columns=columns_axioms
        )
        dfs_WEC.append(df_WEC)

    # Marge all dataframes
    df = pd.concat(dfs_WEC)

    # Print dataframe in latex table format
    # As headers we abbreviate the content of df.columns
    print(tabulate(
        df,
        headers=['Anon.', ' Neut.', 'Condorcet', 'Pareto', 'Indep.', 'Avg.'],
        tablefmt='latex_raw',
        numalign='center',
    ))

    return





def exp3_similarities(
        file_model,
        eval_dataset_size,
        n_best_rules):
    """
    Computes the similarities between the model and the rules 

    `file_model` is the path to the folder with the model and the results.
    `eval_dataset_size` is the number of profiles that are sampled to compute 
    the similarity to the rules. The sample method is the one also used for 
    the model
    `n_best_rules` is a positive integer n. Then the first n many rules,
    ordered by similarity to the model, are considered in the table.

    Outputs a table displaying the similarities.
    """


    # Load data about model
    with open(f"{file_model}/results.json") as json_file:
        exp_data = json.load(json_file)

    architecture = exp_data["architecture"]
    max_num_voters = exp_data["max_num_voters"]
    max_num_alternatives = exp_data["max_num_alternatives"]
    architecture_parameters = exp_data["architecture_parameters"]
    election_sampling = exp_data["election_sampling"]

    if architecture == 'WEC':
        if 'load_embeddings_from' in architecture_parameters:
            load_embeddings_from = architecture_parameters['load_embeddings_from']
        else:
            load_embeddings_from = file_model
        pre_embeddings = Word2Vec.load(f"{load_embeddings_from}/pre_embeddings.bin")


    # Load model

    #First build a new model with the right parameters
    if architecture == 'MLP':
        exp_model = MLP(max_num_voters, max_num_alternatives)


    if architecture == 'CNN':
        exp_model = CNN(
            max_num_voters, 
            max_num_alternatives,
            architecture_parameters['kernel1'],
            architecture_parameters['kernel2'],
            architecture_parameters['channels']
        )

    if architecture == 'WEC':    
        exp_model = WEC(pre_embeddings, max_num_voters, max_num_alternatives)


    # Then load the previous state of the model if given
    checkpoint = torch.load(f'{file_model}/model.pth')
    exp_model.load_state_dict(checkpoint['model_state_dict'])


    # Define the rule computed by the model
    # We only consider neutrality averaged rules
    if architecture == 'MLP':
        model_rule_n = models.MLP2rule_n(exp_model, None)

    if architecture == 'CNN':
        model_rule_n = models.CNN2rule_n(exp_model, None)
            
    if architecture == 'WEC':
        model_rule_n = models.WEC2rule_n(exp_model, None)

    # Collect similarity of model to rules
    similarities_model_to_rules = {
        k:v['identity_accu']
        for k,v in exp_data['rule_comparison']['neut'].items()
    }
    # and pick the `n_best_rules` most similar ones
    similarities_model_to_rules = sorted(
        similarities_model_to_rules, 
        key=similarities_model_to_rules.get, 
        reverse=True
    )[0:n_best_rules]

    # Collect the thus considered rules as dictionary
    considered_rules = {
        k:utils.dict_rules_all[k] for k in similarities_model_to_rules
    }
    
    # Generate the profiles on which similarity is tested
    test_profs, _, _ = generate_profile_data(
        max_num_voters,
        max_num_alternatives,
        eval_dataset_size,
        election_sampling,
        [],
        merge='empty',
    )

    # Initialize the data for the table
    data_identity = []
    data_subset = []

    # First compute the similarity of the model to the rules
    similarities = train_and_eval.rule_similarity(
        model_rule_n,
        considered_rules.keys(),
        test_profs,
        verbose=True)
    data_identity.append(
        [round(100*similarities[rule_name]["identity_accu"],1)
        for rule_name in considered_rules.keys()]
    )
    data_subset.append(
        [round(100*similarities[rule_name]["subset_accu"],1)
        for rule_name in considered_rules.keys()]
    )
    model_superset = [round(100*similarities[rule_name]["superset_accu"],1)
        for rule_name in considered_rules.keys()]

    # Next compute the similarities among the rules
    for name, rule in considered_rules.items():
        similarities = train_and_eval.rule_similarity(
            rule,
            considered_rules.keys(),
            test_profs,
            verbose=True)
        data_identity.append(
            [100*similarities[rule_name]["identity_accu"]
            for rule_name in considered_rules.keys()]
        )
        data_subset.append(
            [100*similarities[rule_name]["subset_accu"]
            for rule_name in considered_rules.keys()]
        )

    # Put everything in a dataframe
    df_identity = pd.DataFrame(
        data_identity,
        index=[f'{architecture} n'] + list(considered_rules.keys()),
        columns=considered_rules.keys()
    )
    df_subset = pd.DataFrame(
        data_subset,
        index=[f'{architecture} n'] + list(considered_rules.keys()),
        columns=considered_rules.keys()
    )

    # For identity version, add the first column mirroring the first row
    df_identity[f'{architecture} n'] = [100] + data_identity[0]
    # and move the newly added column to the front
    df_identity=df_identity[[f'{architecture} n']+list(considered_rules.keys())]

    # For subset version, the first column should be the superset accuracies,
    # i.e., the model being a supset of the rule, or equivalently, the rules 
    # being a subset of the model)
    df_subset[f'{architecture} n'] = [100] + model_superset
    # and move the newly added column to the front
    df_subset = df_subset[[f'{architecture} n']+list(considered_rules.keys())]

    # Print dataframe in latex table format
    # As headers we abbreviate the content of df.columns
    print('IDENTITY ACCURACY')
    print(tabulate(
        df_identity,
        headers=df_identity.columns,
        tablefmt='latex_raw',
        numalign='center',
    ))
    print('SUBSET ACCURACY')
    print(tabulate(
        df_subset,
        headers=df_subset.columns,
        tablefmt='latex_raw',
        numalign='center',
    ))

    return




def exp3_difference_making_profiles(
        file_model,
        eval_dataset_size,
        comparison_rules,
        weak_disagree=True,
        strong_disagree=True,
        print_output_of_rules = None,
        ):
    """
    Finds profiles where model and rules differ 

    We sample `eval_dataset_size`-many profiles and record the outputted 
    winning sets of the trained model stored in the folder `file_model` and 
    known voting rules in `comparison_rules`. Then we can start comparing 
    them in concrete cases, e.g., what is the simplest rule, if any, where the 
    model answer differs from all rule answers. Here 'disagree' can have two 
    meanings:
    * If `weak_disagree`: print the profiles where the winning set outputted 
      by the model is different from every winning set outputted by a rule.
    * If `strong_disagree`: print the profiles where the winning set outputted 
      by the model has empty intersection with every winning set outputted by 
      a rule.
    If `print_output_of_rules` is not None, it should be a list of rule names. 
    Their output on the disagreeing profiles is then also printed. 
        
    Currently only implemented for WEC, but can readily be extended to other 
    architectures.
    """


    # Load data about model
    with open(f"{file_model}/results.json") as json_file:
        exp_data = json.load(json_file)

    architecture = exp_data["architecture"]
    max_num_voters = exp_data["max_num_voters"]
    max_num_alternatives = exp_data["max_num_alternatives"]
    architecture_parameters = exp_data["architecture_parameters"]
    election_sampling = exp_data["election_sampling"]

    if architecture == 'WEC':
        if 'load_embeddings_from' in architecture_parameters:
            load_embeddings_from = architecture_parameters['load_embeddings_from']
        else:
            load_embeddings_from = file_model
        pre_embeddings = Word2Vec.load(f"{load_embeddings_from}/pre_embeddings.bin")


    # Load model
    if architecture == 'WEC':
        exp_model = WEC(pre_embeddings, max_num_voters, max_num_alternatives)


    # Then load the previous state of the model if given
    checkpoint = torch.load(f'{file_model}/model.pth')
    exp_model.load_state_dict(checkpoint['model_state_dict'])


    if architecture == 'WEC':
        model_rule_n = models.WEC2rule_n(exp_model, None)
        model_rule_n_sig = models.WEC2rule_n(
            exp_model,
            None,
            print_sigmoids=True
        )

    # Gather the rules (model and existing) that we consider
    comp_learned_rules = {f'{architecture} n':model_rule_n}
    comp_existing_rules = {
        rule:utils.dict_rules_all[rule] for rule in comparison_rules
    }
    all_rules = comp_learned_rules | comp_existing_rules


    # Generate the profiles on which the rules are compared
    profiles, _, _ = generate_profile_data(
        max_num_voters,
        max_num_alternatives,
        eval_dataset_size,
        election_sampling,
        [],
        merge='empty',
    )

    # Compute the winning sets for the considered rules on the given profiles
    elections = {rule:[] for rule in all_rules.keys()}

    for prof in tqdm(profiles):
        for k,v in all_rules.items():
            elections[k].append(v(prof))



    # (1) Model outputs a winning set that is different from every winning set
    # outputted by a rule
    if weak_disagree:
        model_different_from_all_rules = []
        for row_idx in range(len(profiles)):
            # check if model winning set is not in rule winning sets
            model_wset = elections[f'{architecture} n'][row_idx]
            rule_wsets = [elections[rule][row_idx] 
                        for rule in comp_existing_rules.keys()] 
            if not (model_wset in rule_wsets): 
                a = {
                    'profile_idx':row_idx,
                    'profile':profiles[row_idx],
                    'model_winning_set':model_wset,
                    'rule_winning_sets':{
                        list(comp_existing_rules.keys())[i] : rule_wsets[i] 
                        for i in range(len(rule_wsets))
                    },
                    'num_voters':profiles[row_idx].num_voters 
                }
                model_different_from_all_rules.append(a)
        # Sort output by num_voters of the profiles, so simpler ones come first
        out = sorted(
            model_different_from_all_rules, 
            key=lambda d: d['num_voters']
        )

        print('Profiles where model weakly disagrees (distinct winning sets)\nwith all the provided rules:') 
        num_differences = 0
        for a in out:
            if a['model_winning_set']: #i.e., winning set is not empty
                num_differences += 1
                print('Profile number ', a['profile_idx'])
                profile = a['profile']
                print(tabulate(
                    pd.DataFrame(profile.rankings).transpose(), #transpose to get rankings of voters in columns
                    headers=range(profile.num_voters),
                    tablefmt='latex_raw',
                    showindex='false',
                    numalign='center',
                ))
                sigmoids = model_rule_n_sig(profile)['sigmoids']
                print('The winning set chosen by:')
                print('   model: ', a['model_winning_set'])
                print('      with sigmoids ', sigmoids)
                for k, v in a['rule_winning_sets'].items():
                    print(f'   {k}: {v}')
                if print_output_of_rules is not None:
                    for rule_name in print_output_of_rules:
                        output = utils.dict_rules_all[rule_name](profile)
                        print(f'   {rule_name}: {output}')

        rate = round(100 * num_differences / eval_dataset_size)
        print(f'Percentage of weak disagreement: {rate}% ({num_differences} of {eval_dataset_size} sampled profiles)')

    # (2) For every winning set outputted by a rule, the intersection with
    # the winning set outputted by the model is empty
    if strong_disagree:
        model_different_from_individual_rule_intersection = []
        for row_idx in range(len(profiles)):
            model_wset = elections[f'{architecture} n'][row_idx]
            rule_wsets = [elections[rule][row_idx]
                        for rule in comp_existing_rules.keys()]
            # intersection of model's winning set with rule winning set,
            # for each rule
            intersections = [set(model_wset) & set(rule_wset)
                            for rule_wset in rule_wsets] 
            # check if all these intersections are empty
            if all([not b for b in intersections]): #b is false iff that intersection is empty
                a = {
                    'profile_idx':row_idx,
                    'profile':profiles[row_idx],
                    'model_winning_set':model_wset,
                    'rule_winning_sets':{
                        list(comp_existing_rules.keys())[i] : rule_wsets[i]
                        for i in range(len(rule_wsets))
                    },
                    'num_voters':profiles[row_idx].num_voters 
                }
                model_different_from_individual_rule_intersection.append(a)
        # Sort output by num_voters of the profiles, so simpler ones come first
        out = sorted(model_different_from_individual_rule_intersection,
                    key=lambda d: d['num_voters'])

        print('Profiles where model strongly disagrees (non-intersecting winning sets)\nwith all the provided rules:')
        num_differences = 0
        for a in out:
            if a['model_winning_set']: #i.e., winning set is not empty
                num_differences += 1
                print('Profile number ', a['profile_idx'])
                profile = a['profile']
                print(tabulate(
                    pd.DataFrame(profile.rankings).transpose(), #transpose to get rankings of voters in columns
                    headers=range(profile.num_voters), 
                    tablefmt='latex_raw',
                    showindex='false',
                    numalign='center',
                ))
                sigmoids = model_rule_n_sig(profile)['sigmoids']
                print('The winning set chosen by:')
                print('   model: ', a['model_winning_set'])
                print('      with sigmoids ', sigmoids)
                for k, v in a['rule_winning_sets'].items():
                    print(f'   {k}: {v}')
                if print_output_of_rules is not None:
                    for rule_name in print_output_of_rules:
                        output = utils.dict_rules_all[rule_name](profile)
                        print(f'   {rule_name}: {output}')    
            rate = round(100 * num_differences / eval_dataset_size,3)
        print(f'Percentage of strong disagreement: {rate}% ({num_differences} of {eval_dataset_size} sampled profiles)')
    return





def plot_exp3_loss(file_location, ylim = None):
    # Load result json files
    with open(file_location +'/results.json', 'r') as file:
        results = json.load(file)

    # Load basic data
    architecture = results['architecture']
    max_num_voters = results['max_num_voters']
    max_num_alternatives = results['max_num_alternatives']
    election_sampling = results['election_sampling']['probmodel']
    axiom_opt = results['axiom_opt']

    # The axioms that were optimized for
    opt_choice = []
    if axiom_opt['No_winner'] is not None:
        opt_choice.append('NW')
    if axiom_opt['Condorcet1'] is not None:
        opt_choice.append('C')
    if axiom_opt['Pareto2'] is not None:
        opt_choice.append('P')
    if axiom_opt['Independence'] is not None:
        opt_choice.append('I')
        
    # Gather the data for the plot
    loss_curve = results['loss curve']
    gradient_steps = []
    losses = {axiom : [] for axiom in opt_choice}
    for step, v in loss_curve.items():
        gradient_steps.append(float(step))
        for axiom in opt_choice:
            if axiom == 'NW':
                losses[axiom].append(v['loss_nowi'])
            if axiom == 'C':
                losses[axiom].append(v['loss_con1'])
            if axiom == 'P':
                losses[axiom].append(v['loss_par2'])
            if axiom == 'I':
                losses[axiom].append(v['loss_inde'])

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for axiom in opt_choice:
        ax.plot(
            gradient_steps, 
            losses[axiom], 
            label = f'{axiom}')

    ax.legend(loc=0) 
    if ylim is not None:
        ax.set_ylim(0,ylim)
    ax.grid()
    ax.set_xlabel('Gradient steps')
    ax.set_ylabel('Loss')
    ax.set_title(f'Evolution of losses ({architecture}, {election_sampling}, up to {max_num_voters} vot. & {max_num_alternatives} alt.)')

    # (5) Save the plot
    # time stamp
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # metadata (to save png images this requires backend 'agg',
    # which is default anyways)
    # To access metadata later, use imagemagick and run
    # $ identify -verbose image.png
    metadata = {'plotted_data_from':file_location}

    opt_choice_short = ''
    for axiom in opt_choice:
        opt_choice_short += axiom

    plt.savefig(f'./results/exp3/plots/plot_{current_time}_loss_evolution_{opt_choice_short}.png',
        metadata = metadata,
        backend = 'agg', # default 
        dpi=300, 
        bbox_inches = 'tight'
    )

    return


def plot_exp3_axiom_ablation(result_files):

    plot_data = {}

    for axiom_choice, file in result_files.items():
        # Load result json files
        with open(file +'/results.json', 'r') as file:
            results = json.load(file)

        # Load basic data
        architecture = results['architecture']
        max_num_voters = results['max_num_voters']
        max_num_alternatives = results['max_num_alternatives']
        election_sampling = results['election_sampling']['probmodel']
        axiom_opt = results['axiom_opt']
        admissability = results['admissability']['neut']
        axiom_satisfaction = results['axiom_satisfaction']['model_neut']

        # Check that the axioms in axiom_choice were indeed optimized for
        if axiom_opt['Condorcet1'] is not None:
            assert 'C' in axiom_choice, f'Key {axiom_choice} does not match actual axiom optimization'
        if axiom_opt['Pareto2'] is not None:
            assert 'P' in axiom_choice, f'Key {axiom_choice} does not match actual axiom optimization'
        if axiom_opt['Independence'] is not None:
            assert 'I' in axiom_choice, f'Key {axiom_choice} does not match actual axiom optimization'
        
        # Gather axiom satisfaction and admissability
        sat = {
            'No winner' : 100 * admissability['no_admissible_winner'],
            'Condorcet' : 100 * axiom_satisfaction['Condorcet']['cond_satisfaction'],
            'Pareto' : 100 * axiom_satisfaction['Pareto']['cond_satisfaction'],
            'Independence' : 100 * axiom_satisfaction['Independence']['cond_satisfaction'],
        }
        plot_data[axiom_choice] = sat


    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for axiom in ['No winner', 'Condorcet', 'Pareto', 'Independence']:
        ax.plot(
            result_files.keys(), 
            [0 if np.isnan(sat[axiom]) else sat[axiom]
            for axiom_choice, sat in plot_data.items()],
            'o-',
            label = axiom
        )
        # mark any nan values
        ax.scatter(
            result_files.keys(), 
            [0 if np.isnan(sat[axiom]) else None
            for axiom_choice, sat in plot_data.items()],
            50,
            marker = 's',
            color = 'black',
            zorder=2, # put to foreground
        )
    ax.legend(loc=0) 
    ax.grid()
    ax.set_xlabel('Choices of which axioms to optimize for')
    ax.set_ylabel('Axiom satisfaction')
    ax.set_title(f'Ablation study of axiom optimization ({architecture}, {election_sampling}, up to {max_num_voters} vot. & {max_num_alternatives} alt.)')


    # (5) Save the plot
    # time stamp
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # metadata (to save png images this requires backend 'agg',
    # which is default anyways)
    # To access metadata later, use imagemagick and run
    # $ identify -verbose image.png
    metadata = result_files

    plt.savefig(f'./results/exp3/plots/plot_{current_time}_axiom_ablation.png',
        metadata = metadata,
        backend = 'agg', # default
        dpi=300,
        bbox_inches = 'tight'
    )

    return