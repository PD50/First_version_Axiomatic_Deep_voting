"""
The implementation of experiment 3
"""
import os
import time
from datetime import datetime
import numpy as np
import random
import json
from tqdm import tqdm
import math

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torchinfo import summary

import utils
from utils import flatten_list, flatten_onehot_profile
import generate_data
from generate_data import generate_profile_data, pad_profile_data
from generate_data import onehot_profile_data
import models
from models import MLP, CNN, WEC
from set_transformer import (
    SetTransformer,
    SetTransformer2logits,
    SetTransformer2rule,
    SetTransformer2rule_n,
)
import train_and_eval
import axioms_continuous

from gensim.models import Word2Vec



def experiment3(
        architecture,
        max_num_voters,
        max_num_alternatives,
        election_sampling,
        num_gradient_steps,
        report_intervals,
        eval_dataset_size,
        model_to_rule,
        sample_size_applicable,
        sample_size_maximal,
        architecture_parameters,
        axioms_check_model,
        axioms_check_rule,
        axiom_opt,
        comp_rules_axioms,
        comp_rules_similarity,
        distance='L2',
        random_seed=None,
        batch_size=100,
        learning_rate = 1e-3,
        learning_scheduler = None,
        weight_decay = 0,
        loss_report_intervals = None,
        save_model=False,
        load_model_from = None,
    ):
    """
    Implements our experiment 3

    Inputs:
    * `architecture` can be either `MLP`, `CNN`, `WEC`. The latter 
      ones require additional parameters which can be passed as 
      `architecture_parameters` below (otherwise default values are chosen).

    * `max_num_voters`: a positive integer describing the maximal number of 
      voters that will be considered
    * `max_num_alternatives`: a positive integer describing the maximal number 
      of alternatives that will be considered
    * `election_sampling`: a dictionary describing the parameters for the
      probability model with which profiles are generated. The most important 
      key is `probmodel`. See 
      https://pref-voting.readthedocs.io/en/latest/generate_profiles.html
    
    * `num_gradient_steps`: a positive integer describing the number of 
      gradient steps performed during training the model.
    * `report_intervals` a positive integer n such that after every n gradient 
      steps the current performance (admissibility and axiom satisfaction) of 
      the model is checked and printed
    * `eval_dataset_size`: a positive integer describing the number of profiles
      with their corresponding winning sets that should be used for testing.

    * `model_to_rule`: a dictionary describes which ways of turning the model 
      into a voting rule will be considered (e.g., when considering 
      admissibility, axiom satisfaction, or similarity to other rules) and with
      which parameters:
      {
        'plain':<False or True>
        'neut-averaged':<False or None or positive integer>,
        'neut-anon-averaged':<False or a list [a,b] with a and b either None 
        or a positive integer>
      }  
      If 'plain' is False (resp., True), the plain voting rule obtained from 
      the model won't be (resp., will be) considered. 
      If 'neut-averaged' is False (resp. None, or a positive integer n), the 
      neutrality-averaged voting rule obtained from the model is not considered
      (resp. is checked for all neutrality permutations or for n-many sampled 
      permutations). 
      Similarly for 'neut-anon-averaged': if not False, i.e., if a pair [a,b] 
      is given, then a (either None or a positive integer) describes the 
      neutrality averaging and b (either None or a positive integer) describes 
      the anonymity-averaging.
      
    * `sample_size_applicable`: a positive integer describing the number of 
      sampled profiles on which the axioms are checked and applicable. 
    * `sample_size_maximal`: a positive integer describing how many profiles 
      are at most sampled when trying to find profiles on which the axioms are 
      applicable. 

     * `architecture_parameters` is not needed for MLP, but CNN and WEC they 
      are given the following default values
        * For CNN: 
          {
            'kernel1':[5,1] , 
            'kernel2':[1,5], 
            'channels':64
          } 
          Here `kernel1`: a list [height, width] of the dimension for the 
          kernel/filter in the first convolutional layer of the model. (The 
          height of the images is max_num_alternatives and the width of the 
          images is max_num_voters.) If max num voters/alternatives are quite 
          small, the kernel cannot be too big, otherwise error will be raised.  
          Similarly, `kernel2` is the [height, width]-dimension for the 
          kernel/filter in the second convolutional layer of the model.
          Finally, `channels` is the number of channels of the feature maps of 
          the model.
        * For WEC: 
          {
            'we_corpus_size':int(1e5), 
            'we_size':100, 
            'we_window':5, 
            'we_algorithm':1,
            'load_embeddings_from':'path'
          } 
          Here 'we_corpus_size' is the number of profiles used to pretrain the 
          word embeddings,  `we_size` is the size (i.e., length of vector) of 
          the word embeddings, `we_window` is the size of the window used when 
          training the word embeddings. Finally, `we_algorithm` is either 0 or 
          1 depending on whether one uses the CBOW algorithm or the skip gram 
          algorithm. If 'load_embeddings_from' is specified with a path to 
          stored embeddings, then these are loaded rather than computing new 
          ones. 
              
    * `axioms_check_model` is a list of names of axioms whose satisfaction is 
      checked for the trained model. By default, it is set to all axioms. If 
      empty, no axioms are checked.
    * `axioms_check_rule` is a list of names of axioms whose satisfaction is 
      checked for the rules. By default, it is only the condorcet and the 
      independence axiom, since the other axioms are always satisfied for 
      common voting rules. If empty, no axioms are checked.   

    * `axiom_opt` is a dictionary describing which axioms are optimized to what
      degree during training. This is a dictionary with the following keys 
        axiom_opt = {
            'No_winner':{'weight':10, 'period':'always'},
            'All_winners':None,
            'Inadmissible':None,
            'Resoluteness':None,
            'Anonymity':None, 
            'Neutrality':{'weight':1, 'period':'always', 'sample':25}, 
            'Condorcet1':{'weight':2, 'period':{'from':500, 'until':1000}}, 
            'Condorcet2':None,         
            'Pareto1':None, 
            'Pareto2':{'weight':1, 'period':'always'},             
            'Independence':None
        }    
      where 'None' means the respective axiom is not explicitly optimized, 
      while a dictionary means the axiom is optimized according to the 
      parameters state in the dictionary: 
      * the combined loss is a weighted sum of the losses of each axiom and 
        'weight' described the weight of the axiom in this weighted sum,
      * 'period' described the period during training when this axiom is 
        optimized for. If 'always', then the whole time, and if 
        {'from':500, 'until':1000} then from gradient step 500 until (and 
        excluding) gradient step 1000.
      * If the axiom requires sampling, then 'sample' many are taken.
      
    * `comp_rules_axioms` is a list of rule names whose axiom satisfaction is 
      compared to that of the model.
    * `comp_rules_similarity` is a list of rule names to which the model is 
      compared to in terms of similarity.

    * `distance` describes the notion of distance used when calculating the 
      continuous versions of the voting axioms. The default is 'L2'. Other 
      options are 'cos_sim' (cosine similarity) and 'KLD' (Kullback-Leibler 
      divergence).
      
    * `random_seed` (optional): If not None, set all random seeds to this 
      provided value.
    * `batch_size`: a positive integer describing the size of the batches when 
      training the model. The default is 64.
    * `learning_rate`: a float number describing the learning rate when 
      training the model. The default value is 1e-3.
    * `learning_scheduler`: By default None, but if given, then a positive 
      integer describing the T_0 value for the CosineAnnealingWarmRestarts 
      scheduler (i.e., the number of iterations for the first restart).
    * `weight_decay` of the optimizer which we set by default to 0 since we use 
      synthetic data and hence don't need regularization (its usual default 
      value is 0.01).  
    * `loss_report_intervals` is, if not None, a positive integer describing 
      how often the loss is stored for later printing. 
    * `save_model`: By default False, but if true, the neural network model 
      will be saved.
    * If `load_model_from` is given, it should be a path to a folder with  a 
      (previously trained) pytorch model 'model.pth' which is then used instead
      of a randomly initialized one.   
    """



    # SET UP BASICS

    start_time = time.time()

    assert (
        architecture in ['MLP', 'CNN', 'WEC', 'SetTransformer']
    ), f"The supported architectures are 'MLP', 'CNN', 'WEC', and 'SetTransformer' but {architecture} was given"

    if architecture_parameters is None:
        if architecture == 'CNN':
            architecture_parameters = {
                'kernel1':[5,1] , 
                'kernel2':[1,5], 
                'channels':64} 
        if architecture == 'WEC':
            architecture_parameters = {
                'we_corpus_size':int(1e5),
                'we_size':100, 
                'we_window':5, 
                'we_algorithm':1}
        if architecture == 'SetTransformer':
            architecture_parameters = {
                'd_model': 128,
                'n_heads': 4,
                'd_head': 32,
                'n_encoder_layers': 4,
                'pma_outputs': 1,
            }
 
    # Distance functions for computing continuous versions of axioms 
    KLD = lambda x, y : nn.KLDivLoss(log_target=True, reduction='batchmean')(x.log_softmax(dim=1), y.log_softmax(dim=1))
    L2 = lambda x, y: (1/len(x))*sum(nn.PairwiseDistance(p=2)(x,y))
    cos_sim = lambda x, y: (1/len(x))*sum(nn.CosineSimilarity(dim=1, eps=1e-8)(x,y))

    assert (
        distance in ['L2', 'cos_sim', 'KLD']
    ), f"The supported distances are 'L2', 'cos_sim', and 'KLD' but {distance} was given"

    if distance == 'L2':
        distance_fn = L2
    if distance == 'cos_sim':
        distance_fn = cos_sim
    if distance == 'KLD':
        distance_fn = KLD

    # Set seeds
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False

    # Set up saving of results
    prob_model = election_sampling['probmodel']
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    location = f"./results/exp3/{architecture}/exp3_{current_time}_{prob_model}"
    os.makedirs(location, exist_ok=True)
    print(f'Saving location: {location}')


    results = {
        "location": location,
        "architecture": architecture,
        "max_num_voters": max_num_voters,
        "max_num_alternatives": max_num_alternatives,
        "election_sampling": election_sampling,
        "num_gradient_steps": num_gradient_steps,
        "report_intervals": report_intervals,
        "eval_dataset_size": eval_dataset_size,
        "model_to_rule": model_to_rule,
        "sample_size_applicable": sample_size_applicable,
        "sample_size_maximal": sample_size_maximal,
        "architecture_parameters": architecture_parameters,
        "axioms_check_model_interim": axioms_check_model,
        "axioms_check_rule": axioms_check_rule,
        "axiom_opt": axiom_opt,
        "comp_rules_axioms": comp_rules_axioms,
        "comp_rules_similarity": comp_rules_similarity,
        "distance": distance,
        "random_seed": random_seed,
        "batch_size": batch_size,        
        "learning_rate": learning_rate,
        "learning_scheduler": learning_scheduler,
        "weight_decay": weight_decay,
        "loss_report_intervals": loss_report_intervals,
        "save_model": save_model,
        "load_model_from": load_model_from,
    }

    with open(f"{location}/results.json", "w") as json_file:
        json.dump(results, json_file)




    # GENERATING DATA

    # Training data will be generated before each training batch
    # Only WEC first needs to pretrain word embeddings

    if architecture == 'WEC':
        # First gather architecture parameters
        we_corpus = architecture_parameters['we_corpus_size']
        we_size = architecture_parameters['we_size']
        we_window = architecture_parameters['we_window']
        we_algorithm = architecture_parameters['we_algorithm']
        load_embeddings_from = architecture_parameters.get('load_embeddings_from', None)


        if load_embeddings_from is None:
            print('Now pretraining word embeddings')

            print("First generate profiles and turn them into corpus")
            # Generate profiles and their winning sets
            X_train_profs, _ , _ = generate_profile_data(
                max_num_voters,
                max_num_alternatives,
                we_corpus,
                election_sampling,
                [],
                merge='empty'
            )

            # Turn set of profiles X into a corpus (each profile a sentence)
            train_sentences = [
                [models.ranking_to_string(ranking) for ranking in profile.rankings]
                for profile in X_train_profs
            ]
            # Add the 'UNK' word for future unknown words. And add 'PAD' for
            # padding sentences to desired length. (Adding these after training 
            # the embeddings seems inefficient.)
            train_sentences_with_UNK_and_PAD = train_sentences + [['UNK'], ['PAD']]

            # Pretrain an word embedding on this corpus
            print('Now train word embeddings')
            pre_embeddings = Word2Vec(
                train_sentences_with_UNK_and_PAD,
                vector_size=we_size,
                window=we_window,
                min_count=1,
                workers=8,
                sg=we_algorithm
            )
            print('Done pretraining word embeddings.')

            if save_model:
                print('Save the word embeddings.')
                # We save the pre_embedding word2vec model, so it can be used
                # when initializing a WEC model that we then load with
                # previously trained parameters
                pre_embeddings.save(f"{location}/pre_embeddings.bin")

        # Load back with memory-mapping = read-only, shared across processes.
        if load_embeddings_from is not None:
            print('Load the word embeddings.')
            load_embeddings_from = architecture_parameters['load_embeddings_from']
            pre_embeddings = Word2Vec.load(f"{load_embeddings_from}/pre_embeddings.bin")


    # Dev dataset
    print('Now generate dev and test profiles')

    X_dev_profs, _, _ = generate_profile_data(
        max_num_voters,
        max_num_alternatives,
        eval_dataset_size,
        election_sampling,
        [],
        merge='empty',
    )

    X_test_profs, _, _ = generate_profile_data(
        max_num_voters,
        max_num_alternatives,
        eval_dataset_size,
        election_sampling,
        [],
        merge='empty',
    )




    # NEURAL NETWORK TRAINING


    #Initialize our model for the experiment
    if architecture == 'MLP':
        exp_model = MLP(max_num_voters, max_num_alternatives)
        exp_optimizer = torch.optim.AdamW(
            exp_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        model_on_profiles = lambda X : models.MLP2logits(exp_model,X)

    if architecture == 'CNN':
        exp_model = CNN(
            max_num_voters,
            max_num_alternatives,
            architecture_parameters['kernel1'],
            architecture_parameters['kernel2'],
            architecture_parameters['channels']
        )
        exp_optimizer = torch.optim.AdamW(
            exp_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        model_on_profiles = lambda X : models.CNN2logits(exp_model,X)

    if architecture == 'WEC':    
        exp_model = WEC(pre_embeddings, max_num_voters, max_num_alternatives)
        exp_optimizer = torch.optim.AdamW(
            exp_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        model_on_profiles = lambda X : models.WEC2logits(exp_model,X)

    if architecture == 'SetTransformer':
        exp_model = SetTransformer(
            max_num_voters,
            max_num_alternatives,
            d_model=architecture_parameters.get('d_model', 128),
            n_heads=architecture_parameters.get('n_heads', 4),
            d_head=architecture_parameters.get('d_head', 32),
            n_encoder_layers=architecture_parameters.get('n_encoder_layers', 4),
            pma_outputs=architecture_parameters.get('pma_outputs', 1),
        )
        exp_optimizer = torch.optim.AdamW(
            exp_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        model_on_profiles = lambda X : SetTransformer2logits(exp_model, X)

    # Load the previous state of the model if given
    if load_model_from is not None:
        checkpoint = torch.load(f'{load_model_from}/model.pth')
        exp_model.load_state_dict(checkpoint['model_state_dict'])
        exp_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Set up the learning rate scheduler if given
    if learning_scheduler is not None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            exp_optimizer, 
            T_0 = learning_scheduler
        )



    print('Now starting to train')
    learning_curve = {}
    loss_curve = {}

    
    for step in tqdm(range(num_gradient_steps)):
        
        exp_model.train()

        # Generate data for the batch
        X_train_profs, _, _ = generate_profile_data(
            max_num_voters,
            max_num_alternatives,
            batch_size,
            election_sampling,
            [],
            merge='empty',
        )

        # Compute loss
        if axiom_opt['No_winner'] is not None and (axiom_opt['No_winner']['period'] == 'always' or step in range(axiom_opt['No_winner']['period']['from'], axiom_opt['No_winner']['period']['until'])):
            loss_nowi = axiom_opt['No_winner']['weight'] * axioms_continuous.ax_no_winners_cont(model_on_profiles, X_train_profs)
        else:
            loss_nowi = torch.tensor([0])

        if axiom_opt['All_winners'] is not None and (axiom_opt['All_winners']['period'] == 'always' or step in range(axiom_opt['All_winners']['period']['from'], axiom_opt['All_winners']['period']['until'])):
            loss_alwi = axiom_opt['All_winners']['weight'] * axioms_continuous.ax_all_winners_cont(model_on_profiles, X_train_profs,distance_fn)
        else:
            loss_alwi = torch.tensor([0])

        if axiom_opt['Inadmissible'] is not None and (axiom_opt['Inadmissible']['period'] == 'always' or step in range(axiom_opt['Inadmissible']['period']['from'], axiom_opt['Inadmissible']['period']['until'])):
            loss_inad = axiom_opt['Inadmissible']['weight'] * axioms_continuous.ax_inadmissibility_cont(model_on_profiles, X_train_profs)
        else:
            loss_inad = torch.tensor([0])

        if axiom_opt['Resoluteness'] is not None and (axiom_opt['Resoluteness']['period'] == 'always' or step in range(axiom_opt['Resoluteness']['period']['from'], axiom_opt['Resoluteness']['period']['until'])):
            loss_reso = axiom_opt['Resoluteness']['weight'] * axioms_continuous.ax_resoluteness_cont(model_on_profiles, X_train_profs)
        else:
            loss_reso = torch.tensor([0])

        if axiom_opt['Parity'] is not None and (axiom_opt['Parity']['period'] == 'always' or step in range(axiom_opt['Parity']['period']['from'], axiom_opt['Parity']['period']['until'])):
            loss_pari = axiom_opt['Parity']['weight'] * axioms_continuous.ax_parity_cont(model_on_profiles, X_train_profs)
        else:
            loss_pari = torch.tensor([0])

        if axiom_opt['Anonymity'] is not None and (axiom_opt['Anonymity']['period'] == 'always' or step in range(axiom_opt['Anonymity']['period']['from'], axiom_opt['Anonymity']['period']['until'])):
            loss_anon = axiom_opt['Anonymity']['weight'] * axioms_continuous.ax_anonymity_cont(model_on_profiles, X_train_profs,axiom_opt['Anonymity']['sample'],distance_fn)
        else:
            loss_anon = torch.tensor([0])

        if axiom_opt['Neutrality'] is not None and (axiom_opt['Neutrality']['period'] == 'always' or step in range(axiom_opt['Neutrality']['period']['from'], axiom_opt['Neutrality']['period']['until'])):
            loss_neut = axiom_opt['Neutrality']['weight'] * axioms_continuous.ax_neutrality_cont(model_on_profiles, X_train_profs,axiom_opt['Neutrality']['sample'],distance_fn)
        else:
            loss_neut = torch.tensor([0])

        if axiom_opt['Condorcet1'] is not None and (axiom_opt['Condorcet1']['period'] == 'always' or step in range(axiom_opt['Condorcet1']['period']['from'], axiom_opt['Condorcet1']['period']['until'])):
            loss_con1 = axiom_opt['Condorcet1']['weight'] * axioms_continuous.ax_condorcet1_cont(model_on_profiles, X_train_profs,distance_fn)
        else:
            loss_con1 = torch.tensor([0])

        if axiom_opt['Condorcet2'] is not None and (axiom_opt['Condorcet2']['period'] == 'always' or step in range(axiom_opt['Condorcet2']['period']['from'], axiom_opt['Condorcet2']['period']['until'])):
            loss_con2 = axiom_opt['Condorcet2']['weight'] * axioms_continuous.ax_condorcet2_cont(model_on_profiles, X_train_profs,distance_fn)
        else:
            loss_con2 = torch.tensor([0])

        if axiom_opt['Pareto1'] is not None and (axiom_opt['Pareto1']['period'] == 'always' or step in range(axiom_opt['Pareto1']['period']['from'], axiom_opt['Pareto1']['period']['until'])):
            loss_par1 = axiom_opt['Pareto1']['weight'] * axioms_continuous.ax_pareto1_cont(model_on_profiles, X_train_profs,distance_fn)
        else:
            loss_par1 = torch.tensor([0])

        if axiom_opt['Pareto2'] is not None and (axiom_opt['Pareto2']['period'] == 'always' or step in range(axiom_opt['Pareto2']['period']['from'], axiom_opt['Pareto2']['period']['until'])):
            loss_par2 = axiom_opt['Pareto2']['weight'] * axioms_continuous.ax_pareto2_cont(model_on_profiles, X_train_profs,distance_fn)
        else:
            loss_par2 = torch.tensor([0])

        if axiom_opt['Independence'] is not None and (axiom_opt['Independence']['period'] == 'always' or step in range(axiom_opt['Independence']['period']['from'], axiom_opt['Independence']['period']['until'])):
            loss_inde = axiom_opt['Independence']['weight'] * axioms_continuous.ax_independence_cont(model_on_profiles, X_train_profs,axiom_opt['Independence']['sample'],distance_fn) 
        else:
            loss_inde = torch.tensor([0])

        # Sum up loss
        loss = loss_nowi + loss_alwi + loss_inad + loss_reso + loss_pari + loss_anon + loss_neut + loss_con1 + loss_con2 + loss_par1 + loss_par2 + loss_inde

        # Backpropagation
        exp_optimizer.zero_grad()
        loss.backward()
        exp_optimizer.step()    

        if learning_scheduler is not None:
            scheduler.step()

        if (loss_report_intervals is not None and
                step % loss_report_intervals == 0):
             # Loss on the last gradient step
            latest_loss = {
                'loss_nowi':loss_nowi.item(),
                'loss_alwi':loss_alwi.item(),
                'loss_inad':loss_inad.item(),
                'loss_reso':loss_reso.item(),
                'loss_pari':loss_pari.item(),
                'loss_anon':loss_anon.item(),
                'loss_neut':loss_neut.item(),
                'loss_con1':loss_con1.item(),
                'loss_con2':loss_con2.item(),                
                'loss_par1':loss_par1.item(),
                'loss_par2':loss_par2.item(),
                'loss_inde':loss_inde.item()
            }
            loss_curve[step] = latest_loss


        # Interim evaluation on dev set
        if step % report_intervals == report_intervals - 1:

            # Define the rule computed by the model
            if architecture == 'MLP':
                if model_to_rule['plain'] == True:
                    model_rule = models.MLP2rule(exp_model)
                    model_rule_full = models.MLP2rule(exp_model, full=True)
                if model_to_rule['neut-averaged'] != False:
                    model_rule_n = models.MLP2rule_n(
                        exp_model, 
                        model_to_rule['neut-averaged']
                        )
                    model_rule_n_full = models.MLP2rule_n(
                        exp_model, 
                        model_to_rule['neut-averaged'], 
                        full=True
                        )
                if model_to_rule['neut-anon-averaged'] != False:
                    model_rule_na = models.MLP2rule_na(
                        exp_model, 
                        model_to_rule['neut-anon-averaged'][0],
                        model_to_rule['neut-anon-averaged'][1],
                        )
                    model_rule_na_full = models.MLP2rule_na(
                        exp_model, 
                        model_to_rule['neut-anon-averaged'][0],
                        model_to_rule['neut-anon-averaged'][1],
                        full=True
                        )                      

            if architecture == 'CNN':
                if model_to_rule['plain'] == True:
                    model_rule = models.CNN2rule(exp_model)
                    model_rule_full = models.CNN2rule(exp_model, full=True)
                if model_to_rule['neut-averaged'] != False:
                    model_rule_n = models.CNN2rule_n(
                        exp_model, 
                        model_to_rule['neut-averaged']
                        )
                    model_rule_n_full = models.CNN2rule_n(
                        exp_model, 
                        model_to_rule['neut-averaged'],
                        full=True
                        )
                if model_to_rule['neut-anon-averaged'] != False:
                    model_rule_na = models.CNN2rule_na(
                        exp_model, 
                        model_to_rule['neut-anon-averaged'][0],
                        model_to_rule['neut-anon-averaged'][1],
                        )
                    model_rule_na_full = models.CNN2rule_na(
                        exp_model, 
                        model_to_rule['neut-anon-averaged'][0],
                        model_to_rule['neut-anon-averaged'][1],
                        full=True
                        )                      

            if architecture == 'WEC':    
                if model_to_rule['plain'] == True:
                    model_rule = models.WEC2rule(exp_model)
                    model_rule_full = models.WEC2rule(exp_model, full=True)
                if model_to_rule['neut-averaged'] != False:
                    model_rule_n = models.WEC2rule_n(
                        exp_model, 
                        model_to_rule['neut-averaged']
                        )
                    model_rule_n_full = models.WEC2rule_n(
                        exp_model, 
                        model_to_rule['neut-averaged'],
                        full=True
                        )
                if model_to_rule['neut-anon-averaged'] != False:
                    print('Neut-anon-averaging is not needed for WEC since anonymous by construction')

            if architecture == 'SetTransformer':
                if model_to_rule['plain'] == True:
                    model_rule = SetTransformer2rule(exp_model)
                    model_rule_full = SetTransformer2rule(exp_model, full=True)
                if model_to_rule['neut-averaged'] != False:
                    model_rule_n = SetTransformer2rule_n(
                        exp_model,
                        model_to_rule['neut-averaged']
                        )
                    model_rule_n_full = SetTransformer2rule_n(
                        exp_model,
                        model_to_rule['neut-averaged'],
                        full=True
                        )
                if model_to_rule['neut-anon-averaged'] != False:
                    print('Neut-anon-averaging is not needed for SetTransformer since anonymous by construction')


            # Loss on the last gradient step
            latest_loss = {
                'loss_nowi':loss_nowi.item(),
                'loss_alwi':loss_alwi.item(),
                'loss_inad':loss_inad.item(),
                'loss_reso':loss_reso.item(),
                'loss_pari':loss_pari.item(),
                'loss_anon':loss_anon.item(),
                'loss_neut':loss_neut.item(),
                'loss_con1':loss_con1.item(),
                'loss_con2':loss_con2.item(),                
                'loss_par1':loss_par1.item(),
                'loss_par2':loss_par2.item(),
                'loss_inde':loss_inde.item()
            }
            print('The loss on that last training batch (if nonzero):')
            for k,v in latest_loss.items():
                if v != 0:
                    print('   ', k, v)

            # Admissability

            admissibility_summary_plain = None
            admissibility_summary_neut = None
            admissibility_summary_neut_anon = None

            if model_to_rule['plain'] == True:
                admissibility_summary_plain = train_and_eval.admissibility(model_rule_full,X_dev_profs)
                print('Admissability on dev profiles (plain):')
                for k,v in admissibility_summary_plain.items():
                    print('   ', k, v)
            if model_to_rule['neut-averaged'] != False:
                admissibility_summary_neut = train_and_eval.admissibility(model_rule_n_full,X_dev_profs)
                print('Admissability on dev profiles (neutrality-averaged):')
                for k,v in admissibility_summary_neut.items():
                    print('   ', k, v)            
            if model_to_rule['neut-anon-averaged'] != False:    
                admissibility_summary_neut_anon = train_and_eval.admissibility(model_rule_na_full,X_dev_profs)
                print('Admissability on dev profiles (neut-anon-averaged):')
                for k,v in admissibility_summary_neut_anon.items():
                    print('   ', k, v)

    

            # Axiom satisfaction

            axiom_satisfaction_model_plain = None
            axiom_satisfaction_model_neut = None
            axiom_satisfaction_model_neut_anon = None

            if model_to_rule['plain'] == True:
                print('Axiom satisfaction (plain):')
                axiom_satisfaction_model_plain = {}
                for axiom_name in axioms_check_model:
                    sat = train_and_eval.axiom_satisfaction(model_rule,
                            utils.dict_axioms[axiom_name],
                            max_num_voters,
                            max_num_alternatives,
                            election_sampling,
                            sample_size_applicable,
                            sample_size_maximal,
                            utils.dict_axioms_sample[axiom_name],
                            full_profile=False,
                            comparison_rule=None)
                    axiom_satisfaction_model_plain[axiom_name] = sat
                    cond_sat = sat['cond_satisfaction']
                    print(f'    {axiom_name} {100*cond_sat}%')

            if model_to_rule['neut-averaged'] != False:
                print('Axiom satisfaction (neutrality-averaged):')
                axiom_satisfaction_model_neut = {}
                for axiom_name in axioms_check_model:
                    sat = train_and_eval.axiom_satisfaction(model_rule_n,
                            utils.dict_axioms[axiom_name],
                            max_num_voters,
                            max_num_alternatives,
                            election_sampling,
                            sample_size_applicable,
                            sample_size_maximal,
                            utils.dict_axioms_sample[axiom_name],
                            full_profile=False,
                            comparison_rule=None)
                    axiom_satisfaction_model_neut[axiom_name] = sat
                    cond_sat = sat['cond_satisfaction']
                    print(f'    {axiom_name} {100*cond_sat}%')

            if model_to_rule['neut-anon-averaged'] != False:
                print('Axiom satisfaction (neut-anon-averaged):')
                axiom_satisfaction_model_neut_anon = {}
                for axiom_name in axioms_check_model:
                    sat = train_and_eval.axiom_satisfaction(model_rule_na,
                            utils.dict_axioms[axiom_name],
                            max_num_voters,
                            max_num_alternatives,
                            election_sampling,
                            sample_size_applicable,
                            sample_size_maximal,
                            utils.dict_axioms_sample[axiom_name],
                            full_profile=False,
                            comparison_rule=None)
                    axiom_satisfaction_model_neut_anon[axiom_name] = sat
                    cond_sat = sat['cond_satisfaction']
                    print(f'    {axiom_name} {100*cond_sat}%')               

            # Comparison rules

            similarities_plain = None
            similarities_neut = None
            similarities_neut_anon = None

            if comp_rules_similarity: # True if nonempty

                if model_to_rule['plain'] == True:
                    print('Similarity to other rules on dev set (plain):')
                    similarities_plain = train_and_eval.rule_similarity(
                        model_rule,
                        comp_rules_similarity,
                        X_dev_profs,
                        verbose=True
                    )          
                    for rule_name in comp_rules_similarity:
                        coinc = 100*similarities_plain[rule_name]["identity_accu"]
                        print(f'    {rule_name} {coinc}%')

                if model_to_rule['neut-averaged'] != False:
                    print('Similarity to other rules on dev set (neutrality-averaged):')
                    similarities_neut = train_and_eval.rule_similarity(
                        model_rule_n,
                        comp_rules_similarity,
                        X_dev_profs,
                        verbose=True
                    )          
                    for rule_name in comp_rules_similarity:
                        coinc = 100*similarities_neut[rule_name]["identity_accu"]
                        print(f'    {rule_name} {coinc}%')

                if model_to_rule['neut-anon-averaged'] != False:
                    print('Similarity to other rules on dev set (neut-anon-averaged):')
                    similarities_neut_anon = train_and_eval.rule_similarity(
                        model_rule_na, 
                        comp_rules_similarity, 
                        X_dev_profs,
                        verbose=True
                    )          
                    for rule_name in comp_rules_similarity:
                        coinc = 100*similarities_neut_anon[rule_name]["identity_accu"]
                        print(f'    {rule_name} {coinc}%')


            # Add all interim evaluations to results
            learning_curve[f'{step}'] = {
                'loss_on_last_batch':latest_loss,
                'admissability':{
                    'plain':admissibility_summary_plain,
                    'neut':admissibility_summary_neut,
                    'neut-anon':admissibility_summary_neut_anon,
                },    
                'axiom_satisfaction':{
                    'plain':axiom_satisfaction_model_plain,
                    'neut':axiom_satisfaction_model_neut,
                    'neut-anon':axiom_satisfaction_model_neut_anon,
                },     
                'similarity_to_other_rules':{
                    'plain':similarities_plain,
                    'neut':similarities_neut,
                    'neut-anon':similarities_neut_anon,
            }
            }

    # Store evolution of loss
    if (loss_report_intervals is not None):
        with open(f"{location}/results.json") as json_file:
            data = json.load(json_file)

        data.update({"loss curve": loss_curve})

        with open(f"{location}/results.json", "w") as json_file:
            json.dump(data, json_file)

    # Store evolution of learning
    with open(f"{location}/results.json") as json_file:
        data = json.load(json_file)

    data.update({"learning curve": learning_curve})

    with open(f"{location}/results.json", "w") as json_file:
        json.dump(data, json_file)

    if save_model:
        # We save both the model state and the optimizer state to be able to
        # continue training later on.
        torch.save({
            'arguments' : [max_num_voters, max_num_alternatives, architecture_parameters],
            'model_state_dict': exp_model.state_dict(),
            'optimizer_state_dict': exp_optimizer.state_dict()
            }, f"{location}/model.pth")



    # EVALUATION

    # Define the rule computed by the model
    if architecture == 'MLP':
        if model_to_rule['plain'] == True:
            model_rule = models.MLP2rule(exp_model)
            model_rule_full = models.MLP2rule(exp_model, full=True)
        if model_to_rule['neut-averaged'] != False:
            model_rule_n = models.MLP2rule_n(
                exp_model, 
                model_to_rule['neut-averaged']
                )
            model_rule_n_full = models.MLP2rule_n(
                exp_model, 
                model_to_rule['neut-averaged'],
                full=True
                )
        if model_to_rule['neut-anon-averaged'] != False:
            model_rule_na = models.MLP2rule_na(
                exp_model, 
                model_to_rule['neut-anon-averaged'][0],
                model_to_rule['neut-anon-averaged'][1],
                )
            model_rule_na_full = models.MLP2rule_na(
                exp_model, 
                model_to_rule['neut-anon-averaged'][0],
                model_to_rule['neut-anon-averaged'][1],
                full=True
                )     



    if architecture == 'CNN':
        if model_to_rule['plain'] == True:
            model_rule = models.CNN2rule(exp_model)
            model_rule_full = models.CNN2rule(exp_model, full=True)
        if model_to_rule['neut-averaged'] != False:
            model_rule_n = models.CNN2rule_n(
                exp_model, 
                model_to_rule['neut-averaged']
                )
            model_rule_n_full = models.CNN2rule_n(
                exp_model, 
                model_to_rule['neut-averaged'], 
                full=True)
        if model_to_rule['neut-anon-averaged'] != False:
            model_rule_na = models.CNN2rule_na(
                exp_model, 
                model_to_rule['neut-anon-averaged'][0],
                model_to_rule['neut-anon-averaged'][1],
                )
            model_rule_na_full = models.CNN2rule_na(
                exp_model, 
                model_to_rule['neut-anon-averaged'][0],
                model_to_rule['neut-anon-averaged'][1],
                full=True
                )                      



    if architecture == 'WEC':    
        if model_to_rule['plain'] == True:
            model_rule = models.WEC2rule(exp_model)
            model_rule_full = models.WEC2rule(exp_model, full=True)
        if model_to_rule['neut-averaged'] != False:
            model_rule_n = models.WEC2rule_n(exp_model, model_to_rule['neut-averaged'])
            model_rule_n_full = models.WEC2rule_n(exp_model, model_to_rule['neut-averaged'], full=True)
        if model_to_rule['neut-anon-averaged'] != False:
            print('Neut-anon-averaging is not implemented for WEC')

    if architecture == 'SetTransformer':
        if model_to_rule['plain'] == True:
            model_rule = SetTransformer2rule(exp_model)
            model_rule_full = SetTransformer2rule(exp_model, full=True)
        if model_to_rule['neut-averaged'] != False:
            model_rule_n = SetTransformer2rule_n(exp_model, model_to_rule['neut-averaged'])
            model_rule_n_full = SetTransformer2rule_n(exp_model, model_to_rule['neut-averaged'], full=True)
        if model_to_rule['neut-anon-averaged'] != False:
            print('Neut-anon-averaging is not needed for SetTransformer since anonymous by construction')


    # Admissability

    admissibility_summary_plain = None
    admissibility_summary_neut = None
    admissibility_summary_neut_anon = None

    if model_to_rule['plain'] == True:
        admissibility_summary_plain = train_and_eval.admissibility(model_rule_full,X_test_profs)
        print('Admissability on test profiles (plain):')
        for k,v in admissibility_summary_plain.items():
            print('   ', k, v)
    if model_to_rule['neut-averaged'] != False:
        admissibility_summary_neut = train_and_eval.admissibility(model_rule_n_full,X_test_profs)
        print('Admissability on test profiles (neutrality-averaged):')
        for k,v in admissibility_summary_neut.items():
            print('   ', k, v)            
    if model_to_rule['neut-anon-averaged'] != False:    
        admissibility_summary_neut_anon = train_and_eval.admissibility(model_rule_na_full,X_test_profs)
        print('Admissability on test profiles (neut-anon-averaged):')
        for k,v in admissibility_summary_neut_anon.items():
            print('   ', k, v)


    with open(f"{location}/results.json") as json_file:
        data = json.load(json_file)

    data.update({"admissability": {
                    'plain':admissibility_summary_plain,
                    'neut':admissibility_summary_neut,
                    'neut-anon':admissibility_summary_neut_anon,
                }
                })

    with open(f"{location}/results.json", "w") as json_file:
        json.dump(data, json_file)





    # Initialize dictionary with axiom satisfactions
    axiom_satisfactions = {}

    # Axiom satisfaction of model
    axiom_satisfaction_model_plain = None
    axiom_satisfaction_model_neut = None
    axiom_satisfaction_model_neut_anon = None

    if model_to_rule['plain'] == True:
        print('Axiom satisfaction (plain):')
        axiom_satisfaction_model_plain = {}
        for axiom_name in axioms_check_model:
            sat = train_and_eval.axiom_satisfaction(model_rule,
                    utils.dict_axioms[axiom_name],
                    max_num_voters,
                    max_num_alternatives,
                    election_sampling,
                    sample_size_applicable,
                    sample_size_maximal,
                    utils.dict_axioms_sample[axiom_name],
                    full_profile=False,
                    comparison_rule=None)
            axiom_satisfaction_model_plain[axiom_name] = sat
            cond_sat = sat['cond_satisfaction']
            print(f'    {axiom_name} {100*cond_sat}%')

    if model_to_rule['neut-averaged'] != False:
        print('Axiom satisfaction (neutrality-averaged):')
        axiom_satisfaction_model_neut = {}
        for axiom_name in axioms_check_model:
            sat = train_and_eval.axiom_satisfaction(model_rule_n,
                    utils.dict_axioms[axiom_name],
                    max_num_voters,
                    max_num_alternatives,
                    election_sampling,
                    sample_size_applicable,
                    sample_size_maximal,
                    utils.dict_axioms_sample[axiom_name],
                    full_profile=False,
                    comparison_rule=None)
            axiom_satisfaction_model_neut[axiom_name] = sat
            cond_sat = sat['cond_satisfaction']
            print(f'    {axiom_name} {100*cond_sat}%')

    if model_to_rule['neut-anon-averaged'] != False:
        print('Axiom satisfaction (neut-anon-averaged):')
        axiom_satisfaction_model_neut_anon = {}
        for axiom_name in axioms_check_model:
            sat = train_and_eval.axiom_satisfaction(model_rule_na,
                    utils.dict_axioms[axiom_name],
                    max_num_voters,
                    max_num_alternatives,
                    election_sampling,
                    sample_size_applicable,
                    sample_size_maximal,
                    utils.dict_axioms_sample[axiom_name],
                    full_profile=False,
                    comparison_rule=None)
            axiom_satisfaction_model_neut_anon[axiom_name] = sat
            cond_sat = sat['cond_satisfaction']
            print(f'    {axiom_name} {100*cond_sat}%')       

    axiom_satisfactions['model_plain'] = axiom_satisfaction_model_plain    
    axiom_satisfactions['model_neut'] = axiom_satisfaction_model_neut
    axiom_satisfactions['model_neut_anon'] = axiom_satisfaction_model_neut_anon


    # Axiom satisfaction of rules
    print('Axiom satisfaction of the rules:')
    for rule_name in comp_rules_axioms:
        print(f'Rule {rule_name}')
        axiom_satisfaction_current_rule = {}
        for axiom_name in axioms_check_rule:
            sat = train_and_eval.axiom_satisfaction(utils.dict_rules_all[rule_name],
                    utils.dict_axioms[axiom_name],
                    max_num_voters,
                    max_num_alternatives,
                    election_sampling,
                    sample_size_applicable,
                    sample_size_maximal,
                    utils.dict_axioms_sample[axiom_name],
                    full_profile=False,
                    comparison_rule=None)
            axiom_satisfaction_current_rule[axiom_name] = sat
            cond_sat = sat['cond_satisfaction']
            print(f'    {axiom_name} {100*cond_sat}%')
        axiom_satisfactions[rule_name] = axiom_satisfaction_current_rule

    with open(f"{location}/results.json") as json_file:
        data = json.load(json_file)

    data.update({"axiom_satisfaction": axiom_satisfactions})

    with open(f"{location}/results.json", "w") as json_file:
        json.dump(data, json_file)





    # Comparison rules


    similarities_plain = None
    similarities_neut = None
    similarities_neut_anon = None

    if comp_rules_similarity: # True if nonempty

        if model_to_rule['plain'] == True:
            print('Similarity to other rules on test set (plain):')
            similarities_plain = train_and_eval.rule_similarity(
                model_rule, 
                comp_rules_similarity,
                X_test_profs,verbose=True
            )          
            for rule_name in comp_rules_similarity:
                coinc = 100*similarities_plain[rule_name]["identity_accu"]
                print(f'    {rule_name} {coinc}%')

        if model_to_rule['neut-averaged'] != False:
            print('Similarity to other rules on test set (neutrality-averaged):')
            similarities_neut = train_and_eval.rule_similarity(
                model_rule_n,
                comp_rules_similarity,
                X_test_profs,
                verbose=True
            )          
            for rule_name in comp_rules_similarity:
                coinc = 100*similarities_neut[rule_name]["identity_accu"]
                print(f'    {rule_name} {coinc}%')

        if model_to_rule['neut-anon-averaged'] != False:
            print('Similarity to other rules on test set (neut-anon-averaged):')
            similarities_neut_anon = train_and_eval.rule_similarity(
                model_rule_na,
                comp_rules_similarity,
                X_test_profs,
                verbose=True
            )          
            for rule_name in comp_rules_similarity:
                coinc = 100*similarities_neut_anon[rule_name]["identity_accu"]
                print(f'    {rule_name} {coinc}%')




    with open(f"{location}/results.json") as json_file:
        data = json.load(json_file)

    data.update({"rule_comparison": {
                    'plain':similarities_plain,
                    'neut':similarities_neut,
                    'neut-anon':similarities_neut_anon,
            }
            })

    with open(f"{location}/results.json", "w") as json_file:
        json.dump(data, json_file)



    # Computation time

    end_time = time.time()
    duration = end_time - start_time

    print(f'Runtime (in min): {round(duration/60)}')

    with open(f"{location}/results.json") as json_file:
        data = json.load(json_file)

    data.update({"runtime_in_sec": duration})

    with open(f"{location}/results.json", "w") as json_file:
        json.dump(data, json_file)

    return location