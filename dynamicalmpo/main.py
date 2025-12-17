import warnings
warnings.filterwarnings("ignore", message="to-Python converter for boost::shared_ptr<RDKit::FilterHierarchyMatcher> already registered", category=RuntimeWarning)

import argparse
import os
import random
import torch
import torch.optim as optim
import torch.nn as nn 
import numpy as np
import json 
import csv 
import time 
from tqdm import tqdm 
from typing import List, Dict, Any, Optional
import traceback 
from collections import defaultdict 

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning (main.py): RDKit not found. Some functionalities (constraint checking, scoring) will be limited or unavailable.")



from utils import (load_checkpoint, save_checkpoint, 
                   load_smiles_data, create_vocabulary_from_data,
                   PAD_token, SOS_token, EOS_token, PAD_char, SOS_char, EOS_char,
                   parse_json_config, load_smarts_from_file) 
from vocabulary import load_vocabulary
from model_definition import SmilesRNN

from agent import SmilesGeneratorAgent
from custom_chemprop import CustomChemprop
from scoring import PropertyScorer
from filters import ReinventDiversityFilter, NoDiversityFilter
import constraints 
from mpo import (StaticWeightMPO, DynamicWeightMPO, CyclicalMPO, RandomizedWeightMPO,
                 DynamicImprovementRateMPO, DynamicVarianceMPO, BaseMPOStrategy)

from losses import (BaseLossCalculator, ReinforceLoss, ReinventLoss,
                    AugmentedHillClimbLoss, PPOLoss) 


from RL_trainer import ReinforceTrainer
from supervised_trainer import train_supervised
from sampler import sample_molecules

torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])

def evaluate_smiles_scoring( 
    smiles_list: List[str],
    property_scorer: PropertyScorer,
    mpo_manager: BaseMPOStrategy,
    diversity_filter: ReinventDiversityFilter,
    constraint_config: Dict,
    args: argparse.Namespace,
    device: torch.device
) -> List[Dict]:
    """
    Evaluates a list of SMILES using the configured reward components, including batch docking.
    """
    print(f"\n--- Starting Scoring Mode Evaluation for {len(smiles_list)} SMILES ---")
    results = []
    start_time = time.time()


    desirability_configs_list = parse_json_config(getattr(args, 'desirability_configs', '[]'), "desirability_configs")
    mpo_combination = getattr(args, 'mpo_combination', 'sum')
    mpo_epsilon = getattr(args, 'mpo_product_epsilon', 0.01)
    R_min = getattr(args, 'R_min', 1e-6)
    prop_names = property_scorer.property_names
    num_props = len(prop_names)


    receptor_paths_scoring = getattr(args, 'receptor_paths', [])
    target_names_scoring = getattr(args, 'docking_target_names', [])


    try:
        current_weights = mpo_manager.get_weights(epoch=0).cpu().numpy()
    except Exception:
        print("Warning: Could not get MPO weights for epoch 0, using initial MPO weights for scoring.")
        current_weights = mpo_manager.initial_weights.cpu().numpy()

    if hasattr(property_scorer, '_batch_docking_results'):
        property_scorer._batch_docking_results = None
        property_scorer._batch_indices_processed_docking = None

    print("Performing initial SMILES validation...")
    valid_mols_map = {} 
    mols_for_scorer = []
    original_indices_for_scorer = []
    valid_smiles_list = [] 
    for idx, smi in enumerate(tqdm(smiles_list, desc="Validating SMILES")):
        result_row_init = {
            'InputSMILES': smi, 'IsValid': False, 'PassedConstraints': False,
            'MPO_Score': np.nan, 'DF_Scaffold': None, 'DF_ThresholdMet': False,
            'DF_BucketCount_Current': -1, 'DF_PenaltyApplied': False, 'Final_Reward': R_min
        }
        for p_name in prop_names:
            result_row_init[f'{p_name}_Raw'] = np.nan; result_row_init[f'{p_name}_Desire'] = np.nan
        results.append(result_row_init)

        if not RDKIT_AVAILABLE:
             if idx == 0: print("Warning: RDKit not available, cannot perform validity/constraint checks.")
             continue 

        mol = Chem.MolFromSmiles(smi)
        if mol:
            results[idx]['IsValid'] = True
            valid_mols_map[idx] = mol
            mols_for_scorer.append(mol)
            original_indices_for_scorer.append(idx)
            valid_smiles_list.append(smi) 

    if not mols_for_scorer:
        print("No valid molecules found to score.")
        return results 

    print(f"Checking constraints for {len(valid_smiles_list)} valid SMILES...")
    if RDKIT_AVAILABLE:
        passed_constraints_mask, _ = constraints.check_mandatory_constraints(
            valid_smiles_list, constraint_config, device 
        )
    else:
        passed_constraints_mask = torch.ones(len(valid_smiles_list), dtype=torch.bool, device=device)
        print("Warning: RDKit unavailable, assuming all valid SMILES pass constraints.")

    mols_passed_constraints = []
    original_indices_passed_constraints = []
    for i, mol in enumerate(mols_for_scorer):
        original_idx = original_indices_for_scorer[i]
        passes = passed_constraints_mask[i].item()
        results[original_idx]['PassedConstraints'] = passes
        if passes:
             mols_passed_constraints.append(mol)
             original_indices_passed_constraints.append(original_idx)

    if not mols_passed_constraints:
        print("No valid molecules passed constraints.")
        return results 

    print(f"Calculating scores for {len(mols_passed_constraints)} molecules passing constraints...")

    raw_scores_tensor = property_scorer.get_scores(
        mols_passed_constraints,
        original_indices=original_indices_passed_constraints, 
        receptor_paths=receptor_paths_scoring,         
        target_names=target_names_scoring,          
        apply_desirability=False
    )
   
    desire_scores_tensor = property_scorer.get_scores(
        mols_passed_constraints,
        original_indices=original_indices_passed_constraints, 
        receptor_paths=receptor_paths_scoring,         
        target_names=target_names_scoring,           
        apply_desirability=True
    )

    print("Populating results...")
    raw_scores_np = raw_scores_tensor.cpu().numpy()
    desire_scores_np = desire_scores_tensor.cpu().numpy()

    for i, original_idx in enumerate(tqdm(original_indices_passed_constraints, desc="Processing Scores")):
        result_row = results[original_idx] 
        if not RDKIT_AVAILABLE: 
            mol_object = None
        else:
            mol_object = mols_passed_constraints[i] 

       
        for j, p_name in enumerate(prop_names):
            result_row[f'{p_name}_Raw'] = raw_scores_np[i, j] if not np.isnan(raw_scores_np[i, j]) else 0.0
            result_row[f'{p_name}_Desire'] = desire_scores_np[i, j] if not np.isnan(desire_scores_np[i, j]) else 0.0

        desire_scores = desire_scores_np[i] 
        mpo_score = np.nan
        if mpo_combination == 'product':
            desire_scores_clamped = np.maximum(desire_scores, mpo_epsilon)
            if np.any(current_weights > 1e-9):
                log_scores = np.log(desire_scores_clamped + 1e-9) 
                weighted_log_scores = log_scores * current_weights
                mpo_score = np.exp(np.sum(weighted_log_scores))
            else: mpo_score = 0.0
        else: 
             weighted_sum = np.sum(desire_scores * current_weights)
             sum_weights = np.sum(current_weights)
             mpo_score = weighted_sum / (sum_weights + 1e-9) if sum_weights > 1e-9 else 0.0
        result_row['MPO_Score'] = mpo_score if not np.isnan(mpo_score) else 0.0

        if mol_object and isinstance(diversity_filter, ReinventDiversityFilter):
             df_status = diversity_filter.get_filter_status(mol_object, result_row['MPO_Score'])
             result_row.update({
                 'DF_Scaffold': df_status['scaffold_key'],
                 'DF_ThresholdMet': df_status['threshold_met'],
                 'DF_BucketCount_Current': df_status['current_bucket_count'],
                 'DF_PenaltyApplied': df_status['penalty_would_apply']
             })
        else: 
             df_status = {'penalty_would_apply': False} 
             result_row.update({ 
                 'DF_Scaffold': None, 'DF_ThresholdMet': False,
                 'DF_BucketCount_Current': -1, 'DF_PenaltyApplied': False
             })



        if result_row['PassedConstraints']:
            if df_status['penalty_would_apply'] and not args.disable_diversity_filter:
                result_row['Final_Reward'] = R_min
            else:
                result_row['Final_Reward'] = result_row['MPO_Score']

 

    end_time = time.time()
    print(f"--- Scoring finished in {end_time - start_time:.2f} seconds ---")
    return results

def main():
    
    parser = argparse.ArgumentParser(
        description="Modular SMILES RNN: Supervised Training, Sampling, Scoring, or Reinforcement Learning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    group_arg_map = defaultdict(list)
    def add_arg_to_group(group, *args, **kwargs):
        action = group.add_argument(*args, **kwargs)
        if action.dest:
             group_arg_map[group.title].append(action.dest)
        return action

    mode_action = parser.add_argument('--mode', type=str, required=True,
                    choices=['train_supervised', 'sample', 'reinforce', 'scoring'],
                    help='Execution Mode.')

    common_group = parser.add_argument_group('Common Settings')
    add_arg_to_group(common_group, '--seed', type=int, default=42, help='Random seed.')
    add_arg_to_group(common_group, '--num_workers', type=int, default=0, help='DataLoader workers (used in supervised mode).')

    path_group = parser.add_argument_group('File Paths')
    add_arg_to_group(path_group, '--vocab_path', type=str, default='vocabulary/vocab.json', help='Path to vocabulary file.')
    add_arg_to_group(path_group, '--results_dir', type=str, default='results', help='Base directory for saving results (plots, history, outputs).')
    add_arg_to_group(path_group, '--data_file', type=str, default='data/chembl_subset.csv', help='Input SMILES data (used in "train_supervised" mode).')
    add_arg_to_group(path_group, '--supervised_model_path', type=str, default='models/smiles_rnn_supervised.pth', help='Path to save/load base supervised model weights.')
    add_arg_to_group(path_group, '--prior_model_path', type=str, default=None, help='Path load PRIOR model weights (for Reinvent, AHC, Reinforce+KL). Defaults to supervised_model_path.')
    add_arg_to_group(path_group, '--load_agent_model_path', type=str, default=None, help='Path load AGENT model weights before RL/Sampling. Defaults to supervised_model_path.')
    add_arg_to_group(path_group, '--supervised_checkpoint_dir', type=str, default='checkpoints/supervised', help='Directory for supervised training checkpoints.')
    add_arg_to_group(path_group, '--rl_checkpoint_dir', type=str, default='checkpoints/rl_modular', help='Directory for RL training checkpoints.')
    add_arg_to_group(path_group, '--structural_alerts_path', type=str, default='config/structural_alerts.txt', help='Path to file with alert SMARTS (one per line) for constraints.')
    add_arg_to_group(path_group, '--required_substructures_path', type=str, default='config/required_substructures.txt', help='Path to file with required SMARTS (one per line) for constraints.')
    add_arg_to_group(path_group, '--input_smiles_file', type=str, default='input_smiles_to_score.smi', help='Path to input file with SMILES strings (for "scoring" mode).')
    add_arg_to_group(path_group, '--output_samples_file', type=str, default='generated_smiles.smi', help='Output filename for generated samples (for "sample" mode).')
    add_arg_to_group(path_group, '--output_scores_file', type=str, default='scoring_results.csv', help='Output filename for scoring results (for "scoring" mode).')

    model_group = parser.add_argument_group('Model Architecture (RNN)')
    add_arg_to_group(model_group, '--model', type=str, default='SmilesRNN', help='Model for prior.') 
    add_arg_to_group(model_group,'--embedding_dim', type=int, default=128, help='Character embedding dimension.')
    add_arg_to_group(model_group,'--hidden_dim', type=int, default=512, help='LSTM hidden dimension.')
    add_arg_to_group(model_group,'--num_layers', type=int, default=3, help='Number of LSTM layers.')
    add_arg_to_group(model_group,'--dropout', type=float, default=0.2, help='Dropout rate between LSTM layers/after embedding/LSTM.')

    train_group = parser.add_argument_group('Supervised Training Settings')
    add_arg_to_group(train_group,'--subset_fraction', type=float, default=1.0, help='Fraction of data for supervised training (0.0 to 1.0).')
    add_arg_to_group(train_group,'--val_size', type=float, default=0.15, help='Validation set fraction (e.g., 0.1 for 10%).')
    add_arg_to_group(train_group,'--test_size', type=float, default=0.15, help='Test set fraction (e.g., 0.1 for 10%).')
    add_arg_to_group(train_group,'--batch_size', type=int, default=512, help='Batch size for supervised training.')
    add_arg_to_group(train_group,'--learning_rate', type=float, default=1e-3, help='Learning rate for supervised optimizer (Adam).')
    add_arg_to_group(train_group,'--epochs', type=int, default=25, help='Maximum number of supervised training epochs.')
    add_arg_to_group(train_group,'--patience', type=int, default=5, help='Early stopping patience (epochs without validation improvement). Set to 0 to disable.')
    add_arg_to_group(train_group,'--resume_supervised', action='store_true', help='Resume supervised training from checkpoint.')
    add_arg_to_group(train_group,'--supervised_sample_freq', type=int, default=0, help='Frequency (epochs) to generate samples during supervised training (0=disable).')

    sample_group = parser.add_argument_group('Sampling Settings')
    add_arg_to_group(sample_group,'--num_samples', type=int, default=100, help='Number of SMILES molecules to generate.')
    add_arg_to_group(sample_group,'--sample_batch_size', type=int, default=64, help='Batch size used internally during generation for efficiency.')
    add_arg_to_group(sample_group,'--max_gen_len', type=int, default=120, help='Maximum length for generated SMILES sequences (used in sampling and RL).')
    add_arg_to_group(sample_group,'--temperature', type=float, default=1.0, help='Sampling temperature. >1 increases diversity, <1 decreases diversity.')
    add_arg_to_group(sample_group,'--sampling_mode', type=str, default='multinomial', choices=['multinomial', 'greedy', 'top_k', 'top_p'], help='Strategy for selecting the next token during sampling.')
    add_arg_to_group(sample_group,'--top_k', type=int, default=0, help='K for top-k sampling (if sampling_mode="top_k"). 0 disables it.')
    add_arg_to_group(sample_group,'--top_p', type=float, default=0.0, help='P for nucleus (top-p) sampling (if sampling_mode="top_p"). 0 disables it.')
    add_arg_to_group(sample_group,'--visualize', action='store_true', help='Visualize token probabilities during sampling (forces batch_size=1 for the first sample).')

    scoring_group = parser.add_argument_group('Scoring Mode Settings')
    add_arg_to_group(scoring_group,'--scoring_load_rl_checkpoint', type=str, default=None,
                               help='(Optional) Path to an RL checkpoint (.pth.tar) to load MPO and DF state for scoring mode.')

    rl_group = parser.add_argument_group('Reinforcement Learning Settings')
    add_arg_to_group(rl_group,'--rl_batch_size', type=int, default=64, help='Batch size for REINFORCE trajectory generation.')
    add_arg_to_group(rl_group,'--rl_epochs', type=int, default=500, help='Number of REINFORCE update steps/epochs.')
    add_arg_to_group(rl_group,'--rl_lr', type=float, default=5e-5, help='Learning rate for REINFORCE optimizer (Adam).')
    add_arg_to_group(rl_group,'--grad_clip', type=float, default=1.0, help='Gradient clipping value for RL training.')
    add_arg_to_group(rl_group,'--rl_checkpoint_freq', type=int, default=50, help='Frequency (in RL epochs) to save RL checkpoints (0 disables).')
    add_arg_to_group(rl_group,'--resume_rl_from', type=str, default=None, help='Filename of a specific checkpoint within rl_checkpoint_dir to resume RL training from.')


    loss_group = parser.add_argument_group('Loss Function Settings (Reinforce Mode)')
    add_arg_to_group(loss_group,'--loss_function', type=str, default='reinforce',
                            choices=['reinforce', 'reinvent', 'ahc', 'ppo'],
                            help="Type of loss function for RL.")
    add_arg_to_group(loss_group, '--reinforce_entropy_beta', type=float, default=0.0,
                            help='(Reinforce) Coefficient for entropy regularization bonus.')
    add_arg_to_group(loss_group, '--reinforce_kl_beta', type=float, default=0.0,
                            help='(Reinforce) Coefficient for KL divergence regularization (requires prior).')
    add_arg_to_group(loss_group, '--reinforce_use_baseline', type=bool, default=True,
                            help='(Reinforce) Whether to use a baseline subtraction.')
    add_arg_to_group(loss_group, '--reinvent_sigma', type=float, default=60.0,
                            help='(Reinvent/AHC) Reward scaling factor sigma.')
    add_arg_to_group(loss_group, '--reinvent_kl_beta', type=float, default=0.0,
                             help='(Reinvent) Optional KL divergence coefficient (requires prior).')
    add_arg_to_group(loss_group, '--reinvent_entropy_beta', type=float, default=0.0,
                             help='(Reinvent) Optional entropy regularization coefficient.')
    add_arg_to_group(loss_group, '--ahc_top_k', type=int, default=8,
                             help='(AHC) Number of top-scoring samples to use for loss calculation.')
    add_arg_to_group(loss_group, '--ahc_kl_beta', type=float, default=0.0,
                             help='(AHC) Optional KL divergence coefficient (requires prior).')
    add_arg_to_group(loss_group, '--ahc_entropy_beta', type=float, default=0.0,
                             help='(AHC) Optional entropy regularization coefficient.')
  


    reward_group = parser.add_argument_group('Reward Function Settings')
    add_arg_to_group(reward_group, '--R_min', type=float, default=1e-6, help='Minimal reward assigned for failed constraints or DF penalty.')
    add_arg_to_group(reward_group, '--target_properties', type=str, nargs='+', default=['QED', 'SA'],
                          help='List of property names for scorer. Available built-in: QED, SA, LogP, MW, MolWt, HBD, HBA, TPSA, RotB, AroRings, AliRings, Fsp3, BertzCT, isMolinDB, isScaffinDB. Use "Docking_{target_name}" for docking properties (requires --receptor_paths and --docking_target_names).')
    add_arg_to_group(reward_group, '--target_values', type=float, nargs='+', default=[0.8, 0.9],
                              help='Target values for properties (order matches target_properties). Used by MPO strategies, typically scaled [0,1].')
    add_arg_to_group(reward_group, '--desirability_configs', type=str, default='[]',
                              help='JSON string for desirability functions. Ex: \'[{"property": "LogP", "type": "gaussian", "params": {"mu": 2.5, "sigma": 1.0}}, {"property": "Docking_TargetA", "type": "sigmoid", ...}]\'')
    add_arg_to_group(reward_group, '--custom_model_configs', type=str, default='[]',
                              help='JSON string for custom prediction models. Ex: \'[{"property_name": "CustomActivity", "model_path": "models/activity.pkl", "feature_config": {}}]\'')
  
    add_arg_to_group(reward_group, '--receptor_paths', type=str, nargs='+', default=[],
                          help='List of paths to pre-prepared OpenEye receptor (.oedu) files for docking.')
    add_arg_to_group(reward_group, '--docking_target_names', type=str, nargs='+', default=[],
                          help='List of unique names corresponding to --receptor_paths. Used to identify docking properties (e.g., "Docking_{name}") in --target_properties and --desirability_configs.')
    add_arg_to_group(reward_group, '--omega_exe_path', type=str, default='oeomega',
                          help='Path to the OpenEye Omega executable (or command name if in PATH).')
    add_arg_to_group(reward_group, '--fred_exe_path', type=str, default='fred',
                           help='Path to the OpenEye FRED executable (or command name if in PATH).')
    add_arg_to_group(reward_group, '--omega_args', type=str, default='pose -flipper true',
                          help='Additional arguments string for the Omega command (e.g., "pose -flipper true -maxconfs 200"). "-in" and "-out" are handled automatically.')
    add_arg_to_group(reward_group, '--fred_args', type=str, default='-dock_resolution Standard',
                          help='Additional arguments string for the FRED command (e.g., "-dock_resolution High"). "-receptor", "-dbase", "-scorefile", "-docked_molecule_file" are handled automatically.')

    add_arg_to_group(reward_group, '--reference_db_smiles_path', type=str, default=None,
                          help='Path to file with reference canonical SMILES (one per line) for isMolinDB check.')
    add_arg_to_group(reward_group, '--reference_db_scaffold_path', type=str, default=None,
                          help='Path to file with reference canonical scaffold SMILES (one per line) for isScaffinDB check.')

    constraint_group = parser.add_argument_group('Mandatory Hard Constraint Settings')
    add_arg_to_group(constraint_group, '--enable_structural_alerts', action='store_true', help='Enable structural alert checking.')
    add_arg_to_group(constraint_group, '--enable_required_substructures', action='store_true', help='Enable required substructure checking.')
    add_arg_to_group(constraint_group, '--enable_property_limits', action='store_true', help='Enable hard property limit checking.')
    add_arg_to_group(constraint_group, '--property_limits_config', type=str, default='[{"property": "MW", "op": "<=", "value": 600}]',
                                 help='JSON string defining property limits. Ex: \'[{"property": "MW", "op": "<=", "value": 600}, {"property": "LogP", "op": ">=", "value": 0}]\'')

    df_group = parser.add_argument_group('Diversity Filter Settings')
    add_arg_to_group(df_group, '--diversity_filter_strategy', '--df_strategy', dest='diversity_filter_strategy', type=str, default='IdenticalMurcko',
                          choices=['IdenticalMurcko', 'Topological', 'ScaffoldSimilarity'], help='Scaffold definition strategy for diversity filter.')
    add_arg_to_group(df_group, '--diversity_filter_threshold', '--df_threshold', dest='diversity_filter_threshold', type=float, default=0.5, help='MPO score threshold to interact with diversity filter buckets.')
    add_arg_to_group(df_group, '--diversity_bucket_capacity', '--df_capacity', dest='diversity_bucket_capacity', type=int, default=25, help='Max count per scaffold bucket (<=0 disables filter effect).')
    add_arg_to_group(df_group, '--diversity_similarity_threshold', '--df_sim_threshold', dest='diversity_similarity_threshold', type=float, default=0.7, help='Tanimoto similarity threshold for ScaffoldSimilarity strategy.')
    add_arg_to_group(df_group, '--disable_diversity_filter', action='store_true', help='Completely disable the diversity filter. Ignores other --df_* args')


    mpo_group = parser.add_argument_group('MPO Settings')
    add_arg_to_group(mpo_group, '--mpo_strategy', type=str, default='dynamic',
                          choices=['static', 'dynamic', 'cyclical', 'randomized', 'improvement', 'variance'], 
                          help="MPO weight management strategy.")
    add_arg_to_group(mpo_group, '--mpo_combination', type=str, default='sum', choices=['sum', 'product', 'chebyshev', 'minkowski'],
                          help='Method to combine weighted scores (sum=weighted average, product=weighted product).')
    add_arg_to_group(mpo_group, '--mpo_product_epsilon', type=float, default=0.01,
                          help='Small value clamp/offset for MPO product combination to avoid log(0)/pow(0).')
    add_arg_to_group(mpo_group, '--minkowski_p', type=float, default=1,
                          help='Minkowski p: p=1: Manhattan-Distance, p=2: Euclidean-Distance, ...')

    mpo_dynamic_group = parser.add_argument_group('MPO Settings - Common Dynamic Params')
    add_arg_to_group(mpo_dynamic_group, '--mpo_update_freq', type=int, default=5,
                          help='Weight update frequency in epochs (used by Dynamic strategies: PerfGap, Improve, Variance).')
    add_arg_to_group(mpo_dynamic_group, '--mpo_ema_alpha', type=float, default=0.1,
                          help='EMA smoothing factor for property scores (used by Dynamic strategies: PerfGap, Improve, Variance).')
    add_arg_to_group(mpo_dynamic_group, '--mpo_softmax_temp', type=float, default=0.9,
                          help='Softmax temperature for weight normalization (used by Dynamic strategies: PerfGap, Improve, Variance).')
    add_arg_to_group(mpo_dynamic_group, '--mpo_min_weight', type=float, default=0.01,
                           help='Minimum raw weight value before softmax normalization (for Dynamic strategies).')

    mpo_specific_group = parser.add_argument_group('MPO Settings - Strategy Specific Params')
    add_arg_to_group(mpo_specific_group, '--mpo_static_weights', type=float, nargs='+', default=None,
                          help='(Static) Fixed weights for static MPO (order matches target_properties).')
    add_arg_to_group(mpo_specific_group, '--mpo_cycle_len', type=int, default=50,
                          help='(Cyclical) Epochs spent focusing on each property.')
    add_arg_to_group(mpo_specific_group, '--mpo_dirichlet_alpha', type=float, default=1.0,
                          help='(Randomized) Concentration parameter alpha for Dirichlet distribution.')
    add_arg_to_group(mpo_specific_group, '--mpo_random_update_freq', type=int, default=1,
                          help='(Randomized) Frequency (epochs) to sample new weights.')
    add_arg_to_group(mpo_specific_group, '--mpo_perf_gap_beta', type=float, default=0.05,
                          help='(Dynamic PerfGap) Weight update step size beta.')
    add_arg_to_group(mpo_specific_group, '--mpo_improvement_beta', type=float, default=0.05,
                          help='(Dynamic ImprovementRate) Sensitivity factor beta for improvement rate.')
    add_arg_to_group(mpo_specific_group, '--mpo_variance_window', type=int, default=10,
                          help='(Dynamic Variance) Window size (epochs) for calculating EMA variance.')
    add_arg_to_group(mpo_specific_group, '--mpo_variance_beta', type=float, default=0.05,
                          help='(Dynamic Variance) Sensitivity factor beta for EMA variance.')


 

    args, chemprop_args = parser.parse_known_args()

  


    if args.receptor_paths or args.docking_target_names: 
        if not args.receptor_paths:
            parser.error("--receptor_paths is required if --docking_target_names is provided.")
        if not args.docking_target_names:
             parser.error("--docking_target_names is required if --receptor_paths is provided.")
        if len(args.receptor_paths) != len(args.docking_target_names):
            parser.error("--receptor_paths and --docking_target_names must have the same number of elements.")

 
        docking_props_expected = {f"Docking_{name}" for name in args.docking_target_names}
        docking_props_found_in_targets = set()
        if args.target_properties: 
            docking_props_found_in_targets = {prop for prop in args.target_properties if prop.startswith("Docking_")}

        if not docking_props_expected.issubset(docking_props_found_in_targets):
             missing = docking_props_expected - docking_props_found_in_targets
             parser.error(f"Docking target names {list(args.docking_target_names)} were provided, but corresponding "
                          f"'Docking_{{name}}' entries are missing from --target_properties: {missing}")


        for r_path in args.receptor_paths:
             if not os.path.isfile(r_path):
                  print(f"Warning: Receptor file specified in --receptor_paths not found: {r_path}")


    
    mode_relevant_groups = {
        'train_supervised': [
            'Common Settings', 'File Paths', 'Model Architecture (RNN)',
            'Supervised Training Settings'
        ],
        'sample': [
            'Common Settings', 'File Paths', 'Model Architecture (RNN)',
            'Sampling Settings'
        ],
        'reinforce': [
            'Common Settings', 'File Paths', 'Model Architecture (RNN)',
            'Sampling Settings', 
            'Reinforcement Learning Settings',
            'Loss Function Settings (Reinforce Mode)',
            'Reward Function Settings', 
            'Mandatory Hard Constraint Settings',
            'Diversity Filter Settings',
            'MPO Settings',
            'MPO Settings - Common Dynamic Params',
            'MPO Settings - Strategy Specific Params'
        ],
        'scoring': [
            'Common Settings', 'File Paths', 'Model Architecture (RNN)', 
            'Scoring Mode Settings',
            'Reward Function Settings', 
            'Mandatory Hard Constraint Settings',
            'Diversity Filter Settings',
            'MPO Settings',
            'MPO Settings - Common Dynamic Params',
            'MPO Settings - Strategy Specific Params'
        ]
    }

  
    relevant_arg_names = set()
    if args.mode in mode_relevant_groups:
        for group_title in mode_relevant_groups[args.mode]:
            if group_title in group_arg_map:
                relevant_arg_names.update(group_arg_map[group_title])
    if mode_action and mode_action.dest:
       relevant_arg_names.add(mode_action.dest)
    elif 'mode' in vars(args):
        relevant_arg_names.add('mode')


    if not RDKIT_AVAILABLE and (args.mode == 'reinforce' or args.mode == 'scoring'):
         print(f"ERROR: RDKit is required for {args.mode} mode (and docking) but not found. Exiting.")
         exit(1)
    if not RDKIT_AVAILABLE and args.mode == 'sample':
        print("Warning: RDKit not available for sample mode. Validity check skipped.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dirs_to_create = {args.results_dir, args.supervised_checkpoint_dir, args.rl_checkpoint_dir}
    path_args_to_check = [
        args.vocab_path, args.supervised_model_path, args.prior_model_path,
        args.load_agent_model_path, args.structural_alerts_path, args.required_substructures_path,
        args.input_smiles_file, args.output_samples_file, args.output_scores_file
    ]
    if args.receptor_paths:
        path_args_to_check.extend(args.receptor_paths)

    for p in path_args_to_check:
        if p is not None:
            dir_path = os.path.dirname(p)
            if dir_path: 
                dirs_to_create.add(dir_path)

    for d in dirs_to_create:
        if d and d != '.': 
             os.makedirs(d, exist_ok=True)


    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("\n--- Using MPS device (Apple Silicon GPU) ---")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(args.seed) 
        print(f"\n--- Using CUDA device: {torch.cuda.get_device_name(0)} ---")
    else:
        device = torch.device("cpu")
        print("\n--- Using CPU device ---")


    print(f"\nSelected Mode: {args.mode}")
    print(f"Relevant arguments for mode '{args.mode}':")
    all_args_dict = vars(args)
    printed_args = set()
    for k in sorted(all_args_dict.keys()):
        if k in relevant_arg_names:
            print(f"  {k}: {all_args_dict[k]}")
            printed_args.add(k)
    print("-" * 30)


    if args.mode == 'train_supervised':

        print("Dispatching to supervised_trainer.train_supervised...")
        try:
            train_supervised(args, device)
        except NameError as e:
            if 'train_supervised' in str(e):
                print("ERROR: 'train_supervised' function not found. Ensure 'from supervised_trainer import train_supervised' is at the top.")
            else:
                print(f"\n!!! NameError during Supervised Training dispatch: {e} !!!"); traceback.print_exc()
            exit(1)
        except Exception as e:
            print(f"\n!!! Critical Error during Supervised Training (in supervised_trainer.py?): {e} !!!"); traceback.print_exc()
            exit(1)

    elif args.mode == 'sample':
        print("Setting up for Sampling Mode...")
        try:
            sample_molecules(args, device)
        except FileNotFoundError as e:
             print(f"ERROR: Required file not found for sampling: {e}"); exit(1)
        except ValueError as e:
             print(f"ERROR: Invalid configuration for sampling: {e}"); exit(1)
        except Exception as e:
             print(f"\n!!! Critical Error during Sampling: {e} !!!"); traceback.print_exc(); exit(1)

    elif args.mode == 'reinforce':
        print("Setting up for Reinforcement Learning Mode...")
        start_epoch = 1
        vocab = load_vocabulary(args.vocab_path)
        if vocab is None: print(f"ERROR: Vocabulary not found at {args.vocab_path}. Please run in 'train_supervised' mode first or provide a valid vocab file."); exit(1)

        try:
             agent_model = SmilesRNN(vocab.n_chars, args.embedding_dim, args.hidden_dim, args.num_layers, args.dropout).to(device)
             agent = SmilesGeneratorAgent(agent_model, vocab, device)
             print("Agent model initialized.")
        except Exception as e: print(f"Error initializing agent model: {e}"); traceback.print_exc(); exit(1)

        agent_load_path = args.load_agent_model_path if args.load_agent_model_path else args.supervised_model_path
        if not agent_load_path or not os.path.exists(agent_load_path):
             print(f"ERROR: Agent model weights path '{agent_load_path}' not found or not specified. Need a starting model for RL."); exit(1)
        print(f"Loading AGENT weights from: {agent_load_path}")
        try:
            load_checkpoint_result = load_checkpoint(agent_load_path, model=agent_model, device=device)
            if load_checkpoint_result is None: print(f"Warning: Loading agent checkpoint from {agent_load_path} returned None.")
        except Exception as e:
             print(f"ERROR loading agent weights from {agent_load_path}: {e}"); traceback.print_exc(); exit(1)


        prior_agent = None
        needs_prior = (args.loss_function in ['reinvent', 'ahc'] or
                       (args.loss_function == 'reinforce' and args.reinforce_kl_beta > 0.0) or
                       (args.loss_function == 'ppo' and False) 
                      )

        if needs_prior:
            prior_load_path = args.prior_model_path if args.prior_model_path else args.supervised_model_path
            if not prior_load_path or not os.path.exists(prior_load_path):
                 reason = f"loss function '{args.loss_function}'" 
                 print(f"ERROR: Prior model weights path '{prior_load_path}' not found or not specified (required for {reason})."); exit(1)

            print(f"Loading PRIOR weights from: {prior_load_path}")
            try:
                prior_model = SmilesRNN(vocab.n_chars, args.embedding_dim, args.hidden_dim, args.num_layers, args.dropout).to(device) 
                load_checkpoint_result_prior = load_checkpoint(prior_load_path, model=prior_model, device=device)
                if load_checkpoint_result_prior is None: print(f"Warning: Loading prior checkpoint from {prior_load_path} returned None.")
                prior_model.eval()
                prior_agent = SmilesGeneratorAgent(prior_model, vocab, device)
                print("Prior agent loaded successfully.")
            except Exception as e: print(f"Error loading prior weights from {prior_load_path}: {e}"); traceback.print_exc(); exit(1)
        elif args.prior_model_path:
             print(f"Warning: --prior_model_path '{args.prior_model_path}' was provided, but the chosen loss function/settings ('{args.loss_function}') do not require a prior. It will not be loaded.")


        try:
            desirability_configs_list = parse_json_config(args.desirability_configs, "desirability_configs")
   
            chemprop_models = None
            if 'Chemprop' in args.target_properties:
                print('PROPS BEFORE', args.target_properties)
                print('DESIR BEFORE', desirability_configs_list)
                chemprop_models = CustomChemprop(args=chemprop_args,
                                                 target_properties=args.target_properties,
                                                 desirability_configs_list=desirability_configs_list
                                                 )
  
            property_scorer = PropertyScorer(
                property_names=args.target_properties,
                desirability_configs=desirability_configs_list,
                device=device,
                omega_exe_path=args.omega_exe_path,
                fred_exe_path=args.fred_exe_path,
                omega_args_str=args.omega_args,
                fred_args_str=args.fred_args,
                reference_db_smiles_path=args.reference_db_smiles_path,
                reference_db_scaffold_path=args.reference_db_scaffold_path,
                chemprop=chemprop_models
            )
            print(f"Property scorer initialized for: {property_scorer.property_names}")


            custom_model_configs = parse_json_config(getattr(args, 'custom_model_configs', '[]'), "custom_model_configs")
            if isinstance(custom_model_configs, list):
                 for model_config in custom_model_configs:
                     prop_name = model_config.get('property_name')
                     model_path = model_config.get('model_path')
                     feature_config = model_config.get('feature_config', {})
                     if prop_name and model_path and prop_name in property_scorer.property_names:
                         print(f"Loading custom model for property '{prop_name}' from '{model_path}'...")
                         property_scorer.load_custom_model(prop_name, model_path, feature_config)
                     elif prop_name and prop_name in property_scorer.property_names:
                          print(f"Warning: Custom model config found for '{prop_name}', but 'model_path' is missing.")
            elif custom_model_configs:
                 print(f"Warning: custom_model_configs was provided but is not a valid JSON list: {args.custom_model_configs}")


            if args.disable_diversity_filter:
                 print("Diversity filter is DISABLED.")
                 diversity_filter = NoDiversityFilter()
            else:
                print(f"Diversity filter enabled: Strategy={args.diversity_filter_strategy}, Threshold={args.diversity_filter_threshold}, Capacity={args.diversity_bucket_capacity}, SimThreshold={args.diversity_similarity_threshold}")
                if not RDKIT_AVAILABLE:
                     print("ERROR: RDKit is required for Diversity Filter but not found. Exiting.")
                     exit(1)
                diversity_filter = ReinventDiversityFilter(
                    strategy=args.diversity_filter_strategy,
                    bucket_capacity=args.diversity_bucket_capacity,
                    similarity_threshold=args.diversity_similarity_threshold,
                    mpo_score_threshold=args.diversity_filter_threshold
                )


            num_props = len(args.target_properties)
            target_values_tensor = torch.tensor(args.target_values, dtype=torch.float32, device=device)
            mpo_manager: BaseMPOStrategy
            mpo_common_dynamic_kwargs = {
                'update_freq': args.mpo_update_freq,
                'ema_alpha': args.mpo_ema_alpha,
                'softmax_temp': args.mpo_softmax_temp,
                'min_weight_value': args.mpo_min_weight
            }
            print(f"Initializing MPO Manager: Strategy='{args.mpo_strategy}', Combination='{args.mpo_combination}'")

            if args.mpo_strategy == 'static':
                 if args.mpo_static_weights and len(args.mpo_static_weights) != num_props:
                      print(f"ERROR: Length mismatch for mpo_static_weights ({len(args.mpo_static_weights)}) vs target_properties ({num_props})."); exit(1)
                 print(f"  Static Weights: {args.mpo_static_weights}")
                 mpo_manager = StaticWeightMPO(num_props, target_values_tensor, device, initial_weights=args.mpo_static_weights)
            elif args.mpo_strategy == 'cyclical':
                 print(f"  Cycle Length: {args.mpo_cycle_len}")
                 mpo_manager = CyclicalMPO(num_props, target_values_tensor, device, cycle_len=args.mpo_cycle_len)
            elif args.mpo_strategy == 'randomized':
                 print(f"  Dirichlet Alpha: {args.mpo_dirichlet_alpha}, Update Freq: {args.mpo_random_update_freq}")
                 mpo_manager = RandomizedWeightMPO(num_props, target_values_tensor, device, dirichlet_alpha=args.mpo_dirichlet_alpha, update_frequency=args.mpo_random_update_freq)
            elif args.mpo_strategy == 'dynamic': # Performance Gap
                 print(f"  Dynamic (PerfGap) Beta: {args.mpo_perf_gap_beta}, Update Freq: {args.mpo_update_freq}, EMA Alpha: {args.mpo_ema_alpha}, Softmax Temp: {args.mpo_softmax_temp}")
                 mpo_manager = DynamicWeightMPO(num_props, target_values_tensor, device, beta=args.mpo_perf_gap_beta, **mpo_common_dynamic_kwargs)
            elif args.mpo_strategy == 'improvement':
                 print(f"  Dynamic (Improvement) Beta: {args.mpo_improvement_beta}, Update Freq: {args.mpo_update_freq}, EMA Alpha: {args.mpo_ema_alpha}, Softmax Temp: {args.mpo_softmax_temp}")
                 mpo_manager = DynamicImprovementRateMPO(num_props, target_values_tensor, device, beta=args.mpo_improvement_beta, **mpo_common_dynamic_kwargs)
            elif args.mpo_strategy == 'variance':
                 print(f"  Dynamic (Variance) Beta: {args.mpo_variance_beta}, Window: {args.mpo_variance_window}, Update Freq: {args.mpo_update_freq}, EMA Alpha: {args.mpo_ema_alpha}, Softmax Temp: {args.mpo_softmax_temp}")
                 mpo_manager = DynamicVarianceMPO(num_props, target_values_tensor, device, beta=args.mpo_variance_beta, variance_window=args.mpo_variance_window, **mpo_common_dynamic_kwargs)
            elif args.mpo_strategy == 'pareto':
                 print(f"  Dynamic (Variance) Beta: {args.mpo_variance_beta}, Window: {args.mpo_variance_window}, Update Freq: {args.mpo_update_freq}, EMA Alpha: {args.mpo_ema_alpha}, Softmax Temp: {args.mpo_softmax_temp}")
                 mpo_manager = Pareto(num_props, target_values_tensor, device, beta=args.mpo_variance_beta, variance_window=args.mpo_variance_window, **mpo_common_dynamic_kwargs)
            else: raise ValueError(f"Unknown MPO strategy: {args.mpo_strategy}")


            loss_calculator: BaseLossCalculator
            print(f"Initializing Loss Calculator: Type='{args.loss_function}'")
 
            if args.loss_function == 'reinforce':
                print(f"  Reinforce Params: UseBaseline={args.reinforce_use_baseline}, EntropyBeta={args.reinforce_entropy_beta}, KLBeta={args.reinforce_kl_beta}")
                loss_calculator = ReinforceLoss(args, prior_agent=prior_agent)
            elif args.loss_function == 'reinvent':
                 if prior_agent is None and (args.reinvent_kl_beta > 0 or args.loss_function == 'reinvent'): 
                       raise ValueError("ReinventLoss requires a prior agent (--prior_model_path), but it was not loaded.")
                 print(f"  Reinvent Params: Sigma={args.reinvent_sigma}, KLBeta={args.reinvent_kl_beta}, EntropyBeta={args.reinvent_entropy_beta}")
                 loss_calculator = ReinventLoss(args, prior_agent)
            elif args.loss_function == 'ahc':
                 if prior_agent is None and (args.ahc_kl_beta > 0 or args.loss_function == 'ahc'): 
                      raise ValueError("AHCLoss requires a prior agent (--prior_model_path), but it was not loaded.")
                 print(f"  AHC Params: Sigma={args.reinvent_sigma}, TopK={args.ahc_top_k}, KLBeta={args.ahc_kl_beta}, EntropyBeta={args.ahc_entropy_beta}")
                 loss_calculator = AugmentedHillClimbLoss(args, prior_agent)
            elif args.loss_function == 'ppo':
                 print(f"  PPO Params: (Add relevant PPO params here from args)")
                 loss_calculator = PPOLoss(args, agent=agent, prior_agent=prior_agent) 
            else: raise ValueError(f"Unknown loss function: {args.loss_function}")

        except Exception as e: print(f"Error during RL component initialization: {e}"); traceback.print_exc(); exit(1)

        optimizer = optim.Adam(agent.model.parameters(), lr=args.rl_lr)
        print(f"Optimizer initialized: Adam, LR={args.rl_lr}")

        loaded_checkpoint_data = None
        if args.resume_rl_from:
             resume_path = os.path.join(args.rl_checkpoint_dir, args.resume_rl_from)
             if os.path.isfile(resume_path):
                 print("-" * 30)
                 print(f"Attempting to resume RL training from: {resume_path}")
                 loaded_checkpoint_data = load_checkpoint(
                     resume_path, model=agent_model, optimizer=optimizer,
                     mpo_manager=mpo_manager, diversity_filter=diversity_filter, device=device
                    )
                 if loaded_checkpoint_data:
                     start_epoch = loaded_checkpoint_data.get('epoch', 0) + 1
                     if 'random_rng_state' in loaded_checkpoint_data: random.setstate(loaded_checkpoint_data['random_rng_state'])
                     if 'np_rng_state' in loaded_checkpoint_data: np.random.set_state(loaded_checkpoint_data['np_rng_state'])
                     if 'torch_rng_state' in loaded_checkpoint_data: torch.set_rng_state(loaded_checkpoint_data['torch_rng_state'])
                     if device == torch.device('cuda') and 'torch_cuda_rng_state' in loaded_checkpoint_data: torch.cuda.set_rng_state_all(loaded_checkpoint_data['torch_cuda_rng_state'])
                     print(f"Resuming RL training from epoch {start_epoch}.")
                 else:
                     print(f"Warning: Failed to load RL checkpoint {resume_path}. Starting RL from scratch (epoch 1).")
                     start_epoch = 1
                 print("-" * 30)
             else:
                  print(f"Warning: Resume checkpoint specified (--resume_rl_from='{args.resume_rl_from}') but file not found at '{resume_path}'. Starting RL from scratch.")
                  start_epoch = 1

        try:
            trainer = ReinforceTrainer(
                agent=agent, prior_agent=prior_agent, property_scorer=property_scorer,
                diversity_filter=diversity_filter, mpo_manager=mpo_manager,
                loss_calculator=loss_calculator, optimizer=optimizer, args=args, device=device,
                receptor_paths=args.receptor_paths, 
                docking_target_names=args.docking_target_names 
            )
            print("ReinforceTrainer initialized.")
        except Exception as e: print(f"Error initializing ReinforceTrainer: {e}"); traceback.print_exc(); exit(1)

        if loaded_checkpoint_data and 'rl_history' in loaded_checkpoint_data:
            try:
                 if isinstance(loaded_checkpoint_data['rl_history'], dict):
                      trainer.rl_history = loaded_checkpoint_data['rl_history']
                      print("Loaded RL history from checkpoint.")
                 else: print("Warning: Checkpoint 'rl_history' is not a dictionary. Ignoring.")
            except Exception as e: print(f"Warning: Error loading history from checkpoint: {e}")
        else:
             print("Starting with fresh RL history.")


        print(f"\n--- Starting Reinforcement Learning Training for {args.rl_epochs} epochs ---")
        try:
            trainer.train(start_epoch=start_epoch, num_epochs=args.rl_epochs)
            print("\n--- Reinforcement Learning Training Completed ---")
        except KeyboardInterrupt:
             print("\n--- Training interrupted by user (KeyboardInterrupt) ---")
             print("Attempting to save final state...")
             if hasattr(trainer, '_save_final_results') and callable(trainer._save_final_results):
                  trainer._save_final_results()
                  print("Final state saved (if possible).")
             else:
                 print("Could not save final state.")
             print("Exiting.")
             exit(0)
        except Exception as e:
            print(f"\n!!! Critical Error during RL training: {e} !!!"); traceback.print_exc();
            print("Attempting to save final state...")
            if hasattr(trainer, '_save_final_results') and callable(trainer._save_final_results):
                 trainer._save_final_results()
                 print("Final state saved (if possible).")
            else:
                 print("Could not find or call trainer._save_final_results().")
            print("Exiting due to critical error.")
            exit(1)


    elif args.mode == 'scoring':
        print("Setting up for Scoring Mode...")
        try:
            desirability_configs_list = parse_json_config(args.desirability_configs, "desirability_configs")
            property_scorer = PropertyScorer(
                property_names=args.target_properties,
                desirability_configs=desirability_configs_list,
                device=device,
                omega_exe_path=args.omega_exe_path,
                fred_exe_path=args.fred_exe_path,
                omega_args_str=args.omega_args,
                fred_args_str=args.fred_args,
                reference_db_smiles_path=args.reference_db_smiles_path,
                reference_db_scaffold_path=args.reference_db_scaffold_path,
                chemprop_args=chemprop_args
            )
            print(f"Property scorer initialized for: {property_scorer.property_names}")


            custom_model_configs = parse_json_config(getattr(args, 'custom_model_configs', '[]'), "custom_model_configs")
            if isinstance(custom_model_configs, list):
                 for model_config in custom_model_configs:
                     prop_name = model_config.get('property_name'); model_path = model_config.get('model_path'); feature_config = model_config.get('feature_config', {})
                     if prop_name and model_path and prop_name in property_scorer.property_names:
                          print(f"Loading custom model for property '{prop_name}' from '{model_path}'...")
                          property_scorer.load_custom_model(prop_name, model_path, feature_config)
                     elif prop_name and prop_name in property_scorer.property_names:
                          print(f"Warning: Custom model config found for '{prop_name}', but 'model_path' is missing.")

            if args.disable_diversity_filter:
                 print("Diversity filter is DISABLED for scoring.")
                 diversity_filter = NoDiversityFilter()
            else:
                print(f"Diversity filter enabled for scoring: Strategy={args.diversity_filter_strategy}, Threshold={args.diversity_filter_threshold}, Capacity={args.diversity_bucket_capacity}, SimThreshold={args.diversity_similarity_threshold}")
                if not RDKIT_AVAILABLE:
                     print("Warning: RDKit not available, cannot use structural Diversity Filters.")
                     diversity_filter = NoDiversityFilter() 
                else:
                     diversity_filter = ReinventDiversityFilter(
                         strategy=args.diversity_filter_strategy,
                         bucket_capacity=args.diversity_bucket_capacity,
                         similarity_threshold=args.diversity_similarity_threshold,
                         mpo_score_threshold=args.diversity_filter_threshold
                     )

            num_props = len(args.target_properties)
            target_values_tensor = torch.tensor(args.target_values, dtype=torch.float32, device=device)
            mpo_manager: BaseMPOStrategy
            mpo_common_dynamic_kwargs = {
                'update_freq': args.mpo_update_freq, 'ema_alpha': args.mpo_ema_alpha,
                'softmax_temp': args.mpo_softmax_temp, 'min_weight_value': args.mpo_min_weight
            }
            print(f"Initializing MPO Manager for scoring: Strategy='{args.mpo_strategy}', Combination='{args.mpo_combination}'")

            if args.mpo_strategy == 'static':
                 if args.mpo_static_weights and len(args.mpo_static_weights) != num_props: print(f"ERROR: Length mismatch for mpo_static_weights."); exit(1)
                 print(f"  Static Weights: {args.mpo_static_weights}")
                 mpo_manager = StaticWeightMPO(num_props, target_values_tensor, device, initial_weights=args.mpo_static_weights)
            elif args.mpo_strategy == 'cyclical':
                 print(f"  Cycle Length: {args.mpo_cycle_len}")
                 mpo_manager = CyclicalMPO(num_props, target_values_tensor, device, cycle_len=args.mpo_cycle_len)
            elif args.mpo_strategy == 'randomized':
                 print(f"  Dirichlet Alpha: {args.mpo_dirichlet_alpha}, Update Freq: {args.mpo_random_update_freq}")
                 mpo_manager = RandomizedWeightMPO(num_props, target_values_tensor, device, dirichlet_alpha=args.mpo_dirichlet_alpha, update_frequency=args.mpo_random_update_freq)
            elif args.mpo_strategy == 'dynamic':
                 print(f"  Dynamic (PerfGap) Beta: {args.mpo_perf_gap_beta}")
                 mpo_manager = DynamicWeightMPO(num_props, target_values_tensor, device, beta=args.mpo_perf_gap_beta, **mpo_common_dynamic_kwargs)
            elif args.mpo_strategy == 'improvement':
                 print(f"  Dynamic (Improvement) Beta: {args.mpo_improvement_beta}")
                 mpo_manager = DynamicImprovementRateMPO(num_props, target_values_tensor, device, beta=args.mpo_improvement_beta, **mpo_common_dynamic_kwargs)
            elif args.mpo_strategy == 'variance':
                 print(f"  Dynamic (Variance) Beta: {args.mpo_variance_beta}, Window: {args.mpo_variance_window}")
                 mpo_manager = DynamicVarianceMPO(num_props, target_values_tensor, device, beta=args.mpo_variance_beta, variance_window=args.mpo_variance_window, **mpo_common_dynamic_kwargs)
            else: raise ValueError(f"Unknown MPO strategy: {args.mpo_strategy}")

        except Exception as e: print(f"Error during Scoring component initialization: {e}"); traceback.print_exc(); exit(1)

        if args.scoring_load_rl_checkpoint:
              checkpoint_path = args.scoring_load_rl_checkpoint
              if os.path.isfile(checkpoint_path):
                  print("-" * 30); print(f"Loading state for MPO Manager and Diversity Filter from: {checkpoint_path}")
                  loaded_checkpoint_data = load_checkpoint(checkpoint_path, model=None, optimizer=None, mpo_manager=mpo_manager, diversity_filter=diversity_filter, device=device)
                  if not loaded_checkpoint_data:
                       print(f"Warning: Failed to load MPO/DF state from checkpoint {checkpoint_path}. Using initial component states.")
                  else:
                       print("MPO Manager and Diversity Filter state loaded successfully.")
                  print("-" * 30)
              else:
                   print(f"Warning: Scoring checkpoint specified (--scoring_load_rl_checkpoint='{checkpoint_path}') but file not found. Using initial component states.")
        else:
              print("Scoring with initial MPO Manager and Diversity Filter states (no checkpoint specified).")

        input_smiles = load_smiles_data(args.input_smiles_file)
        if input_smiles is None or not input_smiles:
             print(f"ERROR: Could not load any valid SMILES from '{args.input_smiles_file}'. Exiting scoring mode."); exit(1)
        print(f"Loaded {len(input_smiles)} SMILES to score from {args.input_smiles_file}.")

        if args.enable_structural_alerts and (not args.structural_alerts_path or not os.path.isfile(args.structural_alerts_path)):
             print(f"ERROR: Structural alerts enabled, but path '{args.structural_alerts_path}' is invalid."); exit(1)
        if args.enable_required_substructures and (not args.required_substructures_path or not os.path.isfile(args.required_substructures_path)):
             print(f"ERROR: Required substructures enabled, but path '{args.required_substructures_path}' is invalid."); exit(1)

        constraint_config = {
            'enable_structural_alerts': getattr(args, 'enable_structural_alerts', False),
            'structural_alerts_path': getattr(args, 'structural_alerts_path', None),
            'enable_required_substructures': getattr(args, 'enable_required_substructures', False),
            'required_substructures_path': getattr(args, 'required_substructures_path', None),
            'enable_property_limits': getattr(args, 'enable_property_limits', False),
            'property_limits_config_list': parse_json_config(getattr(args, 'property_limits_config', '[]'), "property_limits_config")
         }
        print(f"Constraint Config for Scoring: {constraint_config}")


        scoring_results = evaluate_smiles_scoring(
            smiles_list=input_smiles, property_scorer=property_scorer, mpo_manager=mpo_manager,
            diversity_filter=diversity_filter, constraint_config=constraint_config, args=args, device=device
        )

        if scoring_results:
            output_path = args.output_scores_file
            headers = list(scoring_results[0].keys()) if scoring_results else []
            if headers:
                print(f"Saving {len(scoring_results)} scoring results to: {output_path}")
                try:
                    output_dir = os.path.dirname(output_path)
                    if output_dir: os.makedirs(output_dir, exist_ok=True)
                    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=headers)
                        writer.writeheader()
                        writer.writerows(scoring_results)
                    print(f"Scoring results successfully saved.")
                except Exception as e: print(f"Error saving scoring results to {output_path}: {e}")
            else:
                 print("Warning: No headers generated from scoring results.")
        else: print("No scoring results were generated to save.")
        print("\n--- Scoring Mode Completed ---")


    else:
        print(f"Error: Unknown mode '{args.mode}' specified.")
        exit(1)

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("Executing main script...")
    main()
    print("Script finished.")
