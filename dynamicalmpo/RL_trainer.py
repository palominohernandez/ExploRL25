# RL_trainer.py
import argparse
import json
import math 
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time
import torch
import torch.nn.functional as F
import traceback 
import random

from collections import deque
from pygmo import hypervolume # TODO move with HV somewhere else
from torch.optim import Optimizer 
from typing import Dict, Any, List, Tuple, Optional

from agent import SmilesGeneratorAgent
from constraints import check_mandatory_constraints
from filters import ReinventDiversityFilter, NoDiversityFilter
from losses import BaseLossCalculator, ReinforceLoss, ReinventLoss 
from mpo import BaseMPOStrategy
from scalarization import ScalarizationFactory, scalarize_reward
from scoring import PropertyScorer
from utils import save_checkpoint, parse_json_config, SOS_token, PAD_token

try:
    from rdkit import Chem
    from rdkit.Chem import Draw 
    RDKIT_AVAILABLE_TRAINER = True
except ImportError:
    RDKIT_AVAILABLE_TRAINER = False
    print("Warning (RL_trainer.py): RDKit not found. Molecule processing/visualization will be limited.")

class HyperVolume: 
    def __init__(self,
                 target_values: torch.Tensor,
                 convergence_window_size: int = 10):
        self.reference_point = (1 - target_values).detach().cpu().numpy()
        self.hv_history = []
        self.diff_history = []
        self.convergence_window = deque(maxlen=convergence_window_size)
        self.best_hv = {
                        'Hypervolume' : None,
                        'Epoch' : None
                        }

    def calc_hv(self, avg_rewards, epoch):
        avg_rewards = (avg_rewards*-1).unsqueeze(0)
        avg_rewards = avg_rewards.detach().cpu().numpy()
        hv = hypervolume(avg_rewards)
        vol = hv.compute(self.reference_point)
        self.hv_history.append(vol)
        self.convergence_window.append(vol)
        if vol > max(self.hv_history):
            self.best_hv['Hypervolume'] = vol
            self.best_hv['Epoch'] = epoch 
        return vol
    def check_convergence(self, current_hv, epoch, convergence_criteria: float = 0.01):
        if epoch < 2: 
            return -1
        else:
            mean_hv = sum(self.convergence_window) / len(self.convergence_window)
            diff = abs(current_hv - mean_hv) 
            self.diff_history.append(diff)
            return diff < convergence_criteria # TODO alternatively conv criteria => diff/mean

    
def smooth_data(data, span=20): 
    """Calculates Exponential Moving Average. Handles lists and numpy arrays."""
    data_size = 0
    if isinstance(data, (list, np.ndarray)):
         try: data_size = data.size if isinstance(data, np.ndarray) else len(data)
         except Exception: data_size = 0
    if data_size < 2: return data.tolist() if isinstance(data, np.ndarray) else data
    try:
        series = pd.Series(data)
        smoothed = series.ewm(span=span, adjust=False, min_periods=1).mean()
        return smoothed.tolist()
    except Exception as e:
        print(f"Warning: Error during EMA smoothing: {e}. Returning original data as list.")
        if isinstance(data, np.ndarray): return data.tolist()
        elif isinstance(data, list): return data
        else: return []

def calculate_shannon_entropy(counts: List[int]) -> Tuple[float, float]: 
    """Calculates raw Shannon Entropy (H) and Scaled Entropy (H_norm)."""
    total_count = sum(counts)
    if total_count == 0: return 0.0, 0.0
    num_unique_scaffolds = len(counts)
    if num_unique_scaffolds <= 1: return 0.0, 0.0
    entropy = 0.0
    probabilities = [c / total_count for c in counts]
    for p in probabilities:
        if p > 0: entropy -= p * math.log2(p)
    max_entropy = math.log2(num_unique_scaffolds)
    scaled_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    return entropy, scaled_entropy


class ReinforceTrainer:
    """Orchestrates the RL training process using modular components."""
    def __init__(self, *, 
                 agent: SmilesGeneratorAgent,
                 prior_agent: SmilesGeneratorAgent | None,
                 property_scorer: PropertyScorer,
                 diversity_filter: ReinventDiversityFilter | NoDiversityFilter,
                 mpo_manager: BaseMPOStrategy,
                 loss_calculator: BaseLossCalculator,
                 optimizer: Optimizer,
                 args: argparse.Namespace,
                 device,
                 receptor_paths: Optional[List[str]] = None,
                 docking_target_names: Optional[List[str]] = None
                 ):
        self.agent = agent
        self.prior_agent = prior_agent
        self.property_scorer = property_scorer
        self.diversity_filter = diversity_filter
        self.mpo_manager = mpo_manager
        self.loss_calculator = loss_calculator
        self.optimizer = optimizer
        self.args = args
        self.device = device
        self.constraint_config = self._compile_constraint_config(args)
        self.desirability_configs = parse_json_config(getattr(args, 'desirability_configs', '[]'), "desirability_configs")
        self.mpo_combination = getattr(args, 'mpo_combination', 'sum')
        self.mpo_epsilon = getattr(args, 'mpo_product_epsilon', 0.01)
        self.R_min = getattr(args, 'R_min', 1e-6)
        self.target_values = torch.tensor(getattr(args, 'target_values', torch.ones(property_scorer.num_properties)), device='cuda')
        self.p = getattr(args, 'minkowski_p', 1)
        self.receptor_paths = receptor_paths if receptor_paths else []
        self.docking_target_names = docking_target_names if docking_target_names else []
        self.rl_history = self._init_history()

    def _compile_constraint_config(self, args) -> Dict[str, Any]:
        """Helper to gather constraint-related args into a config dict."""
        config = {
            'enable_structural_alerts': getattr(args, 'enable_structural_alerts', False),
            'structural_alerts_path': getattr(args, 'structural_alerts_path', None),
            'enable_required_substructures': getattr(args, 'enable_required_substructures', False),
            'required_substructures_path': getattr(args, 'required_substructures_path', None),
            'enable_property_limits': getattr(args, 'enable_property_limits', False),
            'property_limits_config_list': parse_json_config(getattr(args, 'property_limits_config', '[]'), "property_limits_config")
        }
        if config['enable_structural_alerts'] and (not config['structural_alerts_path'] or not os.path.isfile(config['structural_alerts_path'])):
             print(f"Warning: Structural alerts enabled, but path '{config['structural_alerts_path']}' is invalid.")
        if config['enable_required_substructures'] and (not config['required_substructures_path'] or not os.path.isfile(config['required_substructures_path'])):
             print(f"Warning: Required substructures enabled, but path '{config['required_substructures_path']}' is invalid.")
        return config

    def _init_history(self):
        """Initializes the dictionary to store training history."""
        history = {'epoch': [], 'avg_reward_final': [], 'avg_score_mpo': [],
                   'loss': [], 'constraint_pass_rate': [], 'df_penalty_rate': [],
                   'weights': [], 'avg_scores_raw': [], 'avg_scores_desire': [],
                   'hypervolumes': [], 'convergence_crit': []}
        if isinstance(self.loss_calculator, ReinforceLoss):
            history.update({'policy_loss': [], 'entropy': [], 'reward_baseline': []})
        elif isinstance(self.loss_calculator, ReinventLoss):
            history.update({'avg_prior_log_prob': [], 'avg_agent_log_prob': [], 'entropy': []})
        if hasattr(self.mpo_manager, 'property_score_avg'):
            history['avg_scores_ema'] = []
        return history

    def _log_epoch(self, epoch, start_time, loss_info, stats):
        """Logs metrics for the completed epoch."""
        epoch_duration = time.time() - start_time
        current_weights = self.mpo_manager.get_weights(epoch).cpu().numpy()
        self.rl_history['epoch'].append(epoch)
        self.rl_history['avg_reward_final'].append(stats.get('avg_reward_final', np.nan))
        self.rl_history['avg_score_mpo'].append(stats.get('avg_score_mpo_passed', np.nan)) 
        self.rl_history['loss'].append(loss_info.get('loss', np.nan))
        self.rl_history['constraint_pass_rate'].append(stats.get('constraint_pass_rate', np.nan))
        self.rl_history['df_penalty_rate'].append(stats.get('df_penalty_rate', np.nan)) 
        self.rl_history['weights'].append(current_weights.tolist())
        self.rl_history['avg_scores_raw'].append(stats.get('avg_scores_raw_passed', [np.nan]*self.mpo_manager.num_properties))
        self.rl_history['avg_scores_desire'].append(stats.get('avg_scores_desire_passed', [np.nan]*self.mpo_manager.num_properties))

        self.rl_history['hypervolumes'].append(stats.get('hypervolumes', np.nan))
        self.rl_history['convergence_crit'].append(stats.get('convergence_crit', np.nan))

        if isinstance(self.loss_calculator, ReinforceLoss):
             self.rl_history['policy_loss'].append(loss_info.get('policy_loss', np.nan))
             self.rl_history['entropy'].append(loss_info.get('entropy', np.nan))
             self.rl_history['reward_baseline'].append(loss_info.get('reward_baseline', np.nan))
        elif isinstance(self.loss_calculator, ReinventLoss):
             self.rl_history['avg_prior_log_prob'].append(loss_info.get('avg_prior_log_prob', np.nan))
             self.rl_history['avg_agent_log_prob'].append(loss_info.get('avg_agent_log_prob', np.nan))
             self.rl_history['entropy'].append(loss_info.get('entropy', np.nan)) 

        if 'avg_scores_ema' in self.rl_history:
             if hasattr(self.mpo_manager, 'property_score_avg'):
                  try:
                     ema_scores = self.mpo_manager.property_score_avg.cpu().numpy().tolist()
                     self.rl_history['avg_scores_ema'].append(ema_scores)
                  except Exception as e:
                      print(f"Warning: Could not log MPO EMA scores: {e}")
                      self.rl_history['avg_scores_ema'].append([np.nan] * self.mpo_manager.num_properties)
             else:
                  self.rl_history['avg_scores_ema'].append([np.nan] * self.mpo_manager.num_properties)

        log_msg = (
            f"RL Ep {epoch}/{self.args.rl_epochs} | T: {epoch_duration:.2f}s | "
            f"Rwd(Final): {stats.get('avg_reward_final', 0):.3f} | MPO(Pass): {stats.get('avg_score_mpo_passed', 0):.3f} | "
            f"Loss: {loss_info.get('loss', 0):.4f} | "
            f"CnstPass: {stats.get('constraint_pass_rate', 0):.2f} | " 
            f"DFPen: {stats.get('df_penalty_rate', 0):.2f} | " 
        )

        if isinstance(self.loss_calculator, ReinforceLoss):
             log_msg += f"PolLoss: {loss_info.get('policy_loss', 0):.4f} | Ent: {loss_info.get('entropy', 0):.3f} | Base: {loss_info.get('reward_baseline', 0):.3f} | "
        elif isinstance(self.loss_calculator, ReinventLoss):
             log_msg += f"PriorLL: {loss_info.get('avg_prior_log_prob', 0):.3f} | AgentLL: {loss_info.get('avg_agent_log_prob', 0):.3f} | Ent: {loss_info.get('entropy', 0):.3f} | "

        log_msg += f"W: {np.round(current_weights, 3)}"
        print(log_msg)

    def _checkpoint(self, epoch):
        """Saves a checkpoint periodically."""
        checkpoint_freq = getattr(self.args, 'rl_checkpoint_freq', 0)
        if checkpoint_freq > 0 and epoch > 0 and epoch % checkpoint_freq == 0:
            checkpoint_dir = getattr(self.args, 'rl_checkpoint_dir', 'checkpoints/rl_modular')
            if not os.path.isdir(checkpoint_dir):
                try: os.makedirs(checkpoint_dir, exist_ok=True)
                except OSError as e: print(f"Warning: Could not create checkpoint directory {checkpoint_dir}: {e}"); return

            checkpoint_path = os.path.join(checkpoint_dir, f"rl_checkpoint_epoch_{epoch}.pth.tar")

            state = {
                'epoch': epoch,
                'model_state_dict': self.agent.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'mpo_manager_state': self.mpo_manager.get_state(),
                'diversity_filter_state': self.diversity_filter.get_state(),
                'rl_history': self.rl_history,
                'vocab_size': self.agent.vocab.n_chars,
                'args': vars(self.args),
                'random_rng_state': random.getstate(),
                'np_rng_state': np.random.get_state(),
                'torch_rng_state': torch.get_rng_state(),
                'torch_cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            }
            print(f"\n--- Saving Checkpoint Epoch {epoch} ---") 
            save_checkpoint(state, filename=checkpoint_path)
            print(f"Checkpoint saved to: {checkpoint_path}")


    def train(self, start_epoch, num_epochs):
        """Runs the main RL training loop."""
        print(f"\n--- Starting REINFORCE Training ---")
        print(f"   Loss Function: {type(self.loss_calculator).__name__}")
        print(f"   MPO Strategy: {type(self.mpo_manager).__name__}")
        print(f"   Diversity Filter: {type(self.diversity_filter).__name__} (Strategy: getattr(self.diversity_filter, 'strategy', 'N/A'))")
        print(f"   Total Epochs: {num_epochs}")
        print(f"   Starting Epoch: {start_epoch}")
        print(f"   Minimal Reward (R_min): {self.R_min}")
        print(f"   MPO Combination: {self.mpo_combination}")
        print(f"   Constraint Config: {self.constraint_config}")
        print(f"   Desirability Config: {self.desirability_configs}")
        if self.receptor_paths:
            print(f"   Docking Targets ({len(self.receptor_paths)}): {self.docking_target_names}")
        else:
            print("   Docking Targets: None")
        print("-" * 30)
        total_start_time = time.time()

        hypervolume = HyperVolume(target_values=torch.tensor(self.args.target_values, dtype=torch.float32, device=self.device),
                                  convergence_window_size=10)

        for epoch in range(start_epoch, num_epochs + 1):
            epoch_start_time = time.time()
            self.agent.model.train() 
            batch_stats = {} 

            if hasattr(self.property_scorer, '_batch_docking_results'):
                self.property_scorer._batch_docking_results = None
                self.property_scorer._batch_indices_processed_docking = None

            try:
                smiles, agent_log_probs, agent_probs, agent_actions, lengths = \
                    self.agent.generate_trajectories(self.args.rl_batch_size, self.args.max_gen_len)
                batch_size = len(smiles)


                passed_constraints_mask, valid_mols_list_from_smiles = check_mandatory_constraints(
                    smiles, self.constraint_config, self.device
                )
                passed_constraint_indices_tensor = passed_constraints_mask.nonzero(as_tuple=True)[0]
                num_passed_constraints = len(passed_constraint_indices_tensor)
                batch_stats['constraint_pass_rate'] = num_passed_constraints / batch_size if batch_size > 0 else 0.0

                final_rewards = torch.full((batch_size,), self.R_min, dtype=torch.float32, device=self.device)

                mols_passed = []
                original_indices_passed = [] 

                if num_passed_constraints > 0 and RDKIT_AVAILABLE_TRAINER:
                    for idx_tensor in passed_constraint_indices_tensor:
                         original_idx = idx_tensor.item() 
                         mol = valid_mols_list_from_smiles[original_idx]
                         if mol: 
                             mols_passed.append(mol)
                             original_indices_passed.append(original_idx) 
                         else:
                              print(f"Warning: SMILES at original index {original_idx} passed constraints but Mol is None.")

                    if not mols_passed:
                         print("Warning: Constraints passed but no valid Mol objects were collected.")
                         num_passed_constraints = 0 

                    else:
                        raw_scores_passed = self.property_scorer.get_scores(
                            mols_list=mols_passed,
                            apply_desirability=False,
                            original_indices=original_indices_passed,
                            receptor_paths=self.receptor_paths,
                            target_names=self.docking_target_names
                        )
                        batch_stats['avg_scores_raw_passed'] = raw_scores_passed.mean(dim=0).cpu().numpy().tolist()

                        scores_for_mpo = self.property_scorer.get_scores(
                            mols_list=mols_passed,
                            apply_desirability=True, 
                            original_indices=original_indices_passed,
                            receptor_paths=self.receptor_paths,
                            target_names=self.docking_target_names
                        )
                        batch_stats['avg_scores_desire_passed'] = scores_for_mpo.mean(dim=0).cpu().numpy().tolist()
                        current_weights = self.mpo_manager.get_weights(epoch)
                        scalarizer = ScalarizationFactory.get_scalarization_method(self.mpo_combination,
                                                                                    epsilon=self.mpo_epsilon) 
                        mpo_scores_passed = scalarize_reward(scalarizer,
                                                             scores_for_mpo,
                                                             current_weights,
                                                             targets=self.target_values,
                                                             p=self.p)

                        final_rewards[torch.tensor(original_indices_passed, device=self.device)] = mpo_scores_passed
                        batch_stats['avg_score_mpo_passed'] = mpo_scores_passed.mean().item()

                        num_penalized = 0
                        if not isinstance(self.diversity_filter, NoDiversityFilter) and not self.args.disable_diversity_filter:
                            df_penalty_mask = self.diversity_filter.apply_filter(mols_passed, mpo_scores_passed) 
                            original_indices_passed_tensor = torch.tensor(original_indices_passed, device=self.device)
                            penalized_original_indices = original_indices_passed_tensor[df_penalty_mask]
                            if penalized_original_indices.numel() > 0:
                                final_rewards[penalized_original_indices] = self.R_min 
                            num_penalized = penalized_original_indices.numel()
                        batch_stats['df_penalty_rate'] = num_penalized / num_passed_constraints if num_passed_constraints > 0 else 0.0

                        self.mpo_manager.update_state(raw_scores_passed, scores_for_mpo, epoch)

                elif not RDKIT_AVAILABLE_TRAINER and num_passed_constraints > 0:
                     print("Warning: RDKit not available, cannot perform scoring or filtering. Rewards remain R_min.")
                batch_stats['avg_reward_final'] = final_rewards.mean().item()

        
                batch_stats['hypervolumes'] = hypervolume.calc_hv(scores_for_mpo.mean(dim=0), epoch)
                batch_stats['convergence_crit'] = hypervolume.check_convergence(hypervolume.calc_hv(scores_for_mpo.mean(dim=0), epoch), epoch)
                
                max_len_logprob = agent_log_probs.size(1)
                mask = torch.arange(max_len_logprob, device=self.device)[None, :] < lengths[:, None]
                mask = mask.float()

                loss_kwargs = {}
                if isinstance(self.loss_calculator, ReinventLoss) or \
                   (isinstance(self.loss_calculator, BaseLossCalculator) and getattr(self.loss_calculator, 'requires_sequences', False)): 
                     batch_size_local = agent_actions.shape[0]
                     sos_tokens = torch.full((batch_size_local, 1), SOS_token, dtype=torch.long, device=self.device)
                     padded_actions = F.pad(agent_actions, (0, max_len_logprob - agent_actions.shape[1]), value=PAD_token)
                     agent_action_sequences_with_sos = torch.cat([sos_tokens, padded_actions], dim=1)
                     loss_kwargs['agent_action_sequences'] = agent_action_sequences_with_sos


                loss, loss_info = self.loss_calculator.calculate(
                    agent_log_probs=agent_log_probs,
                    agent_probs=agent_probs,
                    final_rewards=final_rewards, 
                    lengths=lengths,
                    mask=mask,
                    **loss_kwargs
                )

                self.optimizer.zero_grad()
                loss.backward()
                grad_clip_value = getattr(self.args, 'grad_clip', 1.0)
                if grad_clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), max_norm=grad_clip_value)
                self.optimizer.step()

                self._log_epoch(epoch, epoch_start_time, loss_info, batch_stats)
                self._checkpoint(epoch)

            except Exception as e:
                 print(f"\n!!! Critical Error during RL epoch {epoch}: {e} !!!")
                 traceback.print_exc()

        total_training_time = time.time() - total_start_time
        print(f"\nREINFORCE training finished. Total time: {total_training_time:.2f}s")
        self._save_final_results()


    def _save_final_results(self):
        """Saves final model, history, plots (smoothed), scaffold grid, and entropy."""
        print("\n--- Saving Final Results ---")
        loss_name = type(self.loss_calculator).__name__.replace("Loss", "").lower()
        mpo_name = type(self.mpo_manager).__name__.replace("MPO", "").lower()
        df_strategy_name = getattr(self.diversity_filter, 'strategy', 'N/A') if not isinstance(self.diversity_filter, NoDiversityFilter) else 'disabled'
        df_name = f"df{df_strategy_name.lower()}"
        run_name = f"{loss_name}_{mpo_name}_{df_name}"
        if self.docking_target_names:
             run_name += f"_dock_{'_'.join(self.docking_target_names)}"

        base_filename = f"agent_{run_name}"
        results_dir = getattr(self.args, 'results_dir', 'results')
        rl_results_dir = os.path.join(results_dir, 'reinforcement_learning', run_name) 
        os.makedirs(rl_results_dir, exist_ok=True)

        final_model_dir = os.path.dirname(getattr(self.args, 'supervised_model_path', 'models/'))
        os.makedirs(final_model_dir, exist_ok=True) # Ensure model dir exists
        final_model_path = os.path.join(final_model_dir, base_filename + ".pth")
        try:
            torch.save(self.agent.model.state_dict(), final_model_path)
            print(f"Saved final agent model state to {final_model_path}")
        except Exception as e: print(f"Error saving final model: {e}")

        final_raw_entropy = 0.0
        final_scaled_entropy = 0.0
        if isinstance(self.diversity_filter, ReinventDiversityFilter):
            scaffold_memory = getattr(self.diversity_filter, 'scaffold_memory', None)
            if scaffold_memory and isinstance(scaffold_memory, dict):
                if scaffold_memory:
                    counts = list(scaffold_memory.values())
                    final_raw_entropy, final_scaled_entropy = calculate_shannon_entropy(counts)
                    print(f"Final Scaffold Shannon Entropy (Raw)  : {final_raw_entropy:.4f}")
                    print(f"Final Scaffold Shannon Entropy (Scaled): {final_scaled_entropy:.4f}")
                else:
                    print("Scaffold memory empty, Shannon Entropy is 0.0")
            else:
                print("Could not access scaffold memory dict for entropy calculation.")
        elif isinstance(self.diversity_filter, NoDiversityFilter):
             print("Diversity filter disabled, cannot calculate scaffold entropy.")
        else: print("Diversity filter type does not support scaffold entropy calculation.")


        history_path = os.path.join(rl_results_dir, base_filename + "_history.json")
        history_to_save = {}
        for key, value in self.rl_history.items():
            if isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                 history_to_save[key] = [item.tolist() for item in value]
            elif isinstance(value, np.ndarray):
                 history_to_save[key] = value.tolist()
            else:
                 history_to_save[key] = value

        history_to_save['final_scaffold_shannon_entropy_raw'] = final_raw_entropy
        history_to_save['final_scaffold_shannon_entropy_scaled'] = final_scaled_entropy

        try:
            with open(history_path, 'w') as f:
                json.dump(history_to_save, f, indent=4)
            print(f"Saved RL training history to {history_path}")
        except Exception as e: print(f"Error saving RL history to {history_path}: {e}")


        print("Generating final plots...")
        try:
            epochs = history_to_save.get('epoch', [])
            if not epochs: print("Warning: No epoch data found in history, skipping plotting."); return

            num_plots = 6 
            plt.figure(figsize=(18, 12)) 
            smoothing_span = max(1, min(20, len(epochs) // 10)) 
            raw_alpha = 0.3
            raw_lw = 1.0
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']

            ax1 = plt.subplot(2, 3, 1)
            reward_final = history_to_save.get('avg_reward_final', [])
            score_mpo = history_to_save.get('avg_score_mpo', []) 
            plot_legend_1 = False
            if reward_final:
                 ax1.plot(epochs, reward_final, color=colors[0 % len(colors)], marker=None, linestyle='-', linewidth=raw_lw, alpha=raw_alpha, label='_nolegend_')
                 ax1.plot(epochs, smooth_data(reward_final, span=smoothing_span), label='Reward (Final Avg)', color=colors[0 % len(colors)])
                 plot_legend_1 = True
            if score_mpo:
                 ax1.plot(epochs, score_mpo, color=colors[1 % len(colors)], marker=None, linestyle='-', linewidth=raw_lw, alpha=raw_alpha, label='_nolegend_')
                 ax1.plot(epochs, smooth_data(score_mpo, span=smoothing_span), label='MPO Score (Avg Passed)', color=colors[1 % len(colors)])
                 plot_legend_1 = True
            ax1.set_title('Average Scores/Rewards'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Value');
            if plot_legend_1: ax1.legend();
            ax1.grid(True)

            ax2 = plt.subplot(2, 3, 2)
            total_loss = history_to_save.get('loss', [])
            color_idx = 0
            plot_legend_2 = False
            if total_loss:
                ax2.plot(epochs, total_loss, color=colors[color_idx % len(colors)], marker=None, linestyle='-', linewidth=raw_lw, alpha=raw_alpha, label='_nolegend_')
                ax2.plot(epochs, smooth_data(total_loss, span=smoothing_span), label='Total Loss', color=colors[color_idx % len(colors)])
                color_idx += 1; plot_legend_2 = True
            if isinstance(self.loss_calculator, ReinforceLoss):
                policy_loss = history_to_save.get('policy_loss', [])
                if policy_loss:
                     ax2.plot(epochs, policy_loss, color=colors[color_idx % len(colors)], marker=None, linestyle='-', linewidth=raw_lw, alpha=raw_alpha, label='_nolegend_')
                     ax2.plot(epochs, smooth_data(policy_loss, span=smoothing_span), label='Policy Loss', color=colors[color_idx % len(colors)])
                     color_idx += 1; plot_legend_2 = True
                reward_baseline = history_to_save.get('reward_baseline', [])
            elif isinstance(self.loss_calculator, ReinventLoss):
                 agent_ll = history_to_save.get('avg_agent_log_prob', [])
                 prior_ll = history_to_save.get('avg_prior_log_prob', [])
                 if agent_ll:
                     ax2.plot(epochs, agent_ll, color=colors[color_idx % len(colors)], marker=None, linestyle='-', linewidth=raw_lw, alpha=raw_alpha, label='_nolegend_')
                     ax2.plot(epochs, smooth_data(agent_ll, span=smoothing_span), label='Agent LogProb', color=colors[color_idx % len(colors)])
                     color_idx += 1; plot_legend_2 = True
                 if prior_ll:
                     ax2.plot(epochs, prior_ll, color=colors[color_idx % len(colors)], marker=None, linestyle='-', linewidth=raw_lw, alpha=raw_alpha, label='_nolegend_')
                     ax2.plot(epochs, smooth_data(prior_ll, span=smoothing_span), label='Prior LogProb', color=colors[color_idx % len(colors)])
                     color_idx += 1; plot_legend_2 = True

            ax2.set_title('Loss Components'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss Value');
            if plot_legend_2: ax2.legend();
            ax2.grid(True)


            ax3 = plt.subplot(2, 3, 3); color1 = 'tab:red'; ax3.set_xlabel('Epoch'); lns1, lbls1 = [], [];
            entropy_data = history_to_save.get('entropy', []) 
            if entropy_data and not np.all(np.isnan(entropy_data)):
                 ax3.plot(epochs, entropy_data, color=color1, marker=None, linestyle='-', linewidth=raw_lw, alpha=raw_alpha, label='_nolegend_')
                 p1, = ax3.plot(epochs, smooth_data(entropy_data, span=smoothing_span), label='Avg Entropy', color=color1);
                 ax3.set_ylabel('Entropy', color=color1); ax3.tick_params(axis='y', labelcolor=color1); ax3.grid(True, axis='y', color=color1, alpha=0.3); lns1.append(p1); lbls1.append(p1.get_label())
            else: ax3.text(0.5, 0.5, 'Entropy N/A', ha='center', va='center', transform=ax3.transAxes, color=color1); ax3.set_ylabel('Entropy (N/A)', color=color1); ax3.tick_params(axis='y', labelcolor=color1)

            ax3b = ax3.twinx(); color2 = 'tab:blue'; color3 = 'tab:green'; lns2, lbls2 = [], [];
            constraint_pass = history_to_save.get('constraint_pass_rate', [])
            df_penalty = history_to_save.get('df_penalty_rate', [])
            if constraint_pass:
                 ax3b.plot(epochs, constraint_pass, color=color2, marker=None, linestyle='-', linewidth=raw_lw, alpha=raw_alpha, label='_nolegend_');
                 p2, = ax3b.plot(epochs, smooth_data(constraint_pass, span=smoothing_span), label='Constraint Pass Rate', color=color2); lns2.append(p2); lbls2.append(p2.get_label())
            if df_penalty:
                 ax3b.plot(epochs, df_penalty, color=color3, marker=None, linestyle='-', linewidth=raw_lw, alpha=raw_alpha, label='_nolegend_');
                 p3, = ax3b.plot(epochs, smooth_data(df_penalty, span=smoothing_span), label='DF Penalty Rate', color=color3); lns2.append(p3); lbls2.append(p3.get_label())

            ax3b.set_ylabel('Rate', color=color2); ax3b.tick_params(axis='y', labelcolor=color2); ax3b.set_ylim(-0.05, 1.05); ax3b.grid(True, axis='y', color=color2, alpha=0.3, linestyle=':')
            all_lns = lns1 + lns2; all_lbls = lbls1 + lbls2;
            if all_lns: ax3.legend(all_lns, all_lbls, loc=0, fontsize='small'); 
            ax3.set_title('Entropy & Filter Rates')

            ax4 = plt.subplot(2, 3, 4)
            plot_title_4 = 'MPO Weights'
            plot_legend_4 = False
            weights_history = history_to_save.get('weights', [])
            if weights_history and isinstance(weights_history, list) and len(weights_history) == len(epochs):
                try:
                    weights_array = np.array(weights_history)
                    if weights_array.ndim == 2:
                        num_props_w = weights_array.shape[1]
                        prop_names_w = getattr(self.property_scorer, 'property_names', [f'P{i+1}' for i in range(num_props_w)])
                        if num_props_w == len(prop_names_w): 
                            colors_w = plt.cm.get_cmap('tab10', max(10, num_props_w)) 
                            for i, prop_name in enumerate(prop_names_w):
                                ax4.plot(epochs, weights_array[:, i], label=f'W_{prop_name}', color=colors_w(i % 10), marker='.', linestyle='-', markersize=2)
                            ax4.set_ylim(-0.05, 1.05)
                            plot_legend_4 = True
                        else: ax4.text(0.5, 0.5, 'Weight/Prop name mismatch', ha='center', va='center')
                    else: ax4.text(0.5, 0.5, 'Weight data format error', ha='center', va='center')
                except Exception as plot_err: ax4.text(0.5, 0.5, f'Plotting error:\n{plot_err}', ha='center', va='center', fontsize=8); plot_title_4 = 'MPO Weights (Plot Error)'
            else: ax4.text(0.5, 0.5, 'No weight data or length mismatch', ha='center', va='center'); plot_title_4 = 'MPO Weights (No Data)'
            ax4.set_title(plot_title_4); ax4.set_xlabel('Epoch'); ax4.set_ylabel('Weight Value');
            if plot_legend_4: ax4.legend(fontsize='small');
            ax4.grid(True)


            ax5 = plt.subplot(2, 3, 5)
            score_key_to_plot = 'avg_scores_desire'
            scores_data = history_to_save.get(score_key_to_plot, [])
            plot_title_5 = f'Avg Scores ({score_key_to_plot}) & Targets (--)'
            plot_legend_5 = False
            if not scores_data or np.all(np.isnan(np.array(scores_data))):
                 score_key_to_plot = 'avg_scores_ema'
                 scores_data = history_to_save.get(score_key_to_plot, [])
                 plot_title_5 = f'Avg Scores ({score_key_to_plot}) & Targets (--)'
                 if not scores_data or np.all(np.isnan(np.array(scores_data))):
                      score_key_to_plot = 'avg_scores_raw' 
                      scores_data = history_to_save.get(score_key_to_plot, [])
                      plot_title_5 = f'Avg Scores ({score_key_to_plot}) & Targets (--)'

            if scores_data and isinstance(scores_data, list) and len(scores_data) == len(epochs):
                 try:
                     avg_scores_array = np.array(scores_data)
                     if avg_scores_array.ndim == 2:
                         num_props_s = avg_scores_array.shape[1]
                         prop_names_s = getattr(self.property_scorer, 'property_names', [f'P{i+1}' for i in range(num_props_s)])
                         targets = self.mpo_manager.target_values.cpu().numpy() if hasattr(self.mpo_manager, 'target_values') else [np.nan]*num_props_s
                         if num_props_s == len(prop_names_s) and num_props_s == len(targets): #
                             colors_s = plt.cm.get_cmap('tab10', max(10, num_props_s))
                             for i, prop_name in enumerate(prop_names_s):
                                 prop_scores = avg_scores_array[:, i]
                                 if not np.all(np.isnan(prop_scores)):
                                     color = colors_s(i % 10)
                                     ax5.plot(epochs, prop_scores, color=color, marker=None, linestyle='-', linewidth=raw_lw, alpha=raw_alpha, label='_nolegend_')
                                     ax5.plot(epochs, smooth_data(prop_scores, span=smoothing_span), label=f'Avg_{prop_name}', color=color)
                                     if not np.isnan(targets[i]):
                                         ax5.axhline(y=targets[i], color=color, linestyle='--', alpha=0.7, label='_nolegend_')
                                     plot_legend_5 = True
                         else: ax5.text(0.5, 0.5, 'Score/Prop/Target mismatch', ha='center', va='center')
                     else: ax5.text(0.5, 0.5, 'Avg score data format error', ha='center', va='center')
                 except Exception as plot_err: ax5.text(0.5, 0.5, f'Plotting error:\n{plot_err}', ha='center', va='center', fontsize=8); plot_title_5 = f'Avg Scores ({score_key_to_plot}) (Plot Error)'
            else: ax5.text(0.5, 0.5, f'No {score_key_to_plot} data or length mismatch', ha='center', va='center'); plot_title_5 = f'Avg Scores ({score_key_to_plot}) (No Data)'

            ax5.set_title(plot_title_5); ax5.set_xlabel('Epoch'); ax5.set_ylabel('Avg Score Value');
            if plot_legend_5: ax5.legend(fontsize='small');
            ax5.grid(True)


            ax6 = plt.subplot(2, 3, 6)
            top_n_barplot = 25
            plot_title_6 = f"Top {top_n_barplot} Scaffold Counts"
            scaffold_memory = None
            df_strategy = 'N/A'
            if isinstance(self.diversity_filter, ReinventDiversityFilter):
                scaffold_memory = getattr(self.diversity_filter, 'scaffold_memory', None)
                df_strategy = getattr(self.diversity_filter, 'strategy', 'Unknown')
            elif isinstance(self.diversity_filter, NoDiversityFilter):
                 df_strategy = 'Disabled'

            if df_strategy == 'Disabled':
                ax6.text(0.5, 0.5, "DF Disabled", ha='center', va='center')
                plot_title_6 = "Scaffold Counts (DF Disabled)"
            elif df_strategy == 'ScaffoldSimilarity':
                num_fps = len(getattr(self.diversity_filter, 'scaffold_fingerprints', []))
                ax6.text(0.5, 0.5, f"Similarity Filter\n{num_fps} fingerprints", ha='center', va='center')
                plot_title_6 = "Similarity Filter State"
            elif scaffold_memory and isinstance(scaffold_memory, dict):
                 if scaffold_memory:
                     sorted_scaffolds = sorted(scaffold_memory.items(), key=lambda item: item[1], reverse=True)
                     top_scaffolds_for_bar = sorted_scaffolds[:top_n_barplot]
                     scaffold_keys_display = [f"Rank {i+1}" for i in range(len(top_scaffolds_for_bar))] 
                     scaffold_counts = [c for s, c in top_scaffolds_for_bar]
                     if scaffold_keys_display:
                         ax6.bar(scaffold_keys_display, scaffold_counts)
                         ax6.set_ylabel("Count in Bucket")
                         plt.setp(ax6.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
                         plot_title_6 = f"Top {len(top_scaffolds_for_bar)} Scaffold Counts ({df_strategy})"
                     else: ax6.text(0.5, 0.5, "No scaffolds logged", ha='center', va='center')
                 else: ax6.text(0.5, 0.5, "Scaffold memory empty", ha='center', va='center')
                 ax6.grid(axis='y')
                 entropy_text = f"Scaled Entropy: {final_scaled_entropy:.3f}"
                 ax6.text(0.95, 0.95, entropy_text, transform=ax6.transAxes, fontsize=9,
                         verticalalignment='top', horizontalalignment='right',
                         bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))
            else: 
                 ax6.text(0.5, 0.5, "DF state N/A", ha='center', va='center')

            ax6.set_title(plot_title_6)

            plt.tight_layout(pad=1.5)
            plot_path = os.path.join(rl_results_dir, base_filename + "_curves.png")
            plt.savefig(plot_path)
            print(f"Saved RL training plot to {plot_path}")
            plt.close() 


            if RDKIT_AVAILABLE_TRAINER and isinstance(self.diversity_filter, ReinventDiversityFilter) and \
               getattr(self.diversity_filter, 'strategy', '') not in ['ScaffoldSimilarity', 'Disabled', 'N/A', 'Unknown'] and \
               scaffold_memory and isinstance(scaffold_memory, dict):
                top_n_grid = min(25, len(scaffold_memory)) 
                if top_n_grid > 0:
                    print(f"Generating top {top_n_grid} scaffold grid image...")
                    if 'sorted_scaffolds' not in locals(): 
                        sorted_scaffolds = sorted(scaffold_memory.items(), key=lambda item: item[1], reverse=True)
                    sorted_scaffolds_for_grid = sorted_scaffolds[:top_n_grid]
                    mols_to_draw = []
                    legends = []
                    valid_smiles_count = 0
                    for i, (smi, count) in enumerate(sorted_scaffolds_for_grid):
                        mol = Chem.MolFromSmiles(smi) if smi else None
                        if mol:
                            mols_to_draw.append(mol)
                            legends.append(f"Rank {i+1} (Count: {count})")
                            valid_smiles_count += 1
                        else: print(f"  Warning: Could not generate Mol from scaffold SMILES: '{smi}' for grid image.")

                    if mols_to_draw:
                        try:
                            molsPerRow = 5
                            subImgSize = (200, 200)
                            img = Draw.MolsToGridImage(
                                mols_to_draw,
                                molsPerRow=molsPerRow,
                                subImgSize=subImgSize,
                                legends=legends
                            )
                            grid_image_path = os.path.join(rl_results_dir, base_filename + f"_top{valid_smiles_count}_scaffolds.png")
                            img.save(grid_image_path)
                            print(f"Saved scaffold grid image to {grid_image_path}")
                        except Exception as e_grid: print(f"Warning: Could not generate scaffold grid image: {e_grid}")
                    else: print("No valid scaffold molecules to generate grid image.")
                else: print("Scaffold memory is empty, skipping grid image.")

        except Exception as e:
            print(f"ERROR: Could not generate or save RL plots/scaffold image: {e}")
            traceback.print_exc()