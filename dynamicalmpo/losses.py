# losses.py
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
import numpy as np

from typing import Dict, Tuple, Optional
from agent import SmilesGeneratorAgent




def calculate_sequence_log_likelihood(log_probs_steps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Calculates the log-likelihood of sequences by summing step log-probs."""

    masked_log_probs = log_probs_steps * mask
    sequence_log_lik = masked_log_probs.sum(dim=1) 
    return sequence_log_lik



def calculate_entropy(agent_probs: Optional[torch.Tensor], mask: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Calculates the policy entropy per whole batch."""

    if agent_probs is None:
        return torch.tensor(0.0, device=mask.device), 0.0

    try:
        log_probs_all_actions = torch.log(agent_probs.clamp(min=1e-20))
        entropy_per_step = - (agent_probs * log_probs_all_actions).sum(dim=-1) 
        masked_entropy = entropy_per_step * mask


        avg_entropy_tensor = masked_entropy.sum(dim=1).mean() 
        avg_entropy_scalar = avg_entropy_tensor.item()
  

        return avg_entropy_tensor, avg_entropy_scalar
    except Exception as e:
        print(f"Warning: Error calculating entropy: {e}")
        return torch.tensor(0.0, device=mask.device), 0.0


class BaseLossCalculator(ABC):
    """Abstract Base Class for loss calculation in RL."""
    def __init__(self, args):
        self.args = args 

    @abstractmethod
    def calculate(self, *,
                  agent_log_probs: torch.Tensor,
                  agent_probs: Optional[torch.Tensor],
                  final_rewards: torch.Tensor,
                  lengths: torch.Tensor,
                  mask: torch.Tensor,
                  prior_log_probs: Optional[torch.Tensor] = None,
                  **kwargs) -> Tuple[torch.Tensor, Dict]:
        """
        Calculates the loss value for backpropagation.

        Args:
            agent_log_probs (torch.Tensor): Log probabilities of actions from the agent. Shape: (batch, seq_len)
            agent_probs (torch.Tensor): Probabilities of actions from the agent. Shape: (batch, seq_len, vocab_size). Optional.
            final_rewards (torch.Tensor): Final scalar reward for each trajectory. Shape: (batch,)
            lengths (torch.Tensor): Actual lengths of the sequences (excluding SOS/padding). Shape: (batch,)
            mask (torch.Tensor): Boolean or float mask for valid steps. Shape: (batch, seq_len)
            prior_log_probs (torch.Tensor): Log probabilities from the prior network. Shape: (batch, seq_len). Optional.
            **kwargs: Additional arguments needed by specific loss functions (e.g., 'agent_action_sequences').

        Returns:
            torch.Tensor: The scalar loss value ready for backpropagation.
            dict: Dictionary containing components of the loss and related metrics for logging.
        """
        pass

class ReinforceLoss(BaseLossCalculator):
    """
    Calculates REINFORCE loss with optional baseline, KL divergence, and entropy regularization.
    Loss = PolicyLoss + kl_beta * KL_Div - entropy_beta * Entropy
    """
    def __init__(self, args, prior_agent: Optional['SmilesGeneratorAgent'] = None):
        super().__init__(args)
        self.prior_agent = prior_agent 
        self.use_baseline = getattr(args, 'reinforce_use_baseline', True)
        self.kl_beta = getattr(args, 'reinforce_kl_beta', 0.0)
        self.entropy_beta = getattr(args, 'reinforce_entropy_beta', 0.0) 

        self.requires_sequences = (self.prior_agent is not None and self.kl_beta > 0)

        if self.kl_beta > 0 and self.prior_agent is None:
            print("Warning: ReinforceLoss initialized with kl_beta > 0 but no prior_agent. KL term cannot be computed unless prior_log_probs are passed directly to calculate().")

    def _get_prior_log_probs(self, agent_action_sequences_with_sos, lengths, device, mask):
        """Internal helper to calculate prior log probs and apply mask."""
        if self.prior_agent is None:
             print("Error: _get_prior_log_probs called but self.prior_agent is None.")
             return None
        if not hasattr(self.prior_agent, 'evaluate_log_probs'):
             raise NotImplementedError("Prior agent does not have 'evaluate_log_probs' method needed by ReinforceLoss.")

        self.prior_agent.model.eval() 
        with torch.no_grad():
            prior_log_probs = self.prior_agent.evaluate_log_probs(agent_action_sequences_with_sos, lengths)
            prior_log_probs = prior_log_probs.to(device)
            masked_prior_log_probs = prior_log_probs * mask
            return masked_prior_log_probs



    def calculate(self, *, agent_log_probs, agent_probs, final_rewards, lengths, mask, prior_log_probs=None, **kwargs):
        """Calculates the REINFORCE loss, handling internal prior calculation if needed."""
        device = agent_log_probs.device
        batch_size = agent_log_probs.size(0)

        actual_prior_log_probs = prior_log_probs 

        if self.kl_beta > 0 and actual_prior_log_probs is None: 
            if self.prior_agent is not None:
                agent_action_sequences_with_sos = kwargs.get('agent_action_sequences')
                if agent_action_sequences_with_sos is None:
                    print("Warning: kl_beta > 0 and no prior_log_probs passed, but 'agent_action_sequences' missing from kwargs for ReinforceLoss. Cannot compute KL term.")
                else:
                    expected_seq_len = agent_log_probs.shape[1]
                    if agent_action_sequences_with_sos.shape[1] != expected_seq_len + 1: 
                        print(f"Warning: Shape mismatch for prior calculation: agent_action_sequences {agent_action_sequences_with_sos.shape} vs agent_log_probs {agent_log_probs.shape}. KL term skipped.")
                    else:
                        try:
                            actual_prior_log_probs = self._get_prior_log_probs(
                                agent_action_sequences_with_sos, lengths, device, mask
                            )
                            if actual_prior_log_probs is None: 
                                print("Warning: Internal calculation of prior log probs failed. KL term skipped.")
                        except Exception as e:
                            print(f"Error calculating prior log probs internally: {e}. KL term skipped.")
                            actual_prior_log_probs = None 

        kl_div_term = torch.tensor(0.0, device=device)
        avg_kl_div = 0.0
        if self.kl_beta > 0:
            if actual_prior_log_probs is not None:
                if actual_prior_log_probs.shape != agent_log_probs.shape:
                     print(f"Warning: Shape mismatch for KL divergence: prior {actual_prior_log_probs.shape} vs agent {agent_log_probs.shape}. KL term skipped.")
                else:
                    kl_div_per_step = (agent_log_probs - actual_prior_log_probs) * mask 
                    kl_div_batch = kl_div_per_step.sum(dim=1) 
                    kl_div_term = kl_div_batch.mean() 
                    avg_kl_div = kl_div_term.item()

        baseline = 0.0
        if self.use_baseline:
            baseline = final_rewards.mean().detach()

        advantage = final_rewards - baseline

        policy_loss_terms = -agent_log_probs * advantage.unsqueeze(1)
        masked_policy_loss = policy_loss_terms * mask
        policy_loss = masked_policy_loss.sum(dim=1).mean()

        entropy_term, avg_entropy = calculate_entropy(agent_probs, mask)

        total_loss = policy_loss + self.kl_beta * kl_div_term - self.entropy_beta * entropy_term

        loss_info = {
            'loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'kl_div': avg_kl_div,
            'entropy': avg_entropy,
            'entropy_bonus': -self.entropy_beta * avg_entropy, 
            'kl_penalty': self.kl_beta * avg_kl_div,
            'avg_reward': final_rewards.mean().item(),
            'reward_baseline': baseline.item() if isinstance(baseline, torch.Tensor) else baseline
        }
        return total_loss, loss_info





class ReinventLoss(BaseLossCalculator):
    """
    Calculates the Reinvent loss based on per-sequence squared error.
    Loss = Mean [ (PriorLL + sigma * Reward - AgentLL)^2 ] + kl_beta*KL - entropy_beta*Entropy
    """

    def __init__(self, args, prior_agent: Optional['SmilesGeneratorAgent']): 
        super().__init__(args)
        self.prior_agent = prior_agent 
        self.sigma = getattr(args, 'reinvent_sigma', 60.0)
        self.kl_beta = getattr(args, 'reinvent_kl_beta', 0.0) 
        self.entropy_beta = getattr(args, 'reinvent_entropy_beta', 0.0) 

        prior_needed_for_base_loss = True 
        prior_needed_for_kl = (self.kl_beta > 0)
        if self.prior_agent is None and (prior_needed_for_base_loss or prior_needed_for_kl):
            print("Warning: ReinventLoss initialized without a prior_agent, but it's needed for base loss and/or KL penalty.")

        self.requires_sequences = (self.prior_agent is not None)


    def calculate(self, *, agent_log_probs, agent_probs, final_rewards, lengths, mask, prior_log_probs=None, **kwargs):
        """Calculates the Reinvent loss."""
        device = agent_log_probs.device
        batch_size = agent_log_probs.size(0)


        if self.prior_agent is None and prior_log_probs is None:
  
            print("ERROR: ReinventLoss requires prior_log_probs (passed in or via prior_agent). Cannot compute loss.")

            entropy_term, avg_entropy = calculate_entropy(agent_probs, mask)
            total_loss = - self.entropy_beta * entropy_term
            loss_info = {'loss': total_loss.item(),
                        'entropy': avg_entropy,
                        'avg_reward': final_rewards.mean().item()}

            return total_loss, loss_info

        if prior_log_probs is None and self.prior_agent:
            agent_action_sequences_with_sos = kwargs.get('agent_action_sequences')
            if agent_action_sequences_with_sos is None:
                raise ValueError("ReinventLoss requires 'agent_action_sequences' in kwargs if prior_log_probs not provided.")
            if agent_action_sequences_with_sos.shape[1] != agent_log_probs.shape[1] + 1:
                 raise ValueError(f"Shape mismatch: agent_action_sequences {agent_action_sequences_with_sos.shape} vs agent_log_probs {agent_log_probs.shape}")



            with torch.no_grad():

                prior_log_probs = self.prior_agent.evaluate_log_probs(agent_action_sequences_with_sos, lengths)
            prior_log_probs = prior_log_probs.to(device)


            if prior_log_probs.shape != agent_log_probs.shape:
                raise ValueError(f"Shape mismatch after prior evaluation: prior {prior_log_probs.shape} vs agent {agent_log_probs.shape}")

        if prior_log_probs is None:
            print("ERROR: ReinventLoss could not obtain prior_log_probs. Cannot compute base loss or KL term.")
            entropy_term, avg_entropy = calculate_entropy(agent_probs, mask)
            total_loss = - self.entropy_beta * entropy_term
            loss_info = {'loss': total_loss.item(),
                        'entropy': avg_entropy,
                        'avg_reward': final_rewards.mean().item()}

            return total_loss, loss_info

        sequence_agent_log_lik = calculate_sequence_log_likelihood(agent_log_probs, mask)
        sequence_prior_log_lik = calculate_sequence_log_likelihood(prior_log_probs, mask)


        target = sequence_prior_log_lik + self.sigma * final_rewards 

        squared_error =  (target - sequence_agent_log_lik)**2
  
        base_loss = torch.mean(squared_error)


        kl_div_term = torch.tensor(0.0, device=device)
        avg_kl_div = 0.0
        if self.kl_beta > 0:
 
            kl_div_batch = sequence_agent_log_lik - sequence_prior_log_lik
            kl_div_term = kl_div_batch.mean() 
            avg_kl_div = kl_div_term.item()


        entropy_term, avg_entropy = calculate_entropy(agent_probs, mask)


        total_loss = base_loss + self.kl_beta * kl_div_term - self.entropy_beta * entropy_term*self.sigma*0.5 # TODO remove

        loss_info = {
            'loss': total_loss.item(),
            'base_loss': base_loss.item(),
            'kl_div': avg_kl_div,
            'entropy': avg_entropy,
            'entropy_bonus': -self.entropy_beta * avg_entropy,
            'kl_penalty': self.kl_beta * avg_kl_div,
            'avg_reward': final_rewards.mean().item(),
            'avg_agent_log_prob': sequence_agent_log_lik.mean().item(),
            'avg_prior_log_prob': sequence_prior_log_lik.mean().item(),
        }


        return total_loss, loss_info

class AugmentedHillClimbLoss(BaseLossCalculator):
    """
    Calculates AHC loss based on the top-k scoring sequences in the batch.
    Uses the same per-sequence squared error as ReinventLoss, applied to subset.
    Loss = Mean_TopK [ (PriorLL + sigma * Reward - AgentLL)^2 ] + kl_beta*KL - entropy_beta*Entropy
    """
    def __init__(self, args, prior_agent: Optional['SmilesGeneratorAgent']):
        super().__init__(args)
        self.prior_agent = prior_agent
        self.sigma = getattr(args, 'reinvent_sigma', 60.0) 
        self.k = getattr(args, 'ahc_top_k', 1) 
        self.kl_beta = getattr(args, 'ahc_kl_beta', 0.0)
        self.entropy_beta = getattr(args, 'ahc_entropy_beta', 0.0)

        if self.prior_agent is None:
             print("Warning: AHC Loss initialized without a prior_agent. Base loss cannot be calculated.")

    def calculate(self, *, agent_log_probs, agent_probs, final_rewards, lengths, mask, prior_log_probs=None, **kwargs):
        """Calculates the AHC loss."""
        device = agent_log_probs.device
        batch_size = agent_log_probs.size(0)

        if self.prior_agent is None and prior_log_probs is None:
             print("ERROR: AHC Loss requires prior_log_probs (or prior_agent). Cannot compute loss.")
             entropy_term, avg_entropy = calculate_entropy(agent_probs, mask)
             total_loss = - self.entropy_beta * entropy_term 
             loss_info = {'loss': total_loss.item(), 'entropy': avg_entropy, 'avg_reward': final_rewards.mean().item()}
             return total_loss, loss_info

        if prior_log_probs is None and self.prior_agent:
            agent_action_sequences_with_sos = kwargs.get('agent_action_sequences')
            if agent_action_sequences_with_sos is None: raise ValueError("AHC requires 'agent_action_sequences' in kwargs if prior_log_probs not provided.")
            if agent_action_sequences_with_sos.shape[1] != agent_log_probs.shape[1] + 1: raise ValueError("Shape mismatch: agent_action_sequences vs agent_log_probs")
            prior_log_probs = self.prior_agent.evaluate_log_probs(agent_action_sequences_with_sos, lengths).to(device)
            if prior_log_probs.shape != agent_log_probs.shape: raise ValueError("Shape mismatch after prior evaluation")

        if prior_log_probs is None:
             print("ERROR: AHC could not obtain prior_log_probs.")
             entropy_term, avg_entropy = calculate_entropy(agent_probs, mask)
             total_loss = - self.entropy_beta * entropy_term
             loss_info = {'loss': total_loss.item(), 'entropy': avg_entropy, 'avg_reward': final_rewards.mean().item()}
             return total_loss, loss_info


        k_actual = min(self.k, batch_size) 
        if k_actual <= 0: 
             print("Warning: AHC top_k is <= 0. Skipping loss calculation.")
             return torch.tensor(0.0, device=device), {'loss': 0.0, 'avg_reward': final_rewards.mean().item()}

        top_k_indices = torch.topk(final_rewards, k=k_actual, dim=0).indices

        top_k_agent_log_probs = agent_log_probs[top_k_indices]
        top_k_prior_log_probs = prior_log_probs[top_k_indices]
        top_k_rewards = final_rewards[top_k_indices]
        top_k_mask = mask[top_k_indices]
        top_k_agent_probs = agent_probs[top_k_indices] if agent_probs is not None else None

        top_k_seq_agent_log_lik = calculate_sequence_log_likelihood(top_k_agent_log_probs, top_k_mask)
        top_k_seq_prior_log_lik = calculate_sequence_log_likelihood(top_k_prior_log_probs, top_k_mask)

        target = top_k_seq_prior_log_lik + self.sigma * top_k_rewards
        squared_error = (target - top_k_seq_agent_log_lik)**2
        base_loss = torch.mean(squared_error)

        kl_div_term = torch.tensor(0.0, device=device)
        avg_kl_div = 0.0
        if self.kl_beta > 0:
            kl_div_top_k = top_k_seq_agent_log_lik - top_k_seq_prior_log_lik
            kl_div_term = kl_div_top_k.mean()
            avg_kl_div = kl_div_term.item()


        entropy_term, avg_entropy = calculate_entropy(top_k_agent_probs, top_k_mask)

      
        total_loss = base_loss + self.kl_beta * kl_div_term - self.entropy_beta * entropy_term

        loss_info = {
            'loss': total_loss.item(),
            'base_loss': base_loss.item(),
            'kl_div': avg_kl_div,
            'entropy': avg_entropy,
            'entropy_bonus': -self.entropy_beta * avg_entropy,
            'kl_penalty': self.kl_beta * avg_kl_div,
            'avg_reward': final_rewards.mean().item(), 
            'avg_reward_topk': top_k_rewards.mean().item(), 
            'avg_seq_agent_log_lik_topk': top_k_seq_agent_log_lik.mean().item(),
            'avg_seq_prior_log_lik_topk': top_k_seq_prior_log_lik.mean().item(),
        }
        return total_loss, loss_info

class PPOLoss(BaseLossCalculator):
    """Placeholder for PPO Loss calculation."""
    def __init__(self, args):
        super().__init__(args)
        print("Warning: PPOLoss is not fully implemented yet.")

    def calculate(self, *, agent_log_probs, agent_probs, final_rewards, lengths, mask, prior_log_probs=None, **kwargs):
        """Placeholder calculation."""
        print("Warning: PPOLoss.calculate() called, but PPO is not implemented.")
        loss = torch.tensor(0.0, device=agent_log_probs.device, requires_grad=True) 
        loss_info = {
            'loss': 0.0,
            'avg_reward': final_rewards.mean().item(),
        }
        return loss, loss_info


