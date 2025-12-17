# mpo.py
import torch
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from abc import ABC, abstractmethod
import numpy as np
from collections import deque 
from typing import Optional, Dict, Any, List 


class BaseMPOStrategy(ABC):
    """Abstract Base Class for Multi-Objective Optimization strategies."""
    is_reward_dependent: bool = False 

    def __init__(self, num_properties: int, target_values: List[float], device: torch.device, **kwargs):
        if num_properties <= 0:
            raise ValueError("Number of properties must be positive.")
        self.num_properties = num_properties
        '''if len(target_values) != num_properties:
            raise ValueError(f"Length of target_values ({len(target_values)}) must match num_properties ({num_properties}).")''' # TODO

        self.target_values = torch.tensor(target_values, dtype=torch.float32, device=device)
        self.device = device
        self.weights = torch.ones(num_properties, dtype=torch.float32, device=device)
        self.initial_weights = self.weights.clone() # Store initial weights if needed

    @abstractmethod
    def get_weights(self, epoch: int) -> torch.Tensor:
        """Returns the current weight tensor."""
        pass

    @abstractmethod
    def update_state(self, scores_of_passed_molecules: Optional[torch.Tensor],final_reward, epoch: int):
        """Updates internal state based on performance (if applicable)."""
        pass

    def get_state(self) -> Dict[str, Any]:
        """Returns state for checkpointing."""
        return {'weights': self.weights.cpu().numpy().tolist()}

    def load_state(self, state: Dict[str, Any]):
        """Loads state from checkpoint."""
        if 'weights' in state and isinstance(state['weights'], list) and len(state['weights']) == self.num_properties:
            try:
                self.weights = torch.tensor(state['weights'], dtype=torch.float32, device=self.device)
                print(f"{self.__class__.__name__}: Weights loaded from state.")
            except Exception as e:
                 print(f"Warning: Error loading MPO weights for {self.__class__.__name__} from state: {e}")
        else:
            print(f"Warning: MPO weights for {self.__class__.__name__} not found in state or length mismatch. Using current/default weights.")


class StaticWeightMPO(BaseMPOStrategy):
    """Uses fixed, predefined weights."""
    is_reward_dependent: bool = False

    def __init__(self, num_properties: int, target_values: List[float], device: torch.device, initial_weights: Optional[List[float]] = None, **kwargs):
        super().__init__(num_properties, target_values, device, **kwargs)
        if initial_weights is not None and len(initial_weights) == num_properties:
             try:
                 loaded_weights = torch.tensor(initial_weights, dtype=torch.float32, device=device)
                 if torch.any(loaded_weights < 0): print("Warning: Initial static weights contain negative values.")
                 self.weights = loaded_weights
                 print(f"Initialized StaticWeightMPO with provided weights: {self.weights.cpu().numpy()}")
             except Exception as e:
                  print(f"Error setting initial static weights: {e}. Using default (all ones).")
                  print(f"Initialized StaticWeightMPO with default weights (all ones): {self.weights.cpu().numpy()}")
        else:
            if initial_weights is not None: 
                 print(f"Warning: StaticWeightMPO requires {num_properties} initial_weights, but {len(initial_weights)} were given. Using default (all ones).")
            print(f"Initialized StaticWeightMPO with default weights (all ones): {self.weights.cpu().numpy()}")


    def get_weights(self, epoch: int) -> torch.Tensor:
        return self.weights 

    def update_state(self, scores_of_passed_molecules: Optional[torch.Tensor],final_reward, epoch: int):
        pass


class CyclicalMPO(BaseMPOStrategy):
    """Cycles through focusing on one objective at a time."""
    is_reward_dependent: bool = False

    def __init__(self, num_properties: int, target_values: List[float], device: torch.device, cycle_len: int = 50, **kwargs):
        super().__init__(num_properties, target_values, device, **kwargs)
        if cycle_len <= 0:
            raise ValueError("Cycle length must be positive.")
        self.cycle_len = cycle_len 
        print(f"Initialized CyclicalMPO with cycle length per property: {self.cycle_len}")

    def get_weights(self, epoch: int) -> torch.Tensor:
        """Determines weights based on the current epoch within the cycle."""
        if self.num_properties == 0: return torch.tensor([], device=self.device) 

        focus_idx = (epoch // self.cycle_len) % self.num_properties

        weights = torch.zeros(self.num_properties, device=self.device)
        weights[focus_idx] = 1.0 
        return weights

    def update_state(self, scores_of_passed_molecules: Optional[torch.Tensor],final_reward, epoch: int):
        pass 

    def get_state(self) -> Dict[str, Any]:
        return {'cycle_len': self.cycle_len}

    def load_state(self, state: Dict[str, Any]):
        if 'cycle_len' in state: self.cycle_len = state['cycle_len']
        print("CyclicalMPO state loaded (config only).")


class RandomizedWeightMPO(BaseMPOStrategy):
    """Samples weights randomly from a Dirichlet distribution periodically."""
    is_reward_dependent: bool = False

    def __init__(self, num_properties: int, target_values: List[float], device: torch.device,
                 dirichlet_alpha: float = 1.0, update_frequency: int = 1, **kwargs):
        super().__init__(num_properties, target_values, device, **kwargs)
        if dirichlet_alpha <= 0: raise ValueError("Dirichlet alpha must be positive.")
        if update_frequency <= 0: raise ValueError("Update frequency must be positive.")

        self.dirichlet_alpha = dirichlet_alpha
        self.update_frequency = update_frequency
        self.concentration = torch.full((self.num_properties,), self.dirichlet_alpha, device=device)
        self._sample_weights()
        print(f"Initialized RandomizedWeightMPO: Alpha={self.dirichlet_alpha}, Update Freq={self.update_frequency}")

    def _sample_weights(self):
        """Samples weights from the Dirichlet distribution."""
        try:
            dist = Dirichlet(self.concentration)
            self.weights = dist.sample()
        except Exception as e:
             print(f"Warning: Error sampling from Dirichlet distribution: {e}. Falling back to uniform.")
             self.weights = torch.ones(self.num_properties, device=self.device) / self.num_properties

    def get_weights(self, epoch: int) -> torch.Tensor:
        """Returns the current weights, sampling new ones if it's an update epoch."""
        if epoch == 0 or epoch % self.update_frequency == 0:
             self._sample_weights()
        return self.weights

    def update_state(self, scores_of_passed_molecules: Optional[torch.Tensor],final_reward, epoch: int):
        pass 

    def get_state(self) -> Dict[str, Any]:
        """Returns state including current weights and config."""
        state = super().get_state() 
        state['dirichlet_alpha'] = self.dirichlet_alpha
        state['update_frequency'] = self.update_frequency
        return state

    def load_state(self, state: Dict[str, Any]):
        """Loads state including current weights and config."""
        super().load_state(state) 
        self.dirichlet_alpha = state.get('dirichlet_alpha', self.dirichlet_alpha)
        self.update_frequency = state.get('update_frequency', self.update_frequency)
        self.concentration = torch.full((self.num_properties,), self.dirichlet_alpha, device=self.device)
        print("RandomizedWeightMPO state loaded.")



class DynamicWeightMPO(BaseMPOStrategy):
    """Adjusts weights based on performance gap relative to targets (EMA of scores)."""
    is_reward_dependent: bool = True

    def __init__(self, num_properties: int, target_values: List[float], device: torch.device,
                 update_freq: int = 5, beta: float = 0.05, ema_alpha: float = 0.1,
                 softmax_temp: float = 0.1, min_weight_value: float = 0.01, **kwargs):
        super().__init__(num_properties, target_values, device, **kwargs)
        if update_freq <= 0: raise ValueError("update_freq must be positive.")
        if not (0 < ema_alpha <= 1): raise ValueError("ema_alpha must be between 0 and 1.")
        if min_weight_value < 0: raise ValueError("min_weight_value cannot be negative.")

        self.update_freq = update_freq
        self.beta = beta
        self.ema_alpha = ema_alpha
        self.softmax_temp = max(softmax_temp, 1e-6) 
        self.min_weight_value = min_weight_value
        self.property_score_avg = self.target_values.clone() * 0.5 
        self.weights = torch.ones(num_properties, device=device) / num_properties

        print(f"Initialized DynamicWeightMPO (Performance Gap): Update Freq={update_freq}, Beta={beta}, "
              f"EMA Alpha={ema_alpha}, Softmax Temp={self.softmax_temp}, Min Weight={self.min_weight_value}")

    def get_weights(self, epoch: int) -> torch.Tensor:
        return self.weights 

    def update_state(self, scores_of_passed_molecules: Optional[torch.Tensor],final_reward, epoch: int):
        """Updates weights based on EMA of scores from molecules passing the filter."""
        if epoch % self.update_freq != 0 or scores_of_passed_molecules is None or scores_of_passed_molecules.numel() == 0:
             return 

        if scores_of_passed_molecules.ndim == 1:
            scores_of_passed_molecules = scores_of_passed_molecules.unsqueeze(0)

        if scores_of_passed_molecules.shape[1] != self.num_properties:
             print(f"Warning (DynamicMPO): Score tensor shape mismatch. Skipping weight update.")
             return

        batch_avg_passed_scores = final_reward.mean(dim=0)

        self.property_score_avg = (self.ema_alpha * batch_avg_passed_scores) + \
                                  ((1.0 - self.ema_alpha) * self.property_score_avg)

        errors = self.target_values - self.property_score_avg

        weight_change = self.beta * errors
        raw_new_weights = self.weights + weight_change
        clamped_weights = torch.clamp(raw_new_weights, min=self.min_weight_value)

        if epoch % self.update_freq == 0:
            with torch.no_grad():
                print(f"  Weights (Before): {self.weights.cpu().numpy()}")
                print(f"  Raw Update    : {raw_new_weights.cpu().numpy()}")
                print(f"  Clamped Update: {clamped_weights.cpu().numpy()}")
                print(f"  Input to Softmax (Clamped/Temp): {(clamped_weights / self.softmax_temp).cpu().numpy()}")
                print("")

        self.weights = F.softmax(clamped_weights / self.softmax_temp, dim=0)

    def get_state(self) -> Dict[str, Any]:
        """Returns current weights and EMA score averages."""
        state = super().get_state() 
        state['property_score_avg'] = self.property_score_avg.cpu().numpy().tolist()
        state['beta'] = self.beta
        state['ema_alpha'] = self.ema_alpha
        state['softmax_temp'] = self.softmax_temp
        state['min_weight_value'] = self.min_weight_value
        return state

    def load_state(self, state: Dict[str, Any]):
        """Loads weights and EMA score averages."""
        super().load_state(state) 
        if 'property_score_avg' in state and isinstance(state['property_score_avg'], list) and len(state['property_score_avg']) == self.num_properties:
            try:
                self.property_score_avg = torch.tensor(state['property_score_avg'], dtype=torch.float32, device=self.device)
                print("DynamicWeightMPO: property_score_avg loaded from state.")
            except Exception as e:
                print(f"Warning: Error loading MPO property_score_avg from state: {e}")
        else:
             print("Warning: MPO property_score_avg not found/valid in state. Using current/default EMA.")

        self.beta = state.get('beta', self.beta)
        self.ema_alpha = state.get('ema_alpha', self.ema_alpha)
        self.softmax_temp = state.get('softmax_temp', self.softmax_temp)
        self.min_weight_value = state.get('min_weight_value', self.min_weight_value)


class DynamicImprovementRateMPO(BaseMPOStrategy):
    """Adjusts weights based on the improvement rate (change in EMA score)."""
    is_reward_dependent: bool = True

    def __init__(self, num_properties: int, target_values: List[float], device: torch.device,
                 update_freq: int = 5, beta: float = 0.05, ema_alpha: float = 0.1,
                 softmax_temp: float = 0.1, min_weight_value: float = 0.01, **kwargs):
        super().__init__(num_properties, target_values, device, **kwargs)
        if update_freq <= 0: raise ValueError("update_freq must be positive.")
        if not (0 < ema_alpha <= 1): raise ValueError("ema_alpha must be between 0 and 1.")
        if min_weight_value < 0: raise ValueError("min_weight_value cannot be negative.")

        self.update_freq = update_freq
        self.beta = beta 
        self.ema_alpha = ema_alpha
        self.softmax_temp = max(softmax_temp, 1e-6)
        self.min_weight_value = min_weight_value
        self.property_score_avg = self.target_values.clone() * 0.5 
        self.previous_property_score_avg = self.property_score_avg.clone()
        self.weights = torch.ones(num_properties, device=device) / num_properties

        print(f"Initialized DynamicImprovementRateMPO: Update Freq={update_freq}, Beta={beta}, "
              f"EMA Alpha={ema_alpha}, Softmax Temp={self.softmax_temp}, Min Weight={self.min_weight_value}")

    def get_weights(self, epoch: int) -> torch.Tensor:
        return self.weights

    def update_state(self, scores_of_passed_molecules: Optional[torch.Tensor],final_reward, epoch: int):
        """Updates weights based on change in EMA scores."""
        if epoch % self.update_freq != 0 or scores_of_passed_molecules is None or scores_of_passed_molecules.numel() == 0:
            if scores_of_passed_molecules is not None and scores_of_passed_molecules.numel() > 0:
                 if scores_of_passed_molecules.ndim == 1: scores_of_passed_molecules = scores_of_passed_molecules.unsqueeze(0)
                 if scores_of_passed_molecules.shape[1] == self.num_properties:
                      batch_avg = final_reward.mean(dim=0)
                      self.property_score_avg = (self.ema_alpha * batch_avg) + ((1.0 - self.ema_alpha) * self.property_score_avg)
            return 

        if scores_of_passed_molecules.ndim == 1: scores_of_passed_molecules = scores_of_passed_molecules.unsqueeze(0)
        if scores_of_passed_molecules.shape[1] != self.num_properties:
             print(f"Warning (ImprovementRateMPO): Score tensor shape mismatch. Skipping weight update.")
             return

        batch_avg_passed_scores = final_reward.mean(dim=0)

        current_ema = (self.ema_alpha * batch_avg_passed_scores) + \
                      ((1.0 - self.ema_alpha) * self.property_score_avg)

        improvement_rate = current_ema - self.previous_property_score_avg

        weight_change = -self.beta * improvement_rate
        raw_new_weights = self.weights + weight_change

        clamped_weights = torch.clamp(raw_new_weights, min=self.min_weight_value)

        if epoch % self.update_freq == 0: 
            with torch.no_grad():
                print(weight_change)
                print(f"  Weights (Before): {self.weights.cpu().numpy()}")
                print(f"  Raw Update    : {raw_new_weights.cpu().numpy()}")
                print(f"  Clamped Update: {clamped_weights.cpu().numpy()}")
                print(f"  Input to Softmax (Clamped/Temp): {(clamped_weights / self.softmax_temp).cpu().numpy()}")
                print("")

        self.weights = F.softmax(clamped_weights / self.softmax_temp, dim=0)

        self.previous_property_score_avg = current_ema.clone() 
        self.property_score_avg = current_ema 

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state['property_score_avg'] = self.property_score_avg.cpu().numpy().tolist()
        state['previous_property_score_avg'] = self.previous_property_score_avg.cpu().numpy().tolist()
        state['beta'] = self.beta
        return state

    def load_state(self, state: Dict[str, Any]):
        super().load_state(state)
        if 'property_score_avg' in state and isinstance(state['property_score_avg'], list) and len(state['property_score_avg']) == self.num_properties:
            self.property_score_avg = torch.tensor(state['property_score_avg'], dtype=torch.float32, device=self.device)
        if 'previous_property_score_avg' in state and isinstance(state['previous_property_score_avg'], list) and len(state['previous_property_score_avg']) == self.num_properties:
            self.previous_property_score_avg = torch.tensor(state['previous_property_score_avg'], dtype=torch.float32, device=self.device)
        else: 
             if 'property_score_avg' in state: self.previous_property_score_avg = self.property_score_avg.clone()
        self.beta = state.get('beta', self.beta)
        print("DynamicImprovementRateMPO state loaded.")


class DynamicVarianceMPO(BaseMPOStrategy):
    """Adjusts weights based on the variance of EMA scores over a window."""
    is_reward_dependent: bool = True

    def __init__(self, num_properties: int, target_values: List[float], device: torch.device,
                 update_freq: int = 5, beta: float = 0.05, ema_alpha: float = 0.1,
                 softmax_temp: float = 0.1, min_weight_value: float = 0.01,
                 variance_window: int = 10, **kwargs):
        super().__init__(num_properties, target_values, device, **kwargs)
        if update_freq <= 0: raise ValueError("update_freq must be positive.")
        if not (0 < ema_alpha <= 1): raise ValueError("ema_alpha must be between 0 and 1.")
        if min_weight_value < 0: raise ValueError("min_weight_value cannot be negative.")
        if variance_window < 2: raise ValueError("variance_window must be at least 2.")

        self.update_freq = update_freq
        self.beta = beta 
        self.ema_alpha = ema_alpha
        self.softmax_temp = max(softmax_temp, 1e-6)
        self.min_weight_value = min_weight_value
        self.variance_window = variance_window

        self.property_score_avg = self.target_values.clone() * 0.5 
        self.ema_history = deque(maxlen=self.variance_window) 
        self.weights = torch.ones(num_properties, device=device) / num_properties

        print(f"Initialized DynamicVarianceMPO: Update Freq={update_freq}, Beta={beta}, "
              f"EMA Alpha={ema_alpha}, Variance Window={self.variance_window}, "
              f"Softmax Temp={self.softmax_temp}, Min Weight={self.min_weight_value}")

    def get_weights(self, epoch: int) -> torch.Tensor:
        return self.weights

    def update_state(self, scores_of_passed_molecules: Optional[torch.Tensor],final_reward, epoch: int):
        """Updates weights based on variance of recent EMA scores."""
        
        if scores_of_passed_molecules is not None and scores_of_passed_molecules.numel() > 0:
            if scores_of_passed_molecules.ndim == 1: scores_of_passed_molecules = scores_of_passed_molecules.unsqueeze(0)
            if scores_of_passed_molecules.shape[1] == self.num_properties:
                batch_avg = final_reward.mean(dim=0)
                self.property_score_avg = (self.ema_alpha * batch_avg) + ((1.0 - self.ema_alpha) * self.property_score_avg)
                self.ema_history.append(self.property_score_avg.clone())

        if epoch % self.update_freq != 0:
            return

 
        if len(self.ema_history) < 2:
             return 

        history_tensor = torch.stack(list(self.ema_history), dim=0)
        variance = torch.var(history_tensor, dim=0, unbiased=True) 
        weight_change = self.beta * variance
        raw_new_weights = self.weights + weight_change

        clamped_weights = torch.clamp(raw_new_weights, min=self.min_weight_value)

        if epoch % self.update_freq == 0: 
            with torch.no_grad():
                print(f"\n[DEBUG MPO Ep {epoch}] Variance : {variance.cpu().numpy()}")
                print(f"  Weights (Before): {self.weights.cpu().numpy()}")
                print(f"  Raw Update    : {raw_new_weights.cpu().numpy()}")
                print(f"  Clamped Update: {clamped_weights.cpu().numpy()}")
                print(f"  Input to Softmax (Clamped/Temp): {(clamped_weights / self.softmax_temp).cpu().numpy()}")
                print("")

        self.weights = F.softmax(clamped_weights / self.softmax_temp, dim=0)

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state['property_score_avg'] = self.property_score_avg.cpu().numpy().tolist()
        state['ema_history'] = [t.cpu().numpy().tolist() for t in self.ema_history]
        state['beta'] = self.beta
        return state

    def load_state(self, state: Dict[str, Any]):
        super().load_state(state)
        if 'property_score_avg' in state and isinstance(state['property_score_avg'], list) and len(state['property_score_avg']) == self.num_properties:
            self.property_score_avg = torch.tensor(state['property_score_avg'], dtype=torch.float32, device=self.device)
        if 'ema_history' in state and isinstance(state['ema_history'], list):
             try:
                  loaded_history = [torch.tensor(ema, dtype=torch.float32, device=self.device) for ema in state['ema_history']]
                  self.ema_history = deque(loaded_history, maxlen=self.variance_window)
             except Exception as e:
                  print(f"Warning: Error loading EMA history for Variance MPO: {e}")
                  self.ema_history = deque(maxlen=self.variance_window)
        else:
             self.ema_history = deque(maxlen=self.variance_window) 

        self.beta = state.get('beta', self.beta)
        print("DynamicVarianceMPO state loaded.")


