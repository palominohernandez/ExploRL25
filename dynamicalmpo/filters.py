import torch
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple


try:
    from rdkit import Chem
    from rdkit.Chem import DataStructs
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem import AllChem 
    RDKIT_AVAILABLE_FILTERS = True
except ImportError:
    RDKIT_AVAILABLE_FILTERS = False
    print("Warning (filters.py): RDKit not found. ReinventDiversityFilter will be disabled.")


def get_murcko_scaffold(mol: Chem.Mol, generic: bool = False) -> Optional[str]:
    """Calculates Murcko scaffold SMILES."""
    if not mol: return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if generic:
            scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        return Chem.MolToSmiles(scaffold) if scaffold and scaffold.GetNumAtoms() > 0 else ""
    except Exception:
        return None 

def get_topological_scaffold(mol: Chem.Mol) -> Optional[str]:
    """Calculates Topological scaffold (atoms converted to generic C)."""
    if not mol: return None
    try:
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        if not scaffold_mol or scaffold_mol.GetNumAtoms() == 0: return ""

        rw_mol = Chem.RWMol(scaffold_mol)
        for atom in rw_mol.GetAtoms():
            if atom.GetAtomicNum() != 6:
                 atom.SetAtomicNum(6)
                 atom.SetFormalCharge(0)
                 atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
                 atom.SetNoImplicit(True) 
                 atom.SetNumExplicitHs(0) 

        final_scaffold = rw_mol.GetMol()

        return Chem.MolToSmiles(final_scaffold)
    except Exception as e:
        print(f"Warning (filters.py): Error getting topological scaffold: {e}")
        return None

def get_morgan_fingerprint(mol: Chem.Mol, radius: int = 2, nBits: int = 2048) -> Optional[DataStructs.ExplicitBitVect]:
     """Calculates Morgan Fingerprint."""
     if not mol: return None
     try:
         mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius,fpSize=nBits)
         return mfpgen.GetFingerprint(mol)
     except Exception:
         return None

class ReinventDiversityFilter:
    """
    Applies diversity filtering based on scaffolds and MPO scores, inspired by REINVENT.

    Organizes compounds into buckets based on scaffolds. Penalizes compounds added
    to full buckets if their MPO score exceeds a threshold.
    Includes methods for applying filter during RL (updates state) and checking
    filter status for scoring mode (does not update state).
    """
    def __init__(self, strategy: str = 'IdenticalMurcko', bucket_capacity: int = 25,
                 similarity_threshold: float = 0.7, mpo_score_threshold: float = 0.5):
        """
        Initializes the diversity filter.

        Args:
            strategy: 'IdenticalMurcko', 'Topological', or 'ScaffoldSimilarity'.
            bucket_capacity: Maximum number of compounds per scaffold bucket.
            similarity_threshold: Tanimoto similarity threshold for 'ScaffoldSimilarity' strategy.
            mpo_score_threshold: MPO score above which compounds interact with buckets.
        """
        if not RDKIT_AVAILABLE_FILTERS:
             raise ImportError("RDKit is required for ReinventDiversityFilter.")

        if strategy not in ['IdenticalMurcko', 'Topological', 'ScaffoldSimilarity']:
            raise ValueError(f"Unknown diversity filter strategy: {strategy}")
        if bucket_capacity <= 0:
            print("Warning (ReinventDiversityFilter): Bucket capacity <= 0 disables the filter effect.")
            self.bucket_capacity = float('inf')
        else:
             self.bucket_capacity = bucket_capacity

        self.strategy = strategy
        self.similarity_threshold = similarity_threshold
        self.mpo_score_threshold = mpo_score_threshold

        self.scaffold_memory = defaultdict(int)
        self.scaffold_fingerprints: List[Tuple[str, int]] = []

        print(f"Initialized ReinventDiversityFilter: Strategy='{self.strategy}', "
              f"Capacity={self.bucket_capacity}, MPO Threshold={self.mpo_score_threshold}, "
              f"Similarity Threshold={self.similarity_threshold if strategy == 'ScaffoldSimilarity' else 'N/A'}")

    def _get_scaffold_key(self, mol: Chem.Mol) -> Optional[Any]:
        """Calculates the scaffold key based on the chosen strategy."""
        if self.strategy == 'IdenticalMurcko':
            return get_murcko_scaffold(mol, generic=False)
        elif self.strategy == 'Topological':
             return get_topological_scaffold(mol)
        elif self.strategy == 'ScaffoldSimilarity':
             fp = get_morgan_fingerprint(mol)
             return fp
        else:
            return None 

    def apply_filter(self,
                     mols_passed_constraints: List[Chem.Mol],
                     mpo_scores_passed_constraints: torch.Tensor
                     ) -> torch.Tensor:
        """
        Applies the diversity filter logic during RL training, updating internal state.

        Args:
            mols_passed_constraints: List of RDKit Mol objects that passed mandatory constraints.
            mpo_scores_passed_constraints: Tensor of corresponding MPO scores.

        Returns:
            penalty_mask (torch.Tensor): Boolean tensor of the same length as input.
                                         True means the molecule's reward should be penalized to R_min (0.0)
                                         due to the diversity filter. False means keep the MPO score.
        """
        num_passed = len(mols_passed_constraints)
        if num_passed == 0 or self.bucket_capacity == float('inf'):
             return torch.zeros(num_passed, dtype=torch.bool, device=mpo_scores_passed_constraints.device)

        penalize_mask = torch.zeros(num_passed, dtype=torch.bool, device=mpo_scores_passed_constraints.device)
        mpo_scores_np = mpo_scores_passed_constraints.cpu().numpy() 

        indices_above_threshold = np.where(mpo_scores_np > self.mpo_score_threshold)[0]

        if len(indices_above_threshold) == 0:
             return penalize_mask 

        for idx in indices_above_threshold:
            mol = mols_passed_constraints[idx]
            if not mol: 
                continue

            scaffold_key = self._get_scaffold_key(mol)
            if scaffold_key is None: 
                print('Failed to get scaffold')
                continue 

            penalize = False
            if self.strategy == 'ScaffoldSimilarity':
                
                fp_query = scaffold_key

                if not fp_query: 
                    continue

                found_similar = False
                penalize = False
                indices_to_increment = []

                for i, (fp_stored, count) in enumerate(self.scaffold_fingerprints):
                    similarity = DataStructs.TanimotoSimilarity(fp_query, fp_stored)
                    
                    if similarity >= self.similarity_threshold:
                        found_similar = True
                        if count >= self.bucket_capacity:
                            penalize = True
                            break 
                        else:
                            indices_to_increment.append(i)
                
                if penalize:
                    penalize_mask[idx] = True
                else:
                    if not found_similar:
                        self.scaffold_fingerprints.append((fp_query, 1))
                    else:
                        for i in indices_to_increment:
                            fp_stored, current_count = self.scaffold_fingerprints[i]
                            self.scaffold_fingerprints[i] = (fp_stored, current_count + 1)
            
            else: 
                scaffold_smiles = scaffold_key
                if not isinstance(scaffold_smiles, str) or scaffold_smiles == "": 
                    continue

                current_count = self.scaffold_memory.get(scaffold_smiles, 0)

                if current_count >= self.bucket_capacity:
                    penalize_mask[idx] = True
                else:
                    self.scaffold_memory[scaffold_smiles] = current_count + 1

        return penalize_mask


    def get_filter_status(self, mol: Chem.Mol, mpo_score: float) -> Dict[str, Any]:
        """
        Checks the diversity filter status for a molecule without updating state.

        Args:
            mol: The RDKit Mol object to check.
            mpo_score: The pre-calculated MPO score for the molecule.

        Returns:
            A dictionary containing status information:
            {
                'scaffold_key': str or None (SMILES or FP Bit String),
                'threshold_met': bool,
                'current_bucket_count': int (count in the relevant bucket, -1 if N/A),
                'penalty_would_apply': bool
            }
        """
        status = {
            'scaffold_key': None,
            'threshold_met': False,
            'current_bucket_count': -1,
            'penalty_would_apply': False
        }
        if not mol or self.bucket_capacity == float('inf'):
            return status

        threshold_met = (mpo_score > self.mpo_score_threshold)
        status['threshold_met'] = threshold_met

        if not threshold_met:
             return status 

        scaffold_key = self._get_scaffold_key(mol)
        status['scaffold_key'] = scaffold_key
        if scaffold_key is None:
             return status 

        current_count = -1 
        bucket_full = False

        if self.strategy == 'ScaffoldSimilarity':
            fp_query_str = scaffold_key
            if not fp_query_str: return status
            fp_query = DataStructs.ExplicitBitVect(fp_query_str)

            max_count_in_similar = 0 
            found_similar = False
            for fp_stored_str, count in self.scaffold_fingerprints:
                fp_stored = DataStructs.ExplicitBitVect(fp_stored_str)
                similarity = DataStructs.TanimotoSimilarity(fp_query, fp_stored)
                if similarity >= self.similarity_threshold:
                    found_similar = True
                    max_count_in_similar = max(max_count_in_similar, count)
                    if count >= self.bucket_capacity:
                        bucket_full = True
                        break
            current_count = max_count_in_similar if found_similar else 0

        else:
            scaffold_smiles = scaffold_key
            if not isinstance(scaffold_smiles, str) or scaffold_smiles == "": return status

            current_count = self.scaffold_memory.get(scaffold_smiles, 0)
            if current_count >= self.bucket_capacity:
                bucket_full = True

        status['current_bucket_count'] = current_count
        status['penalty_would_apply'] = bucket_full

        return status

    def get_state(self) -> Dict[str, Any]:
        """Returns the filter's memory state for checkpointing."""
        state = {
            'scaffold_memory': dict(self.scaffold_memory),
            'scaffold_fingerprints': self.scaffold_fingerprints, 
            'strategy': self.strategy,
            'bucket_capacity': self.bucket_capacity,
            'similarity_threshold': self.similarity_threshold,
            'mpo_score_threshold': self.mpo_score_threshold,
        }
        return state

    def load_state(self, state: Dict[str, Any]):
        """Loads the filter's memory state from a checkpoint."""
        self.strategy = state.get('strategy', self.strategy)
        loaded_capacity = state.get('bucket_capacity', self.bucket_capacity)
        self.bucket_capacity = float('inf') if loaded_capacity == float('inf') or loaded_capacity <= 0 else int(loaded_capacity)

        self.similarity_threshold = state.get('similarity_threshold', self.similarity_threshold)
        self.mpo_score_threshold = state.get('mpo_score_threshold', self.mpo_score_threshold)

        self.scaffold_memory = defaultdict(int, state.get('scaffold_memory', {}))
        loaded_fps = state.get('scaffold_fingerprints', [])
        self.scaffold_fingerprints = [(fp_str, int(count)) for fp_str, count in loaded_fps if isinstance(fp_str, str) and isinstance(count, (int, float))]


        print(f"ReinventDiversityFilter state loaded. Strategy='{self.strategy}', "
              f"Capacity={self.bucket_capacity}, MPO Threshold={self.mpo_score_threshold}, "
              f"Num Scaffold Keys={len(self.scaffold_memory)}, "
              f"Num Fingerprints={len(self.scaffold_fingerprints)}")

    def reset_memory(self):
        """Resets the filter's memory."""
        self.scaffold_memory.clear()
        self.scaffold_fingerprints = []
        print("ReinventDiversityFilter memory reset.")



class NoDiversityFilter:
    """
    A dummy diversity filter that applies no penalty and maintains no state.
    Used when the diversity filter is disabled via arguments.
    """
    strategy = "None" 

    def apply_filter(self,
                     mols_passed_constraints: List[Chem.Mol],
                     mpo_scores_passed_constraints: torch.Tensor
                     ) -> torch.Tensor:
        """Applies no filter, returns all False (no penalty)."""
        num_passed = len(mols_passed_constraints)
        device = mpo_scores_passed_constraints.device
        return torch.zeros(num_passed, dtype=torch.bool, device=device)

    def get_filter_status(self, mol: Chem.Mol, mpo_score: float) -> Dict[str, Any]:
        """Returns a default status indicating no penalty would apply."""
        return {
            'scaffold_key': None,
            'threshold_met': False,       
            'current_bucket_count': -1, 
            'penalty_would_apply': False  
        }

    def get_state(self) -> Dict[str, Any]:
        """Returns an empty state dictionary."""
        return {'strategy': self.strategy} 

    def load_state(self, state: Dict[str, Any]):
        """Loads state (does nothing for the dummy filter)."""
        self.strategy = state.get('strategy', "None") 
        pass 

    def reset_memory(self):
        """Resets memory (does nothing for the dummy filter)."""
        pass 


