# scoring.py

import sys
import os
import torch
import numpy as np
import math
import time
import json
import traceback
from typing import List, Dict, Any, Optional, Tuple

import subprocess
import tempfile
import shutil 

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, DataStructs, AllChem 
    from rdkit.Chem.Scaffolds import MurckoScaffold 
    RDKIT_AVAILABLE_SCORING = True
except ImportError:
    RDKIT_AVAILABLE_SCORING = False
    print("Warning (scoring.py): RDKit not found. Docking and RDKit properties unavailable.")

try:
    sascorer_path = os.path.join(os.environ.get('CONDA_PREFIX', sys.prefix), 'share', 'RDKit', 'Contrib', 'SA_Score')
    if os.path.exists(os.path.join(sascorer_path, 'sascorer.py')):
         sys.path.append(sascorer_path)
         import sascorer
         SASCORE_AVAILABLE_SCORING = True
    else:
        try: 
             import sascorer
             SASCORE_AVAILABLE_SCORING = True
        except ImportError:
            SASCORE_AVAILABLE_SCORING = False
            print("Warning (scoring.py): sascorer not found. SA score will be 0.0.")
except ImportError:
    SASCORE_AVAILABLE_SCORING = False
    print("Warning (scoring.py): sascorer not found. SA score will be 0.0.")

try:
    import joblib 
    PKL_AVAILABLE_SCORING = True
except ImportError:
    PKL_AVAILABLE_SCORING = False
    print("Warning (scoring.py): joblib/pickle not found. Cannot load custom .pkl models.")

from custom_chemprop import CustomChemprop

_props = {
            'MW' : Descriptors.MolWt,
            'LogP' : Descriptors.MolLogP,
            'HBD' : Descriptors.NumHDonors,
            'HBA' : Descriptors.NumHAcceptors,
            'TPSA' : Descriptors.TPSA,
            'RotB' : Descriptors.NumRotatableBonds,
            'AroRings' : Descriptors.NumAromaticRings,
            'AliRings' : Descriptors.NumAliphaticRings,
            'Fsp3' : Descriptors.FractionCSP3,
            'BertzCT' : Descriptors.BertzCT,
            'QED' : Descriptors.qed,
            'SA' : sascorer.calculateScore}


def _identity(x, params=None):
    return x

def _linear_ramp(x, params):
    low, high = params['low'], params['high']
    target = params.get('target_value', 1.0)
    #if low >= high: return float(target) 
    score = (x - low) / (high - low)
    if target == 0.0: score = 1.0 - score
    return max(0.0, min(1.0, float(score))) 

def _step_function(x, params):
    threshold = params['threshold']
    high_val = params.get('high_value', 1.0)
    low_val = params.get('low_value', 0.0)
    side = params.get('side', 'right')
    interval = params['interval'] if params['interval'] is not None else None
    if interval:
        low = float(interval['low'])
        high = float(interval['high'])
        if x >= low and x <= high:
            return float(high_val)
        else:
            return float(low_val)
    else:
        if side == 'right':
            return float(high_val if x >= threshold else low_val)
        else: # side == 'left'
            return float(high_val if x <= threshold else low_val)

def _gaussian_like(x, params):
    mu, sigma = params['mu'], params['sigma']
    if sigma <= 0: return float(1.0 if x == mu else 0.0) 
    try:
        return float(math.exp(-0.5 * ((x - mu) / sigma)**2))
    except OverflowError:
        return 0.0 

def _sigmoid_like(x, params):
    """Applies a sigmoid-like transformation mapping input x to [0, 1]."""
    low = params.get('low', 0.0)
    high = params.get('high', 1.0)
    target = params.get('target_value', 1.0) 
    slope = params.get('slope', 1.0)
    epsilon = 1e-9 

    if abs(high - low) < epsilon:
        return float(target)

    midpoint = (low + high) / 2.0
    x_eff = float(x) 

    try:
        scaled_pos = (x_eff - low) / (high - low)
        exponent_input = slope * (x_eff - midpoint) / abs(high - low) 
        val_logistic = 1.0 / (1.0 + math.exp(-exponent_input))

        if target == 1.0: 
             val = val_logistic
        else: 
             val = 1.0 - val_logistic

    except (OverflowError, ValueError) as e:
        print(f"  DEBUG: Exception during sigmoid calculation (Input {x_eff:.2f}): {e}")

        if target == 1.0: 
            val = 1.0 if x_eff >= midpoint else 0.0
        else: 
            val = 0.0 if x_eff >= midpoint else 1.0

    return max(0.0, min(1.0, float(val)))

DESIRABILITY_FUNCTIONS = {
    'identity': _identity,
    'linear_ramp': _linear_ramp,
    'step': _step_function,
    'gaussian': _gaussian_like,
    'sigmoid': _sigmoid_like,
}


class PropertyScorer:
    """
    Calculates molecular properties and optionally applies desirability functions.
    Includes support for batch docking via external OpenEye tools.
    """
    def __init__(self,
                property_names: List[str],
                desirability_configs: Optional[List[Dict]] = None,
                device: torch.device = torch.device('cpu'),
                omega_exe_path: str = 'oeomega',
                fred_exe_path: str = 'fred',
                omega_args_str: str = 'pose -flipper true',
                fred_args_str: str = '-dock_resolution Standard',
                reference_db_smiles_path: Optional[str] = None,
                reference_db_scaffold_path: Optional[str] = None, 
                chemprop: List[str] = None
                ):
        """
        Initializes the PropertyScorer.

        Args:
            property_names: List of property names to calculate (e.g., "QED", "Docking_TargetA").
            desirability_configs: List of dictionaries defining desirability functions.
            device: PyTorch device.
            omega_exe_path: Path/command for OpenEye Omega executable.
            fred_exe_path: Path/command for OpenEye FRED executable.
            omega_args_str: Additional arguments string for Omega (excluding -in, -out).
            fred_args_str: Additional arguments string for FRED (excluding -receptor, -dbase, -scorefile, -docked_molecule_file).
            reference_db_smiles_path: Path to a file containing reference canonical SMILES (one per line) for isMolinDB check.
            reference_db_scaffold_path: Path to a file containing reference canonical scaffold SMILES (one per line) for isScaffinDB check.
        
        """
        self.property_names = property_names
        self.num_properties = len(property_names)
        self.device = device
        self.desirability_map = self._parse_desirability_configs(desirability_configs)
        self.custom_models = {}
        self.custom_model_feature_configs = {}

        if not RDKIT_AVAILABLE_SCORING:
            print("ERROR (PropertyScorer): RDKit is required for property calculation and docking preprocessing.")

        self.omega_exe_path = omega_exe_path
        self.fred_exe_path = fred_exe_path
        self.omega_args_list = omega_args_str.split() if omega_args_str else []
        self.fred_args_list = fred_args_str.split() if fred_args_str else []
        self._batch_docking_results: Optional[Dict[str, Dict[int, float]]] = None
        self._batch_indices_processed_docking: Optional[List[int]] = None
        self.reference_smiles_set: Optional[set] = None
        self.reference_scaffold_set: Optional[set] = None

        self._load_reference_data(
            reference_db_smiles_path if 'isMolinDB' in self.property_names else None,
            reference_db_scaffold_path if 'isScaffinDB' in self.property_names else None
        )
        self.chemprop: CustomChemprop = chemprop

    def _parse_desirability_configs(self, configs: Optional[List[Dict]]) -> Dict[str, Dict]:
        """Parses and validates desirability configurations."""
        d_map = {}
        if configs:
            for config in configs:
                prop = config.get('property')
                func_type = config.get('type')
                params = config.get('params', {})
                if prop in self.property_names:
                    if func_type in DESIRABILITY_FUNCTIONS:
                        d_map[prop] = {'type': func_type, 'params': params}
                    else:
                        print(f"Warning (PropertyScorer): Unknown desirability function type '{func_type}' for property '{prop}'. Using 'identity'.")
                        d_map[prop] = {'type': 'identity', 'params': {}}

            for prop in self.property_names:
                if prop not in d_map:
                    d_map[prop] = {'type': 'identity', 'params': {}}
        else:
            for prop in self.property_names:
                 d_map[prop] = {'type': 'identity', 'params': {}}
        return d_map


    def _load_reference_data(self, smiles_path: Optional[str], scaffold_path: Optional[str]):
        """Loads reference SMILES and Scaffolds into hash sets."""
        if smiles_path:
            if not RDKIT_AVAILABLE_SCORING:
                 print("Warning (_load_reference_data): RDKit needed to canonicalize/validate reference SMILES, but not found. Cannot load isMolinDB reference.")
                 self.reference_smiles_set = None
            elif os.path.exists(smiles_path):
                print(f"Loading reference SMILES for isMolinDB from: {smiles_path}")
                start_time = time.time()
                self.reference_smiles_set = set()
                count = 0
                valid_count = 0
                try:
                    with open(smiles_path, 'r') as f:
                        for line in f:
                            smi = line.strip()
                            count += 1
                            if not smi: continue
                            mol = Chem.MolFromSmiles(smi)
                            if mol: 
                                try:
                                    canonical_smi = Chem.MolToSmiles(mol, canonical=True)
                                    self.reference_smiles_set.add(canonical_smi)
                                    valid_count += 1
                                except Exception:
                                     pass 

                    duration = time.time() - start_time
                    print(f"Loaded {len(self.reference_smiles_set)} unique canonical reference SMILES ({valid_count}/{count} valid/processed lines) in {duration:.2f}s.")
                except Exception as e:
                    print(f"Error reading reference SMILES file {smiles_path}: {e}")
                    self.reference_smiles_set = None
            else:
                print(f"Warning (_load_reference_data): Reference SMILES file not found: {smiles_path}")
                self.reference_smiles_set = None
        else:
            if 'isMolinDB' in self.property_names:
                print("Warning: 'isMolinDB' requested but no reference_db_smiles_path provided.")
            self.reference_smiles_set = None 

        if scaffold_path:
            if not RDKIT_AVAILABLE_SCORING:
                 print("Warning (_load_reference_data): RDKit needed to canonicalize/validate reference scaffolds, but not found. Cannot load isScaffinDB reference.")
                 self.reference_scaffold_set = None
            elif os.path.exists(scaffold_path):
                print(f"Loading reference scaffolds for isScaffinDB from: {scaffold_path}")
                start_time = time.time()
                self.reference_scaffold_set = set()
                count = 0
                valid_count = 0
                try:
                    with open(scaffold_path, 'r') as f:
                        for line in f:
                            scaf_smi = line.strip()
                            count += 1
                            if not scaf_smi: continue
                            scaf_mol = Chem.MolFromSmiles(scaf_smi) 
                            if scaf_mol:
                                try:
                                    canonical_scaf_smi = Chem.MolToSmiles(scaf_mol, canonical=True)
                                    self.reference_scaffold_set.add(canonical_scaf_smi)
                                    valid_count += 1
                                except Exception:
                                    pass
                    duration = time.time() - start_time
                    print(f"Loaded {len(self.reference_scaffold_set)} unique canonical reference scaffolds ({valid_count}/{count} valid/processed lines) in {duration:.2f}s.")
                except Exception as e:
                    print(f"Error reading reference scaffold file {scaffold_path}: {e}")
                    self.reference_scaffold_set = None
            else:
                print(f"Warning (_load_reference_data): Reference scaffold file not found: {scaffold_path}")
                self.reference_scaffold_set = None
        else:
            if 'isScaffinDB' in self.property_names:
                print("Warning: 'isScaffinDB' requested but no reference_db_scaffold_path provided.")
            self.reference_scaffold_set = None 


    def load_custom_model(self, prop_name: str, model_path: str, feature_config: Dict):
        """Loads a custom model from a .pkl file."""
        if not PKL_AVAILABLE_SCORING:
            print(f"Warning (PropertyScorer): Cannot load model for '{prop_name}', joblib/pickle not installed.")
            return
        if not os.path.exists(model_path):
             print(f"Warning (PropertyScorer): Model file not found for '{prop_name}' at {model_path}.")
             return
        try:
            model = joblib.load(model_path)
            self.custom_models[prop_name] = model
            self.custom_model_feature_configs[prop_name] = feature_config
            print(f"Loaded custom model for property '{prop_name}' from {model_path}")
        except Exception as e:
            print(f"Error loading custom model for '{prop_name}' from {model_path}: {e}")


    def _perform_batch_docking(self,
                               smiles_list: List[str],
                               original_indices: List[int],
                               receptor_paths: List[str],
                               target_names: List[str]
                               ) -> Dict[str, Dict[int, float]]:
        """
        Performs batch docking using OpenEye Omega and FRED via subprocess calls.

        Args:
            smiles_list: List of SMILES strings to dock.
            original_indices: List of original batch indices corresponding to smiles_list.
            receptor_paths: List of paths to prepared receptor (.oedu) files.
            target_names: List of unique names for each receptor.

        Returns:
            A dictionary where keys are target_names and values are dictionaries
            mapping original_batch_idx to the best docking score for that target.
            Returns empty dict if docking cannot proceed or fails completely.
        """
        all_docking_results: Dict[str, Dict[int, float]] = {}
        if not smiles_list or not receptor_paths or not RDKIT_AVAILABLE_SCORING:
            print("Warning (_perform_batch_docking): Missing inputs, RDKit, or receptors. Skipping docking.")
            return all_docking_results

        omega_path = shutil.which(self.omega_exe_path)
        fred_path = shutil.which(self.fred_exe_path)
        if not omega_path:
             print(f"Error: Omega executable not found via command '{self.omega_exe_path}'. Check path or installation. Skipping docking.")
             return all_docking_results
        if not fred_path:
             print(f"Error: FRED executable not found via command '{self.fred_exe_path}'. Check path or installation. Skipping docking.")
             return all_docking_results


        tmpdir = None 
        try:
            tmpdir = tempfile.mkdtemp(prefix="docking_batch_")

            smi_input_path = os.path.join(tmpdir, "input_smiles.smi")
            with open(smi_input_path, 'w') as f:
                for smi in smiles_list:
                    f.write(smi + "\n")

            sdf_output_path = os.path.join(tmpdir, "conformers.oeb.gz")
            omega_cmd = [
                omega_path, 
                *self.omega_args_list,
                "-in", smi_input_path,
                "-out", sdf_output_path
            ]

            start_omega = time.time()
            try:

                omega_timeout = 1800
                omega_process = subprocess.run(omega_cmd, capture_output=True, text=True, check=True, timeout=omega_timeout)
                duration_omega = time.time() - start_omega

            except FileNotFoundError:
                 print(f"Error: Omega executable disappeared? Path: '{omega_path}'. Cannot proceed.")
                 return {}
            except subprocess.CalledProcessError as e:
                duration_omega = time.time() - start_omega
                print(f"Error: Omega execution failed after {duration_omega:.2f}s (Return code: {e.returncode}):")
                print(f"Command: {' '.join(e.cmd)}")
                print("--- Omega stdout ---:\n", e.stdout)
                print("--- Omega stderr ---:\n", e.stderr)
                return {} 
            except subprocess.TimeoutExpired:
                 duration_omega = time.time() - start_omega
                 print(f"Error: Omega timed out after {duration_omega:.2f}s (limit {omega_timeout}s). Skipping docking for this batch.")
                 return {}

            if not os.path.exists(sdf_output_path) or os.path.getsize(sdf_output_path) < 50: 
                 print(f"Warning: Omega ran but output file '{sdf_output_path}' is missing or likely empty. No conformers generated?")
                 return {}

            num_targets = len(receptor_paths)
            for idx, (receptor_path, target_name) in enumerate(zip(receptor_paths, target_names)):
                if not os.path.exists(receptor_path):
                     print(f"Warning: Receptor file '{receptor_path}' for target '{target_name}' not found. Skipping this target.")
                     all_docking_results[target_name] = {} 
                     continue

                receptor_scores: Dict[int, float] = {}
                score_output_path = os.path.join(tmpdir, f"scores_{target_name}.txt")
                docked_output_path = os.path.join(tmpdir, f"docked_{target_name}.sdf.gz") 

                fred_cmd = [
                    fred_path, 
                    *self.fred_args_list,
                    "-receptor", receptor_path,
                    "-dbase", sdf_output_path,
                    "-scorefile", score_output_path,
                    "-docked_molecule_file", docked_output_path
                ]
                start_fred = time.time()
                try:
                    fred_timeout = 3600
                    fred_process = subprocess.run(fred_cmd, capture_output=True, text=True, check=True, timeout=fred_timeout)
                    duration_fred = time.time() - start_fred

                    if os.path.exists(score_output_path):
                        parse_count = 0
                        skipped_header = False
                        with open(score_output_path, 'r') as f:
                            for line_num, line in enumerate(f):
                                line = line.strip()
                                if not line: continue 

                                if not skipped_header and ("Title" in line or "#" in line):
                                    skipped_header = True
                                    continue

                                parts = line.split() 
                                if len(parts) >= 2 and parts[0].startswith("omega_"):
                                    try:
                                        title = parts[0]
                                        score_str = parts[-1]
                                        score = float(score_str)

                                        omega_parts = title.split('_')
                                        if len(omega_parts) >= 2:
                                            one_based_index_str = omega_parts[1]
                                            one_based_index = int(one_based_index_str)

                                            if 1 <= one_based_index <= len(original_indices):
                                                original_batch_idx = original_indices[one_based_index - 1]

                                                if original_batch_idx not in receptor_scores or score < receptor_scores[original_batch_idx]:
                                                    receptor_scores[original_batch_idx] = score
                                                    parse_count += 1 

                                            else:
                                                print(f"Warning (Target {target_name}, Line {line_num+1}): Parsed index {one_based_index} out of range (1-{len(original_indices)}) for title '{title}'")
                                        else:
                                            print(f"Warning (Target {target_name}, Line {line_num+1}): Could not parse 1-based index from omega title: '{title}'")

                                    except (ValueError, IndexError) as parse_err:
                                        print(f"Warning (Target {target_name}, Line {line_num+1}): Could not parse score line '{line}': {parse_err}")
                                else:
                                    if not line.startswith("#"):
                                        print(f"Warning (Target {target_name}, Line {line_num+1}): Skipping unrecognized line format in score file: '{line}'")

                    else:
                        print(f"Warning: Score file '{score_output_path}' not found after FRED run for target '{target_name}'. Assigning 0 scores for this target.")

                except FileNotFoundError:
                    print(f"Error: FRED executable disappeared? Path: '{fred_path}'. Cannot dock for {target_name}.")
                    all_docking_results[target_name] = {}
                    continue 
                except subprocess.CalledProcessError as e:
                    duration_fred = time.time() - start_fred
                    print(f"Error: FRED execution failed for target '{target_name}' after {duration_fred:.2f}s (Return code: {e.returncode}):")
                    print(f"Command: {' '.join(e.cmd)}")
                    print(f"--- FRED stdout ({target_name}) ---:\n", e.stdout)
                    print(f"--- FRED stderr ({target_name}) ---:\n", e.stderr)
                    all_docking_results[target_name] = {}
                except subprocess.TimeoutExpired:
                     duration_fred = time.time() - start_fred
                     print(f"Error: FRED timed out for target '{target_name}' after {duration_fred:.2f}s (limit {fred_timeout}s). Skipping scores for this target.")
                     all_docking_results[target_name] = {}

                all_docking_results[target_name] = receptor_scores

            return all_docking_results

        except Exception as e:
            print(f"!!! Unexpected Error in _perform_batch_docking: {e} !!!")
            traceback.print_exc()
            return {} 
        finally:
            if tmpdir and os.path.exists(tmpdir):
                try:
                    shutil.rmtree(tmpdir)
                except Exception as clean_e:
                    print(f"Warning: Failed to remove temporary directory {tmpdir}: {clean_e}")



    def _calculate_single_property(self, mol: Chem.Mol, prop_name: str) -> Optional[float]:
        """
        Calculates a single raw property value for a molecule.
        Docking properties are handled separately in batch via get_scores.
        """
        if not mol: return None
        if not RDKIT_AVAILABLE_SCORING: return 0.0 

        try:
            if prop_name == 'MW' or prop_name == 'MolWt': return Descriptors.MolWt(mol)
            elif prop_name == 'LogP': return Descriptors.MolLogP(mol)
            elif prop_name == 'HBD': return Descriptors.NumHDonors(mol)
            elif prop_name == 'HBA': return Descriptors.NumHAcceptors(mol)
            elif prop_name == 'TPSA': return Descriptors.TPSA(mol)
            elif prop_name == 'RotB': return Descriptors.NumRotatableBonds(mol)
            elif prop_name == 'AroRings': return Descriptors.NumAromaticRings(mol)
            elif prop_name == 'AliRings': return Descriptors.NumAliphaticRings(mol)
            elif prop_name == 'Fsp3': return Descriptors.FractionCSP3(mol)
            elif prop_name == 'BertzCT': return Descriptors.BertzCT(mol)
            elif prop_name == 'QED': return Descriptors.qed(mol)
            elif prop_name == 'SA':
                if SASCORE_AVAILABLE_SCORING: return sascorer.calculateScore(mol)
                else: return 0.0 
            elif prop_name == 'isMolinDB':
                if mol is None or self.reference_smiles_set is None:
                    return 0.0 
                try:
                    canonical_gen_smiles = Chem.MolToSmiles(mol, canonical=True)
                    return 1.0 if canonical_gen_smiles in self.reference_smiles_set else 0.0
                except Exception as e:
                    return 0.0 

            elif prop_name == 'isScaffinDB':
                if mol is None or self.reference_scaffold_set is None:
                    return 0.0 
                try:
                    gen_scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                    if gen_scaffold.GetNumAtoms() > 0:
                        canonical_gen_scaffold_smiles = Chem.MolToSmiles(gen_scaffold, canonical=True)
                        return 1.0 if canonical_gen_scaffold_smiles in self.reference_scaffold_set else 0.0
                    else:
                        return 0.0 
                except Exception as e:
                    return 0.0 

            elif prop_name.startswith("Docking_"):
                 return None 
            
            elif prop_name.startswith("Chemprop_"):
                 return None 

            else:
                return 0.0 

        except Exception as e:
            return 0.0 

    def get_scores(self,
                   mols_list: List[Optional[Chem.Mol]],
                   apply_desirability: bool = True,
                   original_indices: Optional[List[int]] = None,
                   receptor_paths: Optional[List[str]] = None,
                   target_names: Optional[List[str]] = None,
                   ) -> torch.Tensor:
        """
        Calculates scores for a batch of RDKit Mol objects. Handles batch docking.

        Args:
            mols_list: A list where each element is an RDKit Mol object or None.
                       It's expected this list contains valid molecules relevant
                       to the `original_indices`.
            apply_desirability: If True, applies configured desirability functions.
                                If False, returns raw calculated property values.
            original_indices: List of original batch indices corresponding to mols_list.
                              Required if docking is performed.
            receptor_paths: List of receptor paths. Required if docking is performed.
            target_names: List of target names. Required if docking is performed.


        Returns:
            A torch tensor of shape (batch_size, num_properties) containing scores.
        """
        batch_size = len(mols_list)

        if any('Chemprop_' in name for name in self.property_names):
            smiles_list = [Chem.MolToSmiles(mol) for mol in mols_list]
            cp_values = self.chemprop.infere_chemprop(smiles_list)

        scores = torch.zeros((batch_size, self.num_properties), dtype=torch.float32, device=self.device)

        non_docking_rdkit_props = any(p not in ['SA'] and not p.startswith("Docking_") and p not in self.custom_models for p in self.property_names)
        if non_docking_rdkit_props and not RDKIT_AVAILABLE_SCORING:
            print("Warning (get_scores): RDKit properties requested but RDKit not available. Returning zeros for those properties.")

        needs_docking = any(p.startswith("Docking_") for p in self.property_names)
        if needs_docking:
            if not RDKIT_AVAILABLE_SCORING:
                print("Error: Docking requested but RDKit is required for preprocessing. Cannot perform docking.")
                self._batch_docking_results = {} 
            elif not receptor_paths or not target_names or original_indices is None:
                print("Warning: Docking requested but receptor_paths, target_names, or original_indices not provided to get_scores. Docking scores will be default penalty value.")
                self._batch_docking_results = {} 
            elif self._batch_indices_processed_docking == original_indices and self._batch_docking_results is not None:
                pass
            else:

                smiles_to_dock = []
                valid_indices_for_docking = []
                original_indices_map_to_valid = {} 
                current_valid_idx = 0
                for i, mol in enumerate(mols_list):
                    if mol: 
                        smiles = Chem.MolToSmiles(mol)
                        smiles_to_dock.append(smiles)
                        valid_indices_for_docking.append(original_indices[i])
                        original_indices_map_to_valid[original_indices[i]] = current_valid_idx
                        current_valid_idx += 1
                    else:
                        print(f"Warning: Found None Mol object at index {i} corresponding to original index {original_indices[i]} during docking prep.")

                if not smiles_to_dock:
                    print("Warning: No valid molecules to dock in this batch.")
                    self._batch_docking_results = {}
                else:
                    self._batch_docking_results = self._perform_batch_docking(
                        smiles_list=smiles_to_dock,
                        original_indices=valid_indices_for_docking,
                        receptor_paths=receptor_paths,
                        target_names=target_names
                    )

                self._batch_indices_processed_docking = original_indices

        docking_failure_penalty_value = 0.0

        for i, mol in enumerate(mols_list):
            original_idx = original_indices[i] if original_indices is not None else i 

            if mol is None and not any(p.startswith("Docking_") for p in self.property_names):
                continue

            for j, prop_name in enumerate(self.property_names):
                score_to_use = 0.0 

                if prop_name.startswith("Docking_"):
                    raw_docking_score = None 
                    if self._batch_docking_results is not None and needs_docking:
                        target_name = prop_name.split("Docking_", 1)[1]
                        target_results = self._batch_docking_results.get(target_name, {})
                        raw_docking_score = target_results.get(original_idx, None)
            
                    score_to_use = docking_failure_penalty_value if raw_docking_score is None else raw_docking_score
                elif prop_name.startswith('Chemprop_'):            
                    score_to_use = cp_values[prop_name][i]
                else:

                    if mol is None:
                         raw_value = 0.0 
                    else:
                        raw_value = self._calculate_single_property(mol, prop_name)

                    if raw_value is None or np.isnan(raw_value) or np.isinf(raw_value):
                        score_to_use = 0.0
                    else:
                        score_to_use = float(raw_value)

                if apply_desirability:
                    config = self.desirability_map.get(prop_name, {'type': 'identity', 'params': {}})
                    func = DESIRABILITY_FUNCTIONS.get(config['type'], _identity)
                    try:

                        desire_score = func(float(score_to_use), config['params'])

                        final_score = max(0.0, min(1.0, desire_score))
                    except Exception as e:
                        final_score = 0.0 
                else:
                    final_score = float(score_to_use) 
                scores[i, j] = final_score

        return scores