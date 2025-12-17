import os
import torch
import json
from typing import List, Dict, Any, Optional, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, FilterCatalog
    from rdkit.Chem.FilterCatalog import FilterCatalogParams
    RDKIT_AVAILABLE_CONSTRAINTS = True
except ImportError:
    RDKIT_AVAILABLE_CONSTRAINTS = False
    print("Warning (constraints.py): RDKit not found. Constraint checks will be disabled.")


def load_smarts_from_file(filepath: str) -> List[str]:
    """Loads SMARTS patterns from a file, one per line."""
    patterns = []
    if filepath and os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        patterns.append(line)
        except Exception as e:
            print(f"Warning (constraints.py): Failed to read SMARTS file {filepath}: {e}")
    return patterns

def get_rdkit_mol(smiles: str) -> Optional[Chem.Mol]:
    """Attempts to generate an RDKit Mol object from SMILES."""
    if not RDKIT_AVAILABLE_CONSTRAINTS or not isinstance(smiles, str) or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:

            return mol
        else:
            return None
    except Exception:
        return None 


def check_structural_alerts(mol: Chem.Mol, alert_smarts_list: List[str]) -> bool:
    """Checks if the molecule matches any structural alert SMARTS pattern."""
    if not mol or not alert_smarts_list:
        return True 
    for smarts in alert_smarts_list:
        try:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                return False 
        except Exception as e:
             print(f"Warning (constraints.py): Invalid SMARTS pattern '{smarts}' in alerts: {e}")
    return True 

def check_required_substructures(mol: Chem.Mol, required_smarts_list: List[str]) -> bool:
    """Checks if the molecule contains ALL required substructure SMARTS patterns."""
    if not mol or not required_smarts_list:
        return True 
    for smarts in required_smarts_list:
        try:
            pattern = Chem.MolFromSmarts(smarts)
            if not pattern or not mol.HasSubstructMatch(pattern):
                return False 
        except Exception as e:
             print(f"Warning (constraints.py): Invalid SMARTS pattern '{smarts}' in requirements: {e}")
             return False 
    return True 
def check_property_limits(mol: Chem.Mol, limits_config: List[Dict[str, Any]]) -> bool:
    """Checks if the molecule satisfies all defined property limits."""
    if not mol or not limits_config:
        return True 
    calculated_properties = {}

    def get_prop(prop_name):
        if prop_name not in calculated_properties:
            try:

                if prop_name == 'MW' or prop_name == 'MolWt':
                    val = Descriptors.MolWt(mol)
                elif prop_name == 'LogP':
                    val = Descriptors.MolLogP(mol)
                elif prop_name == 'HBD':
                    val = Descriptors.NumHDonors(mol)
                elif prop_name == 'HBA':
                    val = Descriptors.NumHAcceptors(mol)
                elif prop_name == 'TPSA':
                    val = Descriptors.TPSA(mol)
                elif prop_name == 'RotB':
                    val = Descriptors.NumRotatableBonds(mol)
                else:
                    print(f"Warning (constraints.py): Unknown property '{prop_name}' in limits_config.")
                    return None
                calculated_properties[prop_name] = val
                return val
            except Exception as e:
                print(f"Warning (constraints.py): Failed to calculate property '{prop_name}': {e}")
                return None
        return calculated_properties[prop_name]

    for limit in limits_config:
        prop_name = limit.get('property')
        op_str = limit.get('op')
        value = limit.get('value')

        if not prop_name or not op_str or value is None:
            print(f"Warning (constraints.py): Skipping invalid limit config: {limit}")
            continue

        prop_value = get_prop(prop_name)
        if prop_value is None:
            return False 

        passes = False
        try:
            if op_str == '<=': passes = (prop_value <= value)
            elif op_str == '>=': passes = (prop_value >= value)
            elif op_str == '<': passes = (prop_value < value)
            elif op_str == '>': passes = (prop_value > value)
            elif op_str == '==': passes = (prop_value == value)
            elif op_str == '!=': passes = (prop_value != value)
            else:
                print(f"Warning (constraints.py): Unknown operator '{op_str}' in limit config.")
                return False 

            if not passes:
                return False 
        except TypeError:
             print(f"Warning (constraints.py): Type error comparing {prop_name} ({prop_value}) {op_str} {value}")
             return False 

    return True 



def check_mandatory_constraints(
    smiles_batch: List[str],
    config: Dict[str, Any], 
    device: torch.device
) -> Tuple[torch.Tensor, List[Optional[Chem.Mol]]]:
    """
    Checks a batch of SMILES against configured mandatory constraints.

    Args:
        smiles_batch: List of SMILES strings.
        config: Dictionary containing configuration flags and paths/values
                (e.g., config['enable_structural_alerts'], config['structural_alerts_path'],
                 config['property_limits_config_list'], etc.)
        device: The torch device.

    Returns:
        A tuple containing:
        - passed_mask (torch.Tensor): Boolean tensor (True if passed all checks).
        - valid_mols_list (List[Optional[RDKit Mol]]): List of Mol objects for
          each input SMILES (None if invalid SMILES).
    """
    if not RDKIT_AVAILABLE_CONSTRAINTS:
        print("Error (constraints.py): RDKit not available. Cannot perform constraint checks.")
        passed = torch.zeros(len(smiles_batch), dtype=torch.bool, device=device)
        mols = [None] * len(smiles_batch)
        return passed, mols

    batch_size = len(smiles_batch)
    passed_list = [True] * batch_size
    valid_mols_list: List[Optional[Chem.Mol]] = [None] * batch_size

    structural_alerts = []
    if config.get('enable_structural_alerts', False):
        structural_alerts = load_smarts_from_file(config.get('structural_alerts_path'))

    required_substructures = []
    if config.get('enable_required_substructures', False):
        required_substructures = load_smarts_from_file(config.get('required_substructures_path'))

    property_limits = []
    if config.get('enable_property_limits', False):
        property_limits = config.get('property_limits_config_list', [])


    for i, smi in enumerate(smiles_batch):
        mol = get_rdkit_mol(smi)
        valid_mols_list[i] = mol 
        if mol is None:
            passed_list[i] = False
            continue 

        if structural_alerts and not check_structural_alerts(mol, structural_alerts):
            passed_list[i] = False
            continue

        if required_substructures and not check_required_substructures(mol, required_substructures):
            passed_list[i] = False
            continue

        if property_limits and not check_property_limits(mol, property_limits):
            passed_list[i] = False
            continue


    passed_mask = torch.tensor(passed_list, dtype=torch.bool, device=device)
    return passed_mask, valid_mols_list


