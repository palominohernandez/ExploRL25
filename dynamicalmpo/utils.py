# utils.py
import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import Counter # Keep if used elsewhere
from typing import List, Dict, Any, Optional

# --- RDKit Setup ---
try:
    from rdkit import Chem
    from rdkit import rdBase
    from rdkit.Chem import Descriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
    rdBase.DisableLog('rdApp.error')
    RDKIT_AVAILABLE = True
except ImportError:
    print("Warning: RDKit not found in utils. Functions requiring it will fail.")
    RDKIT_AVAILABLE = False

try:
    sascorer_path = os.path.join(os.environ.get('CONDA_PREFIX', sys.prefix), 'share', 'RDKit', 'Contrib', 'SA_Score')
    if os.path.exists(os.path.join(sascorer_path, 'sascorer.py')):
         sys.path.append(sascorer_path)
         import sascorer
         SASCORE_AVAILABLE = True
    else:
         import sascorer
         SASCORE_AVAILABLE = True
except ImportError:
    print("Warning: SA scorer (sascorer) not found in utils.")
    SASCORE_AVAILABLE = False
# --- End RDKit Setup ---


# --- Token Definitions ---
PAD_token = 0
SOS_token = 1
EOS_token = 2
PAD_char = "<pad>"
SOS_char = "<sos>"
EOS_char = "<eos>"
# --- End Token Definitions ---


# --- SMILES Utilities ---
def canonicalize_smiles(smi: str) -> Optional[str]:
    """Convert SMILES to canonical form using RDKit."""
    if not RDKIT_AVAILABLE or not smi : return None
    try:
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except Exception:
        return None

def calculate_scaffold(smiles_or_mol: Any, scaffold_type: str ='murcko') -> Optional[str]:
    """Calculates the specified scaffold for a molecule."""
    if not RDKIT_AVAILABLE: return None
    mol = None
    if isinstance(smiles_or_mol, str):
        mol = Chem.MolFromSmiles(smiles_or_mol)
    elif isinstance(smiles_or_mol, Chem.Mol):
        mol = smiles_or_mol
    else: return None
    if not mol: return None

    try:
        if scaffold_type == 'murcko':
            core = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(core, canonical=True) if core.GetNumAtoms() > 0 else ""
        else:
            print(f"Warning (utils.calculate_scaffold): Unknown scaffold type '{scaffold_type}'.")
            return None
    except Exception:
        return None
# --- End SMILES Utilities ---


# --- Checkpointing Utilities ---
def save_checkpoint(state: Dict, filename: str ="checkpoint.pth.tar"):
    """Saves checkpoint dictionary."""
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(state, filename)
    except Exception as e:
        print(f"Error saving checkpoint {filename}: {e}")

def load_checkpoint(filename: str, model=None, optimizer=None, mpo_manager=None, diversity_filter=None, device: str ='cpu'):
    """
    Loads checkpoint. Handles both structured dict checkpoints and raw model state_dict files.
    Returns the checkpoint dictionary if structure was found AND model loaded successfully (if provided),
    otherwise returns None.
    """
    if not os.path.isfile(filename):
        print(f"Checkpoint file not found: {filename}")
        return None
    print(f"Loading checkpoint from {filename}")
    map_location = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
    model_loaded_flag = False
    checkpoint_data = None

    try:
        payload = torch.load(filename, map_location=map_location, weights_only=False)

        if isinstance(payload, dict):
            checkpoint_data = payload
            # print("   Checkpoint is a dictionary. Processing structure...") # Less verbose
            if model and 'model_state_dict' in checkpoint_data:
                model_state_dict = checkpoint_data['model_state_dict']
                if list(model_state_dict.keys())[0].startswith('module.'):
                    model_state_dict = {k[7:]: v for k, v in model_state_dict.items()}
                missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
                if missing_keys: print(f"   Warning: Missing keys loading model state: {missing_keys}")
                if unexpected_keys: print(f"   Warning: Unexpected keys loading model state: {unexpected_keys}")
                print("   Model state loaded from checkpoint dict.")
                model_loaded_flag = True
            elif model:
                print("   Warning: Model provided but 'model_state_dict' key not found in checkpoint dict.")

        if model and not model_loaded_flag:
            # print("   Checkpoint not a dict or model state missing. Attempting to load file as raw state_dict...") # Less verbose
            try:
                model_state_dict = payload
                if list(model_state_dict.keys())[0].startswith('module.'):
                     model_state_dict = {k[7:]: v for k, v in model_state_dict.items()}
                missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
                if missing_keys: print(f"   Warning: Missing keys loading raw state_dict: {missing_keys}")
                if unexpected_keys: print(f"   Warning: Unexpected keys loading raw state_dict: {unexpected_keys}")
                print("   Model state loaded directly from file.")
                model_loaded_flag = True
            except Exception as e_state_dict:
                print(f"   ERROR: Could not load file directly as model state_dict: {e_state_dict}")
                return None

        if model and not model_loaded_flag:
            print("   ERROR: Model weights could not be loaded by any method.")
            return None

        if checkpoint_data:
            model_vocab_size = getattr(model, 'vocab_size', None)
            checkpoint_vocab_size = checkpoint_data.get('vocab_size', None)
            if model_vocab_size is not None and checkpoint_vocab_size is not None and model_vocab_size != checkpoint_vocab_size:
                print(f"   ERROR: Vocab size mismatch! Checkpoint: {checkpoint_vocab_size}, Model: {model_vocab_size}.")

            if optimizer and 'optimizer_state_dict' in checkpoint_data:
                try:
                    optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor): state[k] = v.to(device)
                    # print("   Optimizer state loaded.") # Less verbose
                except Exception as e: print(f"   Warning: Could not load optimizer state: {e}.")
            elif optimizer: print("   Warning: Optimizer provided but 'optimizer_state_dict' not found.")

            if mpo_manager and 'mpo_manager_state' in checkpoint_data:
                if hasattr(mpo_manager, 'load_state'):
                    try: mpo_manager.load_state(checkpoint_data['mpo_manager_state'])
                    except Exception as e: print(f"   Warning: Could not load MPO Manager state: {e}.")
                # else: print("   Warning: MPO Manager passed but has no 'load_state' method.")
            elif mpo_manager: print("   Warning: MPO Manager provided but 'mpo_manager_state' not found.")

            if diversity_filter and 'diversity_filter_state' in checkpoint_data:
                if hasattr(diversity_filter, 'load_state'):
                    try: diversity_filter.load_state(checkpoint_data['diversity_filter_state'])
                    except Exception as e: print(f"   Warning: Could not load Diversity Filter state: {e}.")
                # else: print("   Warning: Diversity Filter passed but has no 'load_state' method.")
            elif diversity_filter: print("   Warning: Diversity Filter provided but 'diversity_filter_state' not found.")

            # print(f"   Checkpoint structure processed (Epoch: {checkpoint_data.get('epoch', 'N/A')}).") # Less verbose
            return checkpoint_data
        elif model_loaded_flag:
             # print("   Checkpoint was raw state_dict; only model weights loaded.") # Less verbose
             return None
        else:
             print("   ERROR: Unknown checkpoint format or failed to load required components.")
             return None

    except Exception as e:
        print(f"Error loading checkpoint from {filename}: {e}")
        return None
# --- End Checkpointing Utilities ---


# --- Plotting Utilities ---
def get_color_map(vocab):
    n_chars = max(1, vocab.n_chars); cmap = plt.get_cmap('tab20', n_chars)
    return {i: mcolors.to_hex(cmap(i / n_chars)) for i in range(vocab.n_chars)}
def plot_probabilities(probabilities, chars, index_to_color, current_smiles, step_num):
    plt.figure(figsize=(12, 6)); indices = np.arange(len(probabilities))
    bar_colors = [index_to_color.get(i, '#808080') for i in indices]
    plt.bar(indices, probabilities, color=bar_colors); plt.xlabel('Characters'); plt.ylabel('Probability')
    plt.title(f'Step {step_num}: Next Char Probabilities\nCurrent: {current_smiles}')
    display_indices = indices; display_chars = chars; max_ticks = 50
    if len(chars) > max_ticks: step = (len(chars) + max_ticks - 1) // max_ticks; display_indices = indices[::step]; display_chars = [chars[i] for i in display_indices]
    plt.xticks(display_indices, display_chars, rotation=90, fontsize=10); plt.ylim(0, 1.05); plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout(); plt.show(block=False); plt.pause(0.1); plt.close()


# --- Data Loading Utility ---
# --- REVISED load_smiles_data ---
def load_smiles_data(file_path: str) -> Optional[List[str]]:
    """Loads SMILES strings from a file (CSV or SMI). Assumes SMILES in first column."""
    smiles_data = []
    if not os.path.exists(file_path):
        print(f"Error: Data file not found: {file_path}")
        return None
    try:
        with open(file_path, 'r') as file:
            # Read first line to check for header
            try:
                first_line = next(file).strip()
                line_num = 1
            except StopIteration: # Handle empty file
                print(f"Warning: File {file_path} appears to be empty.")
                return []

            # Check if first line looks like a typical header
            is_header = first_line and any(h in first_line.lower() for h in ['smiles', 'molecule', 'id', 'name'])

            current_line: Optional[str] = first_line
            while current_line is not None:
                process_this_line = True
                if line_num == 1 and is_header:
                    process_this_line = False # Skip header

                if process_this_line:
                    try:
                        # Split by comma or whitespace, take first part
                        parts = current_line.strip().replace(',', ' ').split()
                        if parts: # Ensure parts is not empty
                            smiles = parts[0]
                            if smiles: # Ensure the extracted part is not empty
                                smiles_data.append(smiles)
                        # else: line was blank or malformed after split, ignore
                    except Exception as line_err:
                        # Optionally log the error for the specific line
                        # print(f"Warning: Error processing line {line_num} in {file_path}: {line_err}")
                        pass # Continue to the next line

                # Read the next line
                try:
                    current_line = next(file).strip()
                    line_num += 1
                except StopIteration:
                    current_line = None # End of file

    except Exception as e:
        print(f"Error reading data file {file_path}: {e}")
        return None # Return None on major read error like permissions

    if not smiles_data:
        print(f"Warning: No valid SMILES data loaded from {file_path}.")
        return [] # Return empty list if file exists but no data extracted

    print(f"Loaded {len(smiles_data)} SMILES strings from {file_path}.")
    return smiles_data
# --- End Data Loading Utility ---




# --- Vocabulary Creation Utility (Updated) ---
def create_vocabulary_from_data(smiles_list: List[str]):
    """Creates a Vocabulary object from a list of SMILES strings."""
    # Import Vocabulary class (local import is fine here)
    from vocabulary import SmilesVocabulary

    # 1. Initialize Vocabulary with a name
    #    The __init__ method adds PAD, SOS, EOS automatically.
    vocab = SmilesVocabulary("smiles_auto_vocab") # Provide a suitable name

    count = 0
    # 2. Add characters from each SMILES sequence using addSequence
    for s in smiles_list:
        if isinstance(s, str) and s: # Ensure it's a non-empty string
            vocab.addSequence(s)
            count += 1
        # else: Optional: print warning for non-string/empty item in list

    print(f"Vocabulary created from {count} non-empty sequences. Size: {vocab.n_chars}") # Size should now be > 3
    return vocab


# --- NEW HELPER FUNCTIONS ---
def parse_json_config(json_string: Optional[str], config_name: str = "config") -> Any:
    """Safely parses a JSON string from argparse arguments."""
    default_value = [] if config_name.endswith("s_config") or config_name.endswith("s") else {}
    if not json_string: return default_value
    try:
        corrected_json_string = json_string.replace("'", '"')
        config = json.loads(corrected_json_string)
        return config
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON for '{config_name}': {e}\nReceived string: {json_string}")
        return default_value
    except Exception as e:
        print(f"Unexpected error parsing JSON for '{config_name}': {e}")
        return default_value

def load_smarts_from_file(filepath: str) -> List[str]:
    """Loads SMARTS patterns from a file, one per line, ignoring comments ('#') and empty lines."""
    patterns = []
    if filepath and os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'): patterns.append(line)
        except Exception as e:
            print(f"Warning (utils.py): Failed to read SMARTS file {filepath}: {e}")
    return patterns
# --- End NEW Helper Functions ---


