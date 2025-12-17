# sampler.py
import logging
import matplotlib.pyplot as plt 
import numpy as np
import os
import pandas as pd
import torch

from model_definition import SmilesRNN
from vocabulary import SmilesVocabulary, load_vocabulary
from utils import SOS_char, SOS_token, EOS_token, PAD_token, get_color_map, plot_probabilities, RDKIT_AVAILABLE # TODO rework this rdkit avail shizzl

logger = logging.getLogger(__name__)

@torch.no_grad() 
def sample_smiles(model: SmilesRNN, vocab: SmilesVocabulary, device,
                  max_len=120, temperature=1.0, start_char=SOS_char,
                  batch_size=1, visualize=False, index_to_color=None,
                  sampling_mode='multinomial', top_k=0, top_p=0.0):
    """
    Generates SMILES sequences from the model using various sampling strategies.
    """
    model.eval() 
    generated_smiles_list = [""] * batch_size 

    start_char_idx = vocab.char2index.get(start_char)
    if start_char_idx is None:
         logger.error(f"Error: Start character '{start_char}' not in vocabulary.")
         return [""] * batch_size 

    current_char_idx = torch.full((batch_size, 1), start_char_idx, dtype=torch.long, device=device)
    hidden = model.initHidden(batch_size, device)

    sequences = current_char_idx.clone() 

    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    all_chars = [vocab.index2char.get(i, '?') for i in range(vocab.n_chars)] if visualize else None

    for step in range(max_len):
        if finished.all(): break
        lengths = torch.ones(batch_size, dtype=torch.long) 
        policy_logits, hidden = model(current_char_idx, lengths, hidden)
        output_logits = policy_logits.squeeze(1) 

        if sampling_mode != 'greedy':
            temp = max(temperature, 1e-8) 
            output_logits_temp = output_logits / temp
        else:
            output_logits_temp = output_logits 
        probabilities = torch.softmax(output_logits_temp, dim=1) 

        next_char_idx = None 

        if sampling_mode == 'greedy':
            next_char_idx = torch.argmax(output_logits, dim=1, keepdim=True)

        elif sampling_mode == 'top_k' and top_k > 0:
            k = min(top_k, probabilities.size(-1)) 
            top_k_probs, top_k_indices = torch.topk(probabilities, k, dim=1)
            top_k_probs_norm = top_k_probs / torch.sum(top_k_probs, dim=1, keepdim=True).clamp(min=1e-10)
            sampled_relative_indices = torch.multinomial(top_k_probs_norm, 1)
            next_char_idx = torch.gather(top_k_indices, 1, sampled_relative_indices)

        elif sampling_mode == 'top_p' and top_p > 0.0:
            sorted_probs, sorted_indices = torch.sort(probabilities, dim=-1, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0 
            indices_to_remove = torch.zeros_like(probabilities, dtype=torch.bool).scatter_(1, sorted_indices, sorted_indices_to_remove)
            probs_filtered = probabilities.masked_fill(indices_to_remove, 0.0)
            probs_sum = torch.sum(probs_filtered, dim=1, keepdim=True)
            probs_filtered_norm = probs_filtered / probs_sum.clamp(min=1e-10)
            next_char_idx = torch.multinomial(probs_filtered_norm, 1)

        else: 
            next_char_idx = torch.multinomial(probabilities, 1)

        if visualize and batch_size == 1 and not finished[0]:
             last_chosen_idx_item = next_char_idx[0].item()
             if last_chosen_idx_item not in [EOS_token, SOS_token, PAD_token]:
                  generated_smiles_list[0] += vocab.index2char.get(last_chosen_idx_item, "?")

             if index_to_color: 
                 try:
                     probs_np = probabilities[0].cpu().numpy()
                     plot_probabilities(probs_np, all_chars, index_to_color, generated_smiles_list[0], step + 1)
                 except Exception as e:
                      logger.warning(f"Warning: Error during visualization plotting: {e}")
                      visualize = False 

        sequences = torch.cat([sequences, next_char_idx], dim=1)
        current_char_idx = next_char_idx

        just_finished = (next_char_idx.squeeze(1) == EOS_token) & (~finished)
        finished |= just_finished

    final_smiles = []
    sequences_np = sequences.cpu().numpy() 

    for i in range(batch_size):
        seq_indices = sequences_np[i, 1:] 
        smiles = ""
        for idx in seq_indices:
            if idx == EOS_token: break 
            if idx == PAD_token: break 
            smiles += vocab.index2char.get(idx, "?") 
        final_smiles.append(smiles) 

    return final_smiles


def sample_molecules(args, device):
    """Handles the overall molecule sampling process."""
    logger.info("\n--- Running Sampling Mode ---")

    vocab = load_vocabulary(args.vocab_path)
    if vocab is None: logger.warning(f"ERROR: Vocabulary not found at {args.vocab_path}. Cannot sample."); return

    index_to_color = None
    visualize = getattr(args, 'visualize', False)
    if visualize:
        try: index_to_color = get_color_map(vocab) 
        except Exception as e: logger.warning(f"Warning: could not get color map for visualization: {e}")

    try:
        model = SmilesRNN(vocab.n_chars, args.embedding_dim, args.hidden_dim, args.num_layers, args.dropout).to(device) 
    except Exception as e:
        logging.critical(f"Error initializing model: {e}"); return


    load_path = getattr(args, 'load_agent_model_path', None) 
    if load_path is None:
        load_path = getattr(args, 'supervised_model_path', None) 

    if not load_path or not os.path.exists(load_path):
        logger.critical(f"ERROR: Model weights file not found at specified/default path ('{load_path}'). Cannot sample.")
        return
    
    logger.info(f"Loading model weights for sampling from: {load_path}")

    try:
        loaded_data = torch.load(load_path, map_location=device, weights_only=False)
        state_dict = loaded_data.get('model_state_dict', loaded_data) 

        if isinstance(loaded_data, dict) and 'vocab_size' in loaded_data:
            if model.vocab_size != loaded_data['vocab_size']:
                 logger.critical(f"ERROR: Vocab size mismatch! Loaded model: {loaded_data['vocab_size']}, Current vocab: {model.vocab_size}")
                 return

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys: logger.warning(f"Warning: Missing keys when loading sampling weights: {missing_keys}")
        if unexpected_keys: logger.warning(f"Warning: Unexpected keys when loading sampling weights: {unexpected_keys}")
    except Exception as e:
        logger.critical(f"Error loading weights from {load_path}: {e}")
        return

    model.eval() 

    num_samples_to_gen = getattr(args, 'num_samples', 100)
    sampling_mode = getattr(args, 'sampling_mode', 'multinomial')
    temperature = getattr(args, 'temperature', 1.0)
    top_k = getattr(args, 'top_k', 0)
    top_p = getattr(args, 'top_p', 0.0)
    max_gen_len = getattr(args, 'max_gen_len', 120)
    sample_batch_size = getattr(args, 'sample_batch_size', 50)

    logger.info(f"\nGenerating {num_samples_to_gen} samples using {sampling_mode} sampling...")
    logger.info(f"Settings: Temp={temperature}, Top-k={top_k}, Top-p={top_p}, MaxLen={max_gen_len}")

    all_generated_samples = []
    visualize_this_run = visualize 

    while len(all_generated_samples) < num_samples_to_gen:
        remaining = num_samples_to_gen - len(all_generated_samples)
        current_batch_size = 1 if visualize_this_run else min(sample_batch_size, remaining)
        batch_visualize = visualize_this_run and current_batch_size == 1

        samples_batch = sample_smiles(model, vocab, device, max_gen_len, temperature,
                                      batch_size=current_batch_size,
                                      visualize=batch_visualize, index_to_color=index_to_color,
                                      sampling_mode=sampling_mode, top_k=top_k, top_p=top_p)
        all_generated_samples.extend(samples_batch)

        if visualize_this_run:
            visualize_this_run = False



    output_file = getattr(args, 'output_samples_file', None) 
    results_dir = getattr(args, 'results_dir', 'results/sampling') 
    os.makedirs(results_dir, exist_ok=True) 

    logger.info("-" * 30)
    logger.info(f"Finished generation. Total samples generated: {len(all_generated_samples)}")

    non_empty_samples = [smi for smi in all_generated_samples if isinstance(smi, str) and smi]
    logger.info(f"Number of non-empty SMILES: {len(non_empty_samples)}")


    valid_smiles = []
    if RDKIT_AVAILABLE:  
         from rdkit import Chem 
         valid_count = 0
         for smi in non_empty_samples:
              mol = Chem.MolFromSmiles(smi)
              if mol is not None:
                   valid_count += 1
                   valid_smiles.append(smi) 
         validity_rate = valid_count / len(non_empty_samples) if non_empty_samples else 0
         logger.info(f"RDKit Validity Check: {valid_count} / {len(non_empty_samples)} ({validity_rate:.2%}) non-empty SMILES are valid.")
    else:
        logger.error("RDKit not available, skipping validity check.")
        valid_smiles = non_empty_samples 


    if output_file:
        output_path = os.path.join(results_dir, os.path.basename(output_file))
        logger.info(f"Saving {len(valid_smiles)} valid/non-empty samples to {output_path}...")
        try:
            with open(output_path, 'w') as f:
                for smi in valid_smiles:
                     f.write(smi + '\n')
            logger.info(f"Successfully saved samples.")
        except Exception as e:
            logger.error(f"ERROR: Failed to write samples to {output_path}: {e}")
    else:
        logger.info("Generated valid/non-empty samples (up to 20):")
        for i, smi in enumerate(valid_smiles[:20]):
             logger.info(f"  {i+1}: {smi}")
        if len(valid_smiles) > 20: logger.info("  ...")
    logger.info("-" * 30)

    return pd.DataFrame(valid_smiles, columns=['Smiles'])

