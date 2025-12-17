# datasets.py
import re
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter

from vocabulary import SmilesVocabulary 
from utils import PAD_token, SOS_token, EOS_token, SOS_char, EOS_char, load_smiles_data 

def indexesFromSequence(vocab, sequence):
    """Converts a SMILES sequence to a list of indices, including SOS/EOS tokens."""
    indices = []
    indices.append(SOS_token) 
    for char in sequence:
        index = vocab.char2index.get(char, -1) 
        if index != -1: 
             indices.append(index)

    indices.append(EOS_token) 
    return indices

class SmilesDataset(Dataset):
    """PyTorch Dataset for SMILES sequences."""
    def __init__(self, smiles_list, vocab):
        self.vocab = vocab
        self.sequences = []
        unknown_chars_encountered = Counter()
        skipped_count = 0

        for s in smiles_list:
            if not isinstance(s, str) or not s: 
                 continue
            
            tokens = self.vocab.regex.findall(s)
            if not tokens and s: 
                 skipped_count += 1
                 continue

            indices = [SOS_token]
            unknown_in_seq = False
            for token in tokens:
                index = self.vocab.char2index.get(token, -1)
                if index == -1:
                    unknown_chars_encountered[token] += 1
                    unknown_in_seq = True
                    break 
                indices.append(index)

            if unknown_in_seq:
                skipped_count += 1
                continue 

            indices.append(EOS_token)
            self.sequences.append(torch.tensor(indices, dtype=torch.long))

        if unknown_chars_encountered:
            print(f"Warning: Encountered unknown characters during Dataset creation: {dict(unknown_chars_encountered)}")
            print(f"         Skipped {skipped_count} sequences containing unknown or invalid entries.")
        elif skipped_count > 0:
             print(f"Warning: Skipped {skipped_count} invalid (e.g., empty) entries during Dataset creation.")


    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

def collate_fn(batch):
    """Collates sequences into padded batches for DataLoader."""

    batch = [item for item in batch if item is not None and len(item) > 1] 
    if not batch:
        return None, None, None 

    try:
        inputs = [seq[:-1] for seq in batch]
        targets = [seq[1:] for seq in batch]
    except Exception as e:
         print(f"Error creating inputs/targets in collate_fn: {e}")
         print(f"Problematic batch items (lengths): {[len(x) for x in batch]}")
         return None, None, None


    lengths = torch.tensor([len(seq) for seq in inputs], dtype=torch.long)


    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=PAD_token)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=PAD_token)

    if not torch.all(lengths > 0):
         print(f"Warning: collate_fn detected zero or negative lengths: {lengths.tolist()}")

    return inputs_padded, targets_padded, lengths

