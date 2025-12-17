import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from utils import PAD_token

class SmilesRNN(nn.Module):
    """Recurrent Neural Network model for SMILES generation."""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.vocab_size = vocab_size 
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_token)
        lstm_dropout = dropout if num_layers > 1 else 0
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=lstm_dropout)
        self.dropout_layer = nn.Dropout(dropout)
        self.policy_head = nn.Linear(hidden_dim, vocab_size) 

    def forward(self, input_seq, lengths, hidden=None):
        """Forward pass through the RNN."""
        batch_size = input_seq.size(0)
        seq_len = input_seq.size(1)

        lengths_cpu = lengths.cpu()

        if hidden is None:
            hidden = self.initHidden(batch_size, input_seq.device)

        try:
            embedded = self.embedding(input_seq)
            embedded = self.dropout_layer(embedded)

            packed_embedded = pack_padded_sequence(embedded, lengths_cpu, batch_first=True, enforce_sorted=False)

            packed_output, hidden = self.lstm(packed_embedded, hidden)

            output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=seq_len)

            output = self.dropout_layer(output)

            policy_logits = self.policy_head(output) 

            return policy_logits, hidden

        except Exception as e:
             print(f"Error during SmilesRNN forward pass: {e}")
             print(f"Input shape: {input_seq.shape}, Lengths: {lengths_cpu.tolist()}")
             dummy_logits = torch.zeros(batch_size, seq_len, self.vocab_size, device=input_seq.device)
             if 'hidden' not in locals(): hidden = self.initHidden(batch_size, input_seq.device)
             return dummy_logits, hidden

    def initHidden(self, batch_size, device):
        """Initializes hidden state tensors."""
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device))



