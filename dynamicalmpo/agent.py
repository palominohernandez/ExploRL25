import torch
import torch.nn.functional as F


from model_definition import SmilesRNN
from vocabulary import SmilesVocabulary
from utils import PAD_token, SOS_token, EOS_token

class SmilesGeneratorAgent:
    """Wraps the RNN model and handles trajectory generation and evaluation.""" 
    def __init__(self, model: SmilesRNN, vocab: SmilesVocabulary, device):
        self.model = model
        self.vocab = vocab
        self.device = device

    def generate_trajectories(self, batch_size, max_len):
        """Generates trajectories using the current policy, tracking gradients."""
        self.model.train() 

        sequences = torch.full((batch_size, max_len + 1), PAD_token, dtype=torch.long, device=self.device)
        log_probs = torch.zeros((batch_size, max_len), dtype=torch.float32, device=self.device)
        probabilities = torch.zeros((batch_size, max_len, self.model.vocab_size), dtype=torch.float32, device=self.device)
        actions = torch.full((batch_size, max_len), PAD_token, dtype=torch.long, device=self.device)
        actual_lengths = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        current_char_idx = torch.full((batch_size, 1), SOS_token, dtype=torch.long, device=self.device)
        sequences[:, 0] = SOS_token
        hidden = self.model.initHidden(batch_size, self.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for step in range(max_len):
            active_mask = ~finished
            if not active_mask.any(): break 

            lengths_step = torch.ones(batch_size, dtype=torch.long)
 
            policy_logits, hidden = self.model(current_char_idx, lengths_step, hidden)
            output_logits = policy_logits.squeeze(1) 


            probs_step = F.softmax(output_logits, dim=-1) 
            log_probs_all = F.log_softmax(output_logits/1.4, dim=-1) # TODO remove temp val 


            sampled_action = torch.multinomial(probs_step, 1) 

            action_log_prob = torch.gather(log_probs_all, 1, sampled_action).squeeze(1)

            if active_mask.any():
                 current_active_indices = active_mask.nonzero(as_tuple=True)[0]
                 current_sampled_action = sampled_action[active_mask].squeeze(1)
                 sequences[active_mask, step + 1] = current_sampled_action
                 actions[active_mask, step] = current_sampled_action
                 log_probs[active_mask, step] = action_log_prob[active_mask]
                 probabilities[active_mask, step, :] = probs_step[active_mask]
                 actual_lengths[active_mask] += 1


            just_finished_mask = (sampled_action.squeeze(1) == EOS_token) & active_mask
            finished |= just_finished_mask


            current_char_idx = sampled_action


        final_smiles = []
        with torch.no_grad():
            for i in range(batch_size):
                seq_len = actual_lengths[i].item()
                seq_indices = sequences[i, 1:seq_len+1].cpu().tolist() 
                smiles = ""
                for idx in seq_indices:
                    if idx == EOS_token: break
                    if idx == PAD_token: break 
                    smiles += self.vocab.index2char.get(idx, "?")
                final_smiles.append(smiles)


        return final_smiles, log_probs, probabilities, actions, actual_lengths


    @torch.no_grad() 
    def evaluate_log_probs(self, sequences, lengths):
        """
        Evaluates the log-probability of given action sequences under this agent's model.
        """
        self.model.eval() 
        batch_size, seq_len_incl_sos = sequences.shape
        
        max_len_actions = seq_len_incl_sos - 1
        if max_len_actions <= 0: 
             return torch.zeros((batch_size, 0), dtype=torch.float32, device=self.device)


       
        target_actions = sequences[:, 1:]


        hidden = self.model.initHidden(batch_size, self.device)

        eval_log_probs = torch.zeros((batch_size, max_len_actions), dtype=torch.float32, device=self.device)

        for step in range(max_len_actions):

            current_char_idx = sequences[:, step].unsqueeze(1) 
            lengths_step = torch.ones(batch_size, dtype=torch.long)

            policy_logits, hidden = self.model(current_char_idx, lengths_step, hidden)
            output_logits = policy_logits.squeeze(1) 

     
            log_probs_all = F.log_softmax(output_logits, dim=-1) 

   
            action_this_step = target_actions[:, step].unsqueeze(1) 
            action_this_step = action_this_step.clamp(min=0, max=self.model.vocab_size - 1)

            action_log_prob = torch.gather(log_probs_all, 1, action_this_step).squeeze(1) 

            eval_log_probs[:, step] = action_log_prob

 
        step_indices = torch.arange(max_len_actions, device=self.device).unsqueeze(0) 
        mask = (step_indices < lengths.unsqueeze(1)).float()
        eval_log_probs *= mask 

        return eval_log_probs

