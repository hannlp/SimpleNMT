import torch
import torch.functional as F
'''
translate algorithms, whitch supprot a batch src_tokens input:
    - greedy search
'''

# TODO:!!! 把算法和translator分离出来

def greedy_search(self, src_tokens, f_enc, f_dec, device):
    batch_size = src_tokens.size(0)
    done = torch.tensor([False] * batch_size).to(device)
    
    encoder_out, src_mask = f_enc(src_tokens)

    gen_seqs = torch.full((batch_size, 1), self.tgt_sos_idx).to(device)
    # - gen_seqs: (batch_size, 1) -> <sos>

    probs = F.softmax(self._decode(gen_seqs, encoder_out, src_mask), dim=-1) # TODO: use log_softmax
    _, max_idxs = probs.topk(1) # new words
    
    for step in range(2, self.max_seq_length):           
        done = done | max_idxs.eq(self.tgt_eos_idx).squeeze() #TODO : stop rules
        if all(done):
            break
        
        gen_seqs = torch.cat((gen_seqs, max_idxs.to(device)), dim=1)
        # - gen_seqs: (batch_size, step) -> batch seqs

        probs = F.softmax(self._decode(gen_seqs, encoder_out, src_mask), dim=-1)
        _, max_idxs = probs.topk(1)
    
    return gen_seqs

def beam_search(self, src_tokens):
    
    pass