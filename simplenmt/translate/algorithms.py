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

def beam_search(self, src_tokens, beam_size=4):
    # init
    bos = -1
    MAX_LENGTH = 512
    batch_size = src_tokens.size(0)

    # - gen_seqs: (B x beam_size, step)
    gen_seqs = torch.full((batch_size * beam_size, 1), bos)
    done = torch.tensor([False] * (batch_size * beam_size))
    best_scores = torch.full([batch_size], -1e10)
    _beam_offset = torch.arange(0, batch_size * beam_size, step=beam_size)
    topk_log_probs = torch.tensor([0.0] + [float("-inf")] * (beam_size - 1)).repeat(self.batch_size)
    
    # buffers for the topk scores and 'backpointer'
    topk_scores = torch.empty((self.batch_size, self.beam_size))
    topk_ids = torch.empty((self.batch_size, self.beam_size))
    _batch_index = torch.empty([self.batch_size, self.beam_size])

    encoder_out, src_mask = f_enc(src_tokens) # mask 可以广播，不用repeat
    # - encoder_out: (batch_size, src_len, d_model)
    encoder_outs = encoder_out.repeat(beam_size, 1, 1)
    # - encoder_outs: (batch_size * beam_size, src_len, d_model)

    for step in range(2, MAX_LENGTH):
        
        #decoder_in = gen_seqs[:, -1].view(-1, 1, 1)
        decoder_in = gen_seqs

        # Decoder forward, takes [tgt_len, batch, nfeats] as input
        model_out = f_dec(decoder_in, encoder_outs, src_mask) # log softmax
        # - model_out: (batch_size * beam_size, n_tgt_words)

        log_probs = model_out

        B_times_beam, vocab_size = log_probs.shape
        # using integer division to get an integer _B without casting
        _B = B_times_beam // beam_size

        # Multiply probs by the beam probability.
        log_probs += topk_log_probs.view(_B * self.beam_size, 1)

        # May be length penalty
        curr_scores = log_probs

        # Pick up candidate token by curr_scores
        # Flatten probs into a list of possibilities.
        curr_scores = curr_scores.reshape(-1, self.beam_size * vocab_size)
        topk_scores, topk_ids = torch.topk(curr_scores, self.beam_size, dim=-1)

        
        pass