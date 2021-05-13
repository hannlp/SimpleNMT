import torch
import torch.nn.functional as F

'''
Translate algorithms, whitch supprot a batch input.
    algorithms:
     - greedy search (when args.beam_size <= 0)
     - beam search (when args.beam_size > 0. Support to adjust 
                    these parameters: beam_size and length_penalty)
    
    inputs:
     - src_tokens: (batch_size, src_len)

    outputs:
     - gen_seqs: (batch_size, max_seq_len/tgt_len.max()) (related to the stop rules)
    
'''

MAX_SEQ_LEN, BOS, EOS, PAD = 256, -1, -2, -3

# for all model's encode
def f_encode(model, src_tokens, src_pdx):   
    src_mask = src_tokens.eq(src_pdx)
    enc_kwargs = {'src_tokens': src_tokens, 'src_mask': src_mask}
    encoder_out = model.encoder(**enc_kwargs)
    return encoder_out, src_mask

# for all model's decode
def f_decode(model, prev_tgt_tokens, encoder_out, src_mask, tgt_pdx):
    tgt_mask = prev_tgt_tokens.eq(tgt_pdx)
    dec_kwargs = {'prev_tgt_tokens':prev_tgt_tokens, 'encoder_out': encoder_out, 
                'src_mask': src_mask, 'tgt_mask': tgt_mask}
    decoder_out = model.decoder(**dec_kwargs)
    decoder_out = decoder_out[:, -1, :] # get last token
    model_out = model.out_vocab_proj(decoder_out)
    return model_out

def greedy_search(model, src_tokens, max_seq_len=MAX_SEQ_LEN, bos=BOS, eos=EOS, src_pdx=PAD, tgt_pdx=PAD):
    batch_size = len(src_tokens)
    done = src_tokens.new([False] * batch_size)

    encoder_out, src_mask = f_encode(model, src_tokens, src_pdx)

    gen_seqs = src_tokens.new(batch_size, max_seq_len).fill_(tgt_pdx)
    gen_seqs[:, 0] = bos
    # - gen_seqs: (batch_size, max_seq_len)
    
    for step in range(1, max_seq_len):
        log_probs = F.log_softmax(f_decode(model, gen_seqs[:, :step], encoder_out, src_mask, tgt_pdx), dim=-1)
        _, next_words = log_probs.topk(1)
        
        done = done | next_words.eq(eos).squeeze()
        if all(done):
            break

        gen_seqs[:, step] = next_words.view(-1)

    return gen_seqs

"""
Referenced from facebookresearch/XLM,
 at https://github.com/facebookresearch/XLM/blob/master/xlm/model/transformer.py
"""
class BeamHypotheses(object):
    def __init__(self, n_hyp, length_penalty):
        """
        Initialize n-best list of hypotheses.
        """
        self.length_penalty = length_penalty
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """

        lp = len(hyp) ** self.length_penalty # deafult length penalty
        #lp = (5 + len(hyp)) ** self.length_penalty / (5 + 1) ** self.length_penalty # Google GNMT's length penalty
        score = sum_logprobs / lp

        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]] # delete the worst hyp in beam
                self.worst_score = sorted_scores[1][0] # update worst score with the sencond worst hyp
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """

        if len(self) < self.n_hyp:
            return False
        else:
            return self.worst_score >= best_sum_logprobs / cur_len ** self.length_penalty

def beam_search(model, src_tokens, beam_size, length_penalty, max_seq_len=MAX_SEQ_LEN, bos=BOS, eos=EOS, src_pdx=PAD, tgt_pdx=PAD):
    # batch size
    batch_size = len(src_tokens)

    encoder_out, src_mask = f_encode(model, src_tokens, src_pdx)

    # expand to beam size the source latent representations
    encoder_out = encoder_out.repeat_interleave(beam_size, dim=0)
    src_mask = src_mask.repeat_interleave(beam_size, dim=0)

    # generated sentences (batch with beam current hypotheses)
    generated = src_tokens.new(batch_size * beam_size, max_seq_len).fill_(tgt_pdx)  # upcoming output
    generated[:, 0].fill_(bos)

    # generated hypotheses
    generated_hyps = [BeamHypotheses(n_hyp=beam_size, length_penalty=length_penalty) for _ in range(batch_size)]

    # scores for each sentence in the beam
    beam_scores = encoder_out.new(batch_size, beam_size).fill_(0)
    beam_scores[:, 1:] = -1e9

    # current position
    cur_len = 1

    # done sentences
    done = [False] * batch_size

    while cur_len < max_seq_len:

        # compute word scores
        model_out = f_decode(model, generated[:, :cur_len], encoder_out, src_mask, tgt_pdx) # log softmax
        # - model_out: (batch_size * beam_size, vocab_size)
        scores = F.log_softmax(model_out, dim=-1) # (batch_size * beam_size, n_tgt_words)      
        n_tgt_words = scores.size(-1)
        # - scores: (batch_size * beam_size, n_tgt_words)

        # select next words with scores
        _scores = scores + beam_scores.view(batch_size * beam_size, 1)  # (batch_size * beam_size, n_tgt_words)
        _scores = _scores.view(batch_size, beam_size * n_tgt_words)     # (batch_size, beam_size * n_tgt_words)

        next_scores, next_words = torch.topk(_scores, 2 * beam_size, dim=-1, largest=True, sorted=True)
        # - next_scores, next_words: (batch_size, 2 * beam_size)

        # next batch beam content
        next_batch_beam = []  # list of (batch_size * beam_size) tuple(next hypothesis score, next word, current position in the batch)

        # for each sentence
        for sent_id in range(batch_size):

            # if we are done with this sentence
            done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(next_scores[sent_id].max().item(), cur_len)
            if done[sent_id]:
                next_batch_beam.extend([(0, tgt_pdx, 0)] * beam_size)  # pad the batch
                continue

            # next sentence beam content
            next_sent_beam = []

            # next words for this sentence
            for idx, value in zip(next_words[sent_id], next_scores[sent_id]):

                # get beam and word IDs
                beam_id = idx // n_tgt_words
                word_id = idx % n_tgt_words

                # end of sentence, or next word
                if word_id == eos or cur_len + 1 == max_seq_len:
                    generated_hyps[sent_id].add(generated[sent_id * beam_size + beam_id, :cur_len].clone(), value.item())
                else:
                    next_sent_beam.append((value, word_id, sent_id * beam_size + beam_id))

                # the beam for next step is full
                if len(next_sent_beam) == beam_size:
                    break

            # update next beam content
            if len(next_sent_beam) == 0:
                next_sent_beam = [(0, tgt_pdx, 0)] * beam_size  # pad the batch
            next_batch_beam.extend(next_sent_beam)

        # prepare next batch
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_words = generated.new([x[1] for x in next_batch_beam])
        beam_idx = src_tokens.new([x[2] for x in next_batch_beam])

        # re-order batch and internal states
        generated = generated[beam_idx, :]
        generated[:, cur_len] = beam_words

        # update current length
        cur_len = cur_len + 1
        
        if all(done):
            break

    # select the best hypotheses
    tgt_len = src_tokens.new(batch_size)
    best = []

    for i, hypotheses in enumerate(generated_hyps):
        best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
        tgt_len[i] = len(best_hyp) + 1  # +1 for the <EOS> symbol
        best.append(best_hyp)

    # generate target batch
    gen_seqs = src_tokens.new(batch_size, tgt_len.max().item()).fill_(tgt_pdx)
    for i, hypo in enumerate(best):
        gen_seqs[i, :tgt_len[i] - 1] = hypo
        gen_seqs[i, tgt_len[i] - 1] = eos

    return gen_seqs, tgt_len