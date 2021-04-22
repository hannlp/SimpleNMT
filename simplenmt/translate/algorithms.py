import torch
import torch.nn.functional as F

'''
Translate algorithms, whitch supprot a batch input - src_tokens: (batch_size, src_len):
    - greedy search
    - beam search
'''

MAX_LENGTH = 256
BOS = -1
EOS = -2
PAD = -3
f_enc = None
f_dec = None
# TODO:!!! 把算法和translator分离出来

def f_enc(model, src_tokens, pad):
    # for Transformer's encode
    src_mask = src_tokens.eq(pad)
    encoder_out = model.encoder(src_tokens, src_mask)
    return encoder_out, src_mask

def f_dec(model, prev_tgt_tokens, src_enc, src_mask):
    # for Transformer's decode
    decoder_out = model.decoder(
        prev_tgt_tokens, src_enc, src_mask, None)
    decoder_out = decoder_out[:, -1, :] # get last token
    model_out = model.out_vocab_proj(decoder_out)
    return model_out

def greedy_search(model, src_tokens, max_len=MAX_LENGTH, bos=BOS, eos=EOS, pad=PAD):
    batch_size = len(src_tokens)
    done = src_tokens.new([False] * batch_size)

    encoder_out, src_mask = f_enc(model, src_tokens, pad)

    gen_seqs = src_tokens.new(batch_size, max_len).fill_(pad)
    gen_seqs[:, 0] = bos
    # - gen_seqs: (batch_size, max_len)
    
    for step in range(1, max_len):
        probs = F.log_softmax(f_dec(model, gen_seqs[:, :step], encoder_out, src_mask), dim=-1)
        _, next_words = probs.topk(1)
        
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

def beam_search(model, src_tokens, beam_size, length_penalty, max_len=MAX_LENGTH, bos=BOS, eos=EOS, pad=PAD):
    # batch size
    batch_size = len(src_tokens)

    src_enc, src_mask = f_enc(model, src_tokens, pad)

    # expand to beam size the source latent representations
    src_enc = src_enc.repeat_interleave(beam_size, dim=0)
    src_mask = src_mask.repeat_interleave(beam_size, dim=0)

    # generated sentences (batch with beam current hypotheses)
    generated = src_tokens.new(batch_size * beam_size, max_len).fill_(pad)  # upcoming output
    generated[:, 0].fill_(bos)

    # generated hypotheses
    generated_hyps = [BeamHypotheses(n_hyp=beam_size, length_penalty=length_penalty) for _ in range(batch_size)]

    # scores for each sentence in the beam
    beam_scores = src_enc.new(batch_size, beam_size).fill_(0)
    beam_scores[:, 1:] = -1e9

    # current position
    cur_len = 1

    # done sentences
    done = [False] * batch_size

    while cur_len < max_len:

        # compute word scores
        model_out = f_dec(model, generated[:, :cur_len], src_enc, src_mask) # log softmax
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
                next_batch_beam.extend([(0, pad, 0)] * beam_size)  # pad the batch
                continue

            # next sentence beam content
            next_sent_beam = []

            # next words for this sentence
            for idx, value in zip(next_words[sent_id], next_scores[sent_id]):

                # get beam and word IDs
                beam_id = idx // n_tgt_words
                word_id = idx % n_tgt_words

                # end of sentence, or next word
                if word_id == eos or cur_len + 1 == max_len:
                    generated_hyps[sent_id].add(generated[sent_id * beam_size + beam_id, :cur_len].clone(), value.item())
                else:
                    next_sent_beam.append((value, word_id, sent_id * beam_size + beam_id))

                # the beam for next step is full
                if len(next_sent_beam) == beam_size:
                    break

            # update next beam content
            if len(next_sent_beam) == 0:
                next_sent_beam = [(0, pad, 0)] * beam_size  # pad the batch
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

        # TODO: 优化beam search停止时间
        # if cur_len % 5 == 0:
        #     print(cur_len, done)
        #     try:
        #         for i in range(5):
        #             print()
        #         print(len(generated_hyps[0].hyp[0][1]), generated_hyps[0].hyp[0])
        #         print(len(generated_hyps[1].hyp[0][1]), generated_hyps[1].hyp[0])
        #         print(len(generated_hyps[2].hyp[0][1]), generated_hyps[2].hyp[0])
        #         print(len(generated_hyps[3].hyp[0][1]), generated_hyps[3].hyp[0])
        #     except:
        #         print("haven't generate any sentence!")

        # stop when we are done with each sentence
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
    decoded = src_tokens.new(batch_size, tgt_len.max().item()).fill_(pad)
    for i, hypo in enumerate(best):
        decoded[i, :tgt_len[i] - 1] = hypo
        decoded[i, tgt_len[i] - 1] = eos

    return decoded, tgt_len


"""
Referenced from OpenNMT(unfinished)
"""
def beam_search_(self, src_tokens, beam_size=4):
    # init
    batch_size = src_tokens.size(0)

    # - gen_seqs: (B x beam_size, step)
    gen_seqs = torch.full((batch_size * beam_size, 1), bos)
    done = torch.tensor([False] * (batch_size * beam_size))
    best_scores = torch.full([batch_size], -1e10)
    _beam_offset = torch.arange(0, batch_size * beam_size, step=beam_size)
    # beam偏移：(0, beam, 2*beam, ...) 一共有batch_size个beam _beam_offset[:batch_size]
    # 例子: offset = torch.arange(0, 6 * 5, step=5): tensor([ 0,  5, 10, 15, 20, 25])
    # offset[:5]: tensor([ 0,  5, 10, 15, 20]), offset[:6]: tensor([ 0,  5, 10, 15, 20, 25])

    # 记录每个batch中beam个最高的概率，初始化为这个样子的原因是：一开始每个句子的输入是beam个bos，只需要从其中的一个bos计算topk，不然会重复
    topk_log_probs = torch.tensor([0.0] + [float("-inf")] * (beam_size - 1)).repeat(batch_size)
    
    # buffers for the topk scores and 'backpointer'
    topk_scores, topk_ids, _batch_index = [torch.empty((batch_size, beam_size))] * 3

    encoder_out, src_mask = f_enc(src_tokens)
    # - encoder_out: (batch_size, src_len, d_model)
    encoder_outs = encoder_out.repeat(beam_size, 1, 1)
    # - encoder_outs: (batch_size * beam_size, src_len, d_model)

    for step in range(2, MAX_LENGTH):
        
        #decoder_in = gen_seqs[:, -1].view(-1, 1, 1)
        decoder_in = gen_seqs

        # Decoder forward, takes [tgt_len, batch, nfeats] as input
        model_out = f_dec(decoder_in, encoder_outs, src_mask) # log softmax
        # - model_out: (batch_size * beam_size, vocab_size)

        log_probs = model_out

        B_times_beam, vocab_size = log_probs.shape
        # using integer division to get an integer _B without casting
        _B = B_times_beam // beam_size

        # Multiply probs by the beam probability.
        log_probs += topk_log_probs.view(_B * beam_size, 1) # 此处有广播，topk_log_probs (B * beam, 1) -> (B * beam, vocab_size)
        # 形式是：每个生成的beam中的vocab都会加一遍topk_log_probs
        # - log_probs: (batch_size * beam_size, vocab_size)

        # May be step length penalty
        # ...
        curr_scores = log_probs

        # Pick up candidate token by curr_scores
        
        curr_scores = curr_scores.reshape(-1, beam_size * vocab_size) # Flatten probs into a list of possibilities.
        # 这里的reshape，会按照行优先的顺序，所以这里每一行还是一个句子，一共有batch_size行
        # 每行的形式为: (0+log_p[0], 0+log_p[1], ..., 0+log_p[-1], -inf+log_p[0], -inf+log_p[1],..., -inf+log_p[-1])
        # 后面含有-inf+词表log_p的项，会重复beam_size-1次

        # 在这里已经选出了每一行中概率最大的beam_size个词的索引
        topk_scores, topk_ids = torch.topk(curr_scores, beam_size, dim=-1)
        # - topk_scores, topk_ids: (batch_size, beam_size)

        # Resolve beam origin and map to batch index flat representation.
        _batch_index = topk_ids // vocab_size # 找到索引所在的batch, (batch_size, beam_size(这里的id本来是beam * vocab))
        _batch_index += _beam_offset[:_B].unsqueeze(1) # (0, beam, 2*beam, ...), (1)
        select_indices = _batch_index.view(_B * beam_size)
        topk_ids.fmod_(vocab_size)  # resolve true word ids

        # Append last prediction.
        gen_seqs = torch.cat(
            [gen_seqs.index_select(0, select_indices),
             topk_ids.view(_B * beam_size, 1)], -1)
        
    return gen_seqs

"""
Referenced from attention-is-all-you-need-pytorch(unfinished)
 at https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Translator.py

Only can translate one sentence, and have a lot of bugs
"""
def beam_search__(self, word_list, beam_size=8):
    len_map = torch.arange(1, self.max_seq_len + 1, dtype=torch.long).unsqueeze(0).to(self.device)
    
    def decode(prev_tgt_tokens, encoder_out, src_mask):
        tgt_mask = (prev_tgt_tokens != self.tgt_pdx).to(self.device)
        decoder_out = F.softmax(self.model.decoder(prev_tgt_tokens, encoder_out, src_mask, tgt_mask), dim=-1)
        return decoder_out

    gen_seqs = torch.full((beam_size, 1), self.tgt_sos_idx).to(self.device)

    with torch.no_grad():
        src_tokens = torch.tensor([self.src_stoi[s] for s in word_list]).unsqueeze(0).to(self.device)
        src_mask = (src_tokens != self.src_pdx).to(self.device)

        encoder_out = self.model.encoder(src_tokens, src_mask)

        prev_tgt_tokens = torch.tensor([[self.tgt_pdx]]).to(self.device) # <sos>
        decoder_out = decode(prev_tgt_tokens, encoder_out, src_mask)

        best_k_probs, best_k_idx = decoder_out[:, -1, :].topk(beam_size)
        log_scores = torch.log(best_k_probs).view(beam_size) #-1也行

        gen_seqs = torch.cat((gen_seqs, best_k_idx.transpose(0, 1)), dim=1)

        encoder_out = encoder_out.repeat(beam_size, 1, 1)
        for step in range(2, self.max_seq_len):
            # 这里存在一个src_mask的广播
            decoder_out = decode(gen_seqs, encoder_out, src_mask)

            best_k2_probs, best_k2_idx = decoder_out[:, -1, :].topk(beam_size)

            log_scores_2 = torch.log(best_k2_probs).view(beam_size, -1) + log_scores.view(beam_size, 1)

            log_scores, best_k_idx_in_k2 = log_scores_2.view(-1).topk(beam_size)

            best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
            best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]

            gen_seqs = torch.cat((gen_seqs, best_k_idx.unsqueeze(1)), dim=1)

            eos_locs = gen_seqs == self.tgt_eos_idx
            seq_lens, _ = len_map[:, :eos_locs.size(1)].masked_fill(~eos_locs, self.max_seq_len).min(1)

            if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:
                #_, ans_idx = log_scores.max(0)
                _, ans_idx = log_scores.div(seq_lens.type(torch.float) ** 0.005).max(0)
                ans_idx = ans_idx.item()
                break
    
    return ''.join([self.tgt_itos[s] for s in gen_seqs[ans_idx, :seq_lens[ans_idx]]])