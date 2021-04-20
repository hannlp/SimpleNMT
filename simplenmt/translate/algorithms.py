import torch
import torch.nn.functional as F
'''
translate algorithms, whitch supprot a batch src_tokens input:
    - greedy search
'''
MAX_LENGTH = 512
bos = -1
eos = -2
pad = -3
f_enc = None
f_dec = None
# TODO:!!! 把算法和translator分离出来

def f_enc(model, src_tokens, pad):
    src_mask = src_tokens.eq(pad)
    encoder_out = model.encoder(src_tokens, src_mask)
    return encoder_out, src_mask

def f_dec(model, prev_tgt_tokens, src_enc, src_mask):
    decoder_out = model.decoder(
        prev_tgt_tokens, src_enc, src_mask, None)
    decoder_out = decoder_out[:,-1,:] # get last token
    model_out = model.out_vocab_proj(decoder_out)
    return model_out

def greedy_search(self, src_tokens):
    batch_size = src_tokens.size(0)
    done = torch.tensor([False] * batch_size)
    
    encoder_out, src_mask = f_enc(src_tokens)

    gen_seqs = torch.full((batch_size, 1), bos)
    # - gen_seqs: (batch_size, 1) -> <sos>

    probs = F.softmax(f_dec(gen_seqs, encoder_out, src_mask), dim=-1) # TODO: use log_softmax
    _, max_idxs = probs.topk(1) # new words
    
    for step in range(2, MAX_LENGTH):
        done = done | max_idxs.eq(eos).squeeze() #TODO : stop rules
        if all(done):
            break
        
        gen_seqs = torch.cat((gen_seqs, max_idxs), dim=1)
        # - gen_seqs: (batch_size, step) -> batch seqs

        probs = F.softmax(f_dec(gen_seqs, encoder_out, src_mask), dim=-1)
        _, max_idxs = probs.topk(1)
    
    return gen_seqs


# 参考OpenNMT的实现
def beam_search(self, src_tokens, beam_size=4):
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

    # 记录每个batch中beam个最高的概率，初始化为这个样子的原因是：一开始全是bos，只需要取一条即可
    topk_log_probs = torch.tensor([0.0] + [float("-inf")] * (beam_size - 1)).repeat(batch_size)
    
    # buffers for the topk scores and 'backpointer'
    topk_scores, topk_ids, _batch_index = [torch.empty((batch_size, beam_size))] * 3

    encoder_out, src_mask = f_enc(src_tokens) # mask 可以广播，不用repeat
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

# 参考transformers库的实现
def beam_search_(src_tokens, beam_scorer, beam_size=4):
    # init
    batch_size = src_tokens.size(0)

    # - gen_seqs: (B x beam_size, step)
    gen_seqs = torch.full((batch_size * beam_size, 1), bos)

    # 记录每个batch中beam个最高的概率，初始化为这个样子的原因是：一开始全是bos，只需要取一条即可
    beam_scores = torch.tensor([0.0] + [float("-inf")] * (beam_size - 1)).repeat(batch_size)

    encoder_out, src_mask = f_enc(src_tokens)
    encoder_outs = encoder_out.repeat_interleave(beam_size, 1, 1)
    # - encoder_out: (batch_size, src_len, d_model), encoder_outs: (batch_size * beam_size, src_len, d_model) 
    src_mask = encoder_outs.eq(pad)

    for step in range(2, MAX_LENGTH):
        model_out = f_dec(gen_seqs, encoder_outs, src_mask) # log softmax
        # - model_out: (batch_size * beam_size, vocab_size)
        
        next_token_scores = F.log_softmax(model_out, dim=-1)
        next_token_scores = next_token_scores + beam_scores.view(-1, 1).expand_as(next_token_scores)

        # reshape for beam search
        vocab_size = model_out.size(-1)
        next_token_scores = next_token_scores.view(batch_size, beam_size * vocab_size)
        next_token_scores, next_tokens = torch.topk(
            next_token_scores, 2 * beam_size, dim=1, largest=True, sorted=True
        )
        # - next_token_scores, next_tokens: (batch_size, beam_size[0~beam_size*vocab_size])

        next_indices = next_tokens // vocab_size # 应该是索引的每个batch中的第几个beam
        next_tokens = next_tokens % vocab_size
        
        # 更新上面三个变量
        beam_scores, beam_next_tokens, beam_idx = beam_scorer.process(
            gen_seqs, next_token_scores, next_tokens, next_indices
        )

        gen_seqs = torch.cat([gen_seqs[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

        if beam_scorer.is_done:
            break
    
    pred_tokens = beam_scorer.finalize(
        gen_seqs, beam_scores, next_tokens, next_indices
    )

    return decoded


'''
class BeamHypotheses:
    def __init__(self, beam_size, length_penalty=1.0) -> None:
        self.beam_size = beam_size
        self.length_penalty = length_penalty
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        return len(self.beams)

    def add(self):
        pass
'''
class BeamScorer:
    def __init__(self, batch_size, beam_size, length_penalty) -> None:
        self.beam_size = beam_size
        self._done = torch.tensor([False] * batch_size)
        self._beam_hyps = [
            BeamHypotheses(beam_size=beam_size, length_penalty=length_penalty)
            for _ in range(batch_size)]
        
    @property
    def is_done(self):
        return self._done.all()
    
    def process(self, gen_seqs, next_scores, next_tokens, next_indices):
        batch_size = len(self._beam_hyps)
        next_beam_scores = torch.zeros((batch_size, self.beam_size), dtype=next_scores.dtype)
        next_beam_tokens = torch.zeros((batch_size, self.beam_size), dtype=next_tokens.dtype)
        next_beam_indices = torch.zeros((batch_size, self.beam_size), dtype=next_indices.dtype)

        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            # 这个batch已经结束，则全部填充pad
            if self._done[batch_idx]:
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad
                next_beam_indices[batch_idx, :] = 0
                continue
            



# XLM的实现
def generate_beam(model, src_tokens, beam_size, length_penalty, max_len=256, bos=-1, eos=-2, pad=-3):
    """
    Decode a sentence given initial start.
    `x`:
        - LongTensor(bs, slen)
            <EOS> W1 W2 W3 <EOS> <PAD>
            <EOS> W1 W2 W3   W4  <EOS>
    `lengths`:
        - LongTensor(bs) [5, 6]
    `positions`: ?
        - False, for regular "arange" positions (LM)
        - True, to reset positions from the new generation (MT)

    """

    # check inputs
    assert beam_size >= 1

    # batch size
    bs = len(src_tokens)

    src_enc, src_mask = f_enc(model, src_tokens, pad)

    # expand to beam size the source latent representations
    src_enc = src_enc.repeat_interleave(beam_size, dim=0)
    #print(src_enc[0, :2, :2], src_enc[1, :2, :2], src_enc[2, :2, :2], src_enc[3, :2, :2])
    #print(src_enc[4, :2, :2], src_enc[5, :2, :2], src_enc[8, :2, :2], src_enc[9, :2, :2])
    src_mask = src_mask.repeat_interleave(beam_size, dim=0)

    # generated sentences (batch with beam current hypotheses)
    generated = src_tokens.new(bs * beam_size, max_len)  # upcoming output
    generated.fill_(pad)                   # fill upcoming ouput with <PAD>
    generated[:, 0].fill_(bos)                # we use <EOS> for <BOS> everywhere ????

    # generated hypotheses
    generated_hyps = [BeamHypotheses(beam_size, max_len, length_penalty) for _ in range(bs)]

    # scores for each sentence in the beam
    beam_scores = src_enc.new(bs, beam_size).fill_(0)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view(-1)

    # current position
    cur_len = 1

    # done sentences
    done = [False for _ in range(bs)]

    while cur_len < max_len:

        # compute word scores
        model_out = f_dec(model, generated[:, :cur_len], src_enc, src_mask) # log softmax
        # - model_out: (batch_size * beam_size, vocab_size)
        scores = F.log_softmax(model_out, dim=-1)       # (bs * beam_size, n_words)
        n_words = scores.size(-1)
        assert scores.size() == (bs * beam_size, n_words)

        # select next words with scores
        _scores = scores + beam_scores[:, None].expand_as(scores)  # (bs * beam_size, n_words)
        _scores = _scores.view(bs, beam_size * n_words)            # (bs, beam_size * n_words)

        next_scores, next_words = torch.topk(_scores, 2 * beam_size, dim=-1, largest=True, sorted=True)
        assert next_scores.size() == next_words.size() == (bs, 2 * beam_size)

        # next batch beam content
        # list of (bs * beam_size) tuple(next hypothesis score, next word, current position in the batch)
        next_batch_beam = []

        # for each sentence
        for sent_id in range(bs):

            # if we are done with this sentence
            done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(next_scores[sent_id].max().item())
            if done[sent_id]:
                next_batch_beam.extend([(0, pad, 0)] * beam_size)  # pad the batch
                continue

            # next sentence beam content
            next_sent_beam = []

            # next words for this sentence
            for idx, value in zip(next_words[sent_id], next_scores[sent_id]):

                # get beam and word IDs
                beam_id = idx // n_words
                word_id = idx % n_words

                # end of sentence, or next word
                if word_id == eos or cur_len + 1 == max_len:
                    generated_hyps[sent_id].add(generated[sent_id * beam_size + beam_id, :cur_len].clone(), value.item())
                else:
                    next_sent_beam.append((value, word_id, sent_id * beam_size + beam_id))

                # the beam for next step is full
                if len(next_sent_beam) == beam_size:
                    break

            # update next beam content
            assert len(next_sent_beam) == 0 if cur_len + 1 == max_len else beam_size
            if len(next_sent_beam) == 0:
                next_sent_beam = [(0, pad, 0)] * beam_size  # pad the batch
            next_batch_beam.extend(next_sent_beam)
            assert len(next_batch_beam) == beam_size * (sent_id + 1)

        # sanity check / prepare next batch
        assert len(next_batch_beam) == bs * beam_size
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_words = generated.new([x[1] for x in next_batch_beam])
        beam_idx = src_tokens.new([x[2] for x in next_batch_beam])

        # re-order batch and internal states
        generated = generated[beam_idx, :]
        generated[:, cur_len] = beam_words

        # update current length
        cur_len = cur_len + 1

        # stop when we are done with each sentence
        if all(done):
            break

    # select the best hypotheses
    tgt_len = src_tokens.new(bs)
    best = []

    for i, hypotheses in enumerate(generated_hyps):
        best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
        tgt_len[i] = len(best_hyp) + 1  # +1 for the <EOS> symbol
        best.append(best_hyp)

    #print(tgt_len, tgt_len.max(), tgt_len.shape)
    # generate target batch
    decoded = src_tokens.new(bs, tgt_len.max().item()).fill_(pad)
    for i, hypo in enumerate(best):
        decoded[i, :tgt_len[i] - 1] = hypo
        decoded[i, tgt_len[i] - 1] = eos

    # sanity check
    try:
        assert (decoded == eos).sum() == 2 * bs
    except:
        print('sanity check error!')

    return decoded, tgt_len


class BeamHypotheses(object):
    def __init__(self, n_hyp, max_len, length_penalty):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_len = max_len - 1  # ignoring <BOS>
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
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        else:
            return self.worst_score >= best_sum_logprobs / self.max_len ** self.length_penalty