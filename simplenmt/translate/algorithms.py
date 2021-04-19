import torch
import torch.functional as F
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
def beam_search_(src_tokens, beam_size=4):
    # init
    batch_size = src_tokens.size(0)

    # - gen_seqs: (B x beam_size, step)
    gen_seqs = torch.full((batch_size * beam_size, 1), bos)

    # 记录每个batch中beam个最高的概率，初始化为这个样子的原因是：一开始全是bos，只需要取一条即可
    beam_scores = torch.tensor([0.0] + [float("-inf")] * (beam_size - 1)).repeat(batch_size)

    encoder_out, src_mask = f_enc(src_tokens) # mask 可以广播，不用repeat
    # - encoder_out: (batch_size, src_len, d_model)
    encoder_outs = encoder_out.repeat_interleave(beam_size, 1, 1)
    # - encoder_outs: (batch_size * beam_size, src_len, d_model)
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

        next_indices = next_tokens // vocab_size
        next_tokens = next_tokens % vocab_size
        


    return decoded 