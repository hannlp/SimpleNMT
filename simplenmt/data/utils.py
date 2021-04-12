
def prepare_batch(batch, use_cuda=False):
    # when batch_first == True
    src_tokens = batch.src
    prev_tgt_tokens = batch.trg[:, :-1]
    tgt_tokens = batch.trg[:, 1:]

    if use_cuda:
        src_tokens, prev_tgt_tokens, tgt_tokens = src_tokens.cuda(
        ), prev_tgt_tokens.cuda(), tgt_tokens.cuda()
    return src_tokens, prev_tgt_tokens, tgt_tokens