def prepare_batch(batch, CUDA_OK=False):
    src_tokens = batch.src.transpose(0, 1)
    prev_tgt_tokens = batch.trg.transpose(0, 1)[:, :-1]
    tgt_tokens = batch.trg.transpose(0, 1)[:, 1:]
    if CUDA_OK:
        src_tokens, prev_tgt_tokens, tgt_tokens = src_tokens.cuda(
        ), prev_tgt_tokens.cuda(), tgt_tokens.cuda()
    return src_tokens, prev_tgt_tokens, tgt_tokens