
def prepare_batch(batch, use_cuda=False):

    # 注：如果Field里batch_first为True，这里就都不需要transpose
    src_tokens = batch.src.transpose(0, 1)
    prev_tgt_tokens = batch.trg.transpose(0, 1)[:, :-1]
    tgt_tokens = batch.trg.transpose(0, 1)[:, 1:]

    # 不过，为了简洁起见，Field还是不显式的添加batch_first=True参数了，在这里transpose就行了
    # src_tokens = batch.src
    # prev_tgt_tokens = batch.trg[:, :-1]
    # tgt_tokens = batch.trg[:, 1:]

    if use_cuda:
        src_tokens, prev_tgt_tokens, tgt_tokens = src_tokens.cuda(
        ), prev_tgt_tokens.cuda(), tgt_tokens.cuda()
    return src_tokens, prev_tgt_tokens, tgt_tokens