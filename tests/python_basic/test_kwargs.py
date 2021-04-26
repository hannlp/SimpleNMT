
def forward(src_tokens, tgt_tokens, src_mask, tgt_mask=4, **kwargs):
    # 这种方法可以实现一个_encode()和一个_decode()适配多种模型
    print(src_tokens, tgt_tokens, src_mask, tgt_mask)
    print(kwargs)
    return kwargs

a = {'src_tokens':1, 'tgt_tokens':2, 'src_mask':3, 'srcc_mask':5, 'tgt_mask':6}
ref = forward(**a)
print(ref)

# for luong
def _encode(model, src_tokens, src_pdx):
    src_mask = src_tokens.eq(src_pdx)
    encoder_outs = model.encoder(src_tokens, src_mask)
    return encoder_outs, src_mask

def _decode(model, prev_tgt_tokens, encoder_out, src_mask, **kwargs):
    
    decoder_out = model.decoder(
        prev_tgt_tokens, encoder_out, src_mask)
    decoder_out = decoder_out[:,-1,:] # get last token
    model_out = model.out_vocab_proj(decoder_out)
    return model_out

# for transformer
def f_enc(model, src_tokens, src_pdx):
    # for Transformer's encode
    src_mask = src_tokens.eq(src_pdx)
    encoder_out = model.encoder(src_tokens, src_mask)
    return encoder_out, src_mask

def f_dec(model, prev_tgt_tokens, src_enc, src_mask, tgt_pdx):
    # for Transformer's decode
    tgt_mask = prev_tgt_tokens.eq(tgt_pdx)
    decoder_out = model.decoder(
        prev_tgt_tokens, src_enc, src_mask, tgt_mask)
    decoder_out = decoder_out[:, -1, :] # get last token
    model_out = model.out_vocab_proj(decoder_out)
    return model_out