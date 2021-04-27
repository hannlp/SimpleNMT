
def forward(src_tokens, tgt_tokens, src_mask, tgt_mask=4, **kwargs):
    # 这种方法可以实现一个_encode()和一个_decode()适配多种模型
    print(src_tokens, tgt_tokens, src_mask, tgt_mask)
    print(type(kwargs))
    return kwargs

a = {'src_tokens':1, 'tgt_tokens':2, 'src_mask':3, 'srcc_mask':5, 'tgtt_mask':6}
ref = forward(**a)
print(ref)

ref2 = forward(1, 2, 3, ccc=4, aa=5, bb=6)

# for luong
def _encode(model, src_tokens, src_pdx):
    src_mask = src_tokens.eq(src_pdx)
    encoder_out = model.encoder(src_tokens, src_mask)
    return encoder_out, src_mask

def _decode(model, prev_tgt_tokens, encoder_out, src_mask, **kwargs):
    decoder_out = model.decoder(
        prev_tgt_tokens, encoder_out, src_mask)
    decoder_out = decoder_out[:,-1,:] # get last token
    model_out = model.out_vocab_proj(decoder_out)
    return model_out

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