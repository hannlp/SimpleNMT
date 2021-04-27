import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvS2S(nn.Module):
    def __init__(self, n_src_words, n_tgt_words, src_pdx=-1, tgt_pdx=-1, 
                d_model=512, n_head=8, n_layers=6, p_drop=0.1, 
                max_seq_len=512) -> None:

        super().__init__()
        self.d_model = d_model
        self.src_pdx, self.tgt_pdx = src_pdx, tgt_pdx  # pdx: padding index
        
        self.encoder = Encoder(n_src_words, src_pdx=src_pdx, n_head=n_head, 
                                d_model=d_model, n_layers=n_layers, p_drop=p_drop, 
                                max_seq_len=max_seq_len)

        self.decoder = Decoder(n_tgt_words, tgt_pdx=tgt_pdx, n_head=n_head,
                                d_model=d_model, n_layers=n_layers, p_drop=p_drop, 
                                max_seq_len=max_seq_len)
        self.out_vocab_proj = nn.Linear(d_model, n_tgt_words)

    def forward(self, src_tokens, prev_tgt_tokens):
        '''
        params:
          - src_tokens: (batch_size, src_len)
          - prev_tgt_tokens: (batch_size, tgt_len)
        
        returns:
          - model_out: (batch_size, tgt_len, n_tgt_words)
        '''

        src_mask = src_tokens.eq(self.src_pdx)
        tgt_mask = prev_tgt_tokens.eq(self.tgt_pdx)

        encoder_out = self.encoder(src_tokens, src_mask)
        decoder_out = self.decoder(
            prev_tgt_tokens, encoder_out, src_mask, tgt_mask)
        model_out = self.out_vocab_proj(decoder_out)
        return model_out

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self):
        pass

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self):
        pass