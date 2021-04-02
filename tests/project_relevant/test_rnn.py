import sys
import os
sys.path.append(os.getcwd())
import torch
from simplenmt.models.seq2seq import Seq2Seq
from simplenmt.models.rnnsearch import RNNSearch

MAX_SEQ_LEN = 256

batch_size = 5
n_src_words, n_tgt_words = 20, 25
src_len, tgt_len = 64, 60

src_tokens = torch.randint(n_src_words, (batch_size, src_len))
prev_output_tokens = torch.randint(n_tgt_words, (batch_size, tgt_len))

seq2seq, rnnsearch = False, True
if seq2seq: # Seq2Seq
    model = Seq2Seq(n_src_words=n_src_words, n_tgt_words=n_tgt_words, d_model=256, n_layers=3); print(model)  
    out = model(src_tokens, prev_output_tokens); print(out.shape)
if rnnsearch: # RNNSearch
    model = RNNSearch(n_src_words, n_tgt_words, max_src_len=MAX_SEQ_LEN, max_tgt_len=MAX_SEQ_LEN,
                      d_model=256, n_layers=3, src_pdx=0, tgt_pdx=0, bidirectional=True); #print(model)
    encoder_out, hiddens, cells = model.encoder(src_tokens)
    print(encoder_out.shape, hiddens.shape, cells.shape)