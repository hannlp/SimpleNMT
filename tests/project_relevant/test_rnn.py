import sys
import os
sys.path.append(os.getcwd())
import torch
from simplenmt.models.rnnsearch import RNNSearch
from simplenmt.models.lstm import LSTMModel

MAX_SEQ_LEN = 256


n_src_words = 20
n_tgt_words = 25
batch_size = 5
src_len = 64
tgt_len = 60

src_tokens = torch.randint(n_src_words, (batch_size, src_len))
prev_output_tokens = torch.randint(n_tgt_words, (batch_size, tgt_len))

rnnsearch, lstmmodel = False, True

# RNNSearch
if rnnsearch:
    model = RNNSearch(n_src_words=n_src_words, n_tgt_words=n_tgt_words, d_model=256, n_layers=3); print(model)
    
    out = model(src_tokens, prev_output_tokens); print(out.shape)

# LSTMModel
if lstmmodel:
    model = LSTMModel(n_src_words, n_tgt_words, max_src_len=MAX_SEQ_LEN, max_tgt_len=MAX_SEQ_LEN,
                      d_model=256, n_layers=3, src_pdx=0, tgt_pdx=0, bidirectional=True); #print(model)
    encoder_out, hiddens, cells = model.encoder(src_tokens)
    print(encoder_out.shape, hiddens.shape, cells.shape)