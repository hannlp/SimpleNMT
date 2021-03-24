import sys
import os
print(os.path, os.getcwd())
sys.path.append(os.getcwd())
import torch
from simplenmt.models.RNNSearch import RNNSearch

n_src_words = 20
n_tgt_words = 25
batch_size = 5
src_len = 64
tgt_len = 60

model = RNNSearch(n_src_words=n_src_words, n_tgt_words=n_tgt_words, max_src_len=256, max_tgt_len=256, d_model=512, n_layers=24, bidirectional=False)
print(model)

src_tokens = torch.randint(n_src_words, (batch_size, src_len))
prev_output_tokens = torch.randint(n_tgt_words, (batch_size, tgt_len))

out = model(src_tokens, prev_output_tokens)
print(out.shape)