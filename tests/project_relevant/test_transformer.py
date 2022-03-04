import sys
import os
print(os.path, os.getcwd())
sys.path.append(os.getcwd())
import torch
from simplenmt.models.transformer_legacy import Transformer

n_src_words = 20
n_tgt_words = 25
batch_size = 5
src_len = 64
tgt_len = 60

model = Transformer(n_src_words=n_src_words, n_tgt_words=n_tgt_words)
#print(model)

src_tokens = torch.randint(n_src_words, (batch_size, src_len))
prev_output_tokens = torch.randint(n_tgt_words, (batch_size, tgt_len))

out = model(src_tokens, prev_output_tokens)
print(out.shape)