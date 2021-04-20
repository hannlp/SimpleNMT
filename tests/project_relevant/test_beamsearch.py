import sys
import os
print(os.path, os.getcwd())
sys.path.append(os.getcwd())

import torch
from simplenmt.translate.algorithms import generate_beam
from simplenmt.models.transformer import Transformer


import time
MAX_SEQ_LEN = 256

n_src_words = 20
n_tgt_words = 25
batch_size = 5
src_len = 64
tgt_len = 60

model = Transformer(n_src_words=n_src_words, n_tgt_words=n_tgt_words)
#print(model)

src_tokens = torch.randint(n_src_words, (batch_size, src_len))


start_time = time.time()
a = generate_beam(model=model, src_tokens=src_tokens, beam_size=4, length_penalty=1.0, max_len=10, bos=1, eos=2, pad=0)
elapsed = (time.time() - start_time)
print(a, elapsed, 's')

avg_elapsed = []
for i in range(10):
    start_time = time.time()
    generate_beam(model=model, src_tokens=src_tokens, beam_size=4, length_penalty=1.0, max_len=32, bos=1, eos=2, pad=0)
    avg_elapsed.append((time.time() - start_time))
print(sum(avg_elapsed) / len(avg_elapsed), 's')
