import sys
import os
print(os.path, os.getcwd())
sys.path.append(os.getcwd())

import torch
from simplenmt.translate.algorithms import beam_search, greedy_search
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

beam_test = False
if beam_test:
    start_time = time.time()
    a = beam_search(model=model, src_tokens=src_tokens, beam_size=4, length_penalty=1.0, max_len=10, bos=1, eos=2, pad=0)
    elapsed = (time.time() - start_time)
    print(a, elapsed, 's')

greedy_test = True
if greedy_test:
    start_time = time.time()
    a = greedy_search(model=model, src_tokens=src_tokens, max_len=10, bos=1, eos=2, pad=0)
    elapsed = (time.time() - start_time)
    print(a, elapsed, 's')


# time test
time_test = False
if time_test:
    avg_elapsed = []
    for i in range(10):
        start_time = time.time()
        beam_search(model=model, src_tokens=src_tokens, beam_size=4, length_penalty=3.0, max_len=10, bos=1, eos=2, pad=0)
        avg_elapsed.append((time.time() - start_time))
    print(sum(avg_elapsed) / len(avg_elapsed), 's')
