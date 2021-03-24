import torch
import torch.nn as nn
import torch.nn.functional as F

class Seq2seq(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, src_tokens):
        model_out = src_tokens
        return model_out