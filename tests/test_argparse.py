import argparse
from typing import Tuple

# The parser for Trainer
parser = argparse.ArgumentParser()
parser.add_argument("-dl_path", "--dataloader_save_path", help="the dataloader save path", type=str, default=".")
parser.add_argument("-model_path", "--model_save_path", type=str, default=".")
parser.add_argument("--batch_size", type=int, default=3200)
parser.add_argument("--warmup_steps", help="warmup steps of learning rate update", type=int, default=4000)
parser.add_argument("--n_epochs", type=int, default=20)

# The parser for Transformer
parser.add_argument("--d_model", help="dimension of the model", type=int, default=512)
parser.add_argument("--n_layer", type=int, default=6)
parser.add_argument("--n_head", help="heads number of mutihead-attention", type=int, default=8)
parser.add_argument("--p_drop", help="probability of dropout", type=float, default=0.1)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--betas", type=float, nargs="+", default=(0.9, 0.98))


args = parser.parse_args()

# define arguments after parse
args.a_int = 1
args.a_str = 'this is a string.'
args.a_float = 3.1415926

print(args)