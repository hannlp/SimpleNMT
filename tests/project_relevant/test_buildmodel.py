import sys
import os
print(os.path, os.getcwd())
sys.path.append(os.getcwd())
import argparse
#from simplenmt.models.transformer import Transformer
from simplenmt.utils.builder import build_model

parser = argparse.ArgumentParser()
parser.add_argument("-model", type=str, default="Transformer")
args = parser.parse_args()
print(args)

build_model(args, CUDA_OK=False)