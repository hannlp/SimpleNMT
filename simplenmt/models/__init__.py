#from .transformer import Transformer
from .transformer_fix import Transformer

'''
transformer_fix:

1. use decoder last layer norm
2. move decoder vocab proj to transformer
3. add share vocab function
4. add max_seq_len args
'''

str2model = {"Transformer": Transformer}
#__all__ = ['Transformer']