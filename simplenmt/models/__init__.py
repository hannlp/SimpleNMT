#from .transformer import Transformer
from .transformer_fix import Transformer

'''
transformer_fix:

1. use decoder last layer norm
2. move decoder vocab proj to transformer
3. add share vocab function
4. add max_seq_len args

3/16 update
5. change the mask compute method: != to .eq(), at : transforemr forward, _add_sequence_mask(), and maksed_fill()
'''

str2model = {"Transformer": Transformer}
#__all__ = ['Transformer']