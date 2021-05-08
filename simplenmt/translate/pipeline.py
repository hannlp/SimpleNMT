import fastBPE
from sacremoses import MosesTokenizer, MosesDetokenizer
from pyhanlp import HanLP

def input_pipeline(sentence, lang, bpecode=None):
    """
    1. 分词（zh）
    2. 转小写（en）
    3. tokenzie
    4. bpe
    """
    if lang == 'zh':
        seg = [term.word for term in HanLP.segment(sentence)]
        mt = MosesTokenizer(lang='zh')
        tokenized_str = mt.tokenize(seg, return_str=True)
        if bpecode is not None:
            bpe = fastBPE.fastBPE(bpecode + '.zh')
            bpe_str = bpe.apply([tokenized_str])[0]
            return bpe_str.split()
        return tokenized_str.split()
    elif lang == 'en':
        lower = sentence.lower()
        mt = MosesTokenizer(lang='en')
        tokenized_str = mt.tokenize(lower, return_str=True)
        if bpecode is not None:
            bpe = fastBPE.fastBPE(bpecode + '.en')
            bpe_str = bpe.apply([tokenized_str])[0]
            return bpe_str.split()
        return tokenized_str.split()
    else:
        raise Exception

def output_pipeline(translated, lang):
    """
    1. 去除bpe
    2. de-tokenize
    3. 转首字母大写（en）
    4. 合并(zh)
    """
    joined_str = ' '.join(translated)
    remove_bpe_str = joined_str.replace('@@ ', '')

    if lang == 'zh':
        md = MosesDetokenizer(lang='zh')
        return md.detokenize(remove_bpe_str.split())
    elif lang == 'en':
        md = MosesDetokenizer(lang='en')
        return md.detokenize(remove_bpe_str.split())
    else:
        raise Exception