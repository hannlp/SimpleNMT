import sacremoses
import argparse
import jieba
import subword-nmt

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-src", type=str, default="zh")
    parser.add_argument("-tgt", type=str, default="en")

    parser.add_argument("-zh_seg", help="中文分词", action="store_true")
    parser.add_argument("-tokenize", action="store_true")
    parser.add_argument("-detokenize", action="store_true")
    parser.add_argument("-truecase", action="store_true")
    parser.add_argument("-normalize", action="store_true")
    parser.add_argument("-split_subword", action="store_true")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    pass