import argparse
from translate.translator import Translator

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-src", type=str, default="zh")
    parser.add_argument("-tgt", type=str, default="en")

    parser.add_argument("-dl_path", help="the dataloader save path", type=str, default="./")
    parser.add_argument("-ckpt_path", help="the checkpoint save path", type=str, default="./")
    parser.add_argument("-max_length", help="the max length of sequence", type=int, default=256)
    parser.add_argument("-beam_size", help="the width of beam search", type=int, default=1)
    args = parser.parse_args()
    return args

def main():
    args = parse()
    translator = Translator(args)
    while True:
        sentence = input('Please input a sentence of source:')
        translator.translate(sentence, beam_size=args.beam_size, src_lang=args.src, tgt_lang=args.tgt)

if __name__ == '__main__':
    main()