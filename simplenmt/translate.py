import argparse
from translate.translator import Translator

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-src", type=str, default="zh")
    parser.add_argument("-tgt", type=str, default="en")

    parser.add_argument("-batch_size", type=int, default=3200)
    parser.add_argument("-generate", help="repalce the translate to generate", action="store_true")

    parser.add_argument("-test_path", help="the test corpus path prefix", type=str, default="./")
    parser.add_argument("-save_path", help="the path to save checkpoint, dataloader and log", type=str, default=".")
    parser.add_argument("-ckpt_suffix", help="the checkpoint's suffix, such as best and last", type=str, default="best")

    # parser.add_argument("-dl_path", help="the dataloader save path", type=str, default="./")
    # parser.add_argument("-ckpt_path", help="the checkpoint save path", type=str, default="./")

    parser.add_argument("-max_seq_length", help="the max length of sequence", type=int, default=256)
    parser.add_argument("-beam_size", help="the width of beam search", type=int, default=-1)
    parser.add_argument("-length_penalty", type=float, default=1.0)

    args = parser.parse_args()
    return args

def main():
    args = parse()
    translator = Translator(args)
    if args.generate:
        #python -u translate.py -generate -src zh -tgt en -dl_path E:\yuchen\models\ldc\zh_en.dl -ckpt_path E:\yuchen\models\ldc -test_path E:\yuchen\ldc_test
        #python translate.py -generate -src zh -tgt en -dl_path E:\yuchen\zh_en.dl -ckpt_path E:\yuchen -test_path E:\yuchen\ldc_valid
        translator.generate(
            exts=('.' + args.src, '.' + args.tgt),
            test_path=args.test_path, 
            batch_size=args.batch_size)
    else:
        #python translate.py -src zh -tgt en -dl_path E:\yuchen\zh_en.dl -ckpt_path E:\yuchen
        while True:
            sentence = input('Please input a sentence({}): '.format(args.src))
            translator.translate(sentence, beam_size=args.beam_size)#, src_lang=args.src, tgt_lang=args.tgt)

if __name__ == '__main__':
    main()