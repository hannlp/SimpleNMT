import argparse
from translate.translator import Translator

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-src", type=str, default="zh")
    parser.add_argument("-tgt", type=str, default="en")

    parser.add_argument("-batch_size", type=int, default=4096)
    parser.add_argument("-data_path", help="the test corpus path, witch can be a path or a prefix", type=str, default=".")
    parser.add_argument("-save_path", help="the path to save checkpoint, dataloader and log", type=str, default=".")
    parser.add_argument("-ckpt_suffix", help="the checkpoint's suffix, such as best, last and a id", type=str, default="best")
    parser.add_argument("-max_seq_len", help="the max length of sequence", type=int, default=128)

    parser.add_argument("-precise", help="use some input and output pipeline" ,action="store_true")
    parser.add_argument("-bpecode", help="the bpecode's prefix", type=str, default=None)

    parser.add_argument("-generate", help="repalce the translate to generate", action="store_true")
    parser.add_argument("-quiet", help="don't print the generate result", action="store_true")
    parser.add_argument("-beam_size", help="the width of beam search", type=int, default=-1)
    parser.add_argument("-length_penalty", type=float, default=0.7)

    args = parser.parse_args()
    return args

def main():
    args = parse()
    translator = Translator(args)
    if args.generate:
        translator.generate(
            src=args.src, tgt=args.tgt, result_save_path=args.save_path, 
            batch_size=args.batch_size, data_path=args.data_path, quiet=args.quiet)
    else:
        if args.precise and args.bpecode is not None:
            import fastBPE
            bpe = fastBPE.fastBPE(args.bpecode + '.' + args.src)
        else:
            bpe = None

        while True:
            sentence = input('Please input a sentence({}): '.format(args.src))
            translator.translate(sentence, src=args.src, tgt=args.tgt, 
                precise=args.precise, bpe=bpe)

if __name__ == '__main__':
    main()