import argparse
from translate.translator import Translator

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dl_path", help="the dataloader save path", type=str, default="./")
    parser.add_argument("-ckpt_path", help="the checkpoint save path", type=str, default="./")
    args = parser.parse_args()
    return args

def main():
    args = parse()
    translator = Translator(args)
    translator.generate(valid_iter)

if __name__ == '__main__':
    main()