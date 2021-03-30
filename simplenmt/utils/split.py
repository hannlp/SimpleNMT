import os
import random
import argparse

'''
Usage:
python split.py -src zh -tgt en -data_path e:\clean -ratio 0.9 -fname train,test,valid
'''
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-src", help="the source language", type=str, default="zh")
    parser.add_argument("-tgt", help="the target language", type=str, default="en")
    parser.add_argument("-data_path", help="the path prefix of whole data which will to be split", type=str, default="")
    parser.add_argument("-ratio", help="the train set ratio of hole data", type=float, default=0.95)
    parser.add_argument("-fname", type=str, default="train,test,valid")
    args = parser.parse_args()
    return args

def split(args):
    print('Spliting \'{}\' and \'{}\' ...'.format(args.data_path + '.' + args.src, args.data_path + '.' + args.tgt), end='')
    src_fp = open(args.data_path + '.' + args.src, encoding='utf-8')
    tgt_fp = open(args.data_path + '.' + args.tgt, encoding='utf-8')

    fname = args.fname.split(',')
    save_dir = os.path.dirname(os.path.abspath(args.data_path)) + '/'

    src_train, src_test, src_val = open(save_dir + fname[0] + '.' + args.src, 'w', encoding='utf-8'), \
      open(save_dir + fname[1] + '.' + args.src, 'w', encoding='utf-8'), open(save_dir + fname[2] + '.' + args.src, 'w', encoding='utf-8')
    tgt_train, tgt_test, tgt_val = open(save_dir + fname[0] + '.' + args.tgt, 'w', encoding='utf-8'), \
      open(save_dir + fname[1] + '.' + args.tgt, 'w', encoding='utf-8'), open(save_dir + fname[2] + '.' + args.tgt, 'w', encoding='utf-8')

    train_ratio, test_ratio = args.ratio, (1 - args.ratio) / 2
    count = {'train':0, 'test':0, 'valid':0}
    for s, t in zip(src_fp.readlines(), tgt_fp.readlines()):
        rand = random.random()
        if 0 < rand <= train_ratio:
            src_train.write(s); tgt_train.write(t); count['train'] += 1
        elif train_ratio < rand <= train_ratio + test_ratio:
            src_test.write(s); tgt_test.write(t); count['test'] += 1
        else:
            src_val.write(s); tgt_val.write(t); count['valid'] += 1
    print('Successful.\ntrain set: {} lines, test_set:{} lines, valid set: {} lines'.format(
          count['train'], count['valid'], count['valid']))
    src_fp.close(); tgt_fp.close(); src_train.close(); src_test.close()
    src_val.close(); tgt_train.close(); tgt_test.close(); tgt_val.close()

if __name__ == '__main__':      
    args = parse()
    split(args)