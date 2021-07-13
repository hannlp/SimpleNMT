import os
import argparse

'''
Usage: 
python cut2.py -input .. -outdir path\ -src zh -tgt en -outprefix raw
'''
parser = argparse.ArgumentParser()
parser.add_argument("-input", help="the input file", type=str)
parser.add_argument("-src", help="the source language", type=str, default="zh")
parser.add_argument("-tgt", help="the target language", type=str, default="en")
parser.add_argument("-outdir", help="the dir of output files", type=str)
parser.add_argument("-outprefix", help="the prefix of output files", type=str, default="raw")
args = parser.parse_args()

def cut2(input, outdir, src, tgt, outprefix):
    fp = open(input, encoding='utf-8')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    src_fp = open(outdir + outprefix + '.' + src, 'w', encoding='utf-8')
    tgt_fp = open(outdir + outprefix + '.' + tgt, 'w', encoding='utf-8')
    for line in fp.readlines():
        tgt_line, src_line = line.replace('\n', '').split('\t')
        src_fp.write(src_line + '\n')
        tgt_fp.write(tgt_line + '\n')
    src_fp.close()
    tgt_fp.close()

if __name__ == '__main__':
    cut2(input=args.input, outdir=args.outdir, src=args.src, tgt=args.tgt, outprefix=args.outprefix)