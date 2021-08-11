import os
import argparse

'''
Usage: 
python cut2.py -input .. -outdir path\ -left zh -right en -outprefix raw
'''
parser = argparse.ArgumentParser()
parser.add_argument("-input", help="the input file", type=str)
parser.add_argument("-left", help="the left part of the file", type=str, default="zh")
parser.add_argument("-right", help="the right part of the file", type=str, default="en")
parser.add_argument("-outdir", help="the dir of output files", type=str)
parser.add_argument("-outprefix", help="the prefix of output files", type=str, default="raw")
args = parser.parse_args()

def cut2(input, outdir, left, right, outprefix):
    fp = open(input, encoding='utf-8')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    left_fp = open(outdir + outprefix + '.' + left, 'w', encoding='utf-8')
    right_fp = open(outdir + outprefix + '.' + right, 'w', encoding='utf-8')
    for line in fp.readlines():
        left_line, right_line = line.replace('\n', '').split('\t')
        left_fp.write(left_line + '\n')
        right_fp.write(right_line + '\n')
    left_fp.close()
    right_fp.close()

if __name__ == '__main__':
    cut2(input=args.input, outdir=args.outdir, left=args.left, right=args.right, outprefix=args.outprefix)