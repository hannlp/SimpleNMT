import sys

'''
Usage: 
python cut2.py fpath new_data_dir
'''

def cut2(fpath, new_data_dir, nsrc='zh', ntgt='en'):
    fp = open(fpath, encoding='utf-8')
    src_fp = open(new_data_dir + 'raw.' + nsrc, 'w', encoding='utf-8')
    tgt_fp = open(new_data_dir + 'raw.' + ntgt, 'w', encoding='utf-8')
    for line in fp.readlines():
        tgt_line, src_line = line.replace('\n', '').split('\t')
        src_fp.write(src_line + '\n')
        tgt_fp.write(tgt_line + '\n')
    src_fp.close()
    tgt_fp.close()

if __name__ == '__main__':      
    cut2(fpath=sys.argv[1], new_data_dir=sys.argv[2], nsrc='zh', ntgt='en')