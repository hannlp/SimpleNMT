import argparse
from bs4 import BeautifulSoup

parser = argparse.ArgumentParser()
parser.add_argument("-input", help="the input file", type=str)
parser.add_argument("-output", help="the output file", type=str)
args = parser.parse_args()

with open(args.input, 'r', encoding='utf8') as readfp, open(args.output, 'w', encoding='utf8') as writefp:
    soup = BeautifulSoup(readfp, 'xml')
    writefp.write(soup.get_text().strip())