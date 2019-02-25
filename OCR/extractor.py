#!/usr/bin/env python
import argparse
from tika import parser

def main(data_file):
    raw = parser.from_file(data_file)

    with open("raw.txt") as f:
        f.write(raw['content'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pdf to convert')
    parser.add_argument('--path', type=str, default="sample.pdf",
                        help='path to the menu to update')
    args = parser.parse_args()

    main(args.path)