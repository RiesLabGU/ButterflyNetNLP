#!/usr/bin/env python
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='habitat-phrases')
    parser.add_argument('--path', type=str, default="sample.pdf",
                        help='path to the menu to update')
    args = parser.parse_args()

    main(args.path)