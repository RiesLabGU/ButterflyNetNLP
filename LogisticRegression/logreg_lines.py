import argparse
import numpy as np

def main(tv_file, vocab_file):
    """Train a classification algorithm using function words on a training and validation file.
    Apply classification algorithm to raw data file, line by line."""

    max_lines = 20

    lines = []
    classes = []
    with open(tv_file, 'r') as data_file:
        for i, line in enumerate(data_file):
            fields = line.strip().split("\t")
            lines.append(fields[-1])
            classes.append(str(fields[0]))
            if i == max_lines:
                break

    line_features = np.zeros((len(lines), len(function_words)), dtype=np.int64)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='feature vector homework')
    parser.add_argument('--tv_path', type=str, default="LogisticRegression/line_reads.txt",
                        help='path to the list of words to use as features')
    parser.add_argument('--function_words_path', type=str, default="NaiveBayes/bnet_vocab.txt",
                        help='path to the list of words to use as features')
    args = parser.parse_args()

    main(args.path, args.tv_path, args.function_words_path)