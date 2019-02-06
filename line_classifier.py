#!/usr/bin/env python
import argparse
import random
import nltk
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB


def zero_rule_algorithm_classification(train, test):
    output_values = [row[-1] for row in train]
    prediction = max(set(output_values), key=output_values.count)
    predicted = [prediction for i in range(len(test))]
    return predicted


def load_function_words(resource_path):
    """load a newline separated text file of function words.
    Return a list"""
    f_words = []
    with open(resource_path, 'r') as f:
        for line in f:
            if line.strip():
                f_words.append(line.lower().strip())
    return f_words


def shuffle_in_unison(a, c, b, d):
    """Adapted from StackOverflow solution URL posted in class. Takes in two datasets
    of the same length, shuffles them by retaining the original indices of both
    arrays, and modulating them both by one random permutation."""
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    shuffled_c = np.empty(c.shape, dtype=c.dtype)
    shuffled_d = np.empty(d.shape, dtype=d.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
        shuffled_c[new_index] = c[old_index]
        shuffled_d[new_index] = d[old_index]
    return shuffled_a, shuffled_c, shuffled_b, shuffled_d


def split_dataset(X, X2, y, hold_out_percent, X3):
    """shuffle and split the dataset. Returns two tuples:
    (X_train, y_train, train_indices): train inputs
    (X_val, y_val, val_indices): validation inputs"""

    X, X2, y, X3 = shuffle_in_unison(X, X2, y, X3)
    split_ind = int(len(X)*(1-hold_out_percent))

    X_tr = X[split_ind:]
    X_val = X[:split_ind-1]
    X2_tr = X2[split_ind:]
    X2_val = X2[:split_ind - 1]
    X3_tr = X2[split_ind:]
    X3_val = X2[:split_ind - 1]
    X_tr_indices = range(split_ind, len(X))
    X_val_indices = range(0, split_ind)

    y_tr = y[split_ind:]
    y_val = y[:split_ind-1]

    return (X_tr, X2_tr, X3_tr, y_tr, X_tr_indices), (X_val, X2_val, X3_val, y_val, X_val_indices)


def main(data_file, tv_file, vocab_file):
    """Train a classification algorithm using function words on a training and validation file.
    Apply classification algorithm to raw data file, line by line."""

    function_words = load_function_words(vocab_file)

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

    for i, line in enumerate(lines):
        line = line.lower()
        a = nltk.word_tokenize(line)
        for j, function_word in enumerate(function_words):
            line_features[i, j] = a.count(function_word)

    totals = line_features.sum(axis=0)  # sum columns
    most_common_count = max(totals)  # grab most frequent function word

    # Print most common function word
    index = np.where(totals == np.amax(totals))[0]  # get column location of most common word
    most_common_word = function_words[index[0]]
    print(f"Most common word: {most_common_word}, count: {most_common_count}")

    # Find function words that never occurred and print them
    zero_inds = np.where(totals == 0)[0]
    if len(zero_inds) > 0:
        print("No instances found for: ")
        for ind in zero_inds:
            print(f"  {function_words[ind]}")
    else:
        print("All function words found")

    # Create a binary feature matrix for all target function words
    binary_features = np.copy(line_features)
    for i, review in enumerate(lines):
        for j, function_word in enumerate(function_words):
            if binary_features[i, j] >= 1:
                binary_features[i, j] = 1

    # Create a binary feature matrix for all target function words
    normalized_features = np.empty_like(line_features, dtype=float)
    for i, review in enumerate(lines):
        for j, function_word in enumerate(function_words):
            normalized_features[i,j] = float(line_features[i,j]/len(review.split()))

    #  Create a dictionary of authorship keys, cast to a numpy array
    key_dict = {class_type: i for i, class_type in enumerate(set(classes))}
    key_list = [key_dict[class_type] for class_type in classes]

    labels = np.asarray(key_list)

    # Split the dataset via the function above
    train, val = split_dataset(line_features, binary_features, labels, 0.9, normalized_features)

    baseprediction = zero_rule_algorithm_classification(train[0], val[3])  # get predictions for baseline
    print(baseprediction[0])
    if baseprediction[0] == 0:
        baseline = (len(baseprediction)-np.count_nonzero(val[3]))/(len(baseprediction))
    else:
        baseline = (len(baseprediction)-len(np.where(val[3] == 0)))/(len(baseprediction))

    # Run multinomialNB on features for baseline
    model_1 = MultinomialNB()
    model_1.fit(train[0], train[3])

    model_2 = BernoulliNB()
    model_2.fit(train[1], train[3])

    print(f"Baseline score: {baseline}")
    print(f"Normal Classifier score: {model_1.score(val[0], val[3])}")  # print score
    print(f"Binary Classifier score: {model_2.score(val[1], val[3])}")  # print score

    ################## Fair testing scheme ########################
    lines = []
    classes = []
    data_file = "bnet_raw.txt"
    with open(data_file, 'r') as fair_test:
        for i, line in enumerate(fair_test):
            fields = line.strip().split("\t")
            lines.append(fields[-1])
            classes.append(str(fields[0]))

    line_features = np.zeros((len(lines), len(function_words)), dtype=np.int64)

    for i, line in enumerate(lines):
        line = line.lower()
        a = nltk.word_tokenize(line)
        for j, function_word in enumerate(function_words):
            line_features[i, j] = a.count(function_word)

    # Create a binary feature matrix for all target function words
    binary_features = np.copy(line_features)
    for i, review in enumerate(lines):
        for j, function_word in enumerate(function_words):
            if binary_features[i, j] >= 1:
                binary_features[i, j] = 1

    key_dict = {class_type: i for i, class_type in enumerate(set(classes))}
    key_list = [key_dict[class_type] for class_type in classes]

    labels = np.asarray(key_list)

    print()
    baseprediction = zero_rule_algorithm_classification(line_features, labels)  # get predictions for baseline
    print(baseprediction[0])

    if baseprediction[0] == 0:
        baseline = (len(baseprediction) - np.count_nonzero(labels)) / (len(baseprediction))
    else:
        baseline = (len(baseprediction) - len(np.where(labels == 0))) / (len(baseprediction))

    print(f"Number of training lines: {max_lines*0.9}")
    print(f"Baseline fair score: {baseline}")
    print(f"Normal Classifier fair score: {model_1.score(line_features, labels)}")  # print score
    print(f"Binary Classifier fair score: {model_2.score(binary_features, labels)}")  # print score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='feature vector homework')
    parser.add_argument('--path', type=str, default="bnet_raw.txt",
                        help='path to the menu to update')
    parser.add_argument('--tv_path', type=str, default="tv_file1.txt",
                        help='path to the list of words to use as features')
    parser.add_argument('--function_words_path', type=str, default="bnet_vocab.txt",
                        help='path to the list of words to use as features')
    args = parser.parse_args()

    main(args.path, args.tv_path, args.function_words_path)