import argparse
import string
import nltk
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB


def load_function_words(resource_path):
    """load a newline separated text file of function words.
    Return a list"""
    f_words = []
    with open(resource_path, 'r') as f:
        for line in f:
            if line.strip():
                f_words.append(line.lower().strip())
    return f_words


def shuffle_in_unison(a, c, b, d, e):
    """Adapted from StackOverflow solution URL posted in class. Takes in two datasets
    of the same length, shuffles them by retaining the original indices of both
    arrays, and modulating them both by one random permutation."""
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    shuffled_c = np.empty(c.shape, dtype=c.dtype)
    shuffled_d = np.empty(d.shape, dtype=d.dtype)
    shuffled_e = np.empty(e.shape, dtype=e.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
        shuffled_c[new_index] = c[old_index]
        shuffled_d[new_index] = d[old_index]
        shuffled_e[new_index] = e[old_index]
    return shuffled_a, shuffled_c, shuffled_b, shuffled_d, shuffled_e


def split_dataset(X, X2, y, hold_out_percent, X3, X4):
    """shuffle and split the dataset. Returns two tuples:
    (X_train, y_train, train_indices): train inputs
    (X_val, y_val, val_indices): validation inputs"""

    X, X2, y, X3, X4 = shuffle_in_unison(X, X2, y, X3, X4)
    split_ind = int(len(X)*(1-hold_out_percent))

    X_tr = X[split_ind:]
    X_val = X[:split_ind-1]
    X2_tr = X2[split_ind:]
    X2_val = X2[:split_ind - 1]
    X3_tr = X3[split_ind:]
    X3_val = X3[:split_ind - 1]
    X4_tr = X4[split_ind:]
    X4_val = X4[:split_ind - 1]
    X_tr_indices = range(split_ind, len(X))
    X_val_indices = range(0, split_ind)

    y_tr = y[split_ind:]
    y_val = y[:split_ind-1]

    return (X_tr, X2_tr, X3_tr, X4_tr, y_tr, X_tr_indices), (X_val, X2_val, X3_val, X4_val, y_val, X_val_indices)


def zero_rule_algorithm_classification(train, test):
    output_values = [row[-1] for row in train]
    prediction = max(set(output_values), key=output_values.count)
    predicted = [prediction for i in range(len(test))]
    return predicted


def create_feature_matrix(lines, function_words):
    line_features = np.zeros((len(lines), len(function_words)), dtype=np.int64)

    for i, line in enumerate(lines):
        line = line.lower()
        a = nltk.word_tokenize(line)
        for j, function_word in enumerate(function_words):
            line_features[i, j] = a.count(function_word)

    return line_features


def create_binary_feature_matrix(lines, function_words, line_features):
    binary_features = np.copy(line_features)
    for i, review in enumerate(lines):
        for j, function_word in enumerate(function_words):
            if binary_features[i, j] >= 1:
                binary_features[i, j] = 1

    return binary_features


def main(tv_file, vocab_file_001, vocab_file_002):
    """Train a classification algorithm using function words on a training and validation file.
    Apply classification algorithm to raw data file, line by line."""

    function_words = load_function_words(vocab_file_001)
    ngrams = load_function_words(vocab_file_002)

    max_lines = 600
    translate_table = dict((ord(char), None) for char in string.punctuation)

    lines = []
    classes = []
    with open(tv_file, 'r') as data_file:
        for i, line in enumerate(data_file):
            fields = line.strip().lower().split("\t")
            fields[-1] = fields[-1].translate(translate_table)
            lines.append(fields[-1].strip("\"").strip("\'"))
            classes.append(str(fields[0]))
            if i == max_lines:
                break

    # Create a normal feature matrices for all target function words
    line_features = create_feature_matrix(lines, function_words)
    line_features2 = create_feature_matrix(lines, ngrams)

    # Create a binary feature matrices for all target function words
    binary_features = create_binary_feature_matrix(lines, function_words, line_features)
    binary_features2 = create_binary_feature_matrix(lines, ngrams, line_features2)

    #  Create a dictionary of label keys, cast to a numpy array
    key_dict = {class_type: i for i, class_type in enumerate(set(classes))}
    key_list = [key_dict[class_type] for class_type in classes]
    labels = np.asarray(key_list)

    # Split the dataset into training and validation sets
    train, val = split_dataset(line_features, binary_features, labels, 0.9, line_features2, binary_features2)
    print("------------------Tokenized Model--------------------------------")

    #############################
    ## Tokenized Model Testing ##
    #############################

    # Run GaussianNB on features
    model_1 = GaussianNB()
    model_1.fit(train[0], train[4])
    print(f"Gaussian Classifier score: {model_1.score(val[0], val[4])}")  # print score

    # Run BernoulliNB on features
    model_2 = BernoulliNB()
    model_2.fit(train[1], train[4])
    print(f"Binary Classifier score: {model_2.score(val[1], val[4])}")  # print score

    # Run zero-rule baseline on features
    baseprediction = zero_rule_algorithm_classification(line_features, labels)  # get predictions for baseline

    if baseprediction[0] == 0:
        baseline = (len(baseprediction) - np.count_nonzero(labels)) / (len(baseprediction))
    else:
        baseline = (len(baseprediction) - len(np.where(labels == 0))) / (len(baseprediction))

    print(f"Baseline score: {baseline}")

    ##########################
    ## 2-gram Model Testing ##
    ##########################

    print("------------------------2-gram Model-----------------------------")

    # Run GaussianNB on features
    model_1 = GaussianNB()
    model_1.fit(train[2], train[4])
    print(f"Gaussian Classifier score: {model_1.score(val[2], val[4])}")  # print score

    # Run BernoulliNB on features
    model_2 = BernoulliNB()
    model_2.fit(train[3], train[4])
    print(f"Binary Classifier score: {model_2.score(val[3], val[4])}")  # print score

    print(f"Baseline score: {baseline}")
    print("Divide by zero error caused by feature matrix exhibiting low difference between label classes.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='feature vector homework')
    parser.add_argument('--tv_path', type=str, default="line_reads.txt",
                        help='path to the sentences from the original text')
    parser.add_argument('--function_words_path_001', type=str, default="vocab_001.txt",
                        help='path to the list of words to use as features')
    parser.add_argument('--function_words_path_002', type=str, default="vocab_002.txt",
                        help='path to the list of 2-grams to use as features')
    args = parser.parse_args()

    main(args.tv_path, args.function_words_path_001, args.function_words_path_002)