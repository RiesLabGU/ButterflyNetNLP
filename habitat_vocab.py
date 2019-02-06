#!/usr/bin/env python
import argparse
import re


def main(data_file):

    lines = []
    with open(data_file, 'r') as data_file:
        for i, line in enumerate(data_file):
            fields = line.strip().split("\t")
            lines.append(fields[2])

    hab_list = list(filter(None, lines))
    print(f"The length of the habitat list is {len(hab_list)} descriptions.")

    merged = ' '.join(hab_list)
    merged = merged.lower()
    merged = re.sub(r'[^\w\s]', '', merged)
    words = merged.split()
    dict = {}

    for w in words:
        if w in dict:
            dict[w] += 1
        else:
            dict[w] = 1

    lst = [(dict[w], w) for w in dict]
    lst.sort()
    lst.reverse()

    # Program assumes user has downloaded an imported stopwords from NLTK
    from nltk.corpus import stopwords  # Import the stop word list

    stop_words = set(stopwords.words('english'))
    lst_new = ([word for word in lst if word not in stop_words])

    print("The most frequent words are:")
    print()

    i = 1
    for count, word in lst_new[:150]:
        print('%2s. %4s %s' % (i, count, word))
        i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='habitat-phrases')
    parser.add_argument('--path', type=str, default="habitat_WGS.txt",
                        help='path to the menu to update')
    args = parser.parse_args()

    main(args.path)