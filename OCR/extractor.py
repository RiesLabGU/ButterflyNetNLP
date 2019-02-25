#!/usr/bin/env python
import re
import nltk
import string
from tika import parser

printable = set(string.printable)

raw = parser.from_file('sample.pdf')
text = raw['content']
text = text.lower().replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()
text = re.sub(r'(?<=[a-z]{4})[.]', '\n', text)
text = text.encode('ascii', errors='ignore').decode(encoding='utf8').replace(r'\\', '').replace('"', '')

with open("raw.txt", 'w', encoding='utf8') as f:
    f.write(text)
