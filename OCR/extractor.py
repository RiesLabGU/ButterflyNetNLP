#!/usr/bin/env python
import re
import nltk
import string
from tika import parser

printable = set(string.printable)

# regex cleaning and line setting
raw = parser.from_file('sample.pdf')  # parse OCR text from file
text = raw['content']  # grab textual content
text = text.lower().replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()  #replace all whitespace, lower
text = ' '.join(text.split())  # remove consecutive spaces
text = re.sub(r'(?<=[a-z]{4})[.]', '.\n', text)
text = text.encode('ascii', errors='ignore').decode(encoding='utf8').replace(r'\\', '').replace('"', '')
text = re.sub(r'(?<=nae)', '\n', text)
text = re.sub(r'(?<=dae)', '\n', text)
text = text.replace('life histories of cascadia butterflies', '').replace('\\', '').replace('|', '')
text = text.replace('adult biology', '').replace('immature stage biology', '')
text = text.replace('description of immature stages', '').replace('discussion', '')
text = re.sub(r'([0-9]){3}', '', text)
text = re.sub(r'([i])\b', '', text)
text = re.sub(r'(family|subfamily).*', '', text)
text = text.replace('-', ' ').replace('tlie', 'the').replace('ihis', 'this').replace('ditilicult', 'difficult')
text = re.sub(r'^\s*$', '', text)
text = ' '.join(text.split())  # remove consecutive spaces
text = re.sub(r'(?<=[a-z]{4})[.]', '.\n', text)
text = text.lstrip(' ')


#line cleaing

with open("raw.txt", 'w', encoding='utf8') as f:
    f.write(text)
