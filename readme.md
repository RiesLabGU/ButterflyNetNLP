ButterflyNetNLP
===
This repo contains workflow code for extracting text from butterfly
field guides which have undergone optical character recognition
as PDFs. Multiple subdirectories contain example data, output, and
code with the exception of original PDFs due to copyright issues.


***

Directories:
===

#### .../OCR
This subdirectory contains files related to processing raw OCR text
from scanned PDFs. Many scanned PDFs require pre-processing to get
rid of OCR errors and non-essential phrases (e.g. book titles and
page numbers).
* *extractor.py* - extracts raw text from OCR'd PDFS, performs
text scrubbing on raw text and outputs line by line results.
* *raw.txt* - output file for *extractor.py*.

#### .../Classifier
A Naive Bayes multiclassifier for classifying parsed sentences as describing either
distribution, morphology, hostplant, life-history, or non-target trait data. This directory
contains the files created for ANYL 521: Computational Linguistics in Advances Python and is
the term project for Vaughn M. Shirey. The file descriptions are as follows:

* *classifier.py* - primary file for constructing the Naive Bayes model for discerning the above
classifications of sentence data. 'Main' takes three (3) arguements, a line by line file of sentences
with classification labels to serve for training and validation purposes, a list of function word, and a
secondary list of 2-gram function words for comparing how features impact the NB model. 

Example command line usage is: `classifer.py --tv_path line_reads.txt --function_words_path_001 vocab_001.txt
 --function_words_path_002 vocab_002.txt` The file outputs scores for Gaussian and Bernoulli based models
 using both vocabulary sets. These results are compared to a zero-rule baseline.
 
* *line_reads.txt* - a TSV file containing line by line sentences and their corresponding trait labels.

* *vocab_001.txt* - a text file containing 1-gram vocabulary words to use as features.

* *vocab_002.txt* - a text file containing 2-gram vocabulary words to use as features.

#### .../NaiveBayes
A legacy directory for determining if a particular parsed sentence is refering to 
a habitat description or not.