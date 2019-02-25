ButterflyNetNLP
===
This repo contains workflow code for extracting text from butterfly
field guides which have undergone optical character recognition
as PDFs. Multiple subdirectories contain example data, output, and
code with the exception of original PDFs due to copyright issues.
***

#### OCR
This subdirectory contains files related to processing raw OCR text
from scanned PDFs. Many scanned PDFs require preprocessing to get
rid of OCR errors and non-essential phrases (e.g. book titles and
page numbers).
* *extractor.py* - extracts raw text from OCR'd PDFS, performs
text scrubbing on raw text and outputs line by line results.
* *raw.txt* - output file for *extractor.py*.

#### NaiveBayes