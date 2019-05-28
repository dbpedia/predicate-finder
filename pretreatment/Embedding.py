#!/usr/bin/env/ python3

from torchnlp.word_to_vector import FastText, GloVe
# please install the latest code of pytorch_nlp

# the pre-trained data will be downloaded when you first run, which are nearly 9GB in total
fasttext = FastText()
glove = GloVe()

print(fasttext['hello'], glove['hello'])



