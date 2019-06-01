#!/usr/bin/env/ python3

from torchnlp.word_to_vector import FastText, GloVe
import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable

# please install the latest code of pytorch_nlp

# the pre-trained data will be downloaded when you first run, which are nearly 9GB in total
fasttext = FastText()
glove = GloVe()


EMB_SIZE = 300

def getQuery(query):

    words_list = query.split(' ')
    input = np.zeros(shape=(len(words_list), 1, EMB_SIZE))
    for i in range(len(words_list)):
        input[i, 0] = fasttext[words_list[i]]

    lstm = nn.LSTM(EMB_SIZE, 150, 1, bidirectional=True)

    input = Variable(torch.from_numpy(input).float())
    h0 = Variable(torch.randn(2, 1, 150).float())
    c0 = Variable(torch.randn(2, 1, 150).float())

    output, hn = lstm(input, (h0, c0))

    return output.transpose(0, 1)[0, -1]

getQuery("hello professor")