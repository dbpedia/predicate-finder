# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from cnn import CNN

class Cla(nn.Module):
    def __init__(self, args):
        super(Cla, self).__init__()
        self.embedding_size = args.embedding_size
        self.hidden_size = args.hidden_size
        self.bidirectional = args.bidirectional
        self.vocab_size = args.vocab_size
        self.layer_num = args.layer_num

        if args.rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=args.pad_id)

        if args.pre_embedding:
            self.embedding.weight = nn.Parameter(torch.from_numpy(args.word_embedding))
        self.embedding.requires_grad = args.update_embedding

        self.cnn_args = {}
        self.cnn_args['filter_sizes'] = list(args.filter_sizes)
        self.cnn_args['filter_num'] = args.filter_num
        self.cnn_args['embedding_size'] = self.embedding_size
        self.cnn = CNN(self.cnn_args)
    
        self.sentential_encoder = self.rnn_cell(self.word_embedding_size, self.word_hidden_size, self.layer_num, 
                                                batch_first=True, bidirectional=self.word_bidirectional)


    def lstm_process(self, inputs, input_lengths):
        inputs = nn.utils.rnn.pack_padded_sequence(inputs, input_lengths, batch_first=True)
        output, _ = self.sentential_encoder(inputs)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output

    def mask(self, inputs, input_lengths):
        inputs = torch.transpose(inputs, 1, 2)
        seq_len = inputs.size(2); batch_size = src_p.size(0)
        mask_ = torch.arange(1, seq_len+1).type(torch.LongTensor).repeat(batch_size, 1).cuda()
        mask_ = mask_.le(input_lengths.unsqueeze(1)).unsqueeze(1)
        inputs.data.masked_fill_(1-mask_, 0.0)
        inputs = torch.transpose(inputs, 1, 2)
        return inputs

    def forward(self, query, query_length):

        query = self.embedding(query)

        hidden_states = self.lstm_process(query, query_length)

        hidden_states = self.mask(hidden_states, query_length)

        return hidden_states
