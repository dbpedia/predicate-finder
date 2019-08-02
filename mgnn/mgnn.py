# -*- coding:utf-8 -*-

import sys

sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mgnn.cnn import CNN


class MGNN(nn.Module):
    def __init__(self, args):
        super(MGNN, self).__init__()

        self.query_emb = self.get_emb(args.query_vocab_size, args.query_emb_size, args.pad_id, args.pre_query_emb,
                                      args.query_emb, args.update_query_emb)
        self.syntax_emb = self.get_emb(args.syntax_vocab_size, args.syntax_emb_size, args.pad_id, args.pre_syntax_emb,
                                       args.syntax_emb, args.update_syntax_emb)
        self.hier_emb = self.get_emb(args.hier_vocab_size, args.hier_emb_size, args.pad_id, args.pre_hier_emb,
                                     args.hier_emb, args.update_hier_emb)
        self.rel_emb = self.get_emb(args.rel_vocab_size, args.rel_emb_size, args.pad_id, args.pre_rel_emb,
                                    args.rel_emb, args.update_rel_emb)

        self.query_cnn = self.get_cnn(args.query_hidden_size, args)
        self.syntax_cnn = self.get_cnn(args.syntax_hidden_size, args)
        self.hier_cnn = self.get_cnn(args.hier_hidden_size, args)

        self.query_encoder = nn.LSTM(args.query_emb_size, args.query_hidden_size, args.layer_num,
                                     batch_first=True, bidirectional=args.bidirectional)
        self.syntax_encoder = nn.LSTM(args.syntax_emb_size, args.syntax_hidden_size, args.layer_num,
                                      batch_first=True, bidirectional=args.bidirectional)
        self.hier_encoder = nn.LSTM(args.hier_emb_size, args.hier_hidden_size, args.layer_num,
                                    batch_first=True, bidirectional=args.bidirectional)

        self.fc = nn.Linear(1 * args.filter_num * 3 + args.rel_emb_size, 1, True)

        self.use_cuda = args.use_cuda

    def get_emb(self, vocab_size, emb_size, pad_id, pre_emb, t_emb, update):
        emb = nn.Embedding(vocab_size, emb_size, padding_idx=pad_id)
        if pre_emb:
            emb.weight = nn.Parameter(torch.from_numpy(t_emb))
        emb.requires_grad = update
        return emb

    def get_cnn(self, embedding_size, args):
        cnn_args = {}
        cnn_args['filter_sizes'] = [args.filter_sizes]
        cnn_args['filter_num'] = args.filter_num
        cnn_args['embedding_size'] = embedding_size * 2
        cnn = CNN(cnn_args)
        return cnn

    def lstm_process(self, inputs, input_lengths, model, input_source):
        # if input_source == 'query':
        #     inputs = nn.utils.rnn.pack_padded_sequence(inputs, input_lengths, batch_first=True)
        #     output, _ = model(inputs)
        #     output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # elif input_source == 'syntax' or input_source == 'hier':
        inputs = inputs.data.cpu().numpy()
        input_lengths = input_lengths.data.cpu().numpy()

        sort_idx = np.argsort(-input_lengths)
        inputs = inputs[sort_idx]
        input_lengths = input_lengths[sort_idx]
        unsort_idx = np.argsort(sort_idx)

        inputs = torch.from_numpy(inputs)
        input_lengths = torch.from_numpy(input_lengths)

        if self.use_cuda:
            inputs = inputs.cuda();
            input_lengths = input_lengths.cuda()
        inputs = nn.utils.rnn.pack_padded_sequence(inputs, input_lengths, batch_first=True)
        output, _ = model(inputs)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        output = output[unsort_idx]
        # input_lengths = input_lengths[unsort_idx]

        return output

    def mask(self, inputs, input_lengths):
        inputs = torch.transpose(inputs, 1, 2)
        seq_len = inputs.size(2);
        batch_size = inputs.size(0)
        mask_ = torch.arange(1, seq_len + 1).type(torch.LongTensor).repeat(batch_size, 1)
        if self.use_cuda:
            mask_ = mask_.cuda()
        mask_ = mask_.le(input_lengths.unsqueeze(1)).unsqueeze(1)
        inputs.data.masked_fill_(1 - mask_, 0.0)
        inputs = torch.transpose(inputs, 1, 2)
        return inputs

    def forward(self, query, query_length, syntax, syntax_length, hier, hier_length, rel, rel_length):

        query = self.query_emb(query)
        syntax = self.syntax_emb(syntax)
        hier = self.hier_emb(hier)
        rel = self.rel_emb(rel)

        query = self.lstm_process(query, query_length, self.query_encoder, 'query')
        syntax = self.lstm_process(syntax, syntax_length, self.syntax_encoder, 'syntax')
        hier = self.lstm_process(hier, hier_length, self.hier_encoder, 'hier')

        query = self.mask(query, query_length)
        syntax = self.mask(syntax, syntax_length)
        hier = self.mask(hier, hier_length)

        query = self.query_cnn(query)
        syntax = self.syntax_cnn(syntax)
        hier = self.hier_cnn(hier)
        rel = rel.squeeze(1)

        ques = torch.cat([query, syntax], 1)
        pred = torch.cat([hier, rel], 1)

        q_p = torch.cat([ques, pred], 1)

        sim = F.sigmoid(self.fc(q_p))

        return sim
