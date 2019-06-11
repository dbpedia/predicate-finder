# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.filter_sizes = args['filter_sizes']
        self.filter_num = args['filter_num']
        self.embedding_size = args['embedding_size']
        
        self.convs = nn.ModuleList([nn.Conv2d(1, self.filter_num, (filter_size, self.embedding_size)) for filter_size in self.filter_sizes])
        
    def forward(self, inputs):
        x = inputs.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        return x
