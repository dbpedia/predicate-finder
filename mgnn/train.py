# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
import config_train as args
import pickle
from mgnn import MGNN
import numpy as np
# from embeddings import GloveEmbedding
# from sklearn.metrics import classification_report, precision_recall_fscore_support
import spacy
import time


queryF = Field(sequential=True, batch_first=True, lower=True, include_lengths=True)
syntaxF = Field(sequential=True, batch_first=True, lower=True, include_lengths=True)
hierF = Field(sequential=True, batch_first=True, lower=True, include_lengths=True)
relF = Field(sequential=True, batch_first=True, lower=True, include_lengths=True)
ansF = Field(sequential=True, batch_first=True, lower=True, include_lengths=True)
labelF = Field(sequential=False, batch_first=True, use_vocab=False)


print('load data...')
train = TabularDataset(path=args.train_data, format='tsv', 
                       fields=[('query', queryF), ('syntax', syntaxF), ('hier', hierF), ('rel', relF), ('ans', ansF),
                               ('label', labelF)])
dev = TabularDataset(path=args.dev_data, format='tsv', 
                       fields=[('query', queryF), ('syntax', syntaxF), ('hier', hierF), ('rel', relF), ('ans', ansF),
                               ('label', labelF)])

queryF.build_vocab(train, min_freq=1)
syntaxF.build_vocab(train, min_freq=1)
hierF.build_vocab(train, min_freq=1)
relF.build_vocab(train, min_freq=1)
ansF.build_vocab(train, min_freq=1)

args.query_vocab_size = len(queryF.vocab)
args.syntax_vocab_size = len(syntaxF.vocab)
args.hier_vocab_size = len(hierF.vocab)
args.rel_vocab_size = len(relF.vocab)
args.ans_vocab_size = len(ansF.vocab)

args.pad_id = queryF.vocab.stoi['<pad>']

args.query_emb = None
args.syntax_emb = None
args.hier_emb = None
args.rel_emb = None
args.ans_emb = None

args.update_query_emb = True
args.update_syntax_emb = True
args.update_hier_emb = True
args.update_rel_emb = True
args.update_ans_emb = True


print('build batch iterator...')
train_batch_iterator = BucketIterator(
    dataset=train, batch_size=args.batch_size,
    sort=False, sort_within_batch=True,
    sort_key=lambda x: len(x.query),
    repeat=False
)
dev_batch_iteraor = BucketIterator(
    dataset=dev, batch_size=args.batch_size,
    sort=False, sort_within_batch=True,
    sort_key=lambda x: len(x.query),
    repeat=False
)


mgnn = MGNN(args)
optimizer = torch.optim.Adam(mgnn.parameters(), lr=args.lr)
loss_func = nn.MSELoss()


def get_log(the_F, the_seq, the_len, log):
    for j in range(the_len):
        log += the_F.vocab.itos[the_seq[j]] + ' '
    log += ' ; '
    return log


def run(batch_generator, mode, best_dev_loss):
    if mode == 'train':
        mgnn.train()
    else:
        mgnn.eval()

    last_loss_data = float('inf')
    loss_drop_counter = 0

    logs = []

    for batch in batch_generator:
        
        query, query_length = getattr(batch, 'query')
        syntax, syntax_length = getattr(batch, 'syntax')
        hier, hier_length = getattr(batch, 'hier')
        rel, rel_length = getattr(batch, 'rel')
        ans, ans_length = getattr(batch, 'ans')
        label = getattr(batch, 'label').type(torch.FloatTensor).unsqueeze(1)

        pred_sim = mgnn(query, query_length, syntax, syntax_length, hier, hier_length,
                         rel, rel_length, ans, ans_length)
        loss = loss_func(pred_sim, label)
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mgnn.parameters(), 3.0)
            optimizer.step()

        for i in range(len(query)):
            log = ''
            log = get_log(queryF, query[i], query_length[i], log)
            log = get_log(hierF, hier[i], hier_length[i], log)
            log = get_log(relF, rel[i], rel_length[i], log) 
            log = get_log(ansF, ans[i], ans_length[i], log)
            log += str(label[i].item())+' ; ' + str(pred_sim[i].item()) + '\n'
            logs.append(log)
    
        # if mode == 'train':
        #     loss_data = loss.item()
        #     if loss_data >= last_loss_data:
        #         loss_drop_counter += 1
        #         if loss_drop_counter >= 3:
        #             for param_group in optimizer.param_groups:
        #                 param_group['lr'] = param_group['lr'] * 0.5
        #                 loss_drop_counter = 0
        #     else:
        #         loss_drop_counter = 0
        #     last_loss_data = loss_data

    loss = round(loss.item(), 4)
    print(mode, ' loss: ', loss)

    with open(mode+'.txt', 'w') as f:
        for log in logs:
            f.write(log)

    return loss



print('begin training ...')

best_dev_loss = float('inf')

for epoch in range(1, args.epochs+1):

    print('The ', str(epoch), '-th iter: ')

    batch_generator = train_batch_iterator.__iter__()
    loss = run(batch_generator, 'train', best_dev_loss)

    batch_generator = dev_batch_iteraor.__iter__()
    loss = run(batch_generator, 'dev', best_dev_loss)

    if loss < best_dev_loss:
        best_dev_loss = loss

print('The best dev loss is: ', best_dev_loss)
