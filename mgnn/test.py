# -*- coding:utf-8 -*-

import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
import config_train as args
import pickle
from mgnn import MGNN
from pretreatment.DataExtract import EntityLinking, GetPredicateList
from syntactic_tree import get_syntactic_seq
import numpy as np
import spacy
import time


def get_candicate(sen):
    query = sen.split()

    entities = EntityLinking(sen)
    ent_list = []
    for dic in entities:
        ent_list.append((dic['surfaceForm'], dic['similarityScore']))
    ent_list = sorted(ent_list, key=lambda x: x[1], reverse=True)
    ent = ent_list[0][0].split()[-1]

    # get the predicates candicates
    # predicates = [[item] for item in GetPredicateList(ent)]
    predicates = [['region'], ['test']]

    q_words = ['what', 'how', 'who', 'why', 'where', 'when', 'which', 'whom', 'whose']
    q_word = ''
    for item in sen.split():
        if item.lower() in q_words:
            q_word = item
    assert q_word != ''
    
    syntax = get_syntactic_seq(sen, ent, q_word)

    hier = ['this', 'is', 'test']
    ans = ['test']

    return query, syntax, hier, predicates, ans, ent


def get_vocab(vocab_path):
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab


def get_id_len(text, vocab):
    id_list = []; len_list = [len(text)]
    for word in text:
        the_id = vocab.stoi[word]
        id_list.append(the_id)
    return torch.LongTensor([id_list]), torch.LongTensor(len_list)


def get_input(sen):

    # all these five fields should be list
    # the element of rel_list shoul be list too
    query, syntax, hier, rel_list, ans, ent = get_candicate(sen)

    query_vocab = get_vocab(args.query_vocab)
    syntax_vocab = get_vocab(args.syntax_vocab)
    hier_vocab = get_vocab(args.hier_vocab)
    rel_vocab = get_vocab(args.rel_vocab)
    ans_vocab = get_vocab(args.ans_vocab)

    query, query_length = get_id_len(query, query_vocab)
    syntax, syntax_length = get_id_len(syntax, syntax_vocab)
    hier, hier_vocab = get_id_len(hier, hier_vocab)
    ans, ans_length = get_id_len(ans, ans_vocab)

    rels = []
    for item in rel_list:
        rel, rel_length = get_id_len(item, rel_vocab)
        rels.append((rel, rel_length))

    return query, query_length, syntax, syntax_length, hier, hier_vocab, rels, ans, ans_length, rel_list, ent



print('load model ...')
mgnn = torch.load(args.model_path)

while(1):
    print('please input a sentence:')
    sent = input()
    if sent == '' or sent.isspace():
        break
    else:
        query, query_length, syntax, syntax_length, \
        hier, hier_length, rels, ans, ans_length, rel_list, ent = get_input(sent)

        for (rel, rel_length), ori_rel in zip(rels, rel_list):
            pred_sim = mgnn(query, query_length, syntax, syntax_length, hier, hier_length,
                            rel, rel_length, ans, ans_length)
            print('entity: '+ent, 'predicate: '+ori_rel[0], 'sim:', pred_sim.cpu().item())