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
from pretreatment.DataExtract import EntityLinking, GetPredicateList, GetHierLabel, get_qword
from pretreatment.QueryFilter import get_simple_query
from pretreatment.syntactic_tree import get_syntactic_seq
import numpy as np
import spacy
import time
import json
from nltk.tokenize import word_tokenize
import csv


def pad_data(data, vocab):
    pad_id = vocab.stoi['<pad>']

    max_length = 0
    for item in data:
        max_length = max_length if max_length >= len(item) else len(item)

    for i in range(len(data)):
        while len(data[i]) < max_length:
            data[i].append(pad_id)
    
    return data


def get_vocab(vocab_path):
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab


def get_id_len(text, vocab):
    id_list = []
    for word in text:
        the_id = vocab.stoi[word]
        id_list.append(the_id)
    return id_list, len(text)


def get_input(candicates):

    query_vocab = get_vocab(args.query_vocab)
    syntax_vocab = get_vocab(args.syntax_vocab)
    hier_vocab = get_vocab(args.hier_vocab)
    rel_vocab = get_vocab(args.rel_vocab)

    queries, queries_length = [], []
    syntaxs, syntaxs_length = [], []
    hiers, hiers_length = [], []
    rels, rels_length = [], []

    for query, syntax, hier, rel in candicates:
    
        query, query_length = get_id_len(query, query_vocab); queries.append(query); queries_length.append(query_length)
        syntax, syntax_length = get_id_len(syntax, syntax_vocab); syntaxs.append(syntax); syntaxs_length.append(syntax_length)
        hier, hier_length = get_id_len(hier, hier_vocab); hiers.append(hier); hiers_length.append(hier_length)
        rel, rel_length = get_id_len(rel, rel_vocab); rels.append(rel); rels_length.append(rel_length)

    queries = pad_data(queries, query_vocab)
    syntaxs = pad_data(syntaxs, syntax_vocab)
    hiers = pad_data(hiers, hier_vocab)
    rels = pad_data(rels, rel_vocab)    

    queries = torch.LongTensor(queries); queries_length = torch.LongTensor(queries_length)
    syntaxs = torch.LongTensor(syntaxs); syntaxs_length = torch.LongTensor(syntaxs_length)
    hiers = torch.LongTensor(hiers); hiers_length = torch.LongTensor(hiers_length)
    rels = torch.LongTensor(rels); rels_length = torch.LongTensor(rels_length)

    return queries, queries_length, syntaxs, syntaxs_length, hiers, hiers_length, rels, rels_length




if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()

    print('load model ...')
    mgnn = torch.load(args.model_path)

    if use_cuda:
        mgnn = mgnn.cuda()

    # read data from test_data.json and calculate the score of the model
    simple_queries = get_simple_query(args.test_data)

    all_question_predicates = []

    for item in simple_queries:
        
        entity_predicate = []
        model_input = []

        query = item['corrected_question']
        text_ents, standard_ents= EntityLinking(query) 

        query_words = word_tokenize(query)
        q_word = get_qword(query_words)

        # flag = False

        for text_ent, standard_ent in zip(text_ents, standard_ents):
            
            # if 'Brown_House' in standard_ent:
            #     flag = True

            try:
                syntax = get_syntactic_seq(query, text_ent.split()[0], q_word)  # List[Str]
                if len(syntax) < 2:
                    syntax.append(q_word)
            except Exception as e:
                print('some errors in get_syntactic_seq')
                print(e)
                syntax = [text_ent.split()[0], q_word]

            predicate_uris = GetPredicateList(standard_ent)  # List[URI]

            if not predicate_uris:
                continue

            for predicate_uri in predicate_uris:

                predicate = predicate_uri.split('/')[-1]
                if '#' in predicate:
                    continue

                hier = GetHierLabel(standard_ent, predicate_uri)  # List[Str]
                if not hier:  # If there is not the hier feature, we will use the predicate to be the hier feature
                    hier = [predicate] * 2

                if 'subject' in predicate or 'wiki' in predicate or 'hypernym' in predicate:
                    continue

                model_input.append((query.split(), syntax, hier, [predicate]))

                entity_predicate.append((standard_ent, predicate))
        
        # print(model_input)

        # if flag:
        #     print(model_input)

        if not model_input:
            continue

        query, query_length, syntax, syntax_length, hier, hier_length, rel, rel_length = get_input(model_input)

        if use_cuda:
            query = query.cuda(); query_length = query_length.cuda()
            syntax = syntax.cuda(); syntax_length = syntax_length.cuda()
            hier = hier.cuda(); hier_length = hier_length.cuda()
            rel = rel.cuda(); rel_length = rel_length.cuda()

        pred_sim = mgnn(query, query_length, syntax, syntax_length, hier, hier_length, rel, rel_length)

        res = []
        for (entity, predicate), score in zip(entity_predicate, pred_sim):
            print('entity: '+entity, 'predicate: '+predicate, 'sim: ', score.cpu().item())
            res.append((entity, predicate, score.cpu().item()))
        
        res.sort(key=lambda x: x[2], reverse=True)

        all_question_predicates.append((item['corrected_question'], res[0][0], res[0][1], res[0][2]))

    with open(args.result_path, "w") as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerows(all_question_predicates)

    print('done!!!')
