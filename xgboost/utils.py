# -*- coding:utf-8 -*-

import sys
sys.path.append('..')
import xgboost as xgb
import Levenshtein
import numpy as np
import re
import spotlight as sl
import random
import pickle
import torch
from pretreatment.test import get_id_len, pad_data
from SPARQLWrapper import SPARQLWrapper, JSON

def get_xgb_data(file_path):
    data = xgb.DMatrix(file_path)
    return data


def Question_Relation_Embedding_Sim(query, predicate):
    query = query.split(); predicate = predicate.split()

    query_id = []
    predicate_id = []
    for word in query:
        query_id.append(query_vocab.stoi[word])
    for word in predicate:
        predicate_id.append(rel_vocab.stoi[word])

    query = np.sum(query_emb.take(query_id, axis=0), axis=0)
    predicate = np.sum(rel_emb.take(predicate_id, axis=0), axis=0)

    cos_sim = np.dot(query, predicate)/(np.linalg.norm(query)*(np.linalg.norm(predicate)))
    return cos_sim


def Question_Predicate_Overlap_Number(query, predicate):
    query = query.split(); predicate = predicate.split()

    tmp_q = [item.lower() for item in query]
    tmp_p = [item.lower() for item in predicate]
    common = len(tmp_q & tmp_p)
    return float(common) / len(query)


def Question_Predicate_Jaro_Winkler_Distance(query, predicate):
    query = query.split(); predicate = predicate.split()

    sim = 0.0
    for q_word in query:
        for p_word in predicate:
            sim += Levenshtein.jaro_winkler(q_word.lower(), p_word.lower())
    return sim / len(query)
    

def Question_Predicted_Answer_Sim(query, sparql_query, parql_template_id, entity_uri, predicate_uri):
    entity_uri = '<'+entity_uri+'>'; predicate_uri = '<'+predicate_uri+'>'

    ent = re.compile(u'<(.*?)>',re.M|re.S|re.I)
    all_items = list(ent.finditer(sparql_query))
    if parql_template_id == 2:
        tmp = sparql_query[:all_items[0].start()] + entity_uri + ' ' + predicate_uri + sparql_query[all_items[1].end():]
    elif parql_template_id == 1 or parql_template_id == 101:
        tmp = sparql_query[:all_items[0].start()] + predicate_uri + ' ' + entity_uri + sparql_query[all_items[1].end():]
    elif parql_template_id == 151 or parql_template_id == 152:
        tmp = sparql_query[:all_items[1].start()] + predicate_uri + ' ' + entity_uri + sparql_query[all_items[2].end():]

    try:
        sparql = SPARQLWrapper("https://dbpedia.org/sparql")
        sparql.setQuery(tmp)
        sparql.setReturnFormat(JSON)
        result = sparql.query().convert()
    except Exception:
        print(sparql_query)

    ans = result['results']['bindings'][0]['uri']['value'].split('/')[-1].split('_')  # List[Str]
    query = query.split()

    query_id = []
    ans_id = []
    for word in query:
        query_id.append(query_vocab.stoi[word])
    for word in ans:
        ans_id.append(query_vocab.stoi[word])

    query = np.sum(query_emb.take(query_id, axis=0), axis=0)
    ans = np.sum(rel_emb.take(ans_id, axis=0), axis=0)

    cos_sim = np.dot(query, ans)/(np.linalg.norm(query)*(np.linalg.norm(ans)))
    return cos_sim


def get_mgnn_score(data):
    res = []

    queries, queries_length = [], []
    syntaxs, syntaxs_length = [], []
    hiers, hiers_length = [], []
    rels, rels_length = [], []

    for query, syntax, hier, rel in data:

        query = query.split(); syntax = syntax.split(); hier = hier.split(); rel = [rel]
    
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


    if use_cuda:
        query = query.cuda(); query_length = query_length.cuda()
        syntax = syntax.cuda(); syntax_length = syntax_length.cuda()
        hier = hier.cuda(); hier_length = hier_length.cuda()
        rel = rel.cuda(); rel_length = rel_length.cuda()

    pred_sim = mgnn(query, query_length, syntax, syntax_length, hier, hier_length, rel, rel_length)

    for sim in pred_sim:
        res.append(sim.cpu().item())

    return res


def concat_mgnn_score(mgnn_score, other_feature):
    for i in range(len(other_feature)):
        other_feature[i].append(mgnn_score[0])


query_vocab = pickle.load(open('../data/query_vocab.pkl', 'rb'))
syntax_vocab = pickle.load(open('../data/syntax_vocab.pkl', 'rb'))
hier_vocab = pickle.load(open('../data/hier_vocab.pkl', 'rb'))
rel_vocab = pickle.load(open('../data/rel_vocab.pkl', 'rb'))
mgnn = torch.load('../data/mgnn.pkl')
query_emb = mgnn.query_emb.weight.data.numpy()
rel_emb = mgnn.rel_emb.weight.data.numpy()

use_cuda = torch.cuda.is_available()
if use_cuda:
    mgnn = mgnn.cuda()