# -*- coding:utf-8 -*-

import json
import sys
sys.path.append('..')
import mgnn.config_train as args
import paths
import re
import csv
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
from pretreatment.DataExtract import EntityLinking, GetPredicateList
from pretreatment.QueryFilter import *
from torchnlp.word_to_vector import FastText, GloVe
# fasttext = FastText()
from embeddings import GloveEmbedding
g = GloveEmbedding('common_crawl_840', d_emb=300)
import math

def get_idf():
    simple_queries = get_simple_query(paths.lcquad_test)
    count = 0
    idf = {}

    num = 0
    for item in simple_queries:
        num += 1; print(num)

        query = item['corrected_question']
        text_ents, standard_ents, standard_ent_uries, confs, types = EntityLinking(query, 'more')

        for i in range(len(standard_ents)):
            standard_ent = standard_ents[i]
            predicate_uris = GetPredicateList(standard_ent, template_id=item['sparql_template_id'])
            count += 1

            for predicate_uri in predicate_uris:
                predicate = predicate_uri.split('/')[-1]
                if predicate not in idf:
                    idf[predicate] = 1
                else:
                    idf[predicate] += 1

    for k, v in idf.items():
        idf[k] = math.log2(float(count)/v)
    with open('../data/idf.pkl', 'wb') as f:
        pickle.dump(idf, f)

    return idf

def method1(idf):
    simple_queries = get_simple_query(paths.lcquad_test)

    total_res = []
    count = 0

    for item in simple_queries:

        count += 1; print(count)

        query = item['corrected_question']; query_words = word_tokenize(query)
        query_id = item['_id']

        text_ents, standard_ents, standard_ent_uries, confs, types = EntityLinking(query, 'more')

        final_entity = ''; final_predicate = ''; final_score = float('-inf')
        for i in range(len(standard_ents)):
            text_ent = text_ents[i]; standard_ent = standard_ents[i]
            sentence_emb = []

            for word in query_words:
                if word not in text_ent:
                    if not g.emb(word)[0]:
                        sentence_emb.append(np.random.uniform(-0.01, 0.01, size=(1, 300))[0])
                    else:
                        sentence_emb.append(np.array(g.emb(word)))
                    # sentence_emb.append(fasttext[word])
            sentence_emb = np.array(sentence_emb)
            
            predicate_uris = GetPredicateList(standard_ent, template_id=item['sparql_template_id'])

            ans_predicate = ''; max_score = float('-inf')
            for predicate_uri in predicate_uris:
                predicate = predicate_uri.split('/')[-1]

                if not g.emb(predicate)[0]:
                    predicate_emb = np.array([np.random.uniform(-0.01, 0.01, size=(1, 300))[0]])
                else:
                    predicate_emb = np.array([g.emb(predicate)])
                # predicate_emb = np.array([fasttext[predicate]])

                mat = np.dot(sentence_emb, predicate_emb.transpose())
                sentence_norm = np.sqrt(np.multiply(sentence_emb, sentence_emb).sum(axis=1))[:, np.newaxis]
                predicate_norm = np.sqrt(np.multiply(predicate_emb, predicate_emb).sum(axis=1))[:, np.newaxis]
                scores = np.divide(mat, np.dot(sentence_norm, predicate_norm.transpose()))

                scores = scores.squeeze()
                avg = np.max(scores)
                if avg > max_score:
                    max_score = avg; ans_predicate = predicate

            if max_score > final_score:
                final_entity = standard_ent; final_predicate = ans_predicate; final_score = max_score

        total_res.append((query, final_entity, final_predicate, final_score))

    with open('../data/base_res.txt', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerows(total_res)

    print('done!')


if __name__ == '__main__':
    # a = np.array([[1.,2.], [2.,3.], [3.,4.]])
    # b = np.array([[5.,6.]])
    # mat = np.dot(a, b.transpose())
    # a = np.sqrt(np.multiply(a, a).sum(axis=1))[:, np.newaxis]
    # b = np.sqrt(np.multiply(b, b).sum(axis=1))[:, np.newaxis]
    # scores = np.divide(mat, np.dot(a, b.transpose()))
    # print(scores)

    idf = get_idf()

    method1(idf)
