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
fasttext = FastText()

def method1():
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
                    sentence_emb.append(fasttext[word])
            sentence_emb = np.array(sentence_emb)
            
            predicate_uris = GetPredicateList(standard_ent, template_id=item['sparql_template_id'])

            ans_predicate = ''; max_score = float('-inf')
            for predicate_uri in predicate_uris:
                predicate = predicate_uri.split('/')[-1]

                predicate_emb = np.array([fasttext[predicate]])

                mat = np.dot(sentence_emb, predicate_emb.transpose())
                sentence_norm = np.sqrt(np.multiply(sentence_emb, sentence_emb).sum(axis=1))[:, np.newaxis]
                predicate_norm = np.sqrt(np.multiply(predicate_emb, predicate_emb).sum(axis=1))[:, np.newaxis]
                socres = np.divide(mat, np.dot(sentence_norm, predicate_norm.transpose()))

                socres = socres.squeeze()
                avg = np.mean(scores)
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

    method1()
