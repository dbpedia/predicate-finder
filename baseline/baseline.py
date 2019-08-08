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
from pretreatment.DataExtract import EntityLinking, GetPredicateList, Entity_Link_Falcon
from pretreatment.QueryFilter import *
from torchnlp.word_to_vector import FastText, GloVe
fasttext = FastText()
from embeddings import GloveEmbedding
g = GloveEmbedding('common_crawl_840', d_emb=300)
import math

def get_case():
    with open('../data/base_res.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        inf_case = []
        for row in csv_reader:
            if row[-1] == '-inf':
                inf_case.append(row[0])
        return inf_case

def get_ngram(text, n):
    word_list = text
    res = []
    for i in range(len(word_list)):
        if i+n > len(word_list):
            break
        res.append(word_list[i:i+n])
    return res


def get_ngram_embedding(text, n):
    embeddings = []
    for i in range(len(text)):
        for j in range(1, n+1):
            words = text[i:i+j]
            if len(words) > 0:
                embedding = []
                for word in words:
                    embedding.append(fasttext[word].numpy())
                embeddings.append(sum(np.array(embedding))/len(words))
    return embeddings


def split_predicate(predicate):
    res = []; word = []

    for ch in predicate:
        if ch.isupper():
            res.append(''.join(word))
            word = [ch.lower()]
        else:
            word.append(ch)
    if word:
        res.append(''.join(word))

    return res

def merge_predicate(predicate):
    tmp = predicate.split()
    if len(tmp) == 1:
        return tmp[0]
    for i in range(1, len(tmp)):
        tmp[i] = tmp[i].capitalize()
    return ''.join(tmp)



def get_idf():
    simple_queries = get_simple_query(paths.lcquad_test)
    count = 0
    idf = {}

    num = 0
    for item in simple_queries:
        num += 1; print(num)

        query = item['corrected_question']
        standard_ents, text_ents = Entity_Link_Falcon(query)

        for i in range(len(standard_ents)):
            standard_ent = standard_ents[i]
            predicate_uris = GetPredicateList(standard_ent, template_id=item['sparql_template_id'])
            count += 1

            for predicate_uri in predicate_uris:
                predicate = predicate_uri.split('/')[-1]
                # predicate = predicate_uri
                if predicate not in idf:
                    idf[predicate] = 1
                else:
                    idf[predicate] += 1
            
    for k, v in idf.items():
        idf[k] = math.log2(float(count)/v)
    with open('../data/old_idf.pkl', 'wb') as f:
        pickle.dump(idf, f)
    
    print('get idf done!')

    return idf


def method1(idf):
    simple_queries = get_simple_query(paths.lcquad_test)
    
    # simple_queries = get_simple_query(paths.lcquad_case)

    total_res = []
    count = 0

    for item in simple_queries:

        count += 1;  print(count)

        query = item['corrected_question']
        # if query != "what kind of things play on WBIG FM?": 
        #     continue

        query_words = word_tokenize(query)
        standard_ents, text_ents = Entity_Link_Falcon(query)

        # remove all the entities
        rest_sen = []; all_text_ent = ' '.join(text_ents)
        for word in query_words:
            if word not in all_text_ent:
                rest_sen.append(word)
        sentence_emb = get_ngram_embedding(rest_sen, 2)  # get 1, 2 gram

        final_entity = ''; final_predicate = ''; final_score = float('-inf')
        for i in range(len(standard_ents)):

            standard_ent = standard_ents[i]
            
            predicate_uris = GetPredicateList(standard_ent, template_id=item['sparql_template_id'])

            tmp_predicate = ''; tmp_score = float('-inf')
            for predicate_uri in predicate_uris:
                predicate = predicate_uri.split('/')[-1]
                # predicate = predicate_uri
                if predicate in idf:
                    idf_score = idf[predicate]
                else:
                    idf_score = 2.0

                predicate_words = split_predicate(predicate)
                # predicate_words = predicate.split()
                
                predicate_emb = []
                for pw in predicate_words:
                    predicate_emb.append(fasttext[pw].numpy())
                predicate_emb = np.array([sum(np.array(predicate_emb))/len(predicate_words)])

                mat = np.dot(sentence_emb, predicate_emb.transpose())
                sentence_norm = np.sqrt(np.multiply(sentence_emb, sentence_emb).sum(axis=1))[:, np.newaxis]
                predicate_norm = np.sqrt(np.multiply(predicate_emb, predicate_emb).sum(axis=1))[:, np.newaxis]
                scores = np.divide(mat, np.dot(sentence_norm, predicate_norm.transpose())+1e-9 )

                scores = scores.squeeze()
                # print(scores)
                avg = np.max(scores) * idf_score
                # print(idf_score)
                if avg > tmp_score:
                    tmp_score = avg; tmp_predicate = predicate

            if tmp_score > final_score:
                final_entity = standard_ent
                final_predicate = tmp_predicate
                # final_predicate = merge_predicate(tmp_predicate)
                final_score = tmp_score

        total_res.append((query, final_entity, final_predicate, final_score))

    with open('../data/old_base_res.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(total_res)

    print('get test result done!')


if __name__ == '__main__':

    idf = get_idf()

    with open('../data/old_idf.pkl', 'rb') as f:
        idf = pickle.load(f)
    method1(idf)
