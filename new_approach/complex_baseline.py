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
from pretreatment.DataExtract import EntityLinking, GetPredicateList, Entity_Link_Falcon, GetNextEntity
from pretreatment.QueryFilter import get_simple_query, get_complex_query
from torchnlp.word_to_vector import FastText, GloVe
fasttext = FastText()
from embeddings import GloveEmbedding
g = GloveEmbedding('common_crawl_840', d_emb=300)
import math


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
    complex_queries = get_complex_query(paths.lcquad_test)
    count = 0
    idf = {}

    num = 0
    for item in complex_queries:
        num += 1; print(num)

        query = item['corrected_question']
        standard_ents, text_ents = Entity_Link_Falcon(query)

        for i in range(len(standard_ents)):
            standard_ent = standard_ents[i]

            predicate_uris, predicate_texts = GetPredicateList(standard_ent, template_id=item['sparql_template_id'])
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
    
    print('get idf done!')

    return idf


def get_best_predicate(standard_ent, template_id, sentence_emb, idf):
    predicate_uris, predicate_texts = GetPredicateList(standard_ent, template_id=template_id)

    tmp_predicate = ''; tmp_score = float('-inf')
    for i in range(len(predicate_uris)):
        predicate_uri = predicate_uris[i]; predicate_text = predicate_texts[i]
        predicate = predicate_uri.split('/')[-1]
        if predicate in idf:
            idf_score = idf[predicate]
        else:
            idf_score = 2.0

        predicate_words = predicate_text.split()
        
        predicate_emb = []
        for pw in predicate_words:
            predicate_emb.append(fasttext[pw].numpy())
        predicate_emb = np.array([sum(np.array(predicate_emb))/len(predicate_words)])

        mat = np.dot(sentence_emb, predicate_emb.transpose())
        sentence_norm = np.sqrt(np.multiply(sentence_emb, sentence_emb).sum(axis=1))[:, np.newaxis]
        predicate_norm = np.sqrt(np.multiply(predicate_emb, predicate_emb).sum(axis=1))[:, np.newaxis]
        scores = np.divide(mat, np.dot(sentence_norm, predicate_norm.transpose())+1e-9 )

        scores = scores.squeeze()
        avg = np.max(scores) * idf_score
        if avg > tmp_score:
            tmp_score = avg; tmp_predicate = predicate
    return tmp_predicate, tmp_score



def method(idf):

    complex_queries = get_simple_query(paths.lcquad_test)
    
    total_res = []
    count = 0

    for item in complex_queries:

        count += 1;  print(count)

        query = item['corrected_question']; template_id = item['sparql_template_id']

        query_words = word_tokenize(query)

        standard_ents, text_ents = Entity_Link_Falcon(query)

        # remove all the entities
        rest_sen = []; all_text_ent = ' '.join(text_ents)
        for word in query_words:
            if word not in all_text_ent:
                rest_sen.append(word)
        sentence_emb = get_ngram_embedding(rest_sen, 2)  # get 1, 2 gram

        first_entity = ''; first_predicate = ''; first_score = float('-inf')
        for i in range(len(standard_ents)):

            standard_ent = standard_ents[i]
            
            tmp_predicate, tmp_score = get_best_predicate(standard_ent, item['sparql_template_id'], sentence_emb, idf)
            
            if tmp_score > first_score:
                first_entity, first_predicate, first_score  = standard_ent, tmp_predicate, tmp_score



        # 此时找到了第一个entity和对应的predicate，但是可能还有别的entity和predicate，要分模版讨论

        # 只有一个三元组
        if template_id in [102]:

            total_res.append((query, first_entity, first_predicate, first_score, '', '', 0.0))
        
        # 包含两个三元组
        elif template_id in [15, 16, 7, 108, 8]:

            second_entity = ''; second_predicate = ''; second_score = float('-inf')
            for i in range(len(standard_ents)):

                standard_ent = standard_ents[i]
                if standard_ent == first_entity: continue
                
                tmp_predicate, tmp_score = get_best_predicate(standard_ent, template_id, sentence_emb, idf)

                if tmp_score > second_score:
                    second_entity, second_predicate, second_score = standard_ent, tmp_predicate, tmp_score

            total_res.append((query, first_entity, first_predicate, first_score, second_entity, second_predicate, second_score))

        elif template_id in [111, 5, 105, 6, 106, 3, 11, 103]:

            second_entity = GetNextEntity(first_entity, first_predicate, template_id)
            if not second_entity:
                total_res.append((query, first_entity, first_predicate, first_score, '', '', 0.0))
            else:
                second_predicate, second_score = get_best_predicate(second_entity, template_id, sentence_emb, idf)
                total_res.append((query, first_entity, first_predicate, first_score, second_entity, second_predicate, second_score))

        elif template_id in [301, 401, 601, 402]:

            total_res.append((query, first_entity, first_predicate, first_score, '', '', 0.0))

        # 包含三个三元组
        elif template_id in [305, 403, 405, 311, 406, 306, 303]:

            second_entity = GetNextEntity(first_entity, first_predicate, template_id)
            if not second_entity:
                total_res.append((query, first_entity, first_predicate, first_score, '', '', 0.0))
            else:
                second_predicate, second_score = get_best_predicate(second_entity, template_id, sentence_emb, idf)
                total_res.append((query, first_entity, first_predicate, first_score, second_entity, second_predicate, second_score))

        elif template_id in [307, 308, 315]:

            second_entity = ''; second_predicate = ''; second_score = float('-inf')
            for i in range(len(standard_ents)):

                standard_ent = standard_ents[i]
                if standard_ent == first_entity: continue
                
                tmp_predicate, tmp_score = get_best_predicate(standard_ent, template_id, sentence_emb, idf)

                if tmp_score > second_score:
                    second_entity, second_predicate, second_score = standard_ent, tmp_predicate, tmp_score

            total_res.append((query, first_entity, first_predicate, first_score, second_entity, second_predicate, second_score))


    with open('../data/base_res_complex.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(total_res)

    print('get test result done!')


if __name__ == '__main__':

    idf = get_idf()

    with open('../data/idf.pkl', 'rb') as f:
        idf = pickle.load(f)
    method(idf)
