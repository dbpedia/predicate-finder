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


def get_idf():
    simple_queries = get_simple_query(paths.lcquad_test)
    count = 0
    idf = {}

    num = 0
    for item in simple_queries:
        num += 1; print(num)

        query = item['corrected_question']
        # text_ents, standard_ents, standard_ent_uries, confs, types = EntityLinking(query, 'more')
        standard_ents, text_ents = Entity_Link_Falcon(query)

        for i in range(len(standard_ents)):
            standard_ent = standard_ents[i]
            predicate_uris = GetPredicateList(standard_ent, template_id=item['sparql_template_id'])
            count += 1

            for predicate_uri in predicate_uris:
                predicate = predicate_uri.split('/')[-1]
                predicate_words = split_predicate(predicate)
                for predicate_word in predicate_words:
                    if predicate_word not in idf:
                        idf[predicate_word] = 1
                    else:
                        idf[predicate_word] += 1

    for k, v in idf.items():
        idf[k] = math.log2(float(count)/v)
    with open('../data/idf.pkl', 'wb') as f:
        pickle.dump(idf, f)
    
    print('get idf done!')

    return idf


def method1(idf):
    simple_queries = get_simple_query(paths.lcquad_test)

    total_res = []
    count = 0

    for item in simple_queries:

        count += 1; print(count)
        # if count < 204: continue

        query = item['corrected_question']; query_words = word_tokenize(query)
        query_id = item['_id']

        # text_ents, standard_ents, standard_ent_uries, confs, types = EntityLinking(query, 'more')
        standard_ents, text_ents = Entity_Link_Falcon(query)

        final_entity = ''; final_predicate = ''; final_score = float('-inf')
        for i in range(len(standard_ents)):
            text_ent = text_ents[i]; standard_ent = standard_ents[i]
            sentence_emb = []

            # remove the entitty
            tmp = []
            for word in query_words:
                if word not in text_ent:
                    tmp.append(word)
            
            sentence_emb = get_ngram_embedding(tmp, 2)  # get 1, 2 gram at the same time
            
            predicate_uris = GetPredicateList(standard_ent, template_id=item['sparql_template_id'])

            ans_predicate = ''; max_score = float('-inf')
            for predicate_uri in predicate_uris:
                predicate = predicate_uri.split('/')[-1]
                predicate_words = split_predicate(predicate)

                idf_score = float('-inf'); in_idf = False
                for predicate_word in predicate_words:
                    if predicate in idf:
                        idf_score = max(idf[predicate_word], idf_score); in_idf = True
                if not in_idf:
                    idf_score = 2.0

                # if not g.emb(predicate)[0]:
                #     predicate_emb = np.array([np.random.uniform(-0.01, 0.01, size=(1, 300))[0]])
                # else:
                #     predicate_emb = np.array([g.emb(predicate)])
                predicate_emb = []
                for predicate_word in predicate_words:
                    predicate_emb.append(fasttext[predicate].numpy())
                predicate_emb = np.array([sum(np.array(predicate_emb))/len(predicate_words)])

                mat = np.dot(sentence_emb, predicate_emb.transpose())
                sentence_norm = np.sqrt(np.multiply(sentence_emb, sentence_emb).sum(axis=1))[:, np.newaxis]
                predicate_norm = np.sqrt(np.multiply(predicate_emb, predicate_emb).sum(axis=1))[:, np.newaxis]
                scores = np.divide(mat, np.dot(sentence_norm, predicate_norm.transpose())+1e-9 )

                scores = scores.squeeze()
                avg = np.max(scores) * idf_score
                if avg > max_score:
                    max_score = avg; ans_predicate = predicate

            if max_score > final_score:
                final_entity = standard_ent; final_predicate = ans_predicate; final_score = max_score

        total_res.append((query, final_entity, final_predicate, final_score))

    with open('../data/base_res.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(total_res)

    print('get test result done!')


if __name__ == '__main__':

    # idf = get_idf()

    with open('../data/idf.pkl', 'rb') as f:
        idf = pickle.load(f)
    method1(idf)
