# -*- coding:utf-8 -*- 

import sys
sys.path.append('..')
import paths
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
import csv
import numpy as np
from pretreatment.QueryFilter import get_simple_query
from pretreatment.DataExtract import EntityLinking, GetPredicateList, GetHierLabel, get_qword
from pretreatment.syntactic_tree import get_syntactic_seq_from_tree, parse_sentence
from utils import *
import pickle
import json

with open('../data/test_data.pkl', 'rb') as f:
    processed_test_data = pickle.load(f)

with open('../data/test-data.json', "r") as f:
    ori_test_data = json.load(f)
ori_dict = {}
for tmp in ori_test_data:
    ori_dict[tmp['_id']] = {'sparql_query':tmp['sparql_query'], 'sparql_template_id':tmp['sparql_template_id']}


my_xgb = xgb.Booster(model_file='../data/xgb.m')

all_question_predicates = []

count = 0
for item in processed_test_data:
    count += 1; print(count)

    query_id = item[0]; query = item[1]; feature_items = item[2]
    sparql_query = ori_dict[query_id]['sparql_query']; sparql_template_id = ori_dict[query_id]['sparql_template_id']

    mgnn_feature = []; other_feature = []; entity_predicate = []
    for f_item in feature_items:
        entity_predicate.append((f_item[4].split('/')[-1], f_item[2][0]))
        mgnn_feature.append([query.split(), f_item[0], f_item[1], f_item[2]])
        other_feature.append([f_item[3],
                             Question_Relation_Embedding_Sim(query, f_item[2][0]),
                             Question_Predicate_Overlap_Number(query, f_item[2][0]),
                             Question_Predicate_Jaro_Winkler_Distance(query, f_item[2][0]),
                             Question_Predicted_Answer_Sim(query, sparql_query, sparql_template_id, f_item[4], f_item[5])] )
    
    mgnn_score = get_mgnn_score(mgnn_feature)
    for i in range(len(mgnn_score)):
        other_feature[i].insert(1, mgnn_score[i] )

    print(other_feature)

    
    sim_data = []
    for tmp in other_feature:
        feature = ''
        for index, val in enumerate(tmp):
            feature += ' ' + str(index+1) + ':' + str(val)
        feature = str(1) + feature
        sim_data.append(feature)
    with open('../data/xgb_test_tmp.txt', 'w') as f:
        for line in sim_data:
            f.write(line+'\n')



    # xgb_input = xgb.DMatrix(np.array(other_feature) )
    xgb_input = xgb.DMatrix('../data/xgb_test_tmp.txt')
    scores = my_xgb.predict(xgb_input)

    res = []
    for (entity, predicate), score in zip(entity_predicate, scores):
        print('entity: '+entity, 'predicate: '+predicate, 'sim: ', score)
        res.append((entity, predicate, score))
    
    res.sort(key=lambda x: x[2], reverse=True)

    all_question_predicates.append((query, res[0][0], res[0][1], res[0][2]))


with open('../data/xgb_result.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t')
    writer.writerows(all_question_predicates)

print('done!!!')