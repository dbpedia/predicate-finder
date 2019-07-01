# -*- coding:utf-8 -*- 

import sys
sys.path.append('..')
import paths
from pretreatment.QueryFilter import get_simple_query, get_for_151_152, get_for_1_101, get_for_2
from pretreatment.DataExtract import EntityLinking, GetPredicateList, GetHierLabel, get_qword
from pretreatment.syntactic_tree import get_syntactic_seq_from_tree, parse_sentence
from utils import *
import re
from nltk.tokenize import word_tokenize


def get_hier(entity, predicate, template_id):
    hier = ' '.join(GetHierLabel(entity, predicate, template_id))
    if not hier:
        hier = predicate + ' ' + predicate
    return hier

def get_syntax(entity, query_words):
    ent_in_query = ''
    for word in query_words:
        if entity.lower() == word.lower():
            ent_in_query = word
            break
    return ent_in_query


simple_queries = get_simple_query(paths.lcquad_train)

# preprocess the original data fot calculating the multi similarities

right_ent_right_predicate = []  # data for calculating other similarities
wrong_ent_right_predicate = []
right_ent_wrong_predicate = []
wrong_ent_wrong_predicate = []

r_r = []; w_r = []; r_w = []; w_w = []  # data for mgnn similarity

for item in simple_queries:

    try:
        if item["sparql_template_id"] == 2:
            standard_ent_uri, standard_ent, entity, predicate_uri, predicate = get_for_2(item, 'more')
        elif item["sparql_template_id"] == 1 or item["sparql_template_id"] == 101:
            standard_ent_uri, standard_ent, entity, predicate_uri, predicate = get_for_1_101(item, 'more')
        elif item["sparql_template_id"] == 151 or item["sparql_template_id"] == 152:
            standard_ent_uri, standard_ent, entity, predicate_uri, predicate = get_for_151_152(item, 'more')
        
        standard_ent = re.sub("[\s+\.\!,&$%^*)(+\"\']+|[+——！，。？、~@#￥%……&*（）]+",  "", standard_ent)
        tmp = standard_ent.split('_')
        standard_ent = [a for a in tmp if a != '']
        entity = standard_ent[-1]; standard_ent = '_'.join(standard_ent)

        if standard_ent == '' or predicate_uri == '':
            continue
    except Exception as e:
        print('some error in parse the question!!!')
        print(e)
        continue

    sparql_query = item['sparql_query']
    sparql_template_id = item['sparql_template_id']

    query = item['corrected_question']
    
    # get the wrong entity and the confidence of it 
    _, pred_standard_ents, pred_standard_ent_uries, sims = EntityLinking(query, 'more')
    if not pred_standard_ents:
        continue
    for pred_standard_ent, pred_standard_ent_uri, sim in zip(pred_standard_ents, pred_standard_ent_uries, sims):
        if pred_standard_ent != standard_ent:
                wrong_standard_ent = pred_standard_ent
                wrong_standard_ent_uri = pred_standard_ent_uri
                wrong_sim = sim
                break

    # get the wrong predicate
    pred_predicate_uris = GetPredicateList(standard_ent)  # List[URI]
    if not pred_predicate_uris:
        continue
    for pred_predicate_uri in pred_predicate_uris:
        pred_predicate = pred_predicate_uri.split('/')[-1]
        if pred_predicate != predicate:
            wrong_predicate = pred_predicate
            wrong_predicate_uri = pred_predicate_uri
            break

    # syntax tree for right entity and wrong entity
    query_words = word_tokenize(query)
    q_word = get_qword(query_words)
    
    ent_in_query = get_syntax(entity, query_words)
    if not ent_in_query:
        continue
    wrong_ent_in_query = get_syntax(wrong_standard_ent.split('_')[0], query_words)
    if not wrong_ent_in_query:
        continue

    parse_tree = parse_sentence(query)
    try:
        right_syntax = ' '.join(get_syntactic_seq_from_tree(parse_tree, ent_in_query, q_word))
        wrong_syntax = ' '.join(get_syntactic_seq_from_tree(parse_tree, wrong_ent_in_query, q_word))
    except Exception as e:
        print('some errors in syntax!!!')
        print(e)
        continue

    # get the hier feature
    r_r_hier = get_hier(standard_ent, predicate_uri, item["sparql_template_id"])
    r_w_hier = get_hier(standard_ent, wrong_predicate_uri, item["sparql_template_id"])
    w_r_hier = get_hier(wrong_standard_ent, predicate_uri, item["sparql_template_id"])
    w_w_hier = get_hier(wrong_standard_ent, wrong_predicate_uri, item["sparql_template_id"])

    # the data for mgnn
    r_r.append((query, right_syntax, r_r_hier, predicate))
    r_w.append((query, right_syntax, r_w_hier, wrong_predicate))
    w_r.append((query, wrong_syntax, w_r_hier, predicate))
    w_w.append((query, wrong_syntax, w_w_hier, wrong_predicate))

    # part of features for xgb
    feature_6 = Question_Predicted_Answer_Sim(query, sparql_query, sparql_template_id, standard_ent_uri, predicate_uri)
    right_ent_right_predicate.append([3.0, query, predicate, 1.0, feature_6])

    feature_6 = Question_Predicted_Answer_Sim(query, sparql_query, sparql_template_id, wrong_standard_ent_uri, predicate_uri)
    wrong_ent_right_predicate.append([2.0, query, predicate, wrong_sim, feature_6])

    feature_6 = Question_Predicted_Answer_Sim(query, sparql_query, sparql_template_id, standard_ent_uri, wrong_predicate_uri)
    right_ent_wrong_predicate.append([2.0, query, wrong_predicate, 1.0, feature_6])

    feature_6 = Question_Predicted_Answer_Sim(query, sparql_query, sparql_template_id, wrong_standard_ent_uri, wrong_predicate_uri)
    wrong_ent_wrong_predicate.append([1.0, query, wrong_predicate, wrong_sim, feature_6])


# get the score from mgnn model
r_r_mgnn = get_mgnn_score(r_r)
r_w_mgnn = get_mgnn_score(r_w)
w_r_mgnn = get_mgnn_score(w_r)
w_w_mgnn = get_mgnn_score(w_w)

# concat mgnn score with other feature
concat_mgnn_score(r_r_mgnn, right_ent_right_predicate)
concat_mgnn_score(r_w_mgnn, right_ent_wrong_predicate)
concat_mgnn_score(w_r_mgnn, wrong_ent_right_predicate)
concat_mgnn_score(w_w_mgnn, wrong_ent_wrong_predicate)

processed_data = [right_ent_right_predicate, wrong_ent_right_predicate, \
        right_ent_wrong_predicate, wrong_ent_wrong_predicate]


# get multi similarities
data_for_xgb = []
for sub_data in processed_data:
    sim_data = []
    for item in sub_data:
        tmp = []
        tmp.append(item[3])
        tmp.append(item[5])
        tmp.append(Question_Relation_Embedding_Sim(item[1], item[2]))
        tmp.append(Question_Predicate_Overlap_Number(item[1], item[2]))
        tmp.append(Question_Predicate_Jaro_Winkler_Distance(item[1], item[2]))
        tmp.append(item[4])

        feature = ''
        for index, val in enumerate(tmp):
            feature += ' ' + str(index+1) + ':' + str(val)
        feature = str(item[0]) + feature

        sim_data.append(tmp)

    data_for_xgb.extend(sim_data)

random.shuffle(data_for_xgb)

with open('../data/xgb_train.txt', 'w') as f:
    f.writelines(data_for_xgb)
    
