#!/usr/bin/env/ python3

import json
import sys
sys.path.append('..')
import mgnn.config_train as args
import paths

import re
import csv
import pickle
from pretreatment.syntactic_tree import get_syntactic_seq, get_syntactic_seq_from_tree, parse_sentence
from nltk.tokenize import word_tokenize
from pretreatment.DataExtract import GetHierLabel, get_qword, Entity_Link_Falcon, GetPredicateList
import random
from pretreatment.get_info_from_json import *


def get_simple_query(file_path, temp_ids = []):
    with open(file_path, "r") as f:
        data = json.load(f)
    
    if not temp_ids:
        temp_ids = [1, 2, 101, 151, 152]

    simple_queries = [item for item in data if item['sparql_template_id'] in temp_ids]

    return simple_queries

def get_complex_query(file_path, temp_ids = []):
    with open(file_path, "r") as f:
        data = json.load(f)

    simples = [1, 2, 101, 151, 152]

    if not temp_ids:
        complex_queries = [item for item in data if item['sparql_template_id'] not in simples]
    else:
        complex_queries = [item for item in data if item['sparql_template_id'] in temp_ids]
    
    return complex_queries


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

def get_ent_in_query(query_words, text_ent):
    ent_in_query = ''
    for word in query_words:
        if word.lower() == text_ent.lower():
            ent_in_query = word; break
    return ent_in_query


def GetSimpleQueryForTrain():

    simple_queries = get_simple_query(paths.lcquad_train)
    res = []

    counter = 0
    for item in simple_queries:

        counter += 1; print(counter)

        try:
            if item["sparql_template_id"] == 2:
                standard_ent, entity, predicate_uri, predicate = get_for_2(item)
            elif item["sparql_template_id"] == 1 or item["sparql_template_id"] == 101:
                standard_ent, entity, predicate_uri, predicate = get_for_1_101(item)
            elif item["sparql_template_id"] == 151 or item["sparql_template_id"] == 152:
                standard_ent, entity, predicate_uri, predicate = get_for_151_152(item)

            r_predicate = " ".join(split_predicate(predicate))
            
            text_ent = re.sub("[\s+\.\!,&$%^*)(+\"\']+|[+——！，。？、~@#￥%……&*（）]+",  "", standard_ent)
            text_ent = [item.lower() for item in text_ent.replace(' ', '').split('_') ][-1]

            if standard_ent == '' or predicate_uri == '': continue

        except Exception as e:
            print('some error in parse the question!!!'); print(e)
            continue

        query = item['corrected_question']; print(query)
        query_words = word_tokenize(query)

        ent_in_query = get_ent_in_query(query_words, text_ent)
        if not ent_in_query: continue

        q_word = get_qword(query_words)
        try:
            syntax = " ".join(get_syntactic_seq(query, ent_in_query, q_word))
        except Exception as e:
            print('some error in syntax!!!'); print(e)
            continue

        hier = " ".join(GetHierLabel(standard_ent, predicate_uri, item["sparql_template_id"]))
        if not hier: hier = r_predicate


        all_predicate_uris = GetPredicateList(standard_ent, template_id=item['sparql_template_id'])
        random.shuffle(all_predicate_uris)
        f_predicates = []; f_hiers = []; count = 0
        for t_predicate_uri in all_predicate_uris:
            t_predicate = t_predicate_uri.split('/')[-1]
            if t_predicate != predicate:
                f_predicates.append( " ".join(split_predicate(t_predicate) ) )

                f_hier = " ".join(GetHierLabel(standard_ent, t_predicate_uri, item["sparql_template_id"]))
                if not f_hier: f_hier = " ".join(split_predicate(t_predicate) )
                f_hiers.append(f_hier)

                count += 1
                if count >= 4: break

        more_wrong_pair = []
        all_standard_ents, all_text_ents = Entity_Link_Falcon(query)
        for i in range(len(all_standard_ents)):

            t_stan_ent = all_standard_ents[i]; t_text_ent = all_text_ents[i].split(" ")[-1]
            if t_stan_ent == standard_ent: continue

            t_ent_in_query = get_ent_in_query(query_words, t_text_ent)
            if not t_ent_in_query: continue
            try:
                t_syntax = " ".join(get_syntactic_seq(query, t_ent_in_query, q_word))
            except Exception as e:
                print('some error in syntax!!!'); print(e)
                continue

            all_predicate_uris = GetPredicateList(t_stan_ent, template_id=item['sparql_template_id'])
            random.shuffle(all_predicate_uris); tmp_count = 0
            for t_predicate_uri in all_predicate_uris:
                t_predicate = t_predicate_uri.split('/')[-1]
                t_predicate = " ".join(split_predicate(t_predicate))
                t_hier = " ".join(GetHierLabel(t_stan_ent, t_predicate_uri, item["sparql_template_id"]))
                if not t_hier: t_hier = t_predicate
                more_wrong_pair.append((query, t_syntax, t_hier, t_predicate, 0))
                tmp_count += 1
                if tmp_count >= 2: break

            break  # 只考虑一个错误的entity

        res.append((query, syntax, hier, r_predicate, 1))
        for f_hier, f_predicate in zip(f_hiers, f_predicates):
            res.append((query, syntax, f_hier, f_predicate, 0))
        res.extend(more_wrong_pair)


    train_res = res[:int(0.9*len(res))]
    dev_res = res[int(0.9*len(res)):]

    # with open(args.train_data, 'w') as f:
    #     writer = csv.writer(f, delimiter='\t')
    #     writer.writerows(train_res)
    # with open(args.dev_data, 'w') as f:
    #     writer = csv.writer(f, delimiter='\t')
    #     writer.writerows(dev_res)
    
    with open('../data/big_train_data_good.csv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(train_res)
    with open('../data/big_dev_data_good.csv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(dev_res)

    print('done!')


def GetSimpleQueryForTest():

    simple_queries = get_simple_query(paths.lcquad_train, temp_ids=[1,2,101])
    res = {}

    counter = 0
    for item in simple_queries:

        counter += 1; print(counter); print(item['corrected_question'])

        if counter > 2:
            break

        cur_id = item['_id']

        try:
            if item["sparql_template_id"] == 2:
                standard_ent, entity, predicate_uri, standard_predicate = get_for_2(item)
            elif item["sparql_template_id"] == 1 or item["sparql_template_id"] == 101:
                standard_ent, entity, predicate_uri, standard_predicate = get_for_1_101(item)
            elif item["sparql_template_id"] == 151 or item["sparql_template_id"] == 152:
                standard_ent, entity, predicate_uri, standard_predicate = get_for_151_152(item)

            if standard_ent == '' or predicate_uri == '': continue
            
            res[cur_id] = {'standard_ent':standard_ent, 'standard_predicate':standard_predicate, 'candidates':[]}

        except Exception as e:
            print('some error in parse the question!!!'); print(e)
            continue


        query = item['corrected_question']; query_words = word_tokenize(query); q_word = get_qword(query_words)
        template_id = item['sparql_template_id']
        all_ents, all_text_ents = Entity_Link_Falcon(query)

        for i in range(len(all_ents)):

            ent = all_ents[i]; text_ent = all_text_ents[i].split(" ")[-1]

            ent_in_query = get_ent_in_query(query_words, text_ent)
            if not ent_in_query: continue

            try:
                syntax = get_syntactic_seq(query, ent_in_query, q_word)  # list
            except Exception as e:
                print('some error in syntax!!!'); print(e)
                continue

            all_predicate_uris = GetPredicateList(ent, template_id=template_id)

            for predicate_uri in all_predicate_uris:

                predicate = predicate_uri.split('/')[-1]
                text_predicate = split_predicate(predicate)  # list

                hier = GetHierLabel(ent, predicate_uri, template_id)  # list
                if not hier: hier = text_predicate

                res[cur_id]['candidates'].append((query.split(), syntax, hier, text_predicate, ent, predicate))

    with open('../data/datafortest_train.pkl', 'wb') as f:
        pickle.dump(res, f)


def GetSimpleQueryForTest_xgboost():
    simple_queries = get_simple_query(paths.lcquad_test)

    total_res = []

    count = 0

    for item in simple_queries:

        count += 1; print(count)

        res_item = []

        query = item['corrected_question']
        query_id = item['_id']

        parse_tree = parse_sentence(query)

        query_words = word_tokenize(query)
        q_word = get_qword(query_words)

        text_ents, standard_ents, standard_ent_uries, confs, types = EntityLinking(query, 'more')

        for i in range(len(standard_ents)):
            text_ent = text_ents[i]; standard_ent = standard_ents[i]
            conf = confs[i]; standard_ent_uri = standard_ent_uries[i]

            try:
                syntax = get_syntactic_seq_from_tree(parse_tree, text_ent.split()[0], q_word)  # List[Str]
                if len(syntax) < 2: syntax.append(q_word)
            except Exception as e:
                print('some errors in get_syntactic_seq'); print(e)
                syntax = [text_ent.split()[0], q_word]
            
            predicate_uris = GetPredicateList(standard_ent, template_id=item['sparql_template_id'])

            for predicate_uri in predicate_uris:

                predicate = predicate_uri.split('/')[-1]

                # hier = GetHierLabel(standard_ent, predicate_uri)  # List[Str]
                # if not hier: hier = [predicate] * 2
                hier = [types[i]] * 2
                
                res_item.append((syntax, hier, [predicate], conf, standard_ent_uri, predicate_uri))

        if res_item:
            total_res.append((query_id, query, res_item))

    with open(args.xgb_test, 'wb') as f:
        pickle.dump(total_res, f)

    print('done!')

def get_stand_ans_for_simple():

    simple_queries = get_simple_query(paths.lcquad_test)

    res = []

    counter = 0

    for item in simple_queries:

        counter += 1; print(counter)

        try:
            if item["sparql_template_id"] == 2:
                standard_ent, entity, predicate_uri, predicate = get_for_2(item)
            elif item["sparql_template_id"] == 1 or item["sparql_template_id"] == 101:
                standard_ent, entity, predicate_uri, predicate = get_for_1_101(item)
            elif item["sparql_template_id"] == 151 or item["sparql_template_id"] == 152:
                standard_ent, entity, predicate_uri, predicate = get_for_151_152(item)

            res.append((item['corrected_question'], standard_ent, predicate))
            
        except Exception as e:
            print('some error in parse the question!!!')
            print(e)
            continue

    with open('../data/gold_simple_test.csv', "w") as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerows(res)


def get_stand_ans_for_complex():
    complex_queries = get_complex_query(paths.lcquad_test)

    res = []

    counter = 0

    for item in complex_queries:
        print('hhhhh')

        counter += 1; print(counter)

        try:
            if item["sparql_template_id"] == 102:
                standard_ent_1, standard_pre_1, standard_ent_2, standard_pre_2 = get_for_102(item)
            elif item["sparql_template_id"] in [15,16]:
                standard_ent_1, standard_pre_1, standard_ent_2, standard_pre_2 = get_for_15_16(item)
            elif item["sparql_template_id"] in [111,5,105]:
                standard_ent_1, standard_pre_1, standard_ent_2, standard_pre_2 = get_for_111_5_105(item)
            elif item["sparql_template_id"] in [6,106]:
                standard_ent_1, standard_pre_1, standard_ent_2, standard_pre_2 = get_for_6_106(item)
            elif item["sparql_template_id"] in [7,108,8]:
                standard_ent_1, standard_pre_1, standard_ent_2, standard_pre_2 = get_for_7_108_8(item)
            elif item["sparql_template_id"] in [3,11,103]:
                standard_ent_1, standard_pre_1, standard_ent_2, standard_pre_2 = get_for_3_11_103(item)
            elif item["sparql_template_id"] in [301,401,601]:
                standard_ent_1, standard_pre_1, standard_ent_2, standard_pre_2 = get_for_301_401_601(item)
            elif item["sparql_template_id"] in [402]:
                standard_ent_1, standard_pre_1, standard_ent_2, standard_pre_2 = get_for_402(item)
            elif item["sparql_template_id"] in [305,403,405,311]:
                standard_ent_1, standard_pre_1, standard_ent_2, standard_pre_2 = get_for_305_403_405_311(item)
            elif item["sparql_template_id"] in [307,308]:
                standard_ent_1, standard_pre_1, standard_ent_2, standard_pre_2 = get_for_307_308(item)
            elif item["sparql_template_id"] in [406,306]:
                standard_ent_1, standard_pre_1, standard_ent_2, standard_pre_2 = get_for_406_306(item)
            elif item["sparql_template_id"] in [303]:
                standard_ent_1, standard_pre_1, standard_ent_2, standard_pre_2 = get_for_303(item)
            elif item["sparql_template_id"] in [315]:
                standard_ent_1, standard_pre_1, standard_ent_2, standard_pre_2 = get_for_303(item)

            res.append((item['corrected_question'], standard_ent_1, standard_pre_1, standard_ent_2, standard_pre_2))
            
        except Exception as e:
            print('some error in parse the question!!!')
            print(e)
            continue
    
    with open('../data/gold_complex_test.csv', "w") as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerows(res)


if __name__ == '__main__':

    # GetSimpleQueryForTrain()
    # GetSimpleQueryForTest()
    # get_stand_ans_for_simple()
    get_stand_ans_for_complex()

