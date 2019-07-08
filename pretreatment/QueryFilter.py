#!/usr/bin/env/ python3

import json
import sys
sys.path.append('..')
import paths

# set "predicate_finder" as source root
# from paths import lcquad_train
import re
import csv
import pickle
from pretreatment.syntactic_tree import get_syntactic_seq, get_syntactic_seq_from_tree, parse_sentence
from nltk.tokenize import word_tokenize
from pretreatment.DataExtract import GetHierLabel, get_qword, EntityLinking, GetPredicateList


'''
对于152，predicate是中间那个，entity是第一个，作为subject
对于151，同152
对于101，predicate是中间那个，entity是最后那个，作为object
对于1，同101
对于2，predicate是唯一的那一个，entity是sparql_query中的第一个，作为subject
'''

ent = re.compile(u'<(.*?)>',re.M|re.S|re.I)

def get_simple_query(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    simple_temp_one = [1, 2, 101]
    simple_temp_two = [151, 152]

    simple_queries_one = [item for item in data if item['sparql_template_id'] in simple_temp_one]
    simple_queries_two = [item for item in data if item['sparql_template_id'] in simple_temp_two]

    simple_queries = simple_queries_one + simple_queries_two

    return simple_queries


# In fact, this function will not be used
def get_simple_template(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    
    res = {}
    for item in data:
        res[item['id']] = item['template']
    return res


def get_for_2(data, mode='less'):
    candidates = ent.findall(data['sparql_query'])

    standard_ent = candidates[0].split('/')[-1]
    entity = standard_ent.split('_')[0]

    predicate_uri = candidates[1]
    predicate = predicate_uri.split('/')[-1]

    if mode == 'more':
        return candidates[0], standard_ent, entity, predicate_uri, predicate

    return standard_ent, entity, predicate_uri, predicate



def get_for_1_101(data, mode='less'):
    candidates = ent.findall(data['sparql_query'])

    standard_ent = candidates[1].split('/')[-1]
    entity = standard_ent.split('_')[0]

    predicate_uri = candidates[0]
    predicate = predicate_uri.split('/')[-1]

    if mode == 'more':
        return candidates[0], standard_ent, entity, predicate_uri, predicate

    return standard_ent, entity, predicate_uri, predicate


def get_for_151_152(data, mode='less'):
    candidates = ent.findall(data['sparql_query'])

    standard_ent = candidates[0].split('/')[-1]
    entity = standard_ent.split('_')[0]

    predicate_uri = candidates[1]
    predicate = predicate_uri.split('/')[-1]

    if mode == 'more':
        return candidates[0], standard_ent, entity, predicate_uri, predicate

    return standard_ent, entity, predicate_uri, predicate



def GetSimpleQueryForTrain():

    simple_queries = get_simple_query(paths.lcquad_train)

    res = []

    counter = 0
    for item in simple_queries:

        counter += 1
        print(counter)


        try:
            if item["sparql_template_id"] == 2:
                standard_ent, entity, predicate_uri, predicate = get_for_2(item)
            elif item["sparql_template_id"] == 1 or item["sparql_template_id"] == 101:
                standard_ent, entity, predicate_uri, predicate = get_for_1_101(item)
            elif item["sparql_template_id"] == 151 or item["sparql_template_id"] == 152:
                standard_ent, entity, predicate_uri, predicate = get_for_151_152(item)
            
            standard_ent = re.sub("[\s+\.\!,&$%^*)(+\"\']+|[+——！，。？、~@#￥%……&*（）]+",  "", standard_ent)
            tmp = standard_ent.split('_')
            standard_ent = [a for a in tmp if a != '']
            entity = standard_ent[-1]; standard_ent = '_'.join(standard_ent)
            
            # if '(' in standard_ent or ')' in standard_ent:
            #     standard_ent = standard_ent.replace('(', '').replace(')', '')
            #     tmp = standard_ent.split('_')
            #     standard_ent = [a for a in tmp if a != '']
            #     entity = standard_ent[-1]; standard_ent = '_'.join(standard_ent)
        except Exception as e:
            print('some error in parse the question!!!')
            print(e)
            continue

        if standard_ent == '' or predicate_uri == '':
            continue

        query = item['corrected_question']

        query_words = word_tokenize(query)
        ent_in_query = ''
        for word in query_words:
            if entity.lower() == word.lower():
                ent_in_query = word
                break
        if not ent_in_query:
            continue

        q_word = get_qword(query_words)
        try:
            syntax = ' '.join(get_syntactic_seq(query, ent_in_query, q_word))
        except Exception as e:
            print('some error in syntax!!!')
            print(e)
            continue

        # hier = ' '.join(tmp[:min(length, 3)])  # 获取predicate的层次结构
        hier = ' '.join(GetHierLabel(standard_ent, predicate_uri, item["sparql_template_id"]))
        if not hier:
            hier = predicate + ' ' + predicate

        f_predicate = query_words[-2]  # 伪的负样本
        
        res.append((query, syntax, hier, predicate, 1))
        res.append((query, syntax, hier, f_predicate, 0))

    t_res = res[:int(0.8*len(res))]
    d_res = res[int(0.8*len(res)):]

    with open('../data/train_data.csv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(t_res)
    with open('../data/dev_data.csv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(d_res)


def GetSimpleQueryForTest():
    simple_queries = get_simple_query(paths.lcquad_test)

    total_res = []

    count = 0

    for item in simple_queries:

        count += 1; print(count)

        res_item = []

        query = item['corrected_question']
        query_id = item['_id']

        print('parsing the tree...')
        parse_tree = parse_sentence(query)

        query_words = word_tokenize(query)
        q_word = get_qword(query_words)

        print('EntityLinking...')
        text_ents, standard_ents, standard_ent_uries, confs = EntityLinking(query, 'more')

        for i in range(len(standard_ents)):
            text_ent = text_ents[i]; standard_ent = standard_ents[i]
            conf = confs[i]; standard_ent_uri = standard_ent_uries[i]

            try:
                print('get_syntactic_seq_from_tree...')
                syntax = get_syntactic_seq_from_tree(parse_tree, text_ent.split()[0], q_word)  # List[Str]
                if len(syntax) < 2: syntax.append(q_word)
            except Exception as e:
                print('some errors in get_syntactic_seq'); print(e)
                syntax = [text_ent.split()[0], q_word]
            
            print('GetPredicateList...')
            predicate_uris = GetPredicateList(standard_ent)

            for predicate_uri in predicate_uris:

                predicate = predicate_uri.split('/')[-1]
                if '#' in predicate: continue
                if 'subject' in predicate or 'wiki' in predicate or 'hypernym' in predicate: continue

                print('GetHierLabel...')
                hier = GetHierLabel(standard_ent, predicate_uri)  # List[Str]
                if not hier: hier = [predicate] * 2
                
                res_item.append((syntax, hier, [predicate], conf, standard_ent_uri, predicate_uri))

        if res_item:
            total_res.append((query_id, query, res_item))

    with open('../data/test_data.pkl', 'wb') as f:
        pickle.dump(total_res, f)


def get_stand_ans_for_test():

    simple_queries = get_simple_query(paths.lcquad_test)

    res = []

    counter = 0

    for item in simple_queries:

        counter += 1
        print(counter)


        try:
            if item["sparql_template_id"] == 2:
                standard_ent, entity, predicate_uri, predicate = get_for_2(item)
            elif item["sparql_template_id"] == 1 or item["sparql_template_id"] == 101:
                standard_ent, entity, predicate_uri, predicate = get_for_1_101(item)
            elif item["sparql_template_id"] == 151 or item["sparql_template_id"] == 152:
                standard_ent, entity, predicate_uri, predicate = get_for_151_152(item)

            res.append((item['corrected_question'], standard_ent, predicate))
            
            # standard_ent = re.sub("[\s+\.\!,&$%^*)(+\"\']+|[+——！，。？、~@#￥%……&*（）]+",  "", standard_ent)
            # tmp = standard_ent.split('_')
            # standard_ent = [a for a in tmp if a != '']
            # entity = standard_ent[-1]; standard_ent = '_'.join(standard_ent)
        except Exception as e:
            print('some error in parse the question!!!')
            print(e)
            continue

        
    with open('../data/gold_test.csv', "w") as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerows(res)


if __name__ == '__main__':

    # GetSimpleQueryForTrain()
    GetSimpleQueryForTest()
    # get_stand_ans_for_test()