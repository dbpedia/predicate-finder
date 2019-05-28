#!/usr/bin/env/ python3

import json

# set "predicate_finder" as source root
from paths import lcquad_train


def GetSimpleQuery():

    with open(lcquad_train, "r") as f:
        train_data = json.load(f)

    simple_queries = []

    for item in train_data:
        sparql_q = item['sparql_query']
        flag = False

        if (sparql_q.count('<') < 3):
            flag = True
        elif sparql_q.count('<') == 3 and sparql_q.split(' ')[0] == 'ASK':
            flag = True

        if flag == True:
            simple_queries.append((item['_id'], item['corrected_question']))
            print(item['corrected_question'])

    # print(len(simple_queries), len(train_data))

    return simple_queries


if __name__ == '__main__':

    GetSimpleQuery()
