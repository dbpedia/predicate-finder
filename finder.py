#!/usr/bin/env/ python3

import json
import re

from paths import lcquad_train
import pretreatment.DataExtract as ext
import pretreatment.Embedding as emb
# import numpy as np
import torch

class Pair:

    def __init__(self, s, p, query):

        self.s = s
        self.p = p
        self.label = 0

        self.getlabel(query)
        self.getscore(query["corrected_question"])

    def getlabel(self, query):

        match_rule = re.compile(r'[<](.*?)[>]', re.S)
        correct = re.findall(match_rule, query["sparql_query"])

        if self.s in correct:
            self.label += 1
        if self.p in correct:
            self.label += 1
        if self.label == 2:
            self.label += 1

    def getscore(self, query):

        p_fast = emb.fasttext[p]
        q_fast = emb.getQuery(query)
        self.score = torch.dot(p_fast, q_fast)


class Query:
    def __init__(self, query):

        self.id = query['_id']
        self.pairs = []
        self.get_pairs(query)

    def get_pairs(self, query):

        self.entities = ext.EntityLinking(query["corrected_question"])
        template_id = query["sparql_template_id"]

        for ent in self.entities:
            s_uri = ent['URI']
            p_list = ext.GetPredicateList(s_uri, template_id)

            for p_uri in p_list:
                pair = Pair(s_uri, p_uri, query)
                self.pairs.append(pair)


def build_query(data):
    query_list = []
    for item in data:
        q = Query(item)
        query_list.append(q)

    return query_list



if __name__ == '__main__':
    with open(lcquad_train, "r") as f:
        train_data = json.load(f)

    q_list = build_query(train_data)

