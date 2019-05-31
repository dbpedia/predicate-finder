#!/usr/bin/env/ python3

import json
import re

from paths import lcquad_train
import DataExtract as ext

class Pair:
    def __init__(self, s, p, correct):
        self.s = s
        self.p = p
        self.label = 0

        self.getscore(correct)

    def getlabel(self, correct):

        if self.s in correct:
            self.label += 1
        if self.p in correct:
            self.label += 1
        if self.label == 2:
            self.label += 1

class Query:
    def __init__(self, query, entities):
        self.uri = query
        self.pairs = []
        self.entities = entities

def get_pairs(query):
    entities = ext.EntityLinking(query)
    q = Query(query, entities)
    for ent in entities:
        s_uri = ent['URI']
        p_list = ext.GetPredicateList(s_uri)

        match_rule = re.compile(r'[<](.*?)[>]', re.S)
        correct = re.findall(match_rule, query)
        for p_uri in p_list:
            pair = Pair(s_uri, p_uri, correct)
            q.pairs.append(pair)

    return q




if __name__ = '__main__':
    with open(lcquad_train, "r") as f:
        train_data = json.load(f)
