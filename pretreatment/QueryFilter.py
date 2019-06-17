#!/usr/bin/env/ python3

import json

# set "predicate_finder" as source root
from paths import lcquad_train


def GetSimpleQuery():

    with open(lcquad_train, "r") as f:
        train_data = json.load(f)

    simple_temp_one = [1, 2, 101]
    simple_temp_two = [151, 152]

    simple_queries_one = [item['_id'] for item in train_data if item['sparql_template_id'] in simple_temp_one]
    simple_queries_two = [item['_id'] for item in train_data if item['sparql_template_id'] in simple_temp_two]


    return simple_queries_one, simple_queries_two


if __name__ == '__main__':

    a, b = GetSimpleQuery()
    print(a, '\n', b)
    print(len(a), len(b))