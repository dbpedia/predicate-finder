# -*- coding:utf-8 -*-

import sys
sys.path.append('..')
from stanfordcorenlp import StanfordCoreNLP
# import config_train as args
nlp=StanfordCoreNLP('/home/wangdong/stanford-corenlp-full-2018-02-27', lang='en')


def parse_sentence(sen):

    instance = {}
    instance["src_token"] = [item[0] for item in nlp.ner(sen)]
    dependency_parse = sorted(nlp.dependency_parse(sen), key=lambda item:item[2])
    instance["stanford_head"] = [item[1] for item in dependency_parse]
    instance["stanford_deprel"] = [item[0] for item in dependency_parse]

    assert len(instance["src_token"]) == len(instance["stanford_head"])
    assert len(instance["src_token"]) == len(instance["stanford_deprel"])

    return instance


def get_shortest_path(parse_res, sub, obj):
    tokens = parse_res['src_token']
    heads = parse_res['stanford_head']
    deprels = parse_res["stanford_deprel"]

    index_sub = tokens.index(sub)
    index_obj = tokens.index(obj)

    son = index_sub
    fathers_of_sub = [son]
    father = heads[son]
    counter = 0
    while father > 0:
        fathers_of_sub.append(father-1)
        son = father-1
        father = heads[son]
        counter += 1
        if counter > len(heads):
            print('dead loop!!!')
            return [sub, obj]

    son = index_obj
    if son in fathers_of_sub:
        remove_common_father = 1
        fathers_of_obj = []
        father = son
    else:
        remove_common_father = 0
        fathers_of_obj = []
        father = heads[son] - 1
        counter = 0
        while father >= 0 and father not in fathers_of_sub:
            fathers_of_obj.append(father)
            son = father
            father = heads[son] - 1
            counter += 1
            if counter > len(heads):
                print('dead loop!!!')
                return [sub, obj]
        

    assert father in fathers_of_sub

    common_father = fathers_of_sub.index(father)

    short_path = []
    for i in range(1, common_father+1-remove_common_father):
        short_path.append(fathers_of_sub[i])
    for i in range(len(fathers_of_obj)-1, -1, -1):
        short_path.append(fathers_of_obj[i])

    syntactics = [deprels[index_sub]]
    for item in short_path:
        syntactics.append(tokens[item])
        syntactics.append(deprels[item])

    return syntactics


def get_syntactic_seq(sen, sub, obj):
    parse_res = parse_sentence(sen)
    syntactic_seq = get_shortest_path(parse_res, sub, obj)

    return syntactic_seq

def get_syntactic_seq_from_tree(parse_tree, sub, obj):
    syntactic_seq = get_shortest_path(parse_tree, sub, obj)
    return syntactic_seq

if __name__ == '__main__':
    sen = 'What is the region of Tom Perriello ?'
    sub = 'What'
    obj = 'Perriello'
