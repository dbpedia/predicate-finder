#!/usr/bin/env/ python3

import json
import sys
sys.path.append('..')
import re

ent = re.compile(u'<(.*?)>',re.M|re.S|re.I)

# Here are functions for simple queries
def get_for_2(data, mode='less'):
    candidates = ent.findall(data['sparql_query'])

    standard_ent = candidates[0].split('/')[-1]
    entity = standard_ent.split('_')[0]

    predicate_uri = candidates[1]
    standard_pre = predicate_uri.split('/')[-1]

    if mode == 'more':
        return candidates[0], standard_ent, entity, predicate_uri, standard_pre

    return standard_ent, entity, predicate_uri, standard_pre


def get_for_1_101(data, mode='less'):
    candidates = ent.findall(data['sparql_query'])

    standard_ent = candidates[1].split('/')[-1]
    entity = standard_ent.split('_')[0]

    predicate_uri = candidates[0]
    standard_pre = predicate_uri.split('/')[-1]

    if mode == 'more':
        return candidates[0], standard_ent, entity, predicate_uri, standard_pre

    return standard_ent, entity, predicate_uri, standard_pre


def get_for_151_152(data, mode='less'):
    candidates = ent.findall(data['sparql_query'])

    standard_ent = candidates[0].split('/')[-1]
    entity = standard_ent.split('_')[0]

    predicate_uri = candidates[1]
    standard_pre = predicate_uri.split('/')[-1]

    if mode == 'more':
        return candidates[0], standard_ent, entity, predicate_uri, standard_pre

    return standard_ent, entity, predicate_uri, standard_pre


# Here are funcitons for complex queries
def get_for_102(data, mode='less'):
    candidates = ent.findall(data['sparql_query'])

    standard_ent_1 = candidates[0].split('/')[-1]
    text_entity_1 = standard_ent_1.split('_')[0]

    predicate_uri_1 = candidates[1]
    standard_pre_1 = predicate_uri_1.split('/')[-1]

    standard_ent_2 = ''
    standard_pre_2 = ''

    return standard_ent_1, standard_pre_1, standard_ent_2, standard_pre_2

def get_for_15_16(data, mode='less'):
    candidates = ent.findall(data['sparql_query'])

    standard_ent_1 = candidates[0].split('/')[-1]
    text_entity_1 = standard_ent_1.split('_')[0]

    predicate_uri_1 = candidates[1]
    standard_pre_1 = predicate_uri_1.split('/')[-1]

    standard_ent_2 = candidates[2].split('/')[-1]
    text_entity_2 = standard_ent_2.split('_')[0]

    predicate_uri_2 = candidates[3]
    standard_pre_2 = predicate_uri_2.split('/')[-1]

    return standard_ent_1, standard_pre_1, standard_ent_2, standard_pre_2

def get_for_111_5_105(data, mode='less'):
    candidates = ent.findall(data['sparql_query'])

    standard_ent_1 = candidates[1].split('/')[-1]
    text_entity_1 = standard_ent_1.split('_')[0]

    predicate_uri_1 = candidates[0]
    standard_pre_1 = predicate_uri_1.split('/')[-1]

    standard_ent_2 = ''
    text_entity_2 = ''

    predicate_uri_2 = candidates[2]
    standard_pre_2 = predicate_uri_2.split('/')[-1]

    return standard_ent_1, standard_pre_1, standard_ent_2, standard_pre_2

def get_for_6_106(data, mode='less'):
    candidates = ent.findall(data['sparql_query'])

    standard_ent_1 = candidates[1].split('/')[-1]
    text_entity_1 = standard_ent_1.split('_')[0]

    predicate_uri_1 = candidates[0]
    standard_pre_1 = predicate_uri_1.split('/')[-1]

    predicate_uri_2 = candidates[2]
    standard_pre_2 = predicate_uri_2.split('/')[-1]

    standard_ent_2 = ''
    text_entity_2 = ''

    return standard_ent_1, standard_pre_1, standard_ent_2, standard_pre_2

def get_for_7_108_8(data, mode='less'):
    candidates = ent.findall(data['sparql_query'])

    standard_ent_1 = candidates[1].split('/')[-1]
    text_entity_1 = standard_ent_1.split('_')[0]

    predicate_uri_1 = candidates[0]
    standard_pre_1 = predicate_uri_1.split('/')[-1]

    standard_ent_2 = candidates[3].split('/')[-1]
    text_entity_2 = standard_ent_2.split('_')[0]

    predicate_uri_2 = candidates[2]
    standard_pre_2 = predicate_uri_2.split('/')[-1]

    return standard_ent_1, standard_pre_1, standard_ent_2, standard_pre_2

def get_for_3_11_103(data, mode='less'):
    candidates = ent.findall(data['sparql_query'])

    standard_ent_1 = candidates[0].split('/')[-1]
    text_entity_1 = standard_ent_1.split('_')[0]

    predicate_uri_1 = candidates[1]
    standard_pre_1 = predicate_uri_1.split('/')[-1]

    standard_ent_2 = ''
    text_entity_2 = ''

    predicate_uri_2 = candidates[2]
    standard_pre_2 = predicate_uri_2.split('/')[-1]

    return standard_ent_1, standard_pre_1, standard_ent_2, standard_pre_2

def get_for_301_401_601(data, mode='less'):
    candidates = ent.findall(data['sparql_query'])

    standard_ent_1 = candidates[1].split('/')[-1]
    text_entity_1 = standard_ent_1.split('_')[0]

    predicate_uri_1 = candidates[0]
    standard_pre_1 = predicate_uri_1.split('/')[-1]

    standard_ent_2 = ''
    text_entity_2 = ''

    predicate_uri_2 = ''
    standard_pre_2 = ''

    return standard_ent_1, standard_pre_1, standard_ent_2, standard_pre_2

def get_for_402(data, mode='less'):
    candidates = ent.findall(data['sparql_query'])

    standard_ent_1 = candidates[0].split('/')[-1]
    text_entity_1 = standard_ent_1.split('_')[0]

    predicate_uri_1 = candidates[1]
    standard_pre_1 = predicate_uri_1.split('/')[-1]

    standard_ent_2 = ''
    text_entity_2 = ''

    predicate_uri_2 = ''
    standard_pre_2 = ''

    return standard_ent_1, standard_pre_1, standard_ent_2, standard_pre_2

def get_for_305_403_405_311(data, mode='less'):
    candidates = ent.findall(data['sparql_query'])

    standard_ent_1 = candidates[1].split('/')[-1]
    text_entity_1 = standard_ent_1.split('_')[0]

    predicate_uri_1 = candidates[0]
    standard_pre_1 = predicate_uri_1.split('/')[-1]

    standard_ent_2 = ''
    text_entity_2 = ''

    predicate_uri_2 = candidates[2]
    standard_pre_2 = predicate_uri_2.split('/')[-1]

    return standard_ent_1, standard_pre_1, standard_ent_2, standard_pre_2

def get_for_307_308(data, mode='less'):
    candidates = ent.findall(data['sparql_query'])

    standard_ent_1 = candidates[1].split('/')[-1]
    text_entity_1 = standard_ent_1.split('_')[0]

    predicate_uri_1 = candidates[0]
    standard_pre_1 = predicate_uri_1.split('/')[-1]

    standard_ent_2 = candidates[3].split('/')[-1]
    text_entity_2 = standard_ent_2.split('_')[0]

    predicate_uri_2 = candidates[2]
    standard_pre_2 = predicate_uri_2.split('/')[-1]

    return standard_ent_1, standard_pre_1, standard_ent_2, standard_pre_2

def get_for_406_306(data, mode='less'):
    candidates = ent.findall(data['sparql_query'])

    standard_ent_1 = candidates[1].split('/')[-1]
    text_entity_1 = standard_ent_1.split('_')[0]

    predicate_uri_1 = candidates[0]
    standard_pre_1 = predicate_uri_1.split('/')[-1]

    standard_ent_2 = ''
    text_entity_2 = ''

    predicate_uri_2 = candidates[2]
    standard_pre_2 = predicate_uri_2.split('/')[-1]

    return standard_ent_1, standard_pre_1, standard_ent_2, standard_pre_2

def get_for_303(data, mode='less'):
    candidates = ent.findall(data['sparql_query'])

    standard_ent_1 = candidates[0].split('/')[-1]
    text_entity_1 = standard_ent_1.split('_')[0]

    predicate_uri_1 = candidates[1]
    standard_pre_1 = predicate_uri_1.split('/')[-1]

    standard_ent_2 = ''
    text_entity_2 = ''

    predicate_uri_2 = candidates[2]
    standard_pre_2 = predicate_uri_2.split('/')[-1]

    return standard_ent_1, standard_pre_1, standard_ent_2, standard_pre_2

def get_for_315(data, mode='less'):
    candidates = ent.findall(data['sparql_query'])

    standard_ent_1 = candidates[0].split('/')[-1]
    text_entity_1 = standard_ent_1.split('_')[0]

    predicate_uri_1 = candidates[1]
    standard_pre_1 = predicate_uri_1.split('/')[-1]

    standard_ent_2 = candidates[2].split('/')[-1]
    text_entity_2 = standard_ent_2.split('_')[0]

    predicate_uri_2 = candidates[3]
    standard_pre_2 = predicate_uri_2.split('/')[-1]

    return standard_ent_1, standard_pre_1, standard_ent_2, standard_pre_2

