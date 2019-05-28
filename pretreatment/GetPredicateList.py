#!/usr/bin/env/ python3

from SPARQLWrapper import SPARQLWrapper, JSON

def GetPredicateList(entity):

    query_template = """
        PREFIX dbr:  <http://dbpedia.org/resource/>
    
        SELECT DISTINCT ?p WHERE {
            dbr:%s ?p ?o.
        }
        LIMIT 8
    """ %entity

    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setQuery(query_template)
    sparql.setReturnFormat(JSON)
    result = sparql.query().convert()

    p_list = []

    for sub_result in result["results"]["bindings"]:
        p_list.append(str(sub_result['p']['value']).split('/')[-1].split('#')[-1])

    return p_list