#!/usr/bin/env/ python3

from SPARQLWrapper import SPARQLWrapper, JSON
import spotlight as sl

def GetPredicateList(entity):

    query_template = """
        PREFIX dbr:  <http://dbpedia.org/resource/>

        SELECT DISTINCT ?uri WHERE {
            dbr:%s ?uri ?o.
        }
        LIMIT 8
    """ %entity

    # !!![Errno 54]Connection reset by peer

    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setQuery(query_template)
    sparql.setReturnFormat(JSON)
    result = sparql.query().convert()

    p_list = [sub_result['uri']['value'] for sub_result in result["results"]["bindings"]]

    return p_list

def EntityLinking(text):
    annotation = sl.annotate('http://model.dbpedia-spotlight.org/en/annotate', \
                             text, \
                             confidence=0, support=0)
    return annotation

entities = EntityLinking('Obama here')

