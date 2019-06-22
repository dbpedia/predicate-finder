#!/usr/bin/env/ python3

from SPARQLWrapper import SPARQLWrapper, JSON
import spotlight as sl


def get_qword(query):
    q_words = ['what', 'how', 'who', 'why', 'where', 'when', 'which', 'whom', 'whose']
    q_word = ''

    for item in query:
        if item.lower() in q_words:
            q_word = item
    # assert q_word != ''
    if q_word == '':
        q_word = query[0]
    return q_word


def GetPredicateList(entity, query_template='', template_id=0):

    # the LIMIT will influence the results of query
    query_template = """
        PREFIX dbr:  <http://dbpedia.org/resource/>

        SELECT DISTINCT ?uri WHERE {
            dbr:%s ?uri ?o.
        }
    """ %entity

    # !!![Errno 54]Connection reset by peer
    try:
        sparql = SPARQLWrapper("https://dbpedia.org/sparql")
        sparql.setQuery(query_template)
        sparql.setReturnFormat(JSON)
        result = sparql.query().convert()
    except Exception as e:
        print('some errors in GetPredicateList')
        print(e)
        return []

    p_list = [sub_result['uri']['value'] for sub_result in result["results"]["bindings"]]  # List[URI]

    return p_list


def GetHierLabel(entity, predicate_uri, template_id=2):

    # Be careful that there is not space between dbr and entity
    if template_id == 2 or template_id == 151 or template_id == 152:
        query_template = """
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX dbr: <http://dbpedia.org/resource/>
            SELECT DISTINCT ?label ?parentLabel WHERE {{  
                dbr:{entity} <{uri}> ?o .
                <{uri}> rdfs:label ?label .
                filter(lang(?label) = 'en')
                OPTIONAL {{ 
                    <{uri}> rdfs:range ?parent. ?parent rdfs:label ?parentLabel . filter(lang(?parentLabel) = 'en')}}
            }}
            LIMIT 8
        """.format(entity=entity, uri=predicate_uri)
    elif template_id == 1 or template_id == 101:
        query_template = """
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX dbr: <http://dbpedia.org/resource/>
            SELECT DISTINCT ?label ?parentLabel WHERE {{  
                ?s <{uri}> dbr:{entity} .
                <{uri}> rdfs:label ?label .
                filter(lang(?label) = 'en')
                OPTIONAL {{ 
                    <{uri}> rdfs:range ?parent. ?parent rdfs:label ?parentLabel . filter(lang(?parentLabel) = 'en')}}
            }}
            LIMIT 8
        """.format(entity=entity, uri=predicate_uri)


    try:
        sparql = SPARQLWrapper("https://dbpedia.org/sparql")
        sparql.setQuery(query_template)
        sparql.setReturnFormat(JSON)
        result = sparql.query().convert()
    except Exception:
        print(query_template)
        return []

    res = []  # but, which one should be chosen to be the hier feature?
    for sub_result in result["results"]["bindings"]:
        tmp = []
        label = sub_result['label']['value']
        tmp.extend(label.split())

        if 'parentLabel' in sub_result:
            parentLabel = sub_result['parentLabel']['value']
            if ':' in parentLabel:
                parentLabel = parentLabel.split(':')[-1]
            tmp.extend(parentLabel.split())

        res.append(tmp)

    if len(res) == 1:
        return res[0]
    else:  # there is not the hier feature
        return []


# 取出最有可能的entity
def EntityLinking(text):
    try:
        annotation = sl.annotate('http://model.dbpedia-spotlight.org/en/annotate', \
                                text, \
                                confidence=0.4, support=0)
    except Exception as e:
        print('some errors in EntityLinking')
        print(e)
        return [], []

    text_ent = []; standard_ent = []
    for dic in annotation:
        text_ent.append(dic['surfaceForm'])
        standard_ent.append(dic['URI'].split('/')[-1])
        # ent_list.append((dic['surfaceForm'], dic['similarityScore']))
    # ent_list = sorted(ent_list, key=lambda x: x[1], reverse=True)
    # ent = ent_list[0][0]

    return text_ent, standard_ent


if __name__ == '__main__':
    # text = 'Name the office of Richard Coke ?'
    # res = EntityLinking(text)
    # print(res)

    # p_list = GetHierLabel('Barack_Obama', '<http://dbpedia.org/ontology/part>')

    # print(p_list)


    res = GetPredicateList('Richard_Coke')
    print(res)