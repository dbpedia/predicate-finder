#!/usr/bin/env/ python3

from SPARQLWrapper import SPARQLWrapper, JSON
import spotlight as sl
import requests
import json


def get_qword(query):
    q_words = ['what', 'how', 'who', 'why', 'where', 'when', 'which', 'whom', 'whose']
    q_word = ''

    for item in query:
        if item.lower() in q_words:
            q_word = item
    if q_word == '':
        q_word = query[0]
    return q_word


def GetNextEntity(standard_ent, predicate_uri, template_id):

    if template_id in [111,5,105,3,11,103,305,403,405,311,303]:
        query_template = """
        PREFIX dbr:  <http://dbpedia.org/resource/>

        SELECT DISTINCT ?uri WHERE {{
            dbr:{entity} <{predicate}> ?uri.
        }}
    """.format(entity=standard_ent, predicate=predicate_uri)
    elif template_id in [6,106,406,306]:
        query_template = """
        PREFIX dbr:  <http://dbpedia.org/resource/>

        SELECT DISTINCT ?uri WHERE {{
            ?uri <{predicate}> dbr:{entity}.
        }}
    """.format(entity=standard_ent, predicate=predicate_uri)

    try:
        sparql = SPARQLWrapper("https://dbpedia.org/sparql")
        sparql.setQuery(query_template)
        sparql.setReturnFormat(JSON)
        result = sparql.query().convert()
    except Exception:
        print(query_template)
        return []

    entities = [sub_result['uri']['value'].split('/')[-1] for sub_result in result["results"]["bindings"]]

    if not entities:
        return None
    else:
        return entities[0]

def GetPredicateList(entity, query_template='', template_id=0):

    # process some special symbol in entity, such as [(),']
    special_symbol = ["(", ")", ",", "'", "+", "."]
    for symbol in special_symbol:
        entity = entity.replace(symbol, "\\"+symbol)


    # the LIMIT will influence the results of query
    
    if template_id in [151,152,2,102,15,16,3,11,103,402,303,315]:
        query_template = """
        PREFIX dbr:  <http://dbpedia.org/resource/>

        SELECT DISTINCT ?p ?uri WHERE {
            dbr:%s ?p ?o.
            ?p rdfs:label ?uri.
            FILTER(lang(?uri)='en')
        }
        """ %entity
    elif template_id in [1,101,111,5,105,6,106,7,108,8,301,401,601,305,403,405,311,307,308,406,306]:
        query_template = """
        PREFIX dbr:  <http://dbpedia.org/resource/>

        SELECT DISTINCT ?p ?uri WHERE {
            ?s ?p dbr:%s.
            ?p rdfs:label ?uri.
            FILTER(lang(?uri)='en')
        }
        """ %entity


    try:
        sparql = SPARQLWrapper("https://dbpedia.org/sparql")
        sparql.setQuery(query_template)
        sparql.setReturnFormat(JSON)
        result = sparql.query().convert()
    except Exception as e:
        print('some errors in GetPredicateList')
        print(e)
        return []
    
    p_list = [sub_result['p']['value'] for sub_result in result["results"]["bindings"]]  # List[URI]
    w_list = [sub_result['uri']['value'] for sub_result in result["results"]["bindings"]]

    p_res = []; w_res = []
    for item, w in zip(p_list, w_list):
        if '#' in item or 'subject' in item or 'wiki' in item or 'hypernym' in item or 'gender' in item: continue
        p_res.append(item)
        w_res.append(w)

    return p_res, w_res


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


def EntityLinking(text, return_type='less'):
    try:
        annotation = sl.annotate('http://model.dbpedia-spotlight.org/en/annotate', \
                                text, \
                                confidence=0.4, support=0)
        print('good!')
    except Exception as e:
        print('some errors in EntityLinking')
        print(e)
        if return_type=='less':
            return [], []
        if return_type=='more':
            return [], [], [], [], []

    text_ent = []; standard_ent = []; similarities = []; standard_ent_uri = []; types = []
    for dic in annotation:
        text_ent.append(dic['surfaceForm'])
        standard_ent_uri.append(dic['URI'])
        standard_ent.append(dic['URI'].split('/')[-1])
        similarities.append(dic['similarityScore'])
        flag = True
        for item in dic['types'].split(','):
            tmp = item.split(':')
            if tmp[0] == 'DBpedia':
                types.append(tmp[1]); flag = False
                break
        if flag: 
            print(dic)
            types.append(dic['surfaceForm'].split(' ')[-1])
        # ent_list.append((dic['surfaceForm'], dic['similarityScore']))
    # ent_list = sorted(ent_list, key=lambda x: x[1], reverse=True)
    # ent = ent_list[0][0]

    if return_type == 'less':
        return text_ent, standard_ent
    elif return_type == 'more':
        return text_ent, standard_ent, standard_ent_uri, similarities, types


def Entity_Link_Falcon(text):
    headers = {'Content-Type': 'application/json'}
    url = 'https://labs.tib.eu/falcon/api?mode=long'
    data = {"text": text}

    try:
        response = requests.post(url, data=json.dumps(data), headers=headers).text
        response = json.loads(response)
        standard_entities = []; entities = []
        for item in response['entities']:
            standard_entity = item[0].split('/')[-1]
            standard_entities.append(standard_entity)

            entity = item[1]
            entities.append(entity)

        return standard_entities, entities
    except Exception as e:
        print(e)
        return [], []
    

def Question_Predicted_Answer_Sim_(query, sparql_query):
    try:
        sparql = SPARQLWrapper("https://dbpedia.org/sparql")
        sparql.setQuery(sparql_query)
        sparql.setReturnFormat(JSON)
        result = sparql.query().convert()
    except Exception:
        print(sparql_query)

    return result


if __name__ == '__main__':
    # text = 'Name the office of Barack Obama ?'
    # res = EntityLinking(text)
    # print(res)

    # p_list = GetHierLabel('Barack_Obama', '<http://dbpedia.org/ontology/part>')

    # print(p_list)


    # res = GetPredicateList('Dzogchen_Ponlop_Rinpoche', template_id=2)
    # print(res)

    # res = Question_Predicted_Answer_Sim_('', 'SELECT DISTINCT ?uri WHERE { <http://dbpedia.org/resource/Thorington_railway_station> <http://dbpedia.org/ontology/district> ?uri } ')
    # print(res)

    res = GetNextEntity('Barack_Obama', 'http://dbpedia.org/ontology/party', 111)
    print(res)