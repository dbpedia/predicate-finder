import numpy as np
import config_predicate as args
import pickle

def get_relation_id_pair():
    rel_id = {}
    with open(args.ori_rel_id, 'r') as f:
        for line in f:
            rel_, id_ = line.strip().split('\t')
            rel_id[rel_] = int(id_)
    with open(args.rel_id_dic, 'wb') as f:
        pickle.dump(rel_id, f)
    

def get_relation_emb_pair():
    emb = np.memmap(args.ori_rel_emb, dtype='float64', mode='r')
    with open(args.rel_emb_arr, 'wb') as f:
        pickle.dump(emb, f)

if __name__ == '__main__':
    get_relation_id_pair()
    get_relation_emb_pair()