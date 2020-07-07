from pyHGT.data import *
from pyHGT.utils import *
from ogb.nodeproppred import PygNodePropPredDataset

import argparse

parser = argparse.ArgumentParser(description='Preprocess ogbn-mag graph')

'''
    Dataset arguments
'''
parser.add_argument('--output_dir', type=str, default='/datadrive/dataset/OGB_MAG.pk',
                    help='The address to output the preprocessed graph.')

args = parser.parse_args()


dataset = PygNodePropPredDataset(name='ogbn-mag')
data = dataset[0]
evaluator = Evaluator(name='ogbn-mag')
edge_index_dict = data.edge_index_dict
graph = Graph()
edg   = graph.edge_list
years = data.node_year_dict['paper'].t().numpy()[0]

graph = Graph()
edg   = graph.edge_list
years = data.node_year_dict['paper'].t().numpy()[0]
for key in edge_index_dict:
    print(key)
    edges = edge_index_dict[key]
    s_type, r_type, t_type = key[0], key[1], key[2]
    elist = edg[t_type][s_type][r_type]
    rlist = edg[s_type][t_type]['rev_' + r_type]
    for s_id, t_id in edges.t().tolist():
        year = None
        if s_type == 'paper':
            year = years[s_id]
        elif t_type == 'paper':
            year = years[t_id]
        elist[t_id][s_id] = year
        rlist[s_id][t_id] = year
        
        
edg = {}
deg = {key : np.zeros(data.num_nodes[key]) for key in data.num_nodes}
for k1 in graph.edge_list:
    if k1 not in edg:
        edg[k1] = {}
    for k2 in graph.edge_list[k1]:
        if k2 not in edg[k1]:
            edg[k1][k2] = {}
        for k3 in graph.edge_list[k1][k2]:
            if k3 not in edg[k1][k2]:
                edg[k1][k2][k3] = {}
            for e1 in graph.edge_list[k1][k2][k3]:
                if len(graph.edge_list[k1][k2][k3][e1]) == 0:
                    continue
                
                edg[k1][k2][k3][e1] = {}
                for e2 in graph.edge_list[k1][k2][k3][e1]:
                    edg[k1][k2][k3][e1][e2] = graph.edge_list[k1][k2][k3][e1][e2]
                deg[k1][e1] += len(edg[k1][k2][k3][e1])
            print(k1, k2, k3, len(edg[k1][k2][k3]))
graph.edge_list = edg



cv = data.x_dict['paper'].numpy()
graph.node_feature['paper'] = np.concatenate((cv, np.log10(deg['paper'].reshape(-1, 1))), axis=-1)
for _type in data.num_nodes:
    print(_type)
    if _type not in ['paper', 'institution']:
        i = []
        for _rel in graph.edge_list[_type]['paper']:
            for t in graph.edge_list[_type]['paper'][_rel]:
                for s in graph.edge_list[_type]['paper'][_rel][t]:
                    i += [[t, s]]
        if len(i) == 0:
            continue
        i = np.array(i).T
        v = np.ones(i.shape[1])
        m = normalize(sp.coo_matrix((v, i), \
            shape=(data.num_nodes[_type], data.num_nodes['paper'])))
        out = m.dot(cv)
        graph.node_feature[_type] = np.concatenate((out, np.log10(deg[_type].reshape(-1, 1))), axis=-1)

cv = graph.node_feature['author'][:, :-1]
i = []
for _rel in graph.edge_list['institution']['author']:
    for j in graph.edge_list['institution']['author'][_rel]:
        for t in graph.edge_list['institution']['author'][_rel][j]:
            i += [[j, t]]
i = np.array(i).T
v = np.ones(i.shape[1])
m = normalize(sp.coo_matrix((v, i), \
    shape=(data.num_nodes['institution'], data.num_nodes['author'])))
out = m.dot(cv)
graph.node_feature['institution'] = np.concatenate((out, np.log10(deg['institution'].reshape(-1, 1))), axis=-1)     



y = data.y_dict['paper'].t().numpy()[0]
split_idx = dataset.get_idx_split()
train_paper = split_idx['train']['paper'].numpy()
valid_paper = split_idx['valid']['paper'].numpy()
test_paper  = split_idx['test']['paper'].numpy()


graph.y = y
graph.train_paper = train_paper
graph.valid_paper = valid_paper
graph.test_paper  = test_paper
graph.years       = years

graph.train_mask = np.zeros(len(graph.node_feature['paper']), dtype=bool)
graph.train_mask[graph.train_paper] = True

graph.valid_mask = np.zeros(len(graph.node_feature['paper']), dtype=bool)
graph.valid_mask[graph.valid_paper] = True

graph.test_mask = np.zeros(len(graph.node_feature['paper']),  dtype=bool)
graph.test_mask[graph.test_paper] = True

dill.dump(graph, open(args.output_dir, 'wb'))
