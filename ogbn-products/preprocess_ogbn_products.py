from pyHGT.data import *
from pyHGT.utils import *
from ogb.nodeproppred import PygNodePropPredDataset

import argparse

parser = argparse.ArgumentParser(description='Preprocess ogbn-products graph')

'''
    Dataset arguments
'''
parser.add_argument('--data_dir', type=str, default='./',
                    help='The address to output the preprocessed graph.')
parser.add_argument('--data_name', type=str, default='ogbn-products',
                    help='Name of the dataset')

args = parser.parse_args()


dataset = PygNodePropPredDataset(name=args.data_name, root = args.data_dir)
data = dataset[0]
graph = Graph()

y = data.y.t().numpy()[0]
split_idx = dataset.get_idx_split()
train_paper = split_idx['train'].numpy()
valid_paper = split_idx['valid'].numpy()
test_paper  = split_idx['test'].numpy()

graph.y = data.y
graph.train_paper = train_paper
graph.valid_paper = valid_paper
graph.test_paper  = test_paper


graph.train_mask = np.zeros(len(graph.y), dtype=bool)
graph.train_mask[graph.train_paper] = True

graph.valid_mask = np.zeros(len(graph.y), dtype=bool)
graph.valid_mask[graph.valid_paper] = True

graph.test_mask = np.zeros(len(graph.y),  dtype=bool)
graph.test_mask[graph.test_paper] = True


elist = graph.edge_list['product']['product']['relate']
for s_id, t_id in tqdm(data.edge_index.t().tolist()):
    elist[t_id][s_id] = 1
    elist[s_id][t_id] = 1
    
elist = graph.edge_list['product']['cate']['belong']
rlist = graph.edge_list['cate']['product']['rev_belong']

for s_id, t_id in enumerate(y):
    elist[s_id][t_id] = 1
    rlist[t_id][s_id] = 1
        
        
edg = {}
deg = {'product' : np.zeros(len(data.y)), 'cate' : np.zeros(len(data.y))}
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


cv = data.x.numpy()
graph.node_feature = {}
graph.node_feature['product'] = np.concatenate((cv, np.log10((1e-3 + deg['product']).reshape(-1, 1))), axis=-1)
graph.node_feature['cate']    = None

dill.dump(graph, open(args.data_dir + args.data_name, 'wb'))
