import argparse
from tqdm import tqdm
import sys

from sklearn.metrics import f1_score
from data import *
from utils import *
from model import *
from warnings import filterwarnings
filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, ParameterDict, Parameter
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator


parser = argparse.ArgumentParser(description='Training GNN on ogbn-mag benchmark')



parser.add_argument('--data_dir', type=str, default='./',
                    help='The address to output the preprocessed graph.')
parser.add_argument('--data_name', type=str, default='ogbn-products',
                    help='Name of the dataset')
parser.add_argument('--model_dir', type=int, default='./hello',
                    help='The address for storing the trained models.')
parser.add_argument('--task_type', type=str, default='variance_reduce',
                    help='Whether to use variance_reduce evaluation or sequential evaluation')
parser.add_argument('--vr_num', type=int, default=8,
                    help='Whether to use ensemble evaluation or sequential evaluation')
parser.add_argument('--n_pool', type=int, default=8,
                    help='Number of process to sample subgraph')  
parser.add_argument('--n_batch', type=int, default=32,
                    help='Number of batch (sampled graphs) for each epoch') 
parser.add_argument('--batch_size', type=int, default=256,
                    help='Number of output nodes for training')   
parser.add_argument('--cuda', type=int, default=0,
                    help='Number of output nodes for training') 


parser.add_argument('--conv_name', type=str, default='hgt',
                    choices=['hgt', 'gcn', 'gat', 'rgcn', 'han', 'hetgnn'],
                    help='The name of GNN filter. By default is Heterogeneous Graph Transformer (hgt)')
parser.add_argument('--n_hid', type=int, default=256,
                    help='Number of hidden dimension')
parser.add_argument('--n_heads', type=int, default=4,
                    help='Number of attention head')
parser.add_argument('--n_layers', type=int, default=6,
                    help='Number of GNN layers')
parser.add_argument('--dropout', type=int, default=0.2,
                    help='Dropout ratio')
parser.add_argument('--sample_depth', type=int, default=6,
                    help='How many numbers to sample the graph')
parser.add_argument('--sample_width', type=int, default=2048,
                    help='How many nodes to be sampled per layer per type')
parser.add_argument('--prev_norm', help='Whether to add layer-norm on the previous layers', action='store_true')
parser.add_argument('--last_norm', help='Whether to add layer-norm on the last layers',     action='store_true')
parser.add_argument('--use_RTE',   help='Whether to use RTE',     action='store_true')

args = parser.parse_args()
args_print(args)



def feature_products(layer_data, graph):
    feature = {}
    times   = {}
    indxs   = {}
    texts   = []
    for _type in layer_data:
        if len(layer_data[_type]) == 0:
            continue
        idxs  = np.array(list(layer_data[_type].keys()), dtype = np.int)
        tims  = np.array(list(layer_data[_type].values()))[:,1]
        if _type == 'cate':
            feature[_type] = np.zeros([len(idxs), len(graph.node_feature['product'][0])])
            feature[_type][:,0] = idxs
        else:
            feature[_type] = graph.node_feature[_type][idxs]
        times[_type]   = tims
        indxs[_type]   = idxs
        
    return feature, times, indxs, texts

def node_classification_sample(seed, samp_nodes):
    '''
        sub-graph sampling and label preparation for node classification:
        (1) Sample batch_size number of output nodes (papers), get their time.
    '''
    np.random.seed(seed)
    feature, times, edge_list, _, _ = sample_subgraph(graph, \
                inp = {'product': np.concatenate([samp_nodes, np.ones(len(samp_nodes))]).reshape(2, -1).transpose()}, \
                sampled_depth = args.sample_depth, sampled_number = args.sample_width, \
                    feature_extractor = feature_products)

    masked_edge_list = []
    for i in edge_list['product']['cate']['belong']:
        if i[0] >= args.batch_size:
            masked_edge_list += [i]
    edge_list['product']['cate']['belong'] = masked_edge_list

    masked_edge_list = []
    for i in edge_list['cate']['product']['rev_belong']:
        if i[1] >= args.batch_size:
            masked_edge_list += [i]
    edge_list['cate']['product']['rev_belong'] = masked_edge_list

    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = \
            to_torch(feature, times, edge_list, graph)
    
    ylabel = torch.LongTensor(graph.y[samp_nodes])
    x_ids = np.arange(args.batch_size)
    return node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel
    
def prepare_data(pool, task_type = 'train', s_idx=0, n_batch = args.n_batch, batch_size = args.batch_size):
    '''
        Sampled and prepare training and validation data using multi-process parallization.
    '''
    jobs = []
    if task_type == 'train':
        for batch_id in np.arange(args.n_batch):
            p = pool.apply_async(node_classification_sample, args=(randint(), \
                np.random.choice(graph.train_paper, args.batch_size, replace = False)))
            jobs.append(p)
        p = pool.apply_async(node_classification_sample, args=(randint(), \
               np.random.choice(graph.valid_paper, args.batch_size, replace = False)))
        jobs.append(p)

        p = pool.apply_async(node_classification_sample, args=(randint(), \
               np.random.choice(graph.test_paper, args.batch_size, replace = False)))
        jobs.append(p)
    elif task_type == 'sequential':
        for i in np.arange(n_batch):
            target_papers = graph.test_paper[(s_idx + i) * batch_size : (s_idx + i + 1) * batch_size]
            p = pool.apply_async(node_classification_sample, args=(randint(), target_papers))
            jobs.append(p)
    return jobs


graph = dill.load(open('%s/%s.pk' % (args.data_dir, args.data_name), 'rb'))
evaluator = Evaluator(name=args.data_name)
device = torch.device("cuda:%d" % args.cuda)
model = torch.load(args.model_dir)
model.to(device)
print('Model #Params: %d' % get_n_params(model))
criterion = nn.NLLLoss()
gnn, classifier = model[0], model[1]

model.eval()
with torch.no_grad():
    if args.task_type == 'variance_reduce':
        y_pred = []
        y_true = []
        pool = mp.Pool(args.n_pool)
        jobs = prepare_data(pool, task_type = 'variance_reduce', s_idx = 0, n_batch = args.vr_num)
        with tqdm(np.arange(len(graph.test_paper) // args.batch_size), desc='eval') as monitor:
            for s_idx in monitor:
                ress = []
                test_data = [job.get() for job in jobs]
                pool.close()
                pool.join()
                pool = mp.Pool(args.n_pool)
                jobs = prepare_data(pool, task_type = 'variance_reduce', s_idx = s_idx, n_batch = args.vr_num)

                for node_feature, node_type, edge_time, edge_index, edge_type, (train_mask, valid_mask, test_mask), ylabel in test_data:
                    node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                                                   edge_time.to(device), edge_index.to(device), edge_type.to(device))
                    res  = classifier.forward(node_rep[:args.batch_size])
                    ress += [res]

                y_pred += torch.stack(ress).mean(dim=0).argmax(dim=1).tolist()
                y_true += list(ylabel[:args.batch_size])

                test_acc = evaluator.eval({
                        'y_true': torch.LongTensor(y_true).unsqueeze(-1),
                        'y_pred': torch.LongTensor(y_pred).unsqueeze(-1)
                    })['acc']
                monitor.set_postfix(accuracy = test_acc)
                
    elif args.task_type == 'sequential':
        y_pred = []
        y_true = []
        pool = mp.Pool(args.n_pool)
        jobs = prepare_data(pool, task_type = 'sequential', s_idx = 0)
        with tqdm(np.arange(len(graph.test_paper) / args.n_batch // args.batch_size), desc='eval') as monitor:
            for s_idx in monitor:
                test_data = [job.get() for job in jobs]
                pool.close()
                pool.join()
                pool = mp.Pool(args.n_pool)
                jobs = prepare_data(pool, task_type = 'sequential', s_idx = int(s_idx * args.n_batch))

                for node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel in test_data:
                    ylabel = ylabel[:args.batch_size]
                    node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                                                   edge_time.to(device), edge_index.to(device), edge_type.to(device))
                    res  = classifier.forward(node_rep[:args.batch_size])
                    pred  = res.argmax(dim=1)
                    
                    y_pred += pred.tolist()
                    y_true += ylabel.tolist()
                    
                test_acc = evaluator.eval({
                                'y_true': torch.FloatTensor(y_true),
                                'y_pred': torch.FloatTensor(y_pred).unsqueeze(-1)
                            })['acc']
                monitor.set_postfix(accuracy = test_acc)
    elif args.task_type == 'full_batch':
        pass
#         node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel