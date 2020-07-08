import argparse
from tqdm import tqdm
import sys

from sklearn.metrics import f1_score
from pyHGT.data import *
from pyHGT.model import *
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



parser.add_argument('--data_dir', type=sre, default='/datadrive/dataset/OGB_MAG.pk',
                    help='The address of preprocessed graph.')
parser.add_argument('--model_dir', type=sre, default='./hgt_4layer',
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



parser.add_argument('--conv_name', type=str, default='hgt',
                    choices=['hgt', 'gcn', 'gat', 'rgcn', 'han', 'hetgnn'],
                    help='The name of GNN filter. By default is Heterogeneous Graph Transformer (hgt)')
parser.add_argument('--n_hid', type=int, default=512,
                    help='Number of hidden dimension')
parser.add_argument('--n_heads', type=int, default=8,
                    help='Number of attention head')
parser.add_argument('--n_layers', type=int, default=4,
                    help='Number of GNN layers')
parser.add_argument('--dropout', type=int, default=0.2,
                    help='Dropout ratio')
parser.add_argument('--sample_depth', type=int, default=6,
                    help='How many numbers to sample the graph')
parser.add_argument('--sample_width', type=int, default=520,
                    help='How many nodes to be sampled per layer per type')
parser.add_argument('--prev_norm', help='Whether to add layer-norm on the previous layers', action='store_true')
parser.add_argument('--last_norm', help='Whether to add layer-norm on the last layers',     action='store_true')
parser.add_argument('--use_RTE',   help='Whether to use RTE',     action='store_true')

args = parser.parse_args()
args_print(args)



def ogbn_sample(seed, samp_nodes):
    np.random.seed(seed)
    ylabel      = torch.LongTensor(graph.y[samp_nodes])
    feature, times, edge_list, indxs, _ = sample_subgraph(graph, \
                inp = {'paper': np.concatenate([samp_nodes, graph.years[samp_nodes]]).reshape(2, -1).transpose()}, \
                sampled_depth = args.sample_depth, sampled_number = args.sample_width, \
                    feature_extractor = feature_MAG)
    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = \
            to_torch(feature, times, edge_list, graph)
    train_mask = graph.train_mask[indxs['paper']]
    valid_mask = graph.valid_mask[indxs['paper']]
    test_mask  = graph.test_mask[indxs['paper']]
    ylabel     = graph.y[indxs['paper']]
    return node_feature, node_type, edge_time, edge_index, edge_type, (train_mask, valid_mask, test_mask), ylabel
    
def prepare_data(pool, task_type = 'train', s_idx = 0, n_batch = args.n_batch, batch_size = args.batch_size):
    '''
        Sampled and prepare training and validation data using multi-process parallization.
    '''
    jobs = []
    if task_type == 'train':
        for batch_id in np.arange(n_batch):
            p = pool.apply_async(ogbn_sample, args=([randint(), \
                            np.random.choice(graph.train_paper, args.batch_size, replace = False)]))
            jobs.append(p)
    elif task_type == 'variance_reduce':
        target_papers = graph.test_paper[s_idx * args.batch_size : (s_idx + 1) * args.batch_size]
        for batch_id in np.arange(n_batch):
            p = pool.apply_async(ogbn_sample, args=([randint(), target_papers]))
            jobs.append(p)
    elif task_type == 'sequential':
        for i in np.arange(n_batch):
            target_papers = graph.test_paper[(s_idx + i) * batch_size : (s_idx + i + 1) * batch_size]
            p = pool.apply_async(ogbn_sample, args=([randint(), target_papers]))
            jobs.append(p)
    return jobs

np.random.seed(43)
np.random.shuffle(graph.test_paper)

graph = dill.load(open(args.data_dir, 'rb'))
evaluator = Evaluator(name='ogbn-mag')
device = torch.device("cuda:%d" % args.cuda)
gnn = GNN(conv_name = args.conv_name, in_dim = len(graph.node_feature['paper'][0]), \
          n_hid = args.n_hid, n_heads = args.n_heads, n_layers = args.n_layers, dropout = args.dropout,\
          num_types = len(graph.get_types()), num_relations = len(graph.get_meta_graph()) + 1,\
          prev_norm = args.prev_norm, last_norm = args.last_norm, use_RTE = args.use_RTE)
classifier = Classifier(args.n_hid, graph.y.max().item()+1)

model = nn.Sequential(gnn, classifier)
model.load_state_dict(torch.load(args.model_dir))
model.to(device)
print('Model #Params: %d' % get_n_params(model))
criterion = nn.NLLLoss()

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
        jobs = prepare_data(pool, task_type = 'sequential', s_idx = 0, n_batch = args.n_batch, batch_size=args.batch_size)
        with tqdm(np.arange(len(graph.test_paper) / args.n_batch // args.batch_size), desc='eval') as monitor:
            for s_idx in monitor:
                test_data = [job.get() for job in jobs]
                pool.close()
                pool.join()
                pool = mp.Pool(args.n_pool)
                jobs = prepare_data(pool, task_type = 'sequential', s_idx = int(s_idx * args.n_batch), batch_size=args.batch_size)

                for node_feature, node_type, edge_time, edge_index, edge_type, (train_mask, valid_mask, test_mask), ylabel in test_data:
                    ylabel = ylabel[:args.batch_size]
                    node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                                                   edge_time.to(device), edge_index.to(device), edge_type.to(device))
                    res  = classifier.forward(node_rep[:args.batch_size])
                    pred  = res.argmax(dim=1)
                    
                    y_pred += pred.tolist()
                    y_true += ylabel.tolist()
                    
                test_acc = evaluator.eval({
                                'y_true': torch.FloatTensor(y_true).unsqueeze(-1),
                                'y_pred': torch.FloatTensor(y_pred).unsqueeze(-1)
                            })['acc']
                monitor.set_postfix(accuracy = test_acc)
