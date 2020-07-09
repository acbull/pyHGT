import argparse
from tqdm import tqdm
import sys

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


parser = argparse.ArgumentParser(description='Training GNN on ogbn-products benchmark')



parser.add_argument('--data_dir', type=str, default='./',
                    help='The address to output the preprocessed graph.')
parser.add_argument('--data_name', type=str, default='ogbn-products',
                    help='Name of the dataset')
parser.add_argument('--model_dir', type=str, default='./hello',
                    help='The address for storing the trained models.')
parser.add_argument('--plot', action='store_true',
                    help='Whether to plot the loss/acc curve')
parser.add_argument('--cuda', type=int, default=0,
                    help='Avaiable GPU ID')
parser.add_argument('--conv_name', type=str, default='hgt',
                    choices=['hgt', 'gcn', 'gat', 'rgcn', 'han', 'hetgnn'],
                    help='The name of GNN filter. By default is Heterogeneous Graph Transformer (hgt)')
parser.add_argument('--n_hid', type=int, default=256,
                    help='Number of hidden dimension')
parser.add_argument('--n_heads', type=int, default=4,
                    help='Number of attention head')
parser.add_argument('--n_layers', type=int, default=3,
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

'''
    Optimization arguments
'''
parser.add_argument('--optimizer', type=str, default='adamw',
                    choices=['adamw', 'adam', 'sgd', 'adagrad'],
                    help='optimizer to use.')
parser.add_argument('--n_epoch', type=int, default=200,
                    help='Number of epoch to run')
parser.add_argument('--n_pool', type=int, default=8,
                    help='Number of process to sample subgraph')    
parser.add_argument('--n_batch', type=int, default=32,
                    help='Number of batch (sampled graphs) for each epoch') 
parser.add_argument('--batch_size', type=int, default=256,
                    help='Number of output nodes for training')    
parser.add_argument('--clip', type=int, default=1.0,
                    help='Gradient Norm Clipping') 

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


def prepare_data(pool):
    '''
        Sampled and prepare training and validation data using multi-process parallization.
    '''
    jobs = []
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
    return jobs

class GNN(nn.Module):
    def __init__(self, in_dim, n_hid, num_types, num_relations, n_heads, n_layers,\
                 dropout = 0.2, conv_name = 'hgt', prev_norm = False, last_norm = False, use_RTE = True):
        super(GNN, self).__init__()
        self.gcs = nn.ModuleList()
        self.num_types = num_types
        self.in_dim    = in_dim
        self.n_hid     = n_hid
        self.adapt_ws  = nn.ModuleList()
        self.drop      = nn.Dropout(dropout)
        for t_id in range(num_types):
            if graph.get_types()[t_id] == 'cate':
                self.adapt_ws.append(nn.Embedding(graph.y.max().item()+1, n_hid))
            else:
                self.adapt_ws.append(nn.Linear(in_dim, n_hid))
        for l in range(n_layers - 1):
            self.gcs.append(GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm = prev_norm, use_RTE = use_RTE))
        self.gcs.append(GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm = last_norm, use_RTE = use_RTE))

    def forward(self, node_feature, node_type, edge_time, edge_index, edge_type):
        res = torch.zeros(node_feature.size(0), self.n_hid).to(node_feature.device)
        for t_id in range(self.num_types):
            idx = (node_type == int(t_id))
            if idx.sum() == 0:
                continue
            if graph.get_types()[t_id] == 'cate':
                res[idx] = self.adapt_ws[t_id](node_feature[idx][:,0].long())
            else:
                res[idx] = F.gelu(self.adapt_ws[t_id](node_feature[idx]))
        meta_xs = self.drop(res)
        del res
        for gc in self.gcs:
            meta_xs = gc(meta_xs, node_type, edge_index, edge_type, edge_time)
        return meta_xs  
        
        


graph = dill.load(open(os.path.join(args.data_dir, args.data_name + '.pk'), 'rb'))
evaluator = Evaluator(name=args.data_name)
device = torch.device("cuda:%d" % args.cuda)
print(graph.node_feature.keys())
gnn = GNN(conv_name = args.conv_name, in_dim = len(graph.node_feature['product'][0]), \
          n_hid = args.n_hid, n_heads = args.n_heads, n_layers = args.n_layers, dropout = args.dropout,\
          num_types = len(graph.get_types()), num_relations = len(graph.get_meta_graph()) + 1,\
          prev_norm = args.prev_norm, last_norm = args.last_norm, use_RTE = args.use_RTE)
classifier = Classifier(args.n_hid, graph.y.max().item()+1)

model = nn.Sequential(gnn, classifier).to(device)
print('Model #Params: %d' % get_n_params(model))
criterion = nn.NLLLoss()


param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],     'weight_decay': 0.0}
    ]


optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-06)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, pct_start=0.05, anneal_strategy='linear', final_div_factor=10,\
                        max_lr = 5e-4, total_steps = args.n_batch * args.n_epoch + 1)

stats = []
res   = []
best_val   = 0
train_step = 0

pool = mp.Pool(args.n_pool)
st = time.time()
jobs = prepare_data(pool)


for epoch in np.arange(args.n_epoch) + 1:
    '''
        Prepare Training and Validation Data
    '''
    train_data = [job.get() for job in jobs[:-2]]
    valid_data = jobs[-2].get()
    test_data = jobs[-1].get()
    pool.close()
    pool.join()
    '''
        After the data is collected, close the pool and then reopen it.
    '''
    pool = mp.Pool(args.n_pool)
    jobs = prepare_data(pool)
    et = time.time()
    print('Data Preparation: %.1fs' % (et - st))
    
    '''
        Train
    '''
    model.train()
    train_losses = []
    stat = []
    for node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel in train_data:
        node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                               edge_time.to(device), edge_index.to(device), edge_type.to(device))
        res  = classifier.forward(node_rep[x_ids])
        loss = criterion(res, ylabel.squeeze().to(device))

        optimizer.zero_grad() 
        torch.cuda.empty_cache()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        train_losses += [loss.cpu().detach().tolist()]
        train_step += 1
        scheduler.step(train_step)
        del node_rep, ylabel
        
    model.eval()
    with torch.no_grad():
        node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = valid_data
        node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                                   edge_time.to(device), edge_index.to(device), edge_type.to(device))
        res  = classifier.forward(node_rep[x_ids])
        loss = criterion(res, ylabel.squeeze().to(device))
        
        '''
            Calculate Valid NDCG. Update the best model based on highest NDCG score.
        '''
        valid_acc  = evaluator.eval({
                        'y_true': ylabel,
                        'y_pred': res.argmax(dim=1).unsqueeze(-1)
                    })['acc']
        
        if valid_acc > best_val:
            best_val = valid_acc
            torch.save(model, args.model_dir)
            print('UPDATE!!!')
        
        node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = test_data
        node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                                   edge_time.to(device), edge_index.to(device), edge_type.to(device))
        res  = classifier.forward(node_rep[x_ids])
        
        test_acc  = evaluator.eval({
                        'y_true': ylabel,
                        'y_pred': res.argmax(dim=1).unsqueeze(-1)
                    })['acc']
        
        st = time.time()
        print(("Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  Valid Acc: %.4f  Test Acc: %.4f") % \
              (epoch, (st-et), optimizer.param_groups[0]['lr'], np.average(train_losses), \
                    loss.cpu().detach().tolist(), valid_acc, test_acc))
        stats += [[np.average(train_losses), loss.cpu().detach().tolist()]]
        del res, loss
