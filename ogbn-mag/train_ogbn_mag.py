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



parser.add_argument('--data_dir', type=str, default='/datadrive/dataset/OGB_MAG.pk',
                    help='The address of preprocessed graph.')
parser.add_argument('--model_dir', type=str, default='./hgt_4layer',
                    help='The address for storing the trained models.')
parser.add_argument('--plot', action='store_true',
                    help='Whether to plot the loss/acc curve')
parser.add_argument('--cuda', type=int, default=0,
                    help='Avaiable GPU ID')
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

parser.add_argument('--n_epoch', type=int, default=100,
                    help='Number of epoch to run')
parser.add_argument('--n_pool', type=int, default=8,
                    help='Number of process to sample subgraph')    
parser.add_argument('--n_batch', type=int, default=32,
                    help='Number of batch (sampled graphs) for each epoch') 
parser.add_argument('--batch_size', type=int, default=128,
                    help='Number of output nodes for training')  
parser.add_argument('--clip', type=int, default=1.0,
                    help='Gradient Norm Clipping') 

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
                            np.random.choice(target_nodes, args.batch_size, replace = False)]))
            jobs.append(p)
    elif task_type == 'sequential':
        for i in np.arange(n_batch):
            target_papers = graph.test_paper[(s_idx + i) * batch_size : (s_idx + i + 1) * batch_size]
            p = pool.apply_async(ogbn_sample, args=([randint(), target_papers]))
            jobs.append(p)
    elif task_type == 'variance_reduce':
        target_papers = graph.test_paper[s_idx * args.batch_size : (s_idx + 1) * args.batch_size]
        for batch_id in np.arange(n_batch):
            p = pool.apply_async(ogbn_sample, args=([randint(), target_papers]))
            jobs.append(p)
    return jobs

graph = dill.load(open(args.data_dir, 'rb'))
evaluator = Evaluator(name='ogbn-mag')
device = torch.device("cuda:%d" % args.cuda)
target_nodes = np.arange(len(graph.node_feature['paper']))
gnn = GNN(conv_name = args.conv_name, in_dim = len(graph.node_feature['paper'][0]), \
          n_hid = args.n_hid, n_heads = args.n_heads, n_layers = args.n_layers, dropout = args.dropout,\
          num_types = len(graph.get_types()), num_relations = len(graph.get_meta_graph()) + 1,\
          prev_norm = args.prev_norm, last_norm = args.last_norm, use_RTE = args.use_RTE)
classifier = Classifier(args.n_hid, graph.y.max()+1)

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
    datas = [job.get() for job in jobs]
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
    stat = []
    for node_feature, node_type, edge_time, edge_index, edge_type, (train_mask, valid_mask, test_mask), ylabel in datas:
        node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                               edge_time.to(device), edge_index.to(device), edge_type.to(device))
        ylabel = torch.LongTensor(ylabel).to(device)
        train_res  = classifier.forward(node_rep[:len(ylabel)][train_mask])
        valid_res  = classifier.forward(node_rep[:len(ylabel)][valid_mask])
        test_res   = classifier.forward(node_rep[:len(ylabel)][test_mask])

        train_loss = criterion(train_res, ylabel[train_mask])

        optimizer.zero_grad() 
        train_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        train_step += 1
        scheduler.step(train_step)

        train_acc  = evaluator.eval({
                        'y_true': ylabel[train_mask].unsqueeze(-1),
                        'y_pred': train_res.argmax(dim=1).unsqueeze(-1)
                    })['acc']
        valid_acc  = evaluator.eval({
                        'y_true': ylabel[valid_mask].unsqueeze(-1),
                        'y_pred': valid_res.argmax(dim=1).unsqueeze(-1)
                    })['acc']
        test_acc   = evaluator.eval({
                        'y_true': ylabel[test_mask].unsqueeze(-1),
                        'y_pred': test_res.argmax(dim=1).unsqueeze(-1)
                    })['acc']
        stat += [[train_loss.item(), train_acc, valid_acc, test_acc]]
        del node_rep, train_loss, ylabel
    stats += [stat]
    avgs = np.average(stat, axis=0)
    if avgs[2] > best_val:
        best_val = avgs[2]
        torch.save(model.state_dict(), args.model_dir)
        print('UPDATE!!!  ' + str(best_val))
    print('Epoch: %d LR: %.5f Train Loss: %.4f Train Acc: %.4f Valid Acc: %.4f Test Acc: %.4f' % \
         (epoch,  optimizer.param_groups[0]['lr'], avgs[0], avgs[1], avgs[2], avgs[3]))
    st = time.time()
    if args.plot and epoch % 50 == 0:
        s = np.concatenate(stats)
        for i in range(4):
            data = np.stack((s[-args.n_batch * 100:, i], np.arange(len(s[-args.n_batch * 100:, i])) // args.batch_size), axis=0).transpose()
            sb.lineplot(data = pd.DataFrame(data, columns = ['Value', 'Epoch']), x='Epoch', y='Value')
            plt.show()
            
if args.plot:
    s = np.concatenate(stats)
    for i in range(4):
        data = np.stack((s[args.n_batch * 100:, i], np.arange(len(s[args.n_batch * 100:, i])) // args.n_batch), axis=0).transpose()
        sb.lineplot(data = pd.DataFrame(data, columns = ['Value', 'Epoch']), x='Epoch', y='Value')
        plt.show()
