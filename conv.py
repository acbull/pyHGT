import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax
import math

class RAGCNConv(MessagePassing):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout = 0.3, **kwargs):
        super(RAGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.num_types     = num_types
        self.num_relations = num_relations
        self.total_rel     = num_types * num_relations * num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.att           = None
        
        self.interact_sws   = nn.ModuleList()
        self.interact_tws   = nn.ModuleList()
        self.transfer_sws   = nn.ModuleList()
        
        self.relation_ws   = nn.ModuleList()
        self.aggregat_ws   = nn.ModuleList()
        
        for t in range(num_types):
            self.interact_sws.append(nn.Linear(in_dim,   out_dim))
            self.interact_tws.append(nn.Linear(in_dim,   out_dim))
            self.transfer_sws.append(nn.Linear(in_dim,   out_dim))
            self.aggregat_ws.append(nn.Linear(out_dim,  out_dim))
            
        self.relation_ws   = nn.Parameter(torch.ones(num_types, num_relations, num_types, self.n_heads))
        self.interact_rw   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.transfer_rw   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        
        self.drop          = nn.Dropout(dropout)
        self.emb           = RelTemporalEncoding(in_dim)
        
        glorot(self.interact_rw)
        glorot(self.transfer_rw)
        
    def forward(self, node_inp, node_type, edge_index, edge_type, edge_time):
        return self.propagate(edge_index, node_inp=node_inp, node_type=node_type, \
                              edge_type=edge_type, edge_time = edge_time)

    def message(self, edge_index_i, node_inp_i, node_inp_j, node_type_i, node_type_j, edge_type, edge_time, num_nodes):
        '''
            j: source, i: target; <j, i>
        '''
        data_size = edge_index_i.size(0)
        atts, vals = [], []
        res_att     = torch.zeros(data_size, self.n_heads).to(node_inp_i.device)
        res_val     = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)
        
        for source_id in range(self.num_types):
            sb = (node_type_j == int(source_id))
            interact_sw = self.interact_sws[source_id]
            transfer_sw = self.transfer_sws[source_id] 
            for target_id in range(self.num_types):
                tb = (node_type_i == int(target_id)) & sb
                interact_tw = self.interact_tws[target_id]
                for relation_id in range(self.num_relations):
                    idx = ((edge_type == int(relation_id)) & tb).detach()
                    if idx.sum() == 0:
                        continue
                    _node_inp_i = node_inp_i[idx]
                    _node_inp_j = self.emb(node_inp_j[idx], edge_time[idx])
                    _int_i = interact_tw(_node_inp_i).view(-1, self.n_heads, self.d_k)
                    _int_j = interact_sw(_node_inp_j).view(-1, self.n_heads, self.d_k)
                    
                    _int_s = torch.bmm(_int_j.transpose(1,0), self.interact_rw[relation_id]).transpose(1,0)
                    
                    res_att[idx] = (_int_s * _int_i).sum(dim=-1) * self.relation_ws[target_id][relation_id][source_id] / self.sqrt_dk
                    
                    _tra_j = transfer_sw(_node_inp_j).view(-1, self.n_heads, self.d_k)
                    res_val[idx] = torch.bmm(_tra_j.transpose(1,0), self.transfer_rw[relation_id]).transpose(1,0)
                    
        self.att = softmax(res_att, edge_index_i, data_size)
        res = res_val * self.att.view(-1, self.n_heads, 1)
        del res_att, res_val
        return res.view(-1, self.out_dim)


    def update(self, aggr_out, node_inp, node_type):
        '''
           x = W[node_type] * GNN(x)
        '''
        res = torch.zeros(aggr_out.size(0), self.out_dim).to(node_inp.device)
        for t_id in range(self.num_types):
            idx = (node_type == int(t_id))
            if idx.sum() == 0:
                continue
            res[idx] = F.relu(self.aggregat_ws[t_id](aggr_out[idx])) + node_inp[idx]
        out = self.drop(res)
        del res
        return out

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)


class RelTemporalEncoding(nn.Module):
    '''
        Implement the Temporal Encoding (Sinusoid) function.
    '''
    def __init__(self, n_hid, max_len = 240, dropout = 0.3):
        super(RelTemporalEncoding, self).__init__()
        self.drop = nn.Dropout(dropout)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = 1 / (10000 ** (torch.arange(0., n_hid * 2, 2.)) / n_hid / 2)
        self.emb = nn.Embedding(max_len, n_hid * 2)
        self.emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        self.emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        self.emb.requires_grad = False
        self.lin = nn.Linear(n_hid * 2, n_hid)
    def forward(self, x, t):
        return x + self.lin(self.drop(self.emb(t)))
    
    
    
class GeneralConv(nn.Module):
    def __init__(self, conv_name, in_hid, out_hid, num_types, num_relations, n_heads, dropout):
        super(GeneralConv, self).__init__()
        self.conv_name = conv_name
        if self.conv_name == 'hgt':
            self.base_conv = RAGCNConv(in_hid, out_hid, num_types, num_relations, n_heads, dropout)
        elif self.conv_name == 'gcn':
            self.base_conv = GCNConv(in_hid, out_hid)
        elif self.conv_name == 'gat':
            self.base_conv = GATConv(in_hid, out_hid // n_heads, heads=n_heads)
    def forward(self, meta_xs, node_type, edge_index, edge_type, edge_time):
        if self.conv_name == 'hgt':
            return self.base_conv(meta_xs, node_type, edge_index, edge_type, edge_time)
        elif self.conv_name == 'gcn':
            return self.base_conv(meta_xs, edge_index)
        elif self.conv_name == 'gat':
            return self.base_conv(meta_xs, edge_index)
    
  