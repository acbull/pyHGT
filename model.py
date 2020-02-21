from conv import *

class Classifier(nn.Module):
    def __init__(self, n_hid, n_out, dropout = 0.5):
        super(Classifier, self).__init__()
        self.drop     = nn.Dropout(dropout)
        self.n_hids   = n_hid
        self.n_out    = n_out
        self.linear   = nn.Linear(n_hid,  n_out)
    def forward(self, x):
        tx = self.linear(self.drop(x))
        return torch.log_softmax(tx.squeeze(), dim=-1)
    def __repr__(self):
        return '{}(n_hid={}, n_out={})'.format(
            self.__class__.__name__, self.n_hid, self.n_out)

class Matcher(nn.Module):
    '''
        Matching between a pair of nodes to conduct link prediction.
        Use multi-head attention as matching model.
    '''
    def __init__(self, n_hid, dropout = 0.5):
        super(Matcher, self).__init__()
        self.left_linear    = nn.Linear(n_hid,  n_hid)
        self.right_linear   = nn.Linear(n_hid,  n_hid)
        self.sqrt_hd  = math.sqrt(n_hid)
        self.drop     = nn.Dropout(dropout)
        self.cache      = None
    def forward(self, x, y, infer = False, pair = False):
        ty = self.drop(self.right_linear(y))
        if infer:
            '''
                During testing, we will consider millions or even billions of nodes as candidates (x).
                It's not possible to calculate them again for different query (y)
                Since the model is fixed, we propose to cache them, and dirrectly use the results.
            '''
            if self.cache != None:
                tx = self.cache
            else:
                tx = self.left_linear(x)
                self.cache = tx
        else:
            tx = self.drop(self.left_linear(x))
        if pair:
            res = (tx * ty).sum(dim=-1)
        else:
            res = torch.matmul(tx, ty.transpose(0,1))
        return res / self.sqrt_hd

    


    
class GNN(nn.Module):
    def __init__(self, in_dim, n_hid, num_types, num_relations, n_heads, n_layers, dropout = 0.5, conv_name = 'hgt'):
        super(GNN, self).__init__()
        self.gcs = nn.ModuleList()
        self.num_types = num_types
        self.in_dim    = in_dim
        self.n_hid     = n_hid
        self.adapt_ws  = nn.ModuleList()
        self.drop      = nn.Dropout(dropout)
        for t in range(num_types):
            self.adapt_ws.append(nn.Linear(in_dim, n_hid))
        for l in range(n_layers):
            self.gcs.append(GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout))

    def forward(self, node_feature, node_type, edge_time, edge_index, edge_type):
        res = torch.zeros(node_feature.size(0), self.n_hid).to(node_feature.device)
        for t_id in range(self.num_types):
            idx = (node_type == int(t_id))
            if idx.sum() == 0:
                continue
            res[idx] = torch.tanh(self.adapt_ws[t_id](node_feature[idx]))
        meta_xs = self.drop(res)
        del res
        for gc in self.gcs:
            meta_xs = gc(meta_xs, node_type, edge_index, edge_type, edge_time)
        return meta_xs   
    
    
class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""
    def __init__(self, n_word, ninp, nhid, nlayers, dropout=0.3):
        super(RNNModel, self).__init__()
        self.drop    = nn.Dropout(dropout)
        self.rnn     = nn.LSTM(nhid, nhid, nlayers)
        self.encoder = nn.Embedding(n_word, nhid)
        self.decoder = nn.Linear(nhid, n_word)
        self.decoder.weight = self.encoder.weight
        self.adapt   = nn.Linear(ninp + nhid, nhid)
        self.encoder.weight.require_grad = False
    def forward(self, inp, hidden = None):
        emb = self.drop(self.encoder(inp))
        if hidden is not None:
            emb = torch.cat((emb, hidden), dim=-1)
            emb = F.tanh(self.adapt(emb))
        output, _ = self.rnn(self.drop(emb))
        decoded = self.decoder(self.drop(output))
        return decoded
    def from_w2v(self, w2v):
        initrange = 0.1
        self.encoder.weight.data = w2v