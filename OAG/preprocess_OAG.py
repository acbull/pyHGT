from transformers import *

from data import *
import gensim
from gensim.models import Word2Vec
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm   # Comment this line if using jupyter notebook


import argparse

parser = argparse.ArgumentParser(description='Preprocess OAG (CS/Med/All) Data')

'''
    Dataset arguments
'''
parser.add_argument('--input_dir', type=str, default='./data/oag_raw',
                    help='The address to store the original data directory.')
parser.add_argument('--output_dir', type=str, default='./data/oag_output',
                    help='The address to output the preprocessed graph.')
parser.add_argument('--cuda', type=int, default=0,
                    help='Avaiable GPU ID')
parser.add_argument('--domain', type=str, default='_CS',
                    help='CS, Medical or All: _CS or _Med or (empty)')
parser.add_argument('--citation_bar', type=int, default=1,
                    help='Only consider papers with citation larger than (2020 - year) * citation_bar')

args = parser.parse_args()


test_time_bar = 2016

cite_dict = defaultdict(lambda: 0)
with open(args.input_dir + '/PR%s_20190919.tsv' % args.domain) as fin:
    fin.readline()
    for l in tqdm(fin, total = sum(1 for line in open(args.input_dir + '/PR%s_20190919.tsv' % args.domain))):
        l = l[:-1].split('\t')
        cite_dict[l[1]] += 1
        
        
pfl = defaultdict(lambda: {})
with open(args.input_dir + '/Papers%s_20190919.tsv' % args.domain) as fin:
    fin.readline()
    for l in tqdm(fin, total = sum(1 for line in open(args.input_dir + '/Papers%s_20190919.tsv' % args.domain))):
        l = l[:-1].split('\t')
        bound = min(2020 - int(l[1]), 20) * args.citation_bar
        if cite_dict[l[0]] < bound or l[0] == '' or l[1] == '' or l[2] == '' or l[3] == '' and l[4] == '' or int(l[1]) < 1900:
            continue
        pi = {'id': l[0], 'title': l[2], 'type': 'paper', 'time': int(l[1])}
        pfl[l[0]] = pi
        
      
if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")
        
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
model = XLNetModel.from_pretrained('xlnet-base-cased',
                                    output_hidden_states=True,
                                    output_attentions=True).to(device)
        
               
with open(args.input_dir + '/PAb%s_20190919.tsv' % args.domain) as fin:
    fin.readline()
    for l in tqdm(fin, total = sum(1 for line in open(args.input_dir + '/PAb%s_20190919.tsv' % args.domain, 'r'))):
        try:
            l = l.split('\t')
            if l[0] in pfl:
                input_ids = torch.tensor([tokenizer.encode(pfl[l[0]]['title'])]).to(device)[:, :64]
                if len(input_ids[0]) < 4:
                    continue
                all_hidden_states, all_attentions = model(input_ids)[-2:]
                rep = (all_hidden_states[-2][0] * all_attentions[-2][0].mean(dim=0).mean(dim=0).view(-1, 1)).sum(dim=0)
                pfl[l[0]]['emb'] = rep.tolist()
        except Exception as e:
            print(e)
        
        
        
vfi_ids = {}
with open(args.input_dir + '/vfi_vector.tsv') as fin:
    for l in tqdm(fin, total = sum(1 for line in open(args.input_dir + '/vfi_vector.tsv'))):
        l = l[:-1].split('\t')
        vfi_ids[l[0]] = True        
        
        
graph = Graph()
rem = []
with open(args.input_dir + '/Papers%s_20190919.tsv' % args.domain) as fin:
    fin.readline()
    for l in tqdm(fin, total = sum(1 for line in open(args.input_dir + '/Papers%s_20190919.tsv' % args.domain, 'r'))):
        l = l[:-1].split('\t')
        if l[0] not in pfl or l[4] != 'en' or 'emb' not in pfl[l[0]] or l[3] not in vfi_ids:
            continue
        rem += [l[0]]
        vi = {'id': l[3], 'type': 'venue', 'attr': l[-2]}
        graph.add_edge(pfl[l[0]], vi, time = int(l[1]), relation_type = 'PV_' + l[-2])
pfl = {i: pfl[i] for i in rem}
print(len(pfl))

        
with open(args.input_dir + '/PR%s_20190919.tsv' % args.domain) as fin:
    fin.readline()
    for l in tqdm(fin, total = sum(1 for line in open(args.input_dir + '/PR%s_20190919.tsv' % args.domain))):
        l = l[:-1].split('\t')
        if l[0] in pfl and l[1] in pfl:
            p1 = pfl[l[0]]
            p2 = pfl[l[1]]
            if p1['time'] >= p2['time']:
                graph.add_edge(p1, p2, time = p1['time'], relation_type = 'PP_cite')
        
        
        
ffl = {}
with open(args.input_dir + '/PF%s_20190919.tsv' % args.domain) as fin:
    fin.readline()
    for l in tqdm(fin, total = sum(1 for line in open(args.input_dir + '/PF%s_20190919.tsv' % args.domain))):
        l = l[:-1].split('\t')
        if l[0] in pfl and l[1] in vfi_ids:
            ffl[l[1]] = True        
        
        
        
        
with open(args.input_dir + '/FHierarchy_20190919.tsv') as fin:
    fin.readline()
    for l in tqdm(fin, total = sum(1 for line in open(args.input_dir + '/FHierarchy_20190919.tsv'))):
        l = l[:-1].split('\t')
        if l[0] in ffl and l[1] in ffl:
            fi = {'id': l[0], 'type': 'field', 'attr': l[2]}
            fj = {'id': l[1], 'type': 'field', 'attr': l[3]}
            graph.add_edge(fi, fj, relation_type = 'FF_in')
            ffl[l[0]] = fi
            ffl[l[1]] = fj        
        
        
        
        
with open(args.input_dir + '/PF%s_20190919.tsv' % args.domain) as fin:
    fin.readline()
    for l in tqdm(fin, total = sum(1 for line in open(args.input_dir + '/PF%s_20190919.tsv' % args.domain))):
        l = l[:-1].split('\t')
        if l[0] in pfl and l[1] in ffl and type(ffl[l[1]]) == dict:
            pi = pfl[l[0]]
            fi = ffl[l[1]]
            graph.add_edge(pi, fi, time = pi['time'], relation_type = 'PF_in_' + fi['attr'])        
        
        
        
        
coa = defaultdict(lambda: {})
with open(args.input_dir + '/PAuAf%s_20190919.tsv' % args.domain) as fin:
    fin.readline()
    for l in tqdm(fin, total = sum(1 for line in open(args.input_dir + '/PAuAf%s_20190919.tsv' % args.domain))):
        l = l[:-1].split('\t')
        if l[0] in pfl and l[2] in vfi_ids:
            pi = pfl[l[0]]
            ai = {'id': l[1], 'type': 'author'}
            fi = {'id': l[2], 'type': 'affiliation'}
            coa[l[0]][int(l[-1])] = ai
            graph.add_edge(ai, fi, relation_type = 'in')

for pid in tqdm(coa):
    pi = pfl[pid]
    max_seq = max(coa[pid].keys())
    for seq_i in coa[pid]:
        ai = coa[pid][seq_i]
        if seq_i == 1:
            graph.add_edge(ai, pi, time = pi['time'], relation_type = 'AP_write_first')
        elif seq_i == max_seq:
            graph.add_edge(ai, pi, time = pi['time'], relation_type = 'AP_write_last')
        else:
            graph.add_edge(ai, pi, time = pi['time'], relation_type = 'AP_write_other')
        
        
        
        
with open(args.input_dir + '/vfi_vector.tsv') as fin:
    for l in tqdm(fin, total = sum(1 for line in open(args.input_dir + '/vfi_vector.tsv'))):
        l = l[:-1].split('\t')
        ser = l[0]
        for idx in ['venue', 'field', 'affiliation']:
            if ser in graph.node_forward[idx]:
                graph.node_bacward[idx][graph.node_forward[idx][ser]]['node_emb'] = np.array(l[1].split(' '))        
        
        

        
with open(args.input_dir + '/SeqName%s_20190919.tsv' % args.domain) as fin:
    for l in tqdm(fin, total = sum(1 for line in open(args.input_dir + '/SeqName%s_20190919.tsv' % args.domain))):
        l = l[:-1].split('\t')
        key = l[2]
        if key in ['conference', 'journal', 'repository', 'patent']:
            key = 'venue'
        if key == 'fos':
            key = 'field'
        if l[0] in graph.node_forward[key]:
            s = graph.node_bacward[key][graph.node_forward[key][l[0]]]
            s['name'] = l[1]     
    
'''
    Calculate the total citation information as node attributes.
'''
    
for idx, pi in enumerate(graph.node_bacward['paper']):
    pi['citation'] = len(graph.edge_list['paper']['paper']['PP_cite'][idx])
for idx, ai in enumerate(graph.node_bacward['author']):
    citation = 0
    for rel in graph.edge_list['author']['paper'].keys():
        for pid in graph.edge_list['author']['paper'][rel][idx]:
            citation += graph.node_bacward['paper'][pid]['citation']
    ai['citation'] = citation
for idx, fi in enumerate(graph.node_bacward['affiliation']):
    citation = 0
    for aid in graph.edge_list['affiliation']['author']['in'][idx]:
        citation += graph.node_bacward['author'][aid]['citation']
    fi['citation'] = citation
for idx, vi in enumerate(graph.node_bacward['venue']):
    citation = 0
    for rel in graph.edge_list['venue']['paper'].keys():
        for pid in graph.edge_list['venue']['paper'][rel][idx]:
            citation += graph.node_bacward['paper'][pid]['citation']
    vi['citation'] = citation
for idx, fi in enumerate(graph.node_bacward['field']):
    citation = 0
    for rel in graph.edge_list['field']['paper'].keys():
        for pid in graph.edge_list['field']['paper'][rel][idx]:
            citation += graph.node_bacward['paper'][pid]['citation']
    fi['citation'] = citation
    
    
    

'''
    Since only paper have w2v embedding, we simply propagate its
    feature to other nodes by averaging neighborhoods.
    Then, we construct the Dataframe for each node type.
'''
d = pd.DataFrame(graph.node_bacward['paper'])
graph.node_feature = {'paper': d}
cv = np.array(list(d['emb']))
for _type in graph.node_bacward:
    if _type not in ['paper', 'affiliation']:
        d = pd.DataFrame(graph.node_bacward[_type])
        i = []
        for _rel in graph.edge_list[_type]['paper']:
            for t in graph.edge_list[_type]['paper'][_rel]:
                for s in graph.edge_list[_type]['paper'][_rel][t]:
                    if graph.edge_list[_type]['paper'][_rel][t][s] <= test_time_bar:
                        i += [[t, s]]
        if len(i) == 0:
            continue
        i = np.array(i).T
        v = np.ones(i.shape[1])
        m = normalize(sp.coo_matrix((v, i), \
            shape=(len(graph.node_bacward[_type]), len(graph.node_bacward['paper']))))
        out = m.dot(cv)
        d['emb'] = list(out)
        graph.node_feature[_type] = d
'''
    Affiliation is not directly linked with Paper, so we average the author embedding.
'''
cv = np.array(list(graph.node_feature['author']['emb']))
d = pd.DataFrame(graph.node_bacward['affiliation'])
i = []
for _rel in graph.edge_list['affiliation']['author']:
    for j in graph.edge_list['affiliation']['author'][_rel]:
        for t in graph.edge_list['affiliation']['author'][_rel][j]:
            i += [[j, t]]
i = np.array(i).T
v = np.ones(i.shape[1])
m = normalize(sp.coo_matrix((v, i), \
    shape=(len(graph.node_bacward['affiliation']), len(graph.node_bacward['author']))))
out = m.dot(cv)
d['emb'] = list(out)
graph.node_feature['affiliation'] = d           
      
    
edg = {}
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
            print(k1, k2, k3, len(edg[k1][k2][k3]))
graph.edge_list = edg

        
del graph.node_bacward        
dill.dump(graph, open(args.output_dir + '/graph%s.pk' % args.domain, 'wb'))       
        
        
        
