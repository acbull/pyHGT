import argparse

from transformers import *

from pyHGT.data import *

# from tqdm import tqdm_notebook as tqdm  # Uncomment this line if using jupyter notebook

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

venue_types = ['conference', 'journal', 'repository', 'patent']
test_time_bar = 2016

filename = 'PR%s_20190919.tsv' % args.domain
print(f'Counting paper cites from {filename}...')
filename = f'{args.input_dir}/{filename}'
line_count = sum(1 for line in open(filename))

cite_dict = defaultdict(lambda: 0)

with open(filename) as fin:
    fin.readline()
    for line in tqdm(fin, total=line_count):
        tokens = line.strip().split('\t')
        paper_id = tokens[1]
        cite_dict[paper_id] += 1

filename = 'Papers%s_20190919.tsv' % args.domain
print(f'Reading Paper nodes from {filename}...')
filename = f'{args.input_dir}/{filename}'
line_count = sum(1 for line in open(filename))

paper_nodes = defaultdict(lambda: {})

with open(filename) as fin:
    fin.readline()
    for line in tqdm(fin, total=line_count):
        tokens = line.strip().split('\t')

        paper_id = tokens[0]
        time = tokens[1]
        title = tokens[2]
        venue_id = tokens[3]
        lang = tokens[4]

        bound = min(2020 - int(time), 20) * args.citation_bar

        if ((cite_dict[paper_id] < bound) or paper_id == '' or time == '' or title == '') or \
                (venue_id == '' and lang == '') or \
                int(time) < 1900:
            continue

        paper_node = {'id': paper_id, 'title': title, 'type': 'paper', 'time': int(time)}
        paper_nodes[paper_id] = paper_node

if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")

filename = 'PAb%s_20190919.tsv' % args.domain
print(f'Getting paper abstract embeddings. Abstracts are from {filename}...')
filename = f'{args.input_dir}/{filename}'
line_count = sum(1 for line in open(filename, 'r'))

tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
model = XLNetModel.from_pretrained('xlnet-base-cased',
                                   output_hidden_states=True,
                                   output_attentions=True).to(device)

with open(filename) as fin:
    fin.readline()
    for line in tqdm(fin, total=line_count):
        try:
            tokens = line.split('\t')
            paper_id = tokens[0]
            if paper_id in paper_nodes:
                paper_node = paper_nodes[paper_id]

                input_ids = torch.tensor([tokenizer.encode(paper_node['title'])]).to(device)[:, :64]
                if len(input_ids[0]) < 4:
                    continue
                all_hidden_states, all_attentions = model(input_ids)[-2:]
                rep = (all_hidden_states[-2][0] * all_attentions[-2][0].mean(dim=0).mean(dim=0).view(-1, 1)).sum(dim=0)

                paper_node['emb'] = rep.tolist()
        except Exception as e:
            print(e)

filename = 'vfi_vector.tsv'
print(f'Reading Venue/Filed/Affiliation nodes from {filename}...')
filename = f'{args.input_dir}/{filename}'
line_count = sum(1 for line in open(filename))

vfi_ids = {}

with open(filename) as fin:
    for line in tqdm(fin, total=line_count):
        tokens = line.strip().split('\t')
        node_id = tokens[0]
        vfi_ids[node_id] = True

filename = 'Papers%s_20190919.tsv' % args.domain
print(f'Reading Paper-Venue triples from {filename}...')
filename = f'{args.input_dir}/{filename}'
line_count = sum(1 for line in open(filename, 'r'))

graph = Graph()
remaining_nodes = []

with open(filename) as fin:
    fin.readline()
    for line in tqdm(fin, total=line_count):
        tokens = line.strip().split('\t')

        paper_id = tokens[0]
        venue_id = tokens[3]
        lang = tokens[4]

        if (paper_id not in paper_nodes) or (lang != 'en') or \
                ('emb' not in paper_nodes[paper_id]) or (venue_id not in vfi_ids):
            continue

        remaining_nodes.append(paper_id)
        venue_type = tokens[-2]
        venue_node = {'id': venue_id, 'type': 'venue', 'attr': venue_type}
        graph.add_edge(paper_nodes[paper_id], venue_node, time=int(tokens[1]), relation_type='PV_' + venue_type)

org_count = len(paper_nodes)
paper_nodes = {paper_id: paper_nodes[paper_id] for paper_id in remaining_nodes}
print(f'Removed article count: {(org_count - len(paper_nodes)):,}')

filename = 'PR%s_20190919.tsv' % args.domain
print(f'Reading Paper-Paper triples from {filename}...')
filename = f'{args.input_dir}/{filename}'
line_count = sum(1 for line in open(filename))
with open(filename) as fin:
    fin.readline()
    for line in tqdm(fin, total=line_count):
        tokens = line.strip().split('\t')
        paper_id1 = tokens[0]
        paper_id2 = tokens[1]

        if (paper_id1 in paper_nodes) and (paper_id2 in paper_nodes):
            p1 = paper_nodes[paper_id1]
            p2 = paper_nodes[paper_id2]
            if p1['time'] >= p2['time']:
                graph.add_edge(p1, p2, time=p1['time'], relation_type='PP_cite')

filename = 'PF%s_20190919.tsv' % args.domain
print(f'Reading FieldOfStudyIds from {filename}...')
ffl = {}
filename = f'{args.input_dir}/{filename}'
line_count = sum(1 for line in open(filename))
with open(filename) as fin:
    fin.readline()
    for line in tqdm(fin, total=line_count):
        tokens = line.strip().split('\t')

        paper_id = tokens[0]
        field_id = tokens[1]

        if (paper_id in paper_nodes) and (field_id in vfi_ids):
            ffl[field_id] = True

filename = 'FHierarchy_20190919.tsv'
print(f'Reading field hierarchy from {filename}...')
filename = f'{args.input_dir}/{filename}'
line_count = sum(1 for line in open(filename))
with open(filename) as fin:
    fin.readline()
    for line in tqdm(fin, total=line_count):
        tokens = line.strip().split('\t')

        field_id1 = tokens[0]
        field_id2 = tokens[1]
        child_level = tokens[2]  # L1/L2/L3/L4/L5
        parent_level = tokens[3]  # L0/L1/L2/L3/L4

        if (field_id1 in ffl) and (field_id2 in ffl):
            field_node1 = {'id': field_id1, 'type': 'field', 'attr': child_level}
            field_node2 = {'id': field_id2, 'type': 'field', 'attr': parent_level}

            graph.add_edge(field_node1, field_node2, relation_type='FF_in')

            ffl[field_id1] = field_node1
            ffl[field_id2] = field_node2

filename = 'PF%s_20190919.tsv' % args.domain
print(f'Reading Paper-Field triples from {filename}...')
filename = f'{args.input_dir}/{filename}'
line_count = sum(1 for line in open(filename))
with open(filename) as fin:
    fin.readline()
    for line in tqdm(fin, total=line_count):
        tokens = line.strip().split('\t')

        paper_id = tokens[0]
        field_id = tokens[1]

        if (paper_id in paper_nodes) and (field_id in ffl) and (type(ffl[field_id]) == dict):
            paper_node = paper_nodes[paper_id]
            field_node = ffl[field_id]
            graph.add_edge(paper_node, field_node, time=paper_node['time'],
                           relation_type='PF_in_' + field_node['attr'])

filename = 'PAuAf%s_20190919.tsv' % args.domain
print(f'Reading Author-Affiliation triples from {filename}...')
paper_authors = defaultdict(lambda: {})
filename = f'{args.input_dir}/{filename}'
line_count = sum(1 for line in open(filename))
with open(filename) as fin:
    fin.readline()
    for line in tqdm(fin, total=line_count):
        tokens = line.strip().split('\t')

        paper_id = tokens[0]
        author_id = tokens[1]
        affiliation_id = tokens[2]

        if (paper_id in paper_nodes) and (affiliation_id in vfi_ids):
            paper_node = paper_nodes[paper_id]
            author_node = {'id': author_id, 'type': 'author'}
            affiliation_node = {'id': affiliation_id, 'type': 'affiliation'}

            position_in_author_list = int(tokens[-1])
            paper_authors[paper_id][position_in_author_list] = author_node
            graph.add_edge(author_node, affiliation_node, relation_type='in')

print('Adding Author-Paper triples...')
for paper_id in tqdm(paper_authors):
    paper_node = paper_nodes[paper_id]
    max_seq = max(paper_authors[paper_id].keys())

    for seq_i in paper_authors[paper_id]:
        author_node = paper_authors[paper_id][seq_i]
        if seq_i == 1:
            graph.add_edge(author_node, paper_node, time=paper_node['time'], relation_type='AP_write_first')
        elif seq_i == max_seq:
            graph.add_edge(author_node, paper_node, time=paper_node['time'], relation_type='AP_write_last')
        else:
            graph.add_edge(author_node, paper_node, time=paper_node['time'], relation_type='AP_write_other')

filename = 'vfi_vector.tsv'
print(f'Reading embeddings of Venue/Field/Affiliation nodes from {filename}...')
filename = f'{args.input_dir}/{filename}'
line_count = sum(1 for line in open(filename))
with open(filename) as fin:
    for line in tqdm(fin, total=line_count):
        tokens = line.strip().split('\t')

        node_id = tokens[0]
        node_feature_vector = tokens[1]

        for node_type in ['venue', 'field', 'affiliation']:
            if node_id in graph.node_forward[node_type]:
                node_idx = graph.node_forward[node_type][node_id]
                node = graph.node_backward[node_type][node_idx]
                node['node_emb'] = np.array(node_feature_vector.split(' '))

filename = 'SeqName%s_20190919.tsv' % args.domain
print(f'Reading node names from {filename}...')
filename = f'{args.input_dir}/{filename}'
line_count = sum(1 for line in open(filename))
with open(filename) as fin:
    for line in tqdm(fin, total=line_count):
        tokens = line.strip().split('\t')

        node_id = tokens[0]
        node_type = tokens[2]

        if node_type in venue_types:
            node_type = 'venue'
        if node_type == 'fos':
            node_type = 'field'
        if node_id in graph.node_forward[node_type]:
            node_idx = graph.node_forward[node_type][node_id]
            node = graph.node_backward[node_type][node_idx]
            node['name'] = tokens[1]

'''
    Calculate the total citation information as node attributes.
'''
print('Calculate the total citation information as node attributes...')
for paper_idx, paper_node in enumerate(graph.node_backward['paper']):
    paper_node['citation'] = len(graph.edge_list['paper']['paper']['PP_cite'][paper_idx])

for author_idx, author_node in enumerate(graph.node_backward['author']):
    citation = 0
    for rel in graph.edge_list['author']['paper'].keys():
        for paper_idx in graph.edge_list['author']['paper'][rel][author_idx]:
            paper_node = graph.node_backward['paper'][paper_idx]
            citation += paper_node['citation']

    author_node['citation'] = citation

for affiliation_idx, affiliation_node in enumerate(graph.node_backward['affiliation']):
    citation = 0
    for author_idx in graph.edge_list['affiliation']['author']['in'][affiliation_idx]:
        author_node = graph.node_backward['author'][author_idx]
        citation += author_node['citation']

    affiliation_node['citation'] = citation

for venue_idx, venue_node in enumerate(graph.node_backward['venue']):
    citation = 0
    for rel in graph.edge_list['venue']['paper'].keys():
        for paper_idx in graph.edge_list['venue']['paper'][rel][venue_idx]:
            paper_node = graph.node_backward['paper'][paper_idx]
            citation += paper_node['citation']

    venue_node['citation'] = citation

for field_idx, field_node in enumerate(graph.node_backward['field']):
    citation = 0
    for rel in graph.edge_list['field']['paper'].keys():
        for paper_idx in graph.edge_list['field']['paper'][rel][field_idx]:
            paper_node = graph.node_backward['paper'][paper_idx]
            citation += paper_node['citation']

    field_node['citation'] = citation

print('Done.')

'''
    Since only paper have w2v embedding, we simply propagate its
    feature to other nodes by averaging neighborhoods.
    Then, we construct the DataFrame for each node type.
'''
print('Calculating embeddings for non-Paper nodes...')
df = pd.DataFrame(graph.node_backward['paper'])
graph.node_feature = {'paper': df}
paper_embeddings = np.array(list(df['emb']))

for _type in graph.node_backward:
    if _type in ['paper', 'affiliation']:
        continue

    df = pd.DataFrame(graph.node_backward[_type])
    node_pairs = []
    for _rel in graph.edge_list[_type]['paper']:
        for target_idx in graph.edge_list[_type]['paper'][_rel]:
            for source_idx in graph.edge_list[_type]['paper'][_rel][target_idx]:
                if graph.edge_list[_type]['paper'][_rel][target_idx][source_idx] <= test_time_bar:
                    node_pairs += [[target_idx, source_idx]]
    if len(node_pairs) == 0:
        continue

    node_pairs = np.array(node_pairs).T
    edge_count = node_pairs.shape[1]
    v = np.ones(edge_count)
    m = normalize(sp.coo_matrix((v, node_pairs),
                                shape=(len(graph.node_backward[_type]), len(graph.node_backward['paper']))))

    out = m.dot(paper_embeddings)
    df['emb'] = list(out)
    graph.node_feature[_type] = df

'''
    Affiliation is not directly linked with Paper, so we average the author embedding.
'''
author_embeddings = np.array(list(graph.node_feature['author']['emb']))
df = pd.DataFrame(graph.node_backward['affiliation'])
node_pairs = []
for _rel in graph.edge_list['affiliation']['author']:
    for target_idx in graph.edge_list['affiliation']['author'][_rel]:
        for source_idx in graph.edge_list['affiliation']['author'][_rel][target_idx]:
            node_pairs += [[target_idx, source_idx]]

node_pairs = np.array(node_pairs).T
edge_count = node_pairs.shape[1]
v = np.ones(edge_count)
m = normalize(sp.coo_matrix((v, node_pairs),
                            shape=(len(graph.node_backward['affiliation']), len(graph.node_backward['author']))))
out = m.dot(author_embeddings)
df['emb'] = list(out)
graph.node_feature['affiliation'] = df

print('Done.')
print()
print('Cleaning edge list...')
clean_edge_list = {}
# target_type
for k1 in graph.edge_list:
    if k1 not in clean_edge_list:
        clean_edge_list[k1] = {}
    # source_type
    for k2 in graph.edge_list[k1]:
        if k2 not in clean_edge_list[k1]:
            clean_edge_list[k1][k2] = {}
        # relation_type
        for k3 in graph.edge_list[k1][k2]:
            if k3 not in clean_edge_list[k1][k2]:
                clean_edge_list[k1][k2][k3] = {}

            triple_count = 0
            # target_idx
            for e1 in graph.edge_list[k1][k2][k3]:
                edge_count = len(graph.edge_list[k1][k2][k3][e1])
                triple_count += edge_count
                if edge_count == 0:
                    continue
                clean_edge_list[k1][k2][k3][e1] = {}
                # source_idx
                for e2 in graph.edge_list[k1][k2][k3][e1]:
                    clean_edge_list[k1][k2][k3][e1][e2] = graph.edge_list[k1][k2][k3][e1][e2]
            print(k1, k2, k3, triple_count)

graph.edge_list = clean_edge_list
print()
print('Number of nodes:')
for node_type in graph.node_forward:
    print(f'{node_type}: {len(graph.node_forward[node_type]):,}')
print()

del graph.node_backward

print('Writting graph in file:')
dill.dump(graph, open(args.output_dir + '/graph%s.pk' % args.domain, 'wb'))
print('Done.')
