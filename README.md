# Heterogeneous Graph Transformer (HGT)

Heterogeneous Graph Transformer is a graph neural network architecture that can deal with large-scale heterogeneous and dynamic graphs.

You can see our WWW 2020 (know as The Web Conference) paper () for more details.

## Overview
The most important files in this projects are as follow:
- conv.py: The core of our model, implements the transformer-like heterogeneous graph convolutional layer.
- model.py: The wrap of different model components.
- data.py: The data interface and usage.
  - `class Graph`: The data structure of heterogeneous graph. Stores feature in ``Graph.node_feature`` as pandas.DataFrame; Stores adjacency matrix in ``Graph.edge_list`` as dictionay.
  - `def sample_subgraph`: The sampling algorithm for heterogeneous graph. Each iteration samples a fixed number of nodes per type. All the sampled nodes are within the region of already sampled nodes, with sampling probability as the square of relative degree.
- train_*.py: The training and validation script for a specific downstream task.
  - `def *_sample`: The sampling function for a given task. Remember to mask out existing link within the graph to avoid information leakage.
  - `def prepare_data`: Conduct sampling in parallel with multiple processes, which can seamlessly coordinate with model training.
  
## Setup

This implementation is based on pytorch_geometric. To run the code, you need the following dependencies:

- [Pytorch 1.0](https://pytorch.org/)
- [pytorch_geometric](https://pytorch-geometric.readthedocs.io/)
- [gensim](https://github.com/RaRe-Technologies/gensim)
- [sklearn](https://github.com/scikit-learn/scikit-learn)
- [tqdm](https://github.com/tqdm/tqdm)

Our current experiments are conducted on Open Academic Graph (OAG). For easiness of usage, we split the whole dataset into different granularity: all CS papers (8.1G), all ML papers (1.9G), all NN papers (0.6G).

For raw data, you can download via (https://drive.google.com/open?id=1Np4J0q_ZIYr74IIdLGlAgzwwx00KJJeE). After downloading it, run preprocess_OAG.py to extract features and store them in our data structure.

For preprocessed data, you can download via (https://drive.google.com/open?id=1Np4J0q_ZIYr74IIdLGlAgzwwx00KJJeE).

You can also use our code to process other heteogeneous graph, as long as you load them into our data structure `class Graph` in data.py. Refer to preprocess_OAG.py for a demonstration.

## Usage
Execute the following scripts to train on paper-field(L2) classification task using HGT:

```bash
python3 train_paper_field.py --data_dir PATH_OF_DATASET --model_dir PATH_OF_SAVED_MODEL --conv_name hgt
```
There are some key options of this scrips:
- `conv_name`: Choose corresponding model for training. By default we use HGT.
- `--sample_depth` and `--sample_width`: The depth and width of sampled graph. If the model exceeds the GPU memory, can consider reduce their number; if one wants to train a deeper GNN model, consider adding these numbers.
- `--n_pool`: The number of process to parallely conduct sampling. If one has a machine with large memory, can consider adding this number to reduce batch prepartion time.
- `--repeat`: The number of time to reuse a sampled batch for training. If the training time is much smaller than sampling time, can consider adding this number.

The details of other optional hyperparameters can be found in train_*.py. 
### Citation

Please consider citing the following paper when using our code for your application.

```bibtex
@inproceedings{hgt,
  title={Heterogeneous Graph Transformer},
  author={Ziniu Hu and Yuxiao Dong and Kuansan Wang and Yizhou Sun},
  booktitle={Proceedings of the 2020 World Wide Web Conference},
  year={2020}
}
```
