## Usage

This experiment is based on stanford OGB (1.2.1) benchmark. The description of ogbn-products is [avaiable here](https://ogb.stanford.edu/docs/nodeprop/#ogbn-products). The steps are:

 
 1. run ```python preprocess_ogbn_products.py``` to turn the dataset into our own data structure. As the MAG dataset only have input attributes (features) for paper nodes, for all the other types of nodes (author, affiliation, topic), we simply take the average of their connected paper nodes as their input features.
 
    **Note**: 
       - Original products graph is not heterogeneous. To leverage the label information of training and valid data (as observed nodes) as soft label propagation, [we add the discrete labels as a specific type of nodes called "cate"](https://github.com/acbull/pyHGT/blob/776c4e61c55a859e7ba4322d8bf8c58fedb51079/ogbn-products/preprocess_ogbn_products.py#L54) (as they don't have attributes, the input features are replaced with learnable embedding). Each product node is linked to their corresponding "cate" node (label).
       - *To avoid information leakage*, [we didn't add back these nodes for test data](https://github.com/acbull/pyHGT/blob/776c4e61c55a859e7ba4322d8bf8c58fedb51079/ogbn-products/preprocess_ogbn_products.py#L55). Within each mini-batch training, we'll [delete the edges from these "cate" nodes to output nodes](https://github.com/acbull/pyHGT/blob/c22373575388a0d9250844087ed0f3bc973dcc85/ogbn-products/train_ogbn_products.py#L104). This trick significantly reduce the generarzation gap between the training and test nodes.


  2. train the model by ```python train_ogbn_products.py --data_dir PATH_OF_DATASET --model_dir PATH_OF_SAVED_MODEL --n_layers 4 --prev_norm  --last_norm```. Remember to specify your own data and model path.

  3. evaluate the model by ```python eval_ogbn_products.py --data_dir PATH_OF_DATASET --model_dir PATH_OF_SAVED_MODEL --task_type sequential```. 

The **pre-trained model** is [avaiable here](https://drive.google.com/file/d/1K5blZmIVOBDZk40_CJcnNgRp1TJ__1ka/view?usp=sharing). Detailed hyperparameter is:


```
  --conv_name                      STR     Name of GNN filter (model)                           hgt
  --n_hid                          INT     Number of hidden dimension                           256
  --n_heads                        INT     Number of attention head                             4
  --n_layers                       INT     Number of GNN layers                                 3
  --prev_norm                      BOOL    Whether to use layer-norm on previous layers.        True
  --last_norm                      BOOL    Whether to use layer-norm on the last layer.         True
  --use_RTE                        BOOL    Whether to use RTE                                   False 
```

Reference performance numbers for the ogbn-products dataset:

| Model        | Accuracy (Seq)   | # Parameter     | Hardware         |
| ---------    | ---------------  | --------------  |--------------    |
| 3-layer HGT  | 0.8563           | 2,025,573       | Tesla K80 (12GB) |
