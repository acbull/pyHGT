## Usage

This experiment is based on stanford OGB (1.2.1) benchmark. The description of ogbn-mag is [avaiable here](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag). The steps are:

  1. run ```python preprocess_ogbn_mag.py``` to turn the dataset into our own data structure. As the MAG dataset only have input attributes (features) for paper nodes, for all the other types of nodes (author, affiliation, topic), we simply take the average of their connected paper nodes as their input features.

  2. train the model by ```python train_ogbn_mag.py --data_dir PATH_OF_DATASET --model_dir PATH_OF_SAVED_MODEL --n_layers 4 --prev_norm  --last_norm  --use_RTE```. Remember to specify your own data and model path.

  3. evaluate the model by ```python eval_ogbn_mag.py --data_dir PATH_OF_DATASET --model_dir PATH_OF_SAVED_MODEL --task_type ensemble```. We use mini-batch sampling to get node representation and prediction. Based on it, we provide two evaluation type: 
    - 'sequential': Run the sampling for each batch of test nodes only once, and get one set of prediction results.
    - 'variance_reduce':   Run the sampling for each batch of test nodes multiple times, and get the average prediction score for them as prediction results.

The **pre-trained model** is [avaiable here](https://drive.google.com/file/d/1867u-kG_3HjyWg7AeU2XaGH-qRt8GZoN/view?usp=sharing). Detailed hyperparameter is:


```
  --conv_name                      STR     Name of GNN filter (model)                           hgt
  --n_hid                          INT     Number of hidden dimension                           512
  --n_heads                        INT     Number of attention head                             8
  --n_layers                       INT     Number of GNN layers                                 4
  --prev_norm                      BOOL    Whether to use layer-norm on previous layers.        True
  --last_norm                      BOOL    Whether to use layer-norm on the last layer.         True
  --use_RTE                        BOOL    Whether to use RTE                                   True 
```

Reference performance numbers for the ACM dataset:

| Model        | Accuracy (VR)   | Accuracy (Seq) | # Parameter     | Hardware         |
| ---------    | --------------- | -------------- | --------------  |--------------    |
| 4-layer HGT  | 0.5007          | 0.4940         | 21,173,389      | Tesla K80 (12GB) |
