## Usage

This experiment is based on stanford OGB (1.2.1) benchmark. The description of ogbn-products is [avaiable here](https://ogb.stanford.edu/docs/nodeprop/#ogbn-products). The steps are:

 
 1. run ```python preprocess_ogbn_products.py``` to turn the dataset into our own data structure. As the MAG dataset only have input attributes (features) for paper nodes, for all the other types of nodes (author, affiliation, topic), we simply take the average of their connected paper nodes as their input features.

  2. train the model by ```python train_ogbn_products.py --data_dir PATH_OF_DATASET --model_dir PATH_OF_SAVED_MODEL --n_layers 4 --prev_norm  --last_norm```. Remember to specify your own data and model path.

  3. evaluate the model by ```python eval_ogbn_products.py --data_dir PATH_OF_DATASET --model_dir PATH_OF_SAVED_MODEL --task_type sequential```. We use mini-batch sampling to get node representation and prediction. Based on it, we provide two evaluation type: 
