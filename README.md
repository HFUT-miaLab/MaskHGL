This repository provides the Pytorch implementations for "HyperGraph Learning Network for Weakly Supervised Histopathology Whole Slide Image Classification"

# Download the WSIs

We provide the slide list and dataset partition used for evaluating our methods in ./data.

The WSIs can be found in the TCGA project:

https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga

# Patch Extraction

We use self-supervised learning framework DINO for pretraining a ViT-S as our Patch Encoder. You can follow the offical 
repository [here](https://github.com/facebookresearch/dino) for patch extraction yourselves. Besides, we will share the features extracted soon.

# Graph Construction

After the patch extraction is finished, users need to conduct a global cluster for the preparation of graph construction by run the following command.

```
python global_cluster.py --K <global cluster number> --sample_per_slide <random patch sampling> --features_root <the root path for store features> --csv_path <trainval data list path> --save_path <path to save output>
```

Then you should gain a file named 'Global_Cluster_(XX, XX)_K.pkl' in your save path.

After that, run the following command for graph construction. This step may take a longer time.

```
python graph_construction.py --K <global cluster number> --features_root <the root path for store features> --coordinates_root <the root path for store features> --global_clusterer_path <path to previous global clusterer> --K <global cluster number> --h <distance threshold for Spatial Hypergraph construction> --save_path <path to save graphs> --num_worker <num_worker>
```

# Training HyperGLN Model

The configurations yaml files for each benchmarking dataset is provided in ./configs. You may first modify the respective config files for hyper-parameter settings, and run the following command for training.

```
python train.py --cfg <path to your configration file>
```

Evaluation is performed after every epoch on validation sets and testing sets.

The log file and checkpoints will be saved in WEIGHTS_SAVE_PATH of configration file.
