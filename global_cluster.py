# STL
import os
os.environ["OMP_NUM_THREADS"] = '1'
import random
import time
import csv

# 3rd party library
from tqdm import tqdm
import numpy as np
import joblib
import torch
import argparse

from util.kmeans import KMeans
from util.util import fix_random_seeds


def get_train(features_dataset_root, csv_path):
    train_data_paths = []
    with open(csv_path, 'r') as csv_f:
        train_data = [row[0].split('.')[0] + row[0].split('.')[1] for row in csv.reader(csv_f)]

    for name in train_data:
        if os.path.exists(os.path.join(features_dataset_root, name + '_features.pth')):
            train_data_paths.append(os.path.join(features_dataset_root, name + '_features.pth'))

    return train_data_paths


def global_cluster(feature_paths, save_path, num_global_cluster=100, num_samples=-1):
    print("Data loading...")
    tensor_list = []
    for idx, path in enumerate(tqdm(feature_paths)):
        slide_feats = torch.load(path)
        if num_samples == -1 or slide_feats.shape[0] <= num_samples:
            tensor_list.append(slide_feats)
        else:
            sampled_tensor = torch.from_numpy(np.array(random.sample(slide_feats.numpy().tolist(), num_samples)))
            tensor_list.append(sampled_tensor)
    data = torch.cat(tensor_list, dim=0)
    print("Data loading is Done: cluster_data.shape: ", data.shape)

    cluster_begin_timestamp = time.time()
    clusterer = KMeans(n_clusters=num_global_cluster)
    clusterer.fit(data.cuda())
    print("KMeans Clustering Cost Time: ", time.time() - cluster_begin_timestamp)

    save_file = "Global_Cluster_({}, {})_{}.pkl".format(data.shape[0], data.shape[1], num_global_cluster)
    save_path = os.path.join(save_path, save_file)
    joblib.dump(clusterer, save_path)
    print("Global Cluster Save to: ", save_path)


if __name__ == '__main__':
    fix_random_seeds()

    parser = argparse.ArgumentParser("Perform Kmeans Cluster in Embedding Space. (Require GPU for Speedup)")
    parser.add_argument("--K", type=int, default=100, help='global cluster number for global cluster in graph construction')
    parser.add_argument("--sample_per_slide", type=int, default=500, help='random patch sampling for speedup')
    parser.add_argument("--features_root", type=str, default='./data/TCGA-EGFR/features')
    parser.add_argument("--csv_path", type=str, default='./data/TCGA-EGFR/TCGA-EGFR_trainval_5fold.csv')
    parser.add_argument("--save_path", type=str, default='./')
    args = parser.parse_args()

    train_feature_paths = get_train(args.features_root, args.csv_path)
    global_cluster(feature_paths=train_feature_paths, num_global_cluster=args.K,
                   save_path=args.save_path, num_samples=args.sample_per_slide)
    print("Global Cluster: Done!")
