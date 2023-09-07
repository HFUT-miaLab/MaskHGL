# STL
import os
os.environ['OPENBLAS_NUM_THREADS'] = '8'

import argparse
from concurrent.futures import ThreadPoolExecutor
import time

import warnings
warnings.filterwarnings('ignore')

# 3rd party library
import numpy as np
import torch
import joblib
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import KDTree as sKDTree
from scipy.cluster.hierarchy import fcluster
from scipy.cluster import hierarchy


def euclidean_distances(x):
    return squareform(pdist(x, metric='euclidean'))


def feat_adjacency_matrix_knn(feat, clusterer, n_clusters, k=1, is_prob=False, m_prob=1):
    """
    construct feature space's hypregraph incidence matrix from cluster
    :param feat: slide's patch features. N_patch * D
    :param k: KNN cluster center
    :param clusterer: global clusterer
    :param is_prob: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_patch * N_hyperedge(N_clusters)
    """
    num_patch = feat.shape[0]
    num_hyperedge = n_clusters

    H = np.zeros((num_patch, num_hyperedge))

    cluster_centers = clusterer.cluster_centers_.cpu()
    for feat_index in range(num_patch):
        feat_mat = np.vstack((feat[feat_index], cluster_centers))
        dis_vec = euclidean_distances(feat_mat)[0][1:]

        k_neig_indexs = np.argsort(dis_vec)[:k]

        avg_dis = np.average(dis_vec)
        for neigh_index in k_neig_indexs:
            if is_prob:
                H[feat_index, neigh_index] = np.exp(-dis_vec[neigh_index] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[feat_index, neigh_index] = 1.0
    return H


def spatial_adjacency_matrix_hierarchy(feat, coord, lamda_h, lambda_d=3e-1, lambda_f=1e-1):
    TC = sKDTree(coord)
    # I(index): N * N, D(distance): N * N.
    I, D = TC.query_radius(coord, r=6 / lambda_d, return_distance=True, sort_results=True)
    DX = np.zeros(int(coord.shape[0] * (coord.shape[0] - 1) / 2))
    idx = 0

    for i in range(coord.shape[0] - 1):
        f = np.exp(-lambda_f * np.linalg.norm(feat[i] - feat[I[i]], axis=1))  # 特征相似度
        d = np.exp(-lambda_d * D[i])
        df = 1 - f * d
        dfi = np.ones(coord.shape[0])
        dfi[I[i]] = df
        dfi = dfi[i + 1:]
        DX[idx:idx + len(dfi)] = dfi
        idx = idx + len(dfi)
    d = DX

    Z = hierarchy.linkage(d, method='average')
    clusters = fcluster(Z, lamda_h, criterion='distance')
    clusters_set = list(set(clusters))
    H = np.zeros((coord.shape[0], len(clusters_set)))

    for cluster_index, c in enumerate(clusters_set):
        coord_index = np.where(clusters == c)
        H[coord_index, cluster_index] = 1
    return H


def hyperedge_index_concat(hyperedge_index_feat, hyperedge_index_spatial, num_feat_hyperedge):
    """
    concate two type of hyperedge_index. e.g. feature-level hyperedge and spatial-level hyperedge.
    :return: concated_hyperedge_index.
    :return: split_index: indicate where to concat.
    """
    hyperedge_index_spatial[1][:] += num_feat_hyperedge
    concated_hyperedge_index = np.hstack((hyperedge_index_feat, hyperedge_index_spatial))
    return concated_hyperedge_index


def convert_H2index(H):
    """
    convert Adjacency matrix(H) to Sparse Hyperedge_index.
    :param H: Adjacency matrix, shape like (N, E).
    :return: Sparse Hyperedge_index, shape like (2, ?).
    """
    node_list = []
    edge_list = []

    H = np.array(H, dtype=np.float32)  # N * E
    for edge_index in range(H.shape[1]):
        for node_index in range(H.shape[0]):
            if H[node_index][edge_index] != 0:
                node_list.append(node_index)
                edge_list.append(edge_index)
    hyperedge_index = np.vstack((np.array([node_list]), np.array([edge_list])))

    return hyperedge_index


def construct_edge_index_for_mutiprocess(feature_path, coord_path, graph_save_path, feat_global_clusterer, args):
    """
    construct hyperedge_index.
    """
    # Embedded Hypergraph
    feat = torch.load(feature_path).numpy()
    H_feat = feat_adjacency_matrix_knn(feat, feat_global_clusterer, args.K)
    hyperedge_index_feat = convert_H2index(H_feat)
    hyperedge_index_feat_sorted = hyperedge_index_feat[:, hyperedge_index_feat[0].argsort()]

    # Spatial Hypergraph
    coord = torch.load(coord_path).numpy()
    H_coord = spatial_adjacency_matrix_hierarchy(feat, coord, lamda_h=args.h)
    hyperedge_index_spatial = convert_H2index(H_coord)
    hyperedge_index_spatial_sorted = hyperedge_index_spatial[:, hyperedge_index_spatial[0].argsort()]

    # Hypergraph Stack
    hyperedge_index_concated = hyperedge_index_concat(hyperedge_index_feat_sorted, hyperedge_index_spatial_sorted, args.K)

    # Hyperedge Attribution Generation
    hyperedge_attr = generate_hyperedge_attr(feat, hyperedge_index_concated)
    joblib.dump(value={'hyperedge_index': hyperedge_index_concated, 'hyperedge_attr': hyperedge_attr},
                filename=graph_save_path)

    print(graph_save_path, "is Done! Total Cost:", time.time() - args.begin_timestamp)


def generate_hyperedge_attr(x, hyperedge_index):
    feature_dim = x.shape[1]
    num_hyperedge = np.max(hyperedge_index[1]) + 1

    # Numpy Version
    hyperedge_attr = np.zeros((num_hyperedge, feature_dim))
    for edge_idx in range(num_hyperedge):
        _indexs = hyperedge_index[0][np.argwhere(hyperedge_index[1] == edge_idx).reshape(-1)]

        if _indexs.size == 0:
            continue
        else:
            hyperedge_attr[edge_idx, :] = np.mean(x[_indexs], axis=0)

    return hyperedge_attr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hypergraph Construction")
    parser.add_argument('--features_root', type=str, default='./data/TCGA-EGFR/features')
    parser.add_argument('--coordinates_root', type=str, default='./data/TCGA-EGFR/features')
    parser.add_argument('--global_clusterer_path', type=str, default='./Global_Cluster_(243901, 384)_100.pkl')
    parser.add_argument('--K', type=int, default=100, help='Global Cluster Number K, Number of Embedded Hypergraph')
    parser.add_argument('--h', type=float, default=0.8, help='Distance Threshold for Spatial Hypergraph Construction h_d')
    parser.add_argument('--save_path', type=str, default='./graphs/TCGA-EGFR')
    parser.add_argument('--num_worker', type=int, default=12, help='if 0, multi-threading mode will turn off')
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    global_clusterer = joblib.load(args.global_clusterer_path)

    if args.num_worker != 0:
        thread_executor = ThreadPoolExecutor(max_workers=args.num_worker)
    args.begin_timestamp = time.time()
    print("Perform Graph Construction...")
    for path in os.listdir(args.features_root):
        file_name = path.split(os.sep)[-1]
        if file_name.find("features") != -1:
            slide_name = file_name.split('_')[0]

            feature_path = os.path.join(args.features_root, path)
            coord_path = os.path.join(args.coordinates_root, slide_name + '_coordinates.pth')

            graph_save_path = os.path.join(args.save_path, slide_name + '_graph.pkl')
            if not os.path.exists(graph_save_path):
                if args.num_worker == 0:
                    construct_edge_index_for_mutiprocess(feature_path, coord_path, graph_save_path, global_clusterer, args)
                else:
                    thread_executor.submit(construct_edge_index_for_mutiprocess, feature_path, coord_path, graph_save_path,
                                           global_clusterer, args)
            else:
                print(slide_name, "Exist...")
