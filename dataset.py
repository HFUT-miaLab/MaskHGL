import os
import csv

import numpy as np
import torch
from torch.utils.data import Dataset
import joblib


# Note: dataset specialized function, you might need to write you own function for different dataset.
def get_train_valid_names(train_valid_csv, fold=0, nclass=2):
    train_names = []
    train_labels = []
    valid_names = []
    valid_labels = []

    with open(train_valid_csv, 'r') as csv_f:
        for row in csv.reader(csv_f):
            name = row[0].split('.')[0] + row[0].split('.')[1]
            cls_index = int(row[2])
            if int(row[3]) != fold:
                train_names.append(name)
                train_labels.append(cls_index)
            else:
                valid_names.append(name)
                valid_labels.append(cls_index)

    _, cls_count = np.unique(np.array(train_labels), return_counts=True)
    cls_weights = np.sum(cls_count) / cls_count
    train_weights = [cls_weights[label] for label in train_labels]

    return train_names, train_labels, valid_names, valid_labels, train_weights


# Note: dataset specialized function, you might need to write you own function for different dataset.
def get_test_names(test_csv, nclass=2):
    test_names = []
    test_labels = []

    with open(test_csv, 'r') as csv_f:
        for row in csv.reader(csv_f):
            name = row[0].split('.')[0] + row[0].split('.')[1]
            cls_index = int(row[2])
            test_names.append(name)
            test_labels.append(cls_index)

    return test_names, test_labels


class HyperGLNDataset(Dataset):
    def __init__(self, feat_root, graph_root, names, labels, return_coord=False):
        self.feat_paths = [os.path.join(feat_root, name + '_features.pth') for name in names]
        self.coord_paths = [os.path.join(feat_root, name + '_coordinates.pth') for name in names]
        self.graph_paths = [os.path.join(graph_root, name + '_graph.pkl') for name in names]
        self.labels = labels
        self.return_coord = return_coord
        assert len(self.feat_paths) == len(self.graph_paths) == len(self.labels)

    def __getitem__(self, index):
        feat = torch.load(self.feat_paths[index])
        label = self.labels[index]
        graph = joblib.load(self.graph_paths[index])
        hyperedge_index = graph['hyperedge_index']
        hyperedge_attr = graph['hyperedge_attr']

        if self.return_coord:
            coord = torch.load(self.coord_paths[index])
            return feat, coord, label, torch.tensor(hyperedge_index, dtype=torch.int64), torch.tensor(hyperedge_attr, dtype=torch.float32)
        else:
            return feat, label, torch.tensor(hyperedge_index, dtype=torch.int64), torch.tensor(hyperedge_attr, dtype=torch.float32)

    def __len__(self):
        return len(self.feat_paths)
