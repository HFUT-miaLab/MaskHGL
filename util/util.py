import sys
import datetime
import random

import torch
import numpy as np


class Logger(object):
    def __init__(self, filename='./logs/' + datetime.datetime.now().strftime('%Y-%m-%d %H%M%S.%f') + '.txt', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class BestModelSaver:
    def __init__(self, max_epoch, ratio=0.3):
        self.best_valid_acc = 0
        self.best_valid_auc = 0
        self.best_valid_acc_epoch = 0
        self.best_valid_auc_epoch = 0

        # Only consider selecting the best model after training beyond this round (begin_epoch)
        self.begin_epoch = int(max_epoch * ratio)

    def update(self, valid_acc, valid_auc, current_epoch):
        if current_epoch < self.begin_epoch:
            return

        if valid_acc >= self.best_valid_acc:
            self.best_valid_acc = valid_acc
            self.best_valid_acc_epoch = current_epoch
        if valid_auc >= self.best_valid_auc:
            self.best_valid_auc = valid_auc
            self.best_valid_auc_epoch = current_epoch


def fix_random_seeds(seed=None):
    """
    Fix random seeds.
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    print("Fix Random Seeds:", seed)


def merge_config_to_args(args, cfg):
    # Data
    args.feature_root = cfg.DATA.FEATURE_ROOT
    args.graphs_root = cfg.DATA.GRAPHS_ROOT
    args.train_valid_csv = cfg.DATA.TRAIN_VALID_CSV
    args.test_csv = cfg.DATA.TEST_CSV

    # Model
    args.arch = cfg.MODEL.ARCH
    args.feat_dim = cfg.MODEL.FEATURE_DIM
    args.num_class = cfg.MODEL.NUM_CLASS
    args.trans_dim = cfg.MODEL.TRANS_DIM
    args.mask_ratio = cfg.MODEL.MASK_RATIO
    args.mask_p = cfg.MODEL.MASK_P
    args.dropout = cfg.MODEL.DROPOUT
    args.loss_weights = cfg.MODEL.LOSS_WEIGHTS

    # TRAIN
    args.batch_size = cfg.TRAIN.BATCH_SIZE
    args.workers = cfg.TRAIN.WORKERS
    args.lr = cfg.TRAIN.LR
    args.weight_decay = cfg.TRAIN.WEIGHT_DECAY
    args.max_epoch = cfg.TRAIN.MAX_EPOCH
    args.show_interval = cfg.TRAIN.SHOW_INTERVAL
    args.eval = cfg.TRAIN.EVAL
    args.weights_save_path = cfg.TRAIN.WEIGHTS_SAVE_PATH
