# STL
import os
import argparse

# 3rd party library
from yacs.config import CfgNode
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score

# local library
from dataset import HyperGATDataset, get_test_names
from util.util import merge_config_to_args_MHRC, fix_random_seeds
from models.HyperGLN import HyperGTv5_2wMHRCv3


def _macro_auc(y_true, y_score, multi_class=False, n_classes=None):
    if multi_class:
        return roc_auc_score(y_true, y_score, average="macro", multi_class='ovo')
    else:
        return roc_auc_score(y_true, y_score, average="macro")


def _micro_auc(y_true, y_score, multi_class=False, n_classes=None):
    if multi_class:
        return roc_auc_score(y_true, y_score, average="micro", multi_class='ovr')
    else:
        return roc_auc_score(y_true, y_score, average="micro")


def eval(args, test_loader, model, eval_metric='ACC'):
    model.eval()

    correct = 0
    total = 0

    y_result = []
    pred_probs = []
    y_preds = []
    with torch.no_grad():
        for step, (feat, target, hyperedge_index, hyperedge_attr) in enumerate(test_loader):
            if torch.cuda.is_available():
                feat = torch.squeeze(feat).cuda()
                target = target.cuda()
                hyperedge_index = torch.squeeze(hyperedge_index.long()).cuda()
                hyperedge_attr = torch.squeeze(hyperedge_attr).cuda()

            logits, _ = model(feat, hyperedge_index, hyperedge_attr, train=False)

            y_probs = F.softmax(logits, dim=1)
            y_pred = torch.argmax(logits, 1)

            y_result += target.cpu().tolist()
            pred_probs += y_probs.cpu().tolist()
            y_preds += y_pred.cpu().tolist()

            correct += (y_pred == target).sum().float()
            total += len(target)

    acc = (correct / total).cpu().data.numpy()
    micro_F1 = f1_score(y_true=y_result, y_pred=y_preds, average='micro')
    macro_F1 = f1_score(y_true=y_result, y_pred=y_preds, average='macro')
    if args.num_class == 2:
        class_names = ['neg', 'pos']
        y_result = np.array(y_result)
        y_probs = np.array(pred_probs)[:, 1]
        micro_auc_score = _micro_auc(y_result, y_score=y_probs, multi_class=args.num_class > 2)
        macro_auc_score = _macro_auc(y_result, y_score=y_probs, multi_class=args.num_class > 2)
    else:
        if args.num_class == 4:
            class_names = ['neg', 'mutant', 'wild', 'other']
        elif args.num_class == 5:
            class_names = ['neg', '19del', 'L858R', 'wild', 'other']
        else:
            raise NotImplementedError
        micro_auc_score = _micro_auc(y_result, y_score=pred_probs, multi_class=args.num_class > 2)
        macro_auc_score = _macro_auc(y_result, y_score=pred_probs, multi_class=args.num_class > 2)

    return acc, micro_auc_score, macro_auc_score, micro_F1, macro_F1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--cfg', type=str,
                        default="./save/HyperGTv5.2wMHRCv3_5class_nk16_gc150_slidegraph/2023-06-17 233135.414322/HyperGTv5.2wMHRCv3_5class_nk16_gc150_slidegraph.yaml")
    parser.add_argument('--features_root', type=str,
                        default='D:\WorkGroup\st\Project_MICCAI_1\Patch_Features\EGFR_DINO_ViT-S-16')
    parser.add_argument('--graphs_root', type=str, default='./graphs/EGFR_hyperedge_index_nk16_150_slidegraph+')
    parser.add_argument('--weights_save_path', type=str,
                        default="./save/HyperGTv5.2wMHRCv3_5class_nk16_gc150_slidegraph/2023-06-17 233135.414322")
    parser.add_argument('--test_csv', type=str,
                        default=r"D:\\WorkGroup\\st\\dataset\\EGFR_slideId_alias_label_test_clean.csv")
    parser.add_argument('--num_class', type=int, default=5)
    parser.add_argument('--feat_dim', type=int, default=384)

    args = parser.parse_args()
    if args.cfg:
        cfg = CfgNode(new_allowed=True)
        cfg.merge_from_file(args.cfg)
        merge_config_to_args_MHRC(args, cfg)

    test_acc_5fold_acc_model = []
    test_micro_auc_5fold_acc_model = []
    test_macro_auc_5fold_acc_model = []
    test_micro_f1_5fold_acc_model = []
    test_macro_f1_5fold_acc_model = []
    # Fix_random_seeds
    fix_random_seeds()
    for fold in range(5):
        # Model init
        if args.arch == 'HyperGTv5.2wMHRCv3':
            model = HyperGTv5_2wMHRCv3(feat_dim=args.feat_dim, n_class=args.num_class, trans_dim=args.trans_dim,
                                       mask_ratio=args.mask_ratio, dropout=args.dropout)
            print("Model Arch: ", args.arch)
        else:
            raise NotImplementedError(args.arch)

        args.fold_save_path = os.path.join(args.weights_save_path, 'fold' + str(fold))
        os.makedirs(args.fold_save_path, exist_ok=True)

        print('Testing Folder: {}.\n\tData Loading...'.format(fold))
        if torch.cuda.is_available():
            model = model.cuda()

        test_names, test_labels = get_test_names(args.test_csv, nclass=args.num_class)
        test_dataset = HyperGATDataset(args.feature_root, args.graphs_root, test_names, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                 pin_memory=True)

        best_acc_model_weight = torch.load(os.path.join(args.fold_save_path, 'best_acc.pth'))
        model.load_state_dict(best_acc_model_weight)
        test_acc, test_micro_auc, test_macro_auc, test_micro_f1, test_macro_f1 = eval(args, test_loader, model,
                                                                                      eval_metric='ACC')
        test_acc_5fold_acc_model.append(test_acc)
        test_micro_auc_5fold_acc_model.append(test_micro_auc)
        test_macro_auc_5fold_acc_model.append(test_macro_auc)
        test_micro_f1_5fold_acc_model.append(test_micro_f1)
        test_macro_f1_5fold_acc_model.append(test_macro_f1)
        print(
            '\t(Test)Best ACC model || ACC: {:.6f} || Micro_AUC: {:.6f} || Macro_AUC: {:.6f} || Micro_F1: {:.6f} || Macro_F1: {:.6f}'
            .format(test_acc, test_micro_auc, test_macro_auc, test_micro_f1, test_macro_f1))

    print("Five-Fold-Validation:")
    print(
        "\tBest_ACC_Model: ACC: {:.2f}±{:.2f}, Micro_AUC: {:.2f}±{:.2}, Macro_AUC: {:.2f}±{:.2}, Micro_F1: {:.2f}±{:.2}, Macro_F1: {:.2f}±{:.2}"
        .format(np.mean(test_acc_5fold_acc_model) * 100, np.std(test_acc_5fold_acc_model) * 100,
                np.mean(test_micro_auc_5fold_acc_model) * 100, np.std(test_micro_auc_5fold_acc_model) * 100,
                np.mean(test_macro_auc_5fold_acc_model) * 100, np.std(test_macro_auc_5fold_acc_model) * 100,
                np.mean(test_micro_f1_5fold_acc_model) * 100, np.std(test_micro_f1_5fold_acc_model) * 100,
                np.mean(test_macro_f1_5fold_acc_model) * 100, np.std(test_macro_f1_5fold_acc_model) * 100))
