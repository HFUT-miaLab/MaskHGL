# STL
import copy
import os
import sys
import argparse
import datetime
import shutil

# 3rd party library
from yacs.config import CfgNode
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from sklearn.metrics import f1_score

# local library
from dataset import HyperGLNDataset, get_train_valid_names, get_test_names
from util.util import merge_config_to_args, Logger, fix_random_seeds, BestModelSaver
import util.metric as metric
from models.HyperGLN import HyperGTv5_2wMHRCv3


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

            logits, _, _ = model(feat, hyperedge_index, hyperedge_attr, train=False)

            y_probs = F.softmax(logits, dim=1)
            y_pred = torch.argmax(logits, 1)

            y_result += target.cpu().tolist()
            pred_probs += y_probs.cpu().tolist()
            y_preds += y_pred.cpu().tolist()

            correct += (y_pred == target).sum().float()
            total += len(target)

    acc = (correct / total).cpu().data.numpy()

    if args.num_class == 2:
        y_result = np.array(y_result)
        y_probs = np.array(pred_probs)[:, 1]
        macro_auc_score = metric.macro_auc(y_result, y_score=y_probs, multi_class=args.num_class > 2)
        macro_F1 = f1_score(y_true=y_result, y_pred=y_preds, average='macro')
        metric.draw_binary_roc_curve(y_result, y_probs, save_path=os.path.join(args.fold_save_path,
                                                                               'MC_' + eval_metric + '.jpg'))
        metric.draw_confusion_matrix(y_result, y_preds,
                                     save_path=os.path.join(args.fold_save_path, 'ROC_Curve_' + eval_metric + '.jpg'),
                                     class_names=["Wild", "Mutant"])
    else:
        if args.num_class == 5:
            class_names = ['neg', 'L858R', '19del', 'Wild', 'Other']
        elif args.num_class == 4:
            class_names = ['neg', 'Mutant', 'Wild', 'Other']
        else:
            raise NotImplementedError
        macro_auc_score = metric.macro_auc(y_result, y_score=pred_probs, multi_class=args.num_class > 2)
        macro_F1 = f1_score(y_true=y_result, y_pred=y_preds, average='macro')
        metric.draw_muti_roc_curve(np.array(y_result), np.array(pred_probs),
                                   save_path=os.path.join(args.fold_save_path, 'MC_' + eval_metric + '.jpg'),
                                   class_names=class_names)
        metric.draw_confusion_matrix(y_result, y_preds,
                                     save_path=os.path.join(args.fold_save_path, 'ROC_Curve_' + eval_metric + '.jpg'),
                                     class_names=class_names)

    return acc, macro_auc_score, macro_F1


def valid(args, valid_loader, model):
    model.eval()

    correct = 0
    total = 0
    y_result = []
    pred_probs = []
    with torch.no_grad():
        for step, (feat, target, hyperedge_index, hyperedge_attr) in enumerate(valid_loader):
            if torch.cuda.is_available():
                feat = torch.squeeze(feat).cuda()
                target = target.cuda()
                hyperedge_index = torch.squeeze(hyperedge_index.long()).cuda()
                hyperedge_attr = torch.squeeze(hyperedge_attr).cuda()

            logits, _, _ = model(feat, hyperedge_index, hyperedge_attr, train=False)
            y_probs = torch.softmax(logits, dim=1)
            y_pred = torch.argmax(logits, dim=1)

            y_result += target.tolist()
            pred_probs += y_probs.tolist()

            correct += (y_pred == target).sum().float()
            total += len(target)

    acc = (correct / total).cpu().detach().data.numpy()
    if args.num_class == 2:
        y_result = np.array(y_result)
        y_probs = np.array(pred_probs)[:, 1]
        auc_score = metric.micro_auc(y_result, y_score=y_probs, multi_class=args.num_class > 2,
                                     n_classes=args.num_class)
    else:
        auc_score = metric.micro_auc(np.array(y_result), np.array(pred_probs), multi_class=args.num_class > 2,
                                     n_classes=args.num_class)

    return acc, auc_score


def train(args, model, train_loader, valid_loader):
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
                                                    epochs=args.max_epoch, pct_start=0.3)

    loss_fn = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loss_fn.cuda()

    args.current_epoch = 0
    best_model_saver = BestModelSaver(args.max_epoch, ratio=0.3)
    for epoch in range(args.max_epoch):
        lr = optimizer.param_groups[0]["lr"]
        args.current_lr = lr

        total_loss, total_bag_loss, total_instance_loss, total_mhrc_loss = 0, 0, 0, 0
        correct, total = 0, 0
        total_step = 0
        for step, (feat, target, hyperedge_index, hyperedge_attr) in enumerate(train_loader):
            optimizer.zero_grad()
            bag_loss, instance_loss, mhrc_loss = 0, 0, 0

            if torch.cuda.is_available():
                feat = torch.squeeze(feat).cuda()
                target = target.cuda()
                hyperedge_index = torch.squeeze(hyperedge_index.long()).cuda()
                hyperedge_attr = torch.squeeze(hyperedge_attr).cuda()

            bag_logits, instance_logits, mhrc_loss = model(feat, hyperedge_index, hyperedge_attr, train=True)
            bag_pred = torch.argmax(F.softmax(bag_logits, 1), 1)

            correct += int((bag_pred == target).sum().cpu())
            total = total + len(target)

            bag_loss_factor, instance_loss_factor, mhrc_loss_factor = args.loss_weights
            # Bag-level Loss
            bag_loss = loss_fn(bag_logits, target)
            total_bag_loss += bag_loss.item()
            # Instance-level Loss
            if instance_logits is not None:
                instance_pseudo_labels = torch.full((instance_logits.shape[0],), target.item()).cuda()
                instance_loss = instance_loss_factor * loss_fn(instance_logits, instance_pseudo_labels)
                total_instance_loss += instance_loss.item()
            # MHRC loss
            if mhrc_loss != 0:
                mhrc_loss = mhrc_loss_factor * mhrc_loss
                total_mhrc_loss += mhrc_loss.item()
            # Total Loss
            loss = bag_loss + instance_loss + mhrc_loss
            total_loss += loss

            loss.backward()
            optimizer.step()

            scheduler.step()
            total_step += 1
            if (step + 1) % args.show_interval == 0:
                print("\tEpoch: [{}/{}] || epochiter: [{}/{}] || LR: {:.6f} || Bag_Loss(Avg): {:.6f} ||"
                      " Instance_Loss(Avg): {:.6f} || MHRC_Loss(Avg): {:.6f} || Total_Loss(Avg): {:.6f}"
                      .format(args.current_epoch + 1, args.max_epoch, step + 1, len(train_loader),
                              args.current_lr, total_bag_loss / total_step, total_instance_loss / total_step,
                              total_mhrc_loss / total_step, total_loss / total_step))

        train_acc = correct / total

        valid_acc, valid_auc = valid(args, valid_loader, model)
        best_model_saver.update(valid_acc, valid_auc, args.current_epoch)
        print('\tValidation-Epoch: {} || train_acc: {:.6f} || train_bag_loss: {:.6f} ||'
              ' train_instance_loss: {:.6f} || train_mhrc_loss: {:.6f} || train_avg_loss: {:.6f} ||'
              ' valid_acc: {:.6f} || valid_auc: {:.6f}'
              .format(args.current_epoch + 1, train_acc, total_bag_loss / total_step, total_instance_loss / total_step,
                      total_mhrc_loss / total_step, total_loss / total_step, valid_acc, valid_auc))

        current_model_weight = copy.deepcopy(model.state_dict())
        torch.save(current_model_weight,
                   os.path.join(args.fold_save_path, 'epoch' + str(args.current_epoch) + '.pth'))

        args.current_epoch += 1

    shutil.copyfile(os.path.join(args.fold_save_path, 'epoch' + str(best_model_saver.best_valid_acc_epoch) + '.pth'),
                    os.path.join(args.fold_save_path, 'best_acc.pth'))
    shutil.copyfile(os.path.join(args.fold_save_path, 'epoch' + str(best_model_saver.best_valid_auc_epoch) + '.pth'),
                    os.path.join(args.fold_save_path, 'best_auc.pth'))
    return best_model_saver


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--cfg', type=str, default="./configs/HyperGLN_TCGA-EGFR.yaml")

    args = parser.parse_args()

    if args.cfg:
        cfg = CfgNode(new_allowed=True)
        cfg.merge_from_file(args.cfg)
        merge_config_to_args(args, cfg)

    # Save_dir init
    cfg_name = os.path.split(args.cfg)[-1]
    args.weights_save_path = os.path.join(args.weights_save_path, os.path.splitext(cfg_name)[0],
                                          datetime.datetime.now().strftime('%Y-%m-%d %H%M%S.%f'))
    os.makedirs(args.weights_save_path, exist_ok=True)
    shutil.copyfile(args.cfg, os.path.join(args.weights_save_path, cfg_name))

    sys.stdout = Logger(filename=os.path.join(args.weights_save_path,
                                              datetime.datetime.now().strftime('%Y-%m-%d %H%M%S.%f') + '.txt'))

    test_acc_5fold_acc_model = []
    test_macro_auc_5fold_acc_model = []
    test_macro_f1_5fold_acc_model = []
    # Fix_random_seeds
    fix_random_seeds()
    for fold in range(5):
        # Model init
        if args.arch == 'HyperGTv5.2wMHRCv3':
            model = HyperGTv5_2wMHRCv3(feat_dim=args.feat_dim, n_class=args.num_class, trans_dim=args.trans_dim,
                                       mask_p=args.mask_p, mask_ratio=args.mask_ratio, dropout=args.dropout)
            print("\tModel Arch: ", args.arch)
            print("\tMask_ratio: ", args.mask_ratio, "Mask_P: ", args.mask_p)
        else:
            raise NotImplementedError(args.arch)

        args.fold_save_path = os.path.join(args.weights_save_path, 'fold' + str(fold))
        os.makedirs(args.fold_save_path, exist_ok=True)

        print('Training Folder: {}.\n\tData Loading...'.format(fold))
        train_names, train_labels, valid_names, valid_labels, train_weights = get_train_valid_names(
            args.train_valid_csv,
            fold=fold)
        sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_weights))
        train_dataset = HyperGLNDataset(args.feature_root, args.graphs_root, train_names, train_labels)
        valid_dataset = HyperGLNDataset(args.feature_root, args.graphs_root, valid_names, valid_labels)

        train_loader = DataLoader(train_dataset, sampler=sampler, batch_size=args.batch_size, num_workers=args.workers,
                                  pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                  pin_memory=True)

        if torch.cuda.is_available():
            model = model.cuda()
        best_model_saver = train(args, model, train_loader, valid_loader)
        print('\t(Valid)Best ACC: {:.6f} || Best Micro_AUC: {:.6f}'
              .format(best_model_saver.best_valid_acc, best_model_saver.best_valid_auc))

        test_names, test_labels = get_test_names(args.test_csv, nclass=args.num_class)
        test_dataset = HyperGLNDataset(args.feature_root, args.graphs_root, test_names, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                 pin_memory=True)

        best_acc_model_weight = torch.load(os.path.join(args.fold_save_path, 'best_acc.pth'))
        model.load_state_dict(best_acc_model_weight)
        test_acc, test_macro_auc, test_macro_f1 = eval(args, test_loader, model, eval_metric='ACC')
        test_acc_5fold_acc_model.append(test_acc)
        test_macro_auc_5fold_acc_model.append(test_macro_auc)
        test_macro_f1_5fold_acc_model.append(test_macro_f1)
        print('\t(Test)Best ACC model || ACC: {:.6f} || Macro_AUC: {:.6f} || Macro_F1: {:.6f} '
              .format(test_acc, test_macro_auc, test_macro_f1))

    print("Five-Fold-Validation:")
    print("\tBest_ACC_Model: ACC: {:.2f}±{:.2f}, Macro_AUC: {:.2f}±{:.2}, Macro_F1: {:.2f}±{:.2}"
          .format(np.mean(test_acc_5fold_acc_model) * 100, np.std(test_acc_5fold_acc_model) * 100,
                  np.mean(test_macro_auc_5fold_acc_model) * 100, np.std(test_macro_auc_5fold_acc_model) * 100,
                  np.mean(test_macro_f1_5fold_acc_model) * 100, np.std(test_macro_f1_5fold_acc_model) * 100))
