import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, auc, roc_curve, roc_auc_score
import matplotlib.pyplot as plt


def labels_2_one_hot(y, class_num=5):
    one_hot_labels = []
    for i in range(len(y)):
        one_hot = np.zeros(class_num)
        one_hot[y[i]] = 1
        one_hot_labels.append(one_hot.tolist())
    return np.array(one_hot_labels)


def compute_TP_FP_TN_FN(y, pred_labels):
    assert len(y) == len(pred_labels)

    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(y)):
        if y[i] == 1:
            if pred_labels[i] == 1:
                TP += 1
            else:
                FN += 1
        else:
            if pred_labels[i] == 0:
                TN += 1
            else:
                FP += 1
    return TP, FP, TN, FN


def compute_specificity(TN, FP):
    return float(TN) / (TN + FP)


def compute_sensitivity(TP, FN):
    return float(TP) / (TP + FN)


def draw_confusion_matrix(y, pred_labels, class_names, save_path=None):
    confusion = confusion_matrix(y, pred_labels)
    plt.imshow(confusion, cmap=plt.cm.Blues)
    indices = range(len(confusion))

    plt.xticks(indices, class_names)
    plt.yticks(indices, class_names)
    plt.xlabel('Pred')
    plt.ylabel('Truth')
    for pred_index in range(len(confusion)):
        for truth_index in range(len(confusion[pred_index])):
            plt.text(truth_index, pred_index, "{:.2f}%".format(
                confusion[pred_index][truth_index] / np.sum(confusion[pred_index]) * 100),
                     ha="center", va="center")
    plt.colorbar()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.close()


def draw_binary_roc_curve(y, prob, save_path=None):
    """Draw binary-class roc curve.

    Parameters
    ----------
    y : ground truth labels list.
    prob : pred probabilities list.

    Examples
    --------

        txt = "D:\\WorkGroup\\st\\EGFR\\results\\resnet50_best_epoch41.txt"
        result = PN_parse_prediction(txt)
        draw_roc_curve(result['true_labels'], result['pred_probs'])
    """
    fpr, tpr, threshold = roc_curve(y, prob)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值

    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

    plt.close()
    return roc_auc


def draw_muti_roc_curve(true_labels, pred_probs, class_names=None, save_path=None):
    true_labels = labels_2_one_hot(true_labels, class_num=len(class_names))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(true_labels[0])):
        fpr[i], tpr[i], _ = roc_curve(true_labels[:, i], pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false  positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(class_names)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= len(class_names)
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr["macro"], tpr["macro"], color='darkorange', lw=lw, label='ROC curve (area = %0.3f)' % roc_auc["macro"])  ###假正率为横坐标，真正率为纵坐标做曲线

    colors = ['aqua', 'green', 'cornflowerblue', 'orangered', 'pink']
    classes = class_names
    for i, color in zip(range(len(class_names)), colors[:len(class_names)]):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw, linestyle=':',
                 label='ROC curve of class {0}, (area = {1:.3f})'.format(classes[i], roc_auc[i]))
        # plt.rcParams.update({"font.size": 16})

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('Receiver operating characteristic example', fontsize=16)
    plt.legend(loc="lower right")

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

    plt.close()
    return roc_auc["macro"]


def macro_auc(y_true, y_score, multi_class=False, n_classes=None):
    if multi_class:
        return roc_auc_score(y_true, y_score, average="macro", multi_class='ovo')
    else:
        return roc_auc_score(y_true, y_score, average="macro")


def micro_auc(y_true, y_score, multi_class=False, n_classes=None):
    if multi_class:
        return roc_auc_score(y_true, y_score, average="micro", multi_class='ovr')
    else:
        return roc_auc_score(y_true, y_score, average="micro")
