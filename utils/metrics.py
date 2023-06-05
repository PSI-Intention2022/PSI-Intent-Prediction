from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, accuracy_score, f1_score
import numpy as np
from scipy.special import softmax
from scipy.special import expit as sigmoid
import torch.nn.functional as F

def evaluate_intent(target, target_prob, prediction, args):
    '''
    Here we only predict one 'intention' for one track (15 frame observation). (not a sequence as before)
    :param target: (bs x 1), hard label; target_prob: soft probability, 0-1, agreement mean([0, 0.5, 1]).
    :param prediction: (bs), sigmoid probability, 1-dim, should use 0.5 as threshold
    :return:
    '''
    print("Evaluating Intent ...")
    results = {
        'MSE': 0,
        'Acc': 0,
        'F1': 0,
        'mAcc': 0,
        'ConfusionMatrix': [[]],
    }

    bs = target.shape[0]
    # lbl_target = np.argmax(target, axis=-1) # bs x ts
    lbl_target = target # bs
    lbl_taeget_prob = target_prob
    lbl_pred = np.round(prediction) # bs, use 0.5 as threshold

    MSE = np.mean(np.square(lbl_taeget_prob - prediction))
    # hard label evaluation - acc, f1
    Acc = accuracy_score(lbl_target, lbl_pred) # calculate acc for all samples
    F1 = f1_score(lbl_target, lbl_pred, average='macro')

    intent_matrix = confusion_matrix(lbl_target, lbl_pred)  # [2 x 2]
    intent_cls_acc = np.array(intent_matrix.diagonal() / intent_matrix.sum(axis=-1)) # 2
    intent_cls_mean_acc = intent_cls_acc.mean(axis=0)

    results['MSE'] = MSE
    results['Acc'] = Acc
    results['F1'] = F1
    results['mAcc'] = intent_cls_mean_acc
    results['ConfusionMatrix'] = intent_matrix

    return results

def shannon(data):
    shannon = -np.sum(data*np.log2(data))
    return shannon
