import torch
import torch.nn as nn
import numpy as np
from tqdm import *
from face.utils.utils import AverageMeter
from torch.autograd import Variable
import torch.nn.functional as F
import math
from sklearn.metrics import roc_auc_score

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
def ACE_TDR_Cal(label_list, prob_list, rate=0.01):
    ace_list = []
    tdr_list = [0,]
    label_list = (1 - np.array(label_list)).tolist()
    prob_list = (1 - np.array(prob_list)).tolist()
    total = len(label_list)
    for i in range(len(prob_list)):
        TP = TN = FP = FN = 0
        for j in range(len(prob_list)):
            if prob_list[j] > prob_list[i] and label_list[j] == 1:
                TP = TP + 1.
            elif prob_list[j] <= prob_list[i] and label_list[j] == 0:
                TN = TN + 1.
            elif prob_list[j] < prob_list[i] and label_list[j] == 1:
                FN = FN + 1.
            else:
                FP = FP + 1.
        Ferrlive = FP / (FP + TN + 1e-7)
        Ferrfake = FN / (FN + TP + 1e-7)
        FDR = FP / (FP + TN + 1e-7)
        TDR = TP / (TP + FN + 1e-7)
        if FDR < rate:
            tdr_list.append(TDR)
        ace_list.append((Ferrfake+Ferrlive)/2.)
    return min(ace_list), max(tdr_list)
def get_threshold(probs, grid_density):
    Min, Max = min(probs), max(probs)
    thresholds = []
    for i in range(grid_density + 1):
        thresholds.append(0.0 + i * 1.0 / float(grid_density))
    thresholds.append(1.1)
    return thresholds
def eval_state(probs, labels, thr):
    predict = probs >= thr
    TN = np.sum((labels == 0) & (predict == False))
    FN = np.sum((labels == 1) & (predict == False))
    FP = np.sum((labels == 0) & (predict == True))
    TP = np.sum((labels == 1) & (predict == True))
    return TN, FN, FP, TP
def get_EER_states(probs, labels, grid_density = 10000):
    thresholds = get_threshold(probs, grid_density)
    min_dist = 1.0
    min_dist_states = []
    FRR_list = []
    FAR_list = []
    for thr in thresholds:
        TN, FN, FP, TP = eval_state(probs, labels, thr)
        if(FN + TP == 0):
            FRR = TPR = 1.0
            FAR = FP / float(FP + TN)
            TNR = TN / float(TN + FP)
        elif(FP + TN == 0):
            TNR = FAR = 1.0
            FRR = FN / float(FN + TP)
            TPR = TP / float(TP + FN)
        else:
            FAR = FP / float(FP + TN)
            FRR = FN / float(FN + TP)
            TNR = TN / float(TN + FP)
            TPR = TP / float(TP + FN)
        dist = math.fabs(FRR - FAR)
        FAR_list.append(FAR)
        FRR_list.append(FRR)
        if dist <= min_dist:
            min_dist = dist
            min_dist_states = [FAR, FRR, thr]
    EER = (min_dist_states[0] + min_dist_states[1]) / 2.0
    thr = min_dist_states[2]
    return EER, thr, FRR_list, FAR_list
def get_HTER_at_thr(probs, labels, thr):
    TN, FN, FP, TP = eval_state(probs, labels, thr)
    if (FN + TP == 0):
        FRR = 1.0
        FAR = FP / float(FP + TN)
    elif(FP + TN == 0):
        FAR = 1.0
        FRR = FN / float(FN + TP)
    else:
        FAR = FP / float(FP + TN)
        FRR = FN / float(FN + TP)
    HTER = (FAR + FRR) / 2.0
    return HTER
def eval(test_loader, net):
    criterion = nn.CrossEntropyLoss()
    valid_losses = AverageMeter()
    prob_dict = {}
    label_dict = {}
    output_dict_tmp = {}
    target_dict_tmp = {}
    net.eval()
    t = tqdm(test_loader)
    t.set_description("Evaluate")
    with torch.no_grad():
        for iter, (input, target, videoID) in enumerate(t):
            # print(target)
            input = input.cuda()
            target = torch.from_numpy(np.array(target)).long().cuda()
            cls_out = net(input)
            prob = F.softmax(cls_out, dim=1).cpu().data.numpy()[:, 1]
            label = target.cpu().data.numpy()
            videoID = videoID.cpu().data.numpy()
            for i in range(len(prob)):
                if(videoID[i] in prob_dict.keys()):
                    prob_dict[videoID[i]].append(prob[i])
                    label_dict[videoID[i]].append(label[i])
                    output_dict_tmp[videoID[i]].append(cls_out[i].view(1, 2))
                    target_dict_tmp[videoID[i]].append(target[i].view(1))
                else:
                    prob_dict[videoID[i]] = []
                    label_dict[videoID[i]] = []
                    prob_dict[videoID[i]].append(prob[i])
                    label_dict[videoID[i]].append(label[i])
                    output_dict_tmp[videoID[i]] = []
                    target_dict_tmp[videoID[i]] = []
                    output_dict_tmp[videoID[i]].append(cls_out[i].view(1, 2))
                    target_dict_tmp[videoID[i]].append(target[i].view(1))
    prob_list = []
    label_list = []
    for key in prob_dict.keys():
        avg_single_video_prob = sum(prob_dict[key]) / len(prob_dict[key])
        avg_single_video_label = sum(label_dict[key]) / len(label_dict[key])
        prob_list = np.append(prob_list, avg_single_video_prob)
        label_list = np.append(label_list, avg_single_video_label)
        # compute loss and acc for every video
        avg_single_video_output = sum(output_dict_tmp[key]) / len(output_dict_tmp[key])
        avg_single_video_target = sum(target_dict_tmp[key]) / len(target_dict_tmp[key])
        loss = criterion(avg_single_video_output, avg_single_video_target.long())
        valid_losses.update(loss.item())
    ace, tdr = ACE_TDR_Cal(label_list, prob_list)
    auc_score = roc_auc_score(label_list, prob_list)
    cur_EER_valid, threshold, _, _ = get_EER_states(prob_list, label_list)
    cur_HTER_valid = get_HTER_at_thr(prob_list, label_list, threshold)
    return [valid_losses.avg, cur_HTER_valid, auc_score, tdr]


