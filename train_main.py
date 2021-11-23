# -*- coding: utf-8 -*-
import sys
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import argparse
import os
import random
from tqdm import *
import logging

from utils.utils import get_dataset, AverageMeter, set_log
from utils.evaluate import accuracy, eval
from utils.config import DefaultConfig
from models.pad_model import PA_Detector, Face_Related_Work, Cross_Modal_Adapter
from models.networks import PAD_Classifier


config = DefaultConfig()

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--train_data", type=str, default='om', help='Training data (om/ci)')
parser.add_argument("--test_data", type=str, default='ci', help='Testing data (ci/om)')
parser.add_argument("--downstream", type=str, default='FR', help='FR/FE/FA')
parser.add_argument("--graph_type",type=str, default='direct', help='direct/dense')
args = parser.parse_args()

log_dir = config.root + 'face_log/'+ args.downstream+'/'
logger = set_log(log_dir, args.train_data, args.test_data)
logger.info("Log path:" + log_path)
logger.info("Training Protocol")
logger.info("Epoch Total number:{}".format(config.Epoch_num))
logger.info("Batch Size is {:^.2f}".format(config.batch_size))
logger.info("Shuffle Data for Training is {}".format(config.shuffle_train))
logger.info("Training set is {}".format(config.dataset(args.train_data)))
logger.info("Test set is {}".format(config.dataset(args.test_data)))
logger.info("Face related work is {}".format(config.face_related_work(args.downstream)))
logger.info("Graph type is {}".format(config.graph(args.graph_type)))
logger.info("savedir:{}".format(config.savedir))

def load_net_datasets():
    net_pad = PA_Detector()
    net_downstream = Face_Related_Work(config.face_related_work(args.downstream))
    net_adapter = Cross_Modal_Adapter(config.graph(args.graph_type))
    net = PAD_Classifier(net_pad,net_downstream,net_adapter,args.downstream)
    train_data_loader, test_data_loader = get_dataset('./labels',config.dataset(args.train_data), config.sample_frame, config.dataset(args.test_data), config.sample_frame, config.batch_size)
    return net, train_data_loader, test_data_loader

def train():
    net, train_loader, test_loader = load_net_datasets()

    best_model_TOP1 = 0.0
    best_model_HTER = 1.0
    best_model_AUC = 0.0
    best_model_TDR = 0.0
    # loss,top1 accuracy, hter, auc, tdr
    valid_args = [np.inf, 0, 0, 0, 0]

    logger.info('**************************** start training target model! ******************************\n')
    logger.info(
        '---------|-------------- VALID ---------------|---- Training ----|-------- Current Best -------|\n')
    logger.info(
        '  epoch  |   loss    HTER     AUC      TDR    |   loss   top-1   |    HTER     AUC      TDR    |\n')
    logger.info(
        '-----------------------------------------------------------------------------------------------|\n')

    loss_classifier = AverageMeter()
    classifer_top1 = AverageMeter()
    criterion = nn.CrossEntropyLoss().cuda()

    if config.opt == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, amsgrad=True)
    if config.opt == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)


    for e in range(config.Epoch_num):
        t = tqdm(train_loader)
        t.set_description("Epoch [{}/{}]".format(e +1 ,config.Epoch_num))
        for b, (imgs, labels, _) in enumerate(t):
            imgs = imgs.cuda()
            labels = labels.cuda().view(-1)

            net.train()
            out = net(imgs)

            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_classifier.update(loss.item())
            acc = accuracy(out.narrow(0, 0, imgs.size(0)), labels, topk=(1,))
            classifer_top1.update(acc[0])

        if ((e+1) % config.eval_epoch == 0):
            valid_args = eval(test_loader, net)
            is_best = valid_args[1] <= best_model_HTER

            if (is_best):
                best_model_HTER = valid_args[1]
                best_model_AUC  = valid_args[2]
                best_model_TDR  = valid_args[3]

            logger.info(
                '  %3d  |  %5.3f  %6.3f  %6.3f  %6.3f  |  %6.3f  %6.3f  |  %6.3f  %6.3f  %6.3f  |'
                % (
                e+1,
                valid_args[0], valid_args[1] * 100, valid_args[2] * 100, valid_args[3] *100,
                loss_classifier.avg, classifer_top1.avg,
                float(best_model_HTER * 100), float(best_model_AUC * 100), float(best_model_TDR * 100)))

            if is_best:
                save_dir = config.savedir+args.downstream+'_'+args.graph_type+'_Graph'+'/'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = save_dir + 'Train_' + args.train_data + '_test_' + args.test_data + '_' + str(e+1)+'_HTER_'+str(round(best_model_HTER * 100, 3)) + '.pth'
                torch.save(net.state_dict(), save_path)

if __name__ == "__main__":
    train()
