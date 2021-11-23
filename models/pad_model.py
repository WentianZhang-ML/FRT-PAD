# -*- coding: utf-8 -*-
import sys
sys.path.append('../')
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import argparse
from networks import Face_Recognition, Face_Expression, Face_Attribute_D
from networks import Baseline, GAT

def PA_Detector():
    net = Baseline()
    """
    PA_Detector pre-trained by imagenet (resnet18)
    """
    model_dict = net.state_dict()
    logger.info("load imagenet model path:{}".format('./pretrained_model/resnet18-5c106cde.pth'))
    pretrained_dict = torch.load('./pretrained_model/resnet18-5c106cde.pth')
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    return net
def Face_Related_Work(downstream):
    """
    Pre-trained Face_Related_Work model 
    """
    if downstream == 'Face_Recognition':
        net = Face_Recognition()
    	model_path = './pretrained_model/R18_MS1MV3_backbone.pth'
        model_dict = net.state_dict()
        logger.info("load imagenet model path:{}".format(model_path))
        pretrained_dict = torch.load(model_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
    elif downstream == 'Face_Expression_Recognition':
        net = Face_Expression()
    	model_path = './pretrained_model/ijba_res18_naive.pth.tar'
        model_dict = net.state_dict()
        logger.info("load imagenet model path:{}".format(model_path))
        pretrained_dict = torch.load(model_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
    elif downstream == 'Face_Attribute_Editing':
        net = Face_Attribute_D()
        model_path = './pretrained_model/celeba-128x128-5attrs/200000-D.ckpt'
        model_dict = net.state_dict()
        logger.info("load imagenet model path:{}".format(model_path))
        pretrained_dict = torch.load(model_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
    net.eval()
    return net

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def Cross_Modal_Adapter(graph_type):

    """
    Two Graph Attention Networks
    """
    if graph_type == 'Step_by_Step_Graph':
        edges = np.array([[0,1],[1,2],[2,3],[3,4]],dtype=np.int32)
    elif graph_type == 'Dense_Graph':
        edges = np.array([[0,1],[0,2],[0,3],[0,4],[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]],dtype=np.int32)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), 
                        shape=(5, 5), 
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = torch.tensor(np.array(adj.todense()),dtype=torch.float32, requires_grad=True,device=torch.device('cuda:'+args.device))
    net = GAT(batch_size= args.batch_size, adj=adj)
    return net