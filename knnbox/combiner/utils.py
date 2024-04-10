r""" some utils function used for combiner """

import torch
import torch.nn.functional as F
from collections import Counter
import json
import os
import numpy as np
def calculate_knn_prob(vals, distances, probability_dim, temperature, device, datastore_path, **kwargs):
#def calculate_knn_prob(vals, distances, probability_dim, temperature, device, **kwargs):
    r"""
    How vanilla knn-mt calculates knn probs using retrieved vals and distances.
    """
    scaled_dists = - distances / temperature
    #knn_weights = torch.softmax(scaled_dists, dim=-1)

    B, S, K = vals.size()
    """
    B 代表批大小（batch size），
    S 代表序列长度，
    K 代表每个序列元素的k个最近邻。
    """
    CF_total = []
    tf = open(os.path.join( datastore_path ,"dictionary.json"), "r")
    #tf = open("/data/qirui/KNN-BOX-copy-copy/datastore/vanilla/koran/dictionary.json", "r")
    D_dic = json.load(tf)
    #N_dic = Counter(vals)
    
    # with open(os.path.join('/data/qirui/z-testdata','B_S_K.txt'), 'a') as file:
    #                                 string = "first:"+str(vals.size())+" "
    #                                 file.write(string)


    D_num = 0
    N_num = K
    for i in D_dic.values():
        D_num += i
    
    for i, item_tensor in enumerate(vals):
        for item2 in item_tensor:#一个tensor数组
            it = item2.tolist()
            N_dic = Counter(it)
            # 确保item是整数
            for item in item2:
                item = item.item() if isinstance(item, torch.Tensor) else item
                psmall = cal_p(item, N_dic,N_num)
                plarge = cal_p(item, D_dic,D_num)
                if psmall >= plarge:
                    CF = (psmall- plarge)/(1-plarge)
                    #CF = (psmall)/(1-plarge)
                else:
                    CF = (psmall- plarge)/plarge
                    #CF = (psmall)/plarge
                
                CF_total.append(CF)
    CF_total_tensor = torch.tensor(CF_total, dtype=torch.float, device=device)
    CF_total_tensor = CF_total_tensor.view(B, S, K)

    # # construct prob(vanilla版)
    # knn_weights = torch.softmax(scaled_dists, dim=-1)
    # knn_probs = torch.zeros(B, S, probability_dim, device=device)
    # knn_probs.scatter_add_(dim=-1, index=vals, src=knn_weights)
    
    # construct prob(直接版)
    # knn_probs = torch.zeros(B, S, probability_dim, device=device)
    # knn_probs.scatter_add_(dim=-1, index=vals, src=CF_total_tensor)

    

    # # construct prob(softmax版)
    # knn_weights = torch.softmax(CF_total_tensor, dim=-1)
    # knn_probs = torch.zeros(B, S, probability_dim, device=device)
    # knn_probs.scatter_add_(dim=-1, index=vals, src=knn_weights)
    
    # #construct prob(直接相减版)
    # scaled_dists2 = scaled_dists -  CF_total_tensor 
    # knn_weights = torch.softmax(scaled_dists2, dim=-1)
    # knn_probs = torch.zeros(B, S, probability_dim, device=device)
    # knn_probs.scatter_add_(dim=-1, index=vals, src=knn_weights)

    # # construct prob(so之后so版)(不太行)
    # scaled_dists2 = torch.softmax(scaled_dists, dim=-1)
    # CF2 = torch.softmax(CF_total_tensor, dim=-1)
    # knn_weights = scaled_dists2 - CF2
    # knn_probs = torch.zeros(B, S, probability_dim, device=device)
    # knn_probs.scatter_add_(dim=-1, index=vals, src=knn_weights)

    # #construct prob(指数版)
    # scaled_dists2 = torch.exp(scaled_dists) +torch.exp(CF_total_tensor/ temperature) 
    # knn_weights = torch.softmax(scaled_dists2, dim=-1)
    # knn_probs = torch.zeros(B, S, probability_dim, device=device)
    # knn_probs.scatter_add_(dim=-1, index=vals, src=knn_weights)

    #construct prob(论文版)
    scaled_dists2 =  1.6 * scaled_dists -  0.6* CF_total_tensor 
    knn_weights = torch.softmax(scaled_dists2, dim=-1)
    knn_probs = torch.zeros(B, S, probability_dim, device=device)
    knn_probs.scatter_add_(dim=-1, index=vals, src=knn_weights)


    return knn_probs

def cal_p(item, dic, num):
    item = item if isinstance(item, (int, float, str)) else item[0]
    count = dic.get(item, 0)
    p = count * 1.0 / num
    return p

def calculate_combined_prob(knn_prob, neural_model_logit, lambda_, log_probs):
    r""" 
    How vanilla knn-mt calculate the combining probability.
    """
    neural_model_prob = F.softmax(neural_model_logit, dim=-1)

    # with open(os.path.join('/data/qirui/z-testdata','calculate2.txt'), 'a') as file:
    #                                 string = "first:"+str(neural_model_prob.cpu().numpy())+" "
    #                                 file.write(string)

    # with open(os.path.join('/data/qirui/z-testdata','calculate22.txt'), 'a') as file:
    #                                 string = "first:"+str(knn_prob.cpu().numpy())+" "
    #                                 file.write(string)
    knn_prob = knn_prob.to(torch.float16)
    neural_model_prob = neural_model_prob.to(torch.float16)

    combined_probs = knn_prob * lambda_ + neural_model_prob * (1 - lambda_)
    

    # some extra infomation
    extra = {}
    extra["neural_probs"] = neural_model_prob
    extra["unlog_combined_probs"] = combined_probs


    if log_probs:
        combined_probs =  torch.log(combined_probs)
    return combined_probs, extra


def calculate_knn_prob_with_merge_weight(vals, distances, merge_weights, probability_dim, temperature, device, **kwargs):
    r""" 
    when the key-value pair has a merge weight.
    used by greedy-merge knn-mt
    """
    # consider merge weights here
    scaled_dists = - distances / temperature + torch.log(merge_weights.float())
    knn_weights = torch.softmax(scaled_dists, dim=-1)
    
    B, S, K = vals.size()

    # construct prob
    knn_probs = torch.zeros(B, S, probability_dim, device=device)
    knn_probs.scatter_add_(dim=-1, index=vals, src=knn_weights)

    return knn_probs