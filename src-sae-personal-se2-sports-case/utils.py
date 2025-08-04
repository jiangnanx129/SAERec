# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

import numpy as np
import math
import random
import os
import json
import pickle
from scipy.sparse import csr_matrix

import torch
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def nCr(n, r):
    f = math.factorial
    return f(n) // f(r) // f(n - r)


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"{path} created")


def neg_sample(item_set, item_size):  # []
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        for i in range(len(score)):
            # 有一个指标增加了就认为是还在涨
            if score[i] > self.best_score[i] + self.delta:
                return False
        return True

    def __call__(self, score, model):
        # score HIT@10 NDCG@10

        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0] * len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            # ({self.score_min:.6f} --> {score:.6f}) # 这里如果是一个值的话输出才不会有问题
            print(f"Validation score increased.  Saving model ...")
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index).squeeze(dim)


def avg_pooling(x, dim):
    return x.sum(dim=dim) / x.size(dim)


def generate_rating_matrix_valid(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]:  # val每条序列剩下最后两个item
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def generate_rating_matrix_test(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]:  # test每个序列剩一个item
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def get_user_seqs(data_file):
    lines = open(data_file).readlines()
    user_seq = []
    user_id=[]
    item_set = set()
    for line in lines:
        user, items = line.strip().split(" ", 1)
        items = items.split(" ")
        items = [int(item) for item in items]
        user_seq.append(items)
        user_id.append(int(user))
        item_set = item_set | set(items)
    max_item = max(item_set)
    num_users = len(lines)
    num_items=max_item+2
    valid_rating_matrix = generate_rating_matrix_valid(user_seq, num_users, num_items)
    test_rating_matrix = generate_rating_matrix_test(user_seq, num_users, num_items)
    return user_id,user_seq, max_item, valid_rating_matrix, test_rating_matrix


def get_user_seqs_long(data_file):
    lines = open(data_file).readlines()
    user_seq = []
    long_sequence = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(" ", 1)
        items = items.split(" ")
        items = [int(item) for item in items]
        long_sequence.extend(items)
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    return user_seq, max_item, long_sequence


def get_user_seqs_and_sample(data_file, sample_file):
    lines = open(data_file).readlines()
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(" ", 1)
        items = items.split(" ")
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    lines = open(sample_file).readlines()
    sample_seq = []
    for line in lines:
        user, items = line.strip().split(" ", 1)
        items = items.split(" ")
        items = [int(item) for item in items]
        sample_seq.append(items)

    assert len(user_seq) == len(sample_seq)

    return user_seq, max_item, sample_seq


def get_item2attribute_json(data_file):
    item2attribute = json.loads(open(data_file).readline())
    attribute_set = set()
    for item, attributes in item2attribute.items():
        attribute_set = attribute_set | set(attributes)
    attribute_size = max(attribute_set)  # 331
    return item2attribute, attribute_size


def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    # [batch] the answer's rank
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT / len(pred_list), NDCG / len(pred_list), MRR / len(pred_list)


def precision_at_k_per_sample(actual, predicted, topk):
    num_hits = 0
    for place in predicted:
        if place in actual:
            num_hits += 1
    return num_hits / (topk + 0.0)


def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return sum_precision / num_users


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in set(actual[user_id])) / math.log(j + 2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0 / math.log(i + 2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res



# 计算align loss，让用户序列
def compute_alignment_loss(query_repr, attention_scores, memory_repr, topk=5):

    # memory_tensor(1000,64), query_repr(b,64), attention_scores_mean(b,1000)
    batch_size, hidden_size = query_repr.size()
    memory_size = memory_repr.size(0) # 1000

    # 选 topk
    topk = attention_scores.shape[1]
    # print(attention_scores.shape, topk)
    topk_values, topk_indices = attention_scores.topk(k=topk, dim=-1)  # (batch_size, topk)

    align_losses = []

    for b in range(batch_size):
        q = query_repr[b]  # (hidden_size,)
        idx = topk_indices[b]  # (topk,)
        w = topk_values[b]  # (topk,)
        selected_basis = memory_repr[idx]  # (topk, hidden_size)

        # Step 2: 计算query和每个selected basis的cosine similarity
        q_expand = q.unsqueeze(0).expand_as(selected_basis)  # (topk, hidden_size)
        cosine_sim = F.cosine_similarity(q_expand, selected_basis, dim=-1)  # (topk,)

        # Step 3: loss是 (1-cosine_sim)，attention weight加权
        loss = (1 - cosine_sim) * w
        align_losses.append(loss.sum())

    align_loss = torch.stack(align_losses).mean()

    return align_loss


def compute_attention_loss(attention_scores, eps=1e-12):
    """
    Calculate attention entropy loss in a numerically stable way.
    
    Args:
        attention_scores: (batch_size, memory_len)
    
    Returns:
        attn_loss: scalar
    """
    attention_scores = attention_scores.clamp(min=eps, max=1.0)  # 防止负数，保证在[eps, 1]

    entropy = - (attention_scores * torch.log(attention_scores)).sum(dim=-1)  # (batch_size,)
    attn_loss = entropy.mean()  # (scalar)

    return attn_loss







# batch内计算tfidf，batchsize越大越稳定
def compute_tfidf_batch(u, intent_pool, topk=10, threshold=0.0): # threshold=0.0
    """
    Batch-only TF-IDF: 使用当前 batch 中的统计估计 IDF
    Args:
        u: (b, d) 用户向量（序列池化后） -1或mean
        intent_pool: (n, d) 全局 latent intent
        topk: 每个用户保留的 intent 个数
    Returns:
        local_intents: (b, topk, d)
        tfidf_scores: (b, n)
        topk_indices: (b, topk)
    """
    b, d = u.shape
    n = intent_pool.shape[0]

    # Cosine similarity = TF
    u_norm = torch.nn.functional.normalize(u, dim=-1)
    z_norm = torch.nn.functional.normalize(intent_pool, dim=-1)
    sim_scores = torch.matmul(u_norm, z_norm.T)  # (b, n)

    # IDF from batch
    sim_scores = torch.relu(sim_scores) # 负数至0
    activated = (sim_scores > threshold).float()  # (b, n)

    #df = activated.sum(dim=0)  # (n,) # 1. 按照激活值求和
    df = (activated > 0).sum(dim=0) # 2. 返回布尔，以1计数

    idf = torch.log((1 + b) / (1 + df))  # (n,) # 稀有 latent 得高分，常见 latent 得低分

    # TF-IDF
    tfidf = sim_scores * idf.unsqueeze(0)  # (b, n)

    # Top-k
    topk_scores, topk_indices = torch.topk(tfidf, k=topk, dim=-1) # (b topk)

    local_intents = torch.gather(
        intent_pool.unsqueeze(0).expand(b, -1, -1),  # (b, n, d)
        dim=1,
        index=topk_indices.unsqueeze(-1).expand(-1, -1, d)  # (b, topk, d)
    )

    return local_intents, tfidf, topk_indices



# ema累计idf
def compute_tfidf_ema(u, intent_pool, ema_tracker, topk=10, threshold=0.2):
    b, d = u.shape
    n = intent_pool.shape[0]

    u_norm = torch.nn.functional.normalize(u, dim=-1)
    z_norm = torch.nn.functional.normalize(intent_pool, dim=-1)
    sim_scores = torch.matmul(u_norm, z_norm.T)  # (b, n)

    # 更新 EMA tracker
    sim_scores = torch.relu(sim_scores) # 负数至0,加的
    activated = (sim_scores > threshold).float() # (b,1000)
    ema_tracker.update(activated)

    # 使用 EMA IDF
    idf = ema_tracker.get_idf().to(sim_scores.device)  # (n,)
    tfidf = sim_scores * idf.unsqueeze(0)  # (b, n)

    topk_scores, topk_indices = torch.topk(tfidf, k=topk, dim=-1)

    #print("1:", intent_pool.shape, topk_indices.shape, topk_indices.unsqueeze(-1).expand(-1, -1, d).shape, tfidf.shape)
    local_intents = torch.gather(
        intent_pool.unsqueeze(0).expand(b, -1, -1),
        dim=1,
        index=topk_indices.unsqueeze(-1).expand(-1, -1, d)
    )
   

    return local_intents, tfidf, topk_indices, idf

