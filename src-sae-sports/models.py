# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import math
import os
import pickle
from tqdm import tqdm
import random
import copy

import torch
import torch.nn as nn
import gensim
import faiss
import time
from modules import Encoder, LayerNorm, LateFusionEncoder_gating, LateFusionEncoder, LateFusionEncoder3, LateFusionEncoder_c



class KMeans(object):
    def __init__(self, num_cluster, seed, hidden_size, gpu_id=0, device="cpu"):
        """
        Args:
            k: number of clusters
        """
        self.seed = seed
        self.num_cluster = num_cluster
        self.max_points_per_centroid = 4096
        self.min_points_per_centroid = 0
        self.gpu_id = 0
        self.device = device
        self.first_batch = True
        self.hidden_size = hidden_size
        self.clus, self.index = self.__init_cluster(self.hidden_size)
        self.centroids = []

    def __init_cluster(
        self, hidden_size, verbose=False, niter=20, nredo=5, max_points_per_centroid=4096, min_points_per_centroid=0
    ):
        print(" cluster train iterations:", niter)
        clus = faiss.Clustering(hidden_size, self.num_cluster)
        clus.verbose = verbose
        clus.niter = niter
        clus.nredo = nredo
        clus.seed = self.seed
        clus.max_points_per_centroid = max_points_per_centroid
        clus.min_points_per_centroid = min_points_per_centroid

        res = faiss.StandardGpuResources()
        res.noTempMemory()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = self.gpu_id
        index = faiss.GpuIndexFlatL2(res, hidden_size, cfg)
        return clus, index

    def train(self, x):
        # train to get centroids
        if x.shape[0] > self.num_cluster:
            self.clus.train(x, self.index)
        # get cluster centroids
        centroids = faiss.vector_to_array(self.clus.centroids).reshape(self.num_cluster, self.hidden_size)
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).to(self.device)
        self.centroids = nn.functional.normalize(centroids, p=2, dim=1)

    def query(self, x):
        # self.index.add(x)
        D, I = self.index.search(x, 1)  # for each sample, find cluster distance and assignments
        seq2cluster = [int(n[0]) for n in I]
        # print("cluster number:", self.num_cluster,"cluster in batch:", len(set(seq2cluster)))
        seq2cluster = torch.LongTensor(seq2cluster).to(self.device)
        return seq2cluster, self.centroids[seq2cluster]




class SASRecModel(nn.Module):
    def __init__(self, args, latent_init_tensor): # latent_init_tensor (1000,768)
        super(SASRecModel, self).__init__()
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0) # (item_num, 64), padding_idx=0 表示 item ID 为 0 时，embedding 始终为 0
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = LateFusionEncoder_gating(args, latent_init_tensor, trainable_latent=args.trainable_latent, fusion_type=args.fusion_type)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        self.criterion = nn.BCELoss(reduction="none")
        self.apply(self.init_weights)

    # Positional Embedding
    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1) # (b,l)取l
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device) # 顺序位置
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)  # (b,l)

        item_embeddings = self.item_embeddings(sequence) # (b,l)--(b,l,c)
        position_embeddings = self.position_embeddings(position_ids) # (b,l)--(b,l,c)
        sequence_emb = item_embeddings + position_embeddings # (b,l,c)--(b,l,c)
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    # model same as SASRec
    def forward(self, input_ids):

        attention_mask = (input_ids > 0).long() # (b,l)--(b,l)的01矩阵，有些位置是padding，没有实际item，不能attention
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64  (b, 1, 1, l)
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8 (1,l,l) 生成上三角矩阵（对角线以上为1）
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1) # (1,1,l,l) # 下三角
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask # (b,1,l,l)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # fused_output, memory_tensor, query_repr, attention_scores_mean
        sequence_emb = self.add_position_embedding(input_ids) # sequence_emb是(b,l,c)

        # self, hidden_states, attention_mask, output_all_encoded_layers=True
        item_encoded_layers, memory_tensor, query_repr, attention_scores_mean = self.item_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)

        # sequence_output = item_encoded_layers[-1] 
        sequence_output = item_encoded_layers # 没有list保存两次的结果

        return sequence_output, memory_tensor, query_repr, attention_scores_mean


    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()



class SASRecModel_item(nn.Module):
    def __init__(self, args, latent_init_tensor, item_embedding): # latent_init_tensor (1000,768), item_embedding:(12101,768)
        super(SASRecModel_item, self).__init__()
        # 假设 latent_init_tensor 是 shape (item_num, 768)
        self.item_text_proj = nn.Linear(item_embedding.size(1), args.hidden_size)  # 768 → 64
        self.item_text_emb = nn.Parameter(item_embedding, requires_grad=True)  # False可设为True微调 (12101, 768)
        # 融合后压缩成 args.hidden_size
        self.item_fusion = nn.Linear(args.hidden_size * 2, args.hidden_size)

        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0) # (item_num, 64), padding_idx=0 表示 item ID 为 0 时，embedding 始终为 0
        
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = LateFusionEncoder_gating(args, latent_init_tensor, trainable_latent=args.trainable_latent, fusion_type=args.fusion_type)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        self.criterion = nn.BCELoss(reduction="none")
        self.apply(self.init_weights)

    # Positional Embedding
    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1) # (b,l)取l
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device) # 顺序位置
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)  # (b,l)

        #item_embeddings = self.item_embeddings(sequence) # (b,l)--(b,l,c)

        item_ids = sequence  # (b, l)
        id_emb = self.item_embeddings(item_ids)  # (b, l, 64)
        # 安全地取语义 embedding（加一行padding zero向量）
        pad = torch.zeros(1, self.item_text_emb.size(1), device=self.item_text_emb.device) # item_text_emb (12101,768)
        full_text_emb = torch.cat([pad, self.item_text_emb], dim=0)  # (item_num+1, 768)
        text_emb = self.item_text_proj(full_text_emb[item_ids])  # (b, l, 64)
        # 拼接 + 融合
        fused_emb = torch.cat([id_emb, text_emb], dim=-1)  # (b, l, 128)
        item_embeddings = self.item_fusion(fused_emb)  # (b, l, 64)


        position_embeddings = self.position_embeddings(position_ids) # (b,l)--(b,l,c)
        sequence_emb = item_embeddings + position_embeddings # (b,l,c)--(b,l,c)
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    # model same as SASRec
    def forward(self, input_ids):

        attention_mask = (input_ids > 0).long() # (b,l)--(b,l)的01矩阵，有些位置是padding，没有实际item，不能attention
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64  (b, 1, 1, l)
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8 (1,l,l) 生成上三角矩阵（对角线以上为1）
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1) # (1,1,l,l) # 下三角
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask # (b,1,l,l)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # fused_output, memory_tensor, query_repr, attention_scores_mean
        sequence_emb = self.add_position_embedding(input_ids) # sequence_emb是(b,l,c)

        # self, hidden_states, attention_mask, output_all_encoded_layers=True
        item_encoded_layers, memory_tensor, query_repr, attention_scores_mean = self.item_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)

        # sequence_output = item_encoded_layers[-1] 
        sequence_output = item_encoded_layers # 没有list保存两次的结果

        return sequence_output, memory_tensor, query_repr, attention_scores_mean


    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class SASRecModel_item2(nn.Module):
    def __init__(self, args, latent_init_tensor, item_embedding): # latent_init_tensor (1000,768), item_embedding:(12101,768)
        super(SASRecModel_item2, self).__init__()
        # 假设 latent_init_tensor 是 shape (item_num, 768)
        self.item_text_proj = nn.Linear(item_embedding.size(1), args.hidden_size)  # 768 → 64
        # device = item_embedding.device
        # self.item_text_proj = self.item_text_proj.to(device)
        
        item_text_emb_64 = self.item_text_proj(item_embedding)
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0) #.to(device) # (item_num, 64), padding_idx=0 表示 item ID 为 0 时，embedding 始终为 0
        self.item_embeddings.weight.data[1:item_embedding.size(0)+1] = item_text_emb_64

        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = LateFusionEncoder_gating(args, latent_init_tensor, trainable_latent=args.trainable_latent, fusion_type=args.fusion_type)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        self.criterion = nn.BCELoss(reduction="none")
        self.apply(self.init_weights)

    # Positional Embedding
    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1) # (b,l)取l
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device) # 顺序位置
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)  # (b,l)

        item_embeddings = self.item_embeddings(sequence) # (b,l)--(b,l,c)

        position_embeddings = self.position_embeddings(position_ids) # (b,l)--(b,l,c)
        sequence_emb = item_embeddings + position_embeddings # (b,l,c)--(b,l,c)
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    # model same as SASRec
    def forward(self, input_ids):

        attention_mask = (input_ids > 0).long() # (b,l)--(b,l)的01矩阵，有些位置是padding，没有实际item，不能attention
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64  (b, 1, 1, l)
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8 (1,l,l) 生成上三角矩阵（对角线以上为1）
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1) # (1,1,l,l) # 下三角
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask # (b,1,l,l)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # fused_output, memory_tensor, query_repr, attention_scores_mean
        sequence_emb = self.add_position_embedding(input_ids) # sequence_emb是(b,l,c)

        # self, hidden_states, attention_mask, output_all_encoded_layers=True
        item_encoded_layers, memory_tensor, query_repr, attention_scores_mean = self.item_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)

        # sequence_output = item_encoded_layers[-1] 
        sequence_output = item_encoded_layers # 没有list保存两次的结果

        return sequence_output, memory_tensor, query_repr, attention_scores_mean


    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()



class SASRecModel_item3(nn.Module):
    def __init__(self, args, latent_init_tensor, item_embedding): # latent_init_tensor (1000,768), item_embedding:(12101,768)
        super(SASRecModel_item3, self).__init__()
        # 假设 latent_init_tensor 是 shape (item_num, 768)
        self.item_text_proj = nn.Linear(item_embedding.size(1), args.hidden_size)  # 768 → 64
        self.item_text_emb = nn.Parameter(item_embedding, requires_grad=False)  # False可设为True微调 (12101, 768)
    
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0) # (item_num, 64), padding_idx=0 表示 item ID 为 0 时，embedding 始终为 0
        
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = LateFusionEncoder_gating(args, latent_init_tensor, trainable_latent=args.trainable_latent, fusion_type=args.fusion_type)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        self.criterion = nn.BCELoss(reduction="none")
        self.apply(self.init_weights)

    # Positional Embedding
    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1) # (b,l)取l
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device) # 顺序位置
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)  # (b,l)

        #item_embeddings = self.item_embeddings(sequence) # (b,l)--(b,l,c)

        item_ids = sequence  # (b, l)
        id_emb = self.item_embeddings(item_ids)  # (b, l, 64)
        # 安全地取语义 embedding（加一行padding zero向量）
        pad = torch.zeros(1, self.item_text_emb.size(1), device=self.item_text_emb.device) # item_text_emb (12101,768)
        full_text_emb = torch.cat([pad, self.item_text_emb], dim=0)  # (item_num+1, 768)
        text_emb = self.item_text_proj(full_text_emb[item_ids])  # (b, l, 64)
        # 拼接 + 融合
        fused_emb = id_emb + text_emb  # (b, l, 64)
        item_embeddings = self.LayerNorm(fused_emb)  # self.LayerNorm(fused_emb)  # (b, l, 64)


        position_embeddings = self.position_embeddings(position_ids) # (b,l)--(b,l,c)
        sequence_emb = item_embeddings + position_embeddings # (b,l,c)--(b,l,c)
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    # model same as SASRec
    def forward(self, input_ids):

        attention_mask = (input_ids > 0).long() # (b,l)--(b,l)的01矩阵，有些位置是padding，没有实际item，不能attention
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64  (b, 1, 1, l)
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8 (1,l,l) 生成上三角矩阵（对角线以上为1）
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1) # (1,1,l,l) # 下三角
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask # (b,1,l,l)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # fused_output, memory_tensor, query_repr, attention_scores_mean
        sequence_emb = self.add_position_embedding(input_ids) # sequence_emb是(b,l,c)

        # self, hidden_states, attention_mask, output_all_encoded_layers=True
        item_encoded_layers, memory_tensor, query_repr, attention_scores_mean = self.item_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)

        # sequence_output = item_encoded_layers[-1] 
        sequence_output = item_encoded_layers # 没有list保存两次的结果

        return sequence_output, memory_tensor, query_repr, attention_scores_mean


    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()



class SASRecModel_nogating(nn.Module):
    def __init__(self, args, latent_init_tensor): # latent_init_tensor (1000,768)
        super(SASRecModel_nogating, self).__init__()
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0) # (item_num, 64), padding_idx=0 表示 item ID 为 0 时，embedding 始终为 0
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        # self.item_encoder1 = Encoder(args) # 正常序列embedding做自注意
        # self.item_encoder2 = Encoder_latentkv(args, select_intent) # 序列embedding做query, latent embedding做key和value
        #self.item_encoder = LateFusionEncoder(args, latent_init_tensor, trainable_latent=args.trainable_latent, fusion_type=args.fusion_type)
        self.item_encoder = LateFusionEncoder_c(args, latent_init_tensor, trainable_latent=args.trainable_latent, fusion_type=args.fusion_type)
        
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        self.criterion = nn.BCELoss(reduction="none")
        self.apply(self.init_weights)

    # Positional Embedding
    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1) # (b,l)取l
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device) # 顺序位置
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)  # (b,l)

        item_embeddings = self.item_embeddings(sequence) # (b,l)--(b,l,c)
        position_embeddings = self.position_embeddings(position_ids) # (b,l)--(b,l,c)
        sequence_emb = item_embeddings + position_embeddings # (b,l,c)--(b,l,c)
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    # model same as SASRec
    def forward(self, input_ids):

        attention_mask = (input_ids > 0).long() # (b,l)--(b,l)的01矩阵，有些位置是padding，没有实际item，不能attention
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64  (b, 1, 1, l)
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8 (1,l,l) 生成上三角矩阵（对角线以上为1）
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1) # (1,1,l,l) # 下三角
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask # (b,1,l,l)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # fused_output, memory_tensor, query_repr, attention_scores_mean
        sequence_emb = self.add_position_embedding(input_ids) # sequence_emb是(b,l,c)

        # self, hidden_states, attention_mask, output_all_encoded_layers=True
        item_encoded_layers, memory_tensor, query_repr, attention_scores_mean = self.item_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)

        # sequence_output = item_encoded_layers[-1] 
        sequence_output = item_encoded_layers # 没有list保存两次的结果

        return sequence_output, memory_tensor, query_repr, attention_scores_mean


    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()



class SASRecModel3(nn.Module):
    def __init__(self, args, latent_init_tensor): # latent_init_tensor (1000,768)
        super(SASRecModel3, self).__init__()
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0) # (item_num, 64), padding_idx=0 表示 item ID 为 0 时，embedding 始终为 0
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        # self.item_encoder1 = Encoder(args) # 正常序列embedding做自注意
        # self.item_encoder2 = Encoder_latentkv(args, select_intent) # 序列embedding做query, latent embedding做key和value
        self.item_encoder = LateFusionEncoder3(args, latent_init_tensor, trainable_latent=args.trainable_latent, fusion_type=args.fusion_type)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        self.criterion = nn.BCELoss(reduction="none")
        self.apply(self.init_weights)

    # Positional Embedding
    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1) # (b,l)取l
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device) # 顺序位置
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)  # (b,l)

        item_embeddings = self.item_embeddings(sequence) # (b,l)--(b,l,c)
        position_embeddings = self.position_embeddings(position_ids) # (b,l)--(b,l,c)
        sequence_emb = item_embeddings + position_embeddings # (b,l,c)--(b,l,c)
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    # model same as SASRec
    def forward(self, input_ids):

        attention_mask = (input_ids > 0).long() # (b,l)--(b,l)的01矩阵，有些位置是padding，没有实际item，不能attention
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64  (b, 1, 1, l)
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8 (1,l,l) 生成上三角矩阵（对角线以上为1）
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1) # (1,1,l,l) # 下三角
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask # (b,1,l,l)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # fused_output, memory_tensor, query_repr, attention_scores_mean
        sequence_emb = self.add_position_embedding(input_ids) # sequence_emb是(b,l,c)

        # self, hidden_states, attention_mask, output_all_encoded_layers=True
        user_intent_repr, loss_align = self.item_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)

        return user_intent_repr, loss_align # (b,d)


    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()



# GRU Encoder
class GRUEncoder(nn.Module):
    r"""GRU4Rec is a model that incorporate RNN for recommendation.

    Note:

        Regarding the innovation of this article,we can only achieve the data augmentation mentioned
        in the paper and directly output the embedding of the item,
        in order that the generation method we used is common to other sequential models.
    """

    def __init__(self, args):
        super(GRUEncoder, self).__init__()

        # load parameters info
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.args = args
        self.embedding_size = args.hidden_size #64
        self.hidden_size = args.hidden_size*2  #128
        self.num_layers = args.num_hidden_layers-1 #1
        self.dropout_prob =args.hidden_dropout_prob #0.3

        # define layers and loss
        self.emb_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)


    def forward(self, item_seq):
        item_seq_emb = self.item_embeddings(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        # the embedding of the predicted item, shape of (batch_size, embedding_size)
        # seq_output = self.gather_indexes(gru_output, item_seq_len - 1)
        seq_output=gru_output
        return seq_output




