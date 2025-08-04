# -*- coding:utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import numpy as np
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
        (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Embeddings(nn.Module):
    """Construct the embeddings from item, position.
    """

    def __init__(self, args):
        super(Embeddings, self).__init__()

        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)  # 不要乱用padding_idx
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)

        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.args = args

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        items_embeddings = self.item_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = items_embeddings + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads)
            )
        self.num_attention_heads = args.num_attention_heads # 2
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads) # 64/2=32
        self.all_head_size = self.num_attention_heads * self.attention_head_size # 2*32=64

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) # (b,l,2,32)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # (b,2,l,32)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        # [batch_size heads seq_len head_size]
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        #print(mixed_query_layer.shape, query_layer.shape) # torch.Size([256, 50, 64]) torch.Size([256, 2, 50, 32])

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class Intermediate(nn.Module):
    def __init__(self, args):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(args.hidden_size, args.hidden_size * 4)
        if isinstance(args.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[args.hidden_act]
        else:
            self.intermediate_act_fn = args.hidden_act

        self.dense_2 = nn.Linear(args.hidden_size * 4, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, input_tensor):

        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class Layer(nn.Module):
    def __init__(self, args):
        super(Layer, self).__init__()
        self.attention = SelfAttention(args)
        self.intermediate = Intermediate(args)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return intermediate_output


# 原来只考虑自注意的SASRec
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        layer = Layer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers










'''
将latent维度，match到和序列embedding一样: 768-->64
'''
class MemoryProjector(nn.Module):
    def __init__(self, memory_dim, hidden_size):
        super().__init__()
        self.projector = nn.Linear(memory_dim, hidden_size)

    def forward(self, memory_tensor):
        return self.projector(memory_tensor)


'''
CrossAttention: 序列embedding做query, latent embedding做key和value
'''
class CrossAttention(nn.Module):
    def __init__(self, args): # intent_dim
        super(CrossAttention, self).__init__()
        self.num_attention_heads = args.num_attention_heads # 2
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads) # 64/2=32
        self.all_head_size = self.num_attention_heads * self.attention_head_size # 2*32=64
        
        # query的线性变换
        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, intent_tensor):
        # input_tensor: (batch_size, seq_len, hidden_size=64)
        # intent_tensor: (intent_len, intent_dim=768)

        batch_size = input_tensor.size(0)
        intent_tensor = intent_tensor.unsqueeze(0).expand(batch_size, -1, -1) 
        
        mixed_query_layer = self.query(input_tensor)  # (batch, seq_len, all_head_size)
        mixed_key_layer = self.key(intent_tensor)
        mixed_value_layer = self.value(intent_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer) # (b,2,l,32)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # (b, 2, l, l)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer) # (b,2,l,l) (b,2,l,32)-->(b,2,l,32)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # (b,l,2,32)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states, attention_probs # 返回权重(b, head, l, latent_num)


'''
CrossAttention_gating: 序列embedding做query, latent embedding做key和value
'''
class CrossAttention_gating(nn.Module):
    def __init__(self, args, memory_len=None):  # memory_len=1000, latnet 数目
        super(CrossAttention_gating, self).__init__()
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads) # 64/2=32
        self.all_head_size = self.num_attention_heads * self.attention_head_size # 2*32=64

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12)

        # memory gating: 对每个slot（1000个）单独学习缩放参数
        self.memory_gate = nn.Parameter(torch.ones(memory_len))  # memory_len = 1000


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(*new_x_shape).permute(0, 2, 1, 3)

    def forward(self, input_tensor, memory_tensor):
        # memory_tensor: (1000, 64)
        
        # Gating
        gated_memory = memory_tensor * self.memory_gate.unsqueeze(-1)  # (1000, 1) × (1000, 64) → (1000, 64)
        # print(memory_tensor.shape, self.memory_gate.shape, gated_memory.shape) # torch.Size([1208, 768]) torch.Size([1208]) torch.Size([1208, 768])
        
        # ✅ 扩 batch维
        batch_size = input_tensor.size(0)
        gated_memory = gated_memory.unsqueeze(0).expand(batch_size, -1, -1)  # (b, l₂, c)

        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(gated_memory)
        mixed_value_layer = self.value(gated_memory)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states, attention_probs # 返回权重(b, head, l, latent_num)


class CrossAttentionLayer(nn.Module):
    def __init__(self, args):
        super(CrossAttentionLayer, self).__init__()
        self.cross_attention = CrossAttention(args)
        self.intermediate = Intermediate(args)

    def forward(self, hidden_states, intent_tensor):
        attention_output, attention_probs = self.cross_attention(hidden_states, intent_tensor)
        intermediate_output = self.intermediate(attention_output)
        return intermediate_output, attention_probs


class CrossAttentionLayer_gating(nn.Module):
    def __init__(self, args, memory_len):
        super(CrossAttentionLayer_gating, self).__init__()
        self.cross_attention = CrossAttention_gating(args, memory_len) # latent数目
        self.intermediate = Intermediate(args)

    def forward(self, hidden_states, intent_tensor):
        attention_output, attention_probs = self.cross_attention(hidden_states, intent_tensor)
        intermediate_output = self.intermediate(attention_output)
        return intermediate_output, attention_probs

'''
最终函数：不带gating
'''
class LateFusionEncoder(nn.Module):
    def __init__(self, args, latent_init_tensor, trainable_latent=False, fusion_type='add'):
        super(LateFusionEncoder, self).__init__()

        # 控制 latent embedding 是否可训练
        if trainable_latent:
            self.memory_tensor = nn.Parameter(latent_init_tensor.clone())
            with open(args.log_file, "a") as f:
                f.write("latent embedding 可训练" + "\n") # 参数写入log文件
        else:
            self.register_buffer('memory_tensor', latent_init_tensor.clone())
            with open(args.log_file, "a") as f:
                f.write("latent embedding 不可训练" + "\n") # 参数写入log文件

        # ✅ MemoryProjector：将 latent 768 -> hidden_size=64
        self.memory_projector = MemoryProjector(latent_init_tensor.size(1), args.hidden_size)
       
        # Self-attention分支
        self.self_layers = nn.ModuleList([
            Layer(args) for _ in range(args.num_hidden_layers)
        ])

        # Cross-attention分支
        self.cross_layers = nn.ModuleList([
            CrossAttentionLayer(args) for _ in range(args.num_hidden_layers)
        ])

        # 融合方式
        self.fusion_type = fusion_type.lower()
        if self.fusion_type == 'concat':
            self.fusion_dense = nn.Linear(args.hidden_size * 2, args.hidden_size)
        elif self.fusion_type == 'gate':
            self.gate_layer = nn.Linear(args.hidden_size * 2, 1)

        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12)



    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        
        memory_tensor = self.memory_tensor
        # ✅ 先 projector 到 (1000, 64)
        memory_tensor = self.memory_projector(memory_tensor)

        # 1. Self-attention branch
        self_branch = hidden_states
        for layer_module in self.self_layers:
            self_branch = layer_module(self_branch, attention_mask)

        # 2. Cross-attention branch
        cross_branch = hidden_states
        attention_list = []
        for layer_module in self.cross_layers:
            cross_branch, attention_probs = layer_module(cross_branch, memory_tensor)
            attention_list.append(attention_probs)

        # 3. 融合
        if self.fusion_type == 'add':
            fused_output = self_branch + cross_branch
            fused_output = self.LayerNorm(fused_output)
        elif self.fusion_type == 'concat':
            fused_output = torch.cat([self_branch, cross_branch], dim=-1)
            fused_output = self.fusion_dense(fused_output)
            fused_output = self.dropout(fused_output)
            fused_output = self.LayerNorm(fused_output)
        elif self.fusion_type == 'gate':
            concat_output = torch.cat([self_branch, cross_branch], dim=-1)
            gate = torch.sigmoid(self.gate_layer(concat_output))
            fused_output = gate * self_branch + (1 - gate) * cross_branch
            fused_output = self.dropout(fused_output)
            fused_output = self.LayerNorm(fused_output)
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")
        
        if output_all_encoded_layers: # return_latent_info: 表示计算align loss
            # 拿序列最后一个位置作为query
            query_repr = fused_output[:, -1, :]  # (batch_size, hidden_size)
            # 平均所有层attention
            attention_scores_mean = torch.stack(attention_list, dim=0).mean(dim=0)  # (batch_size, num_heads, seq_len, memory_len)
            attention_scores_mean = attention_scores_mean.mean(dim=1)  # 跨head取均值 (batch_size, seq_len, memory_len)
            attention_scores_mean = attention_scores_mean[:, -1, :]  # 只取最后一个位置query的分数

            return fused_output, memory_tensor, query_repr, attention_scores_mean
        else:
            return fused_output

        # fused_output = self.LayerNorm(fused_output)

        # if output_all_encoded_layers:
        #     return [fused_output]
        # else:
        # return fused_output



class LateFusionEncoder_gating(nn.Module):
    def __init__(self, args, latent_init_tensor, trainable_latent=False, fusion_type='add'):
        super(LateFusionEncoder_gating, self).__init__() # latent_init_tensor: (1000, 768)
        
        # 控制 latent embedding 是否可训练
        if trainable_latent:
            self.memory_tensor = nn.Parameter(latent_init_tensor.clone())
            with open(args.log_file, "a") as f:
                f.write("latent embedding 可训练" + "\n") # 参数写入log文件
        else:
            self.register_buffer('memory_tensor', latent_init_tensor.clone())
            with open(args.log_file, "a") as f:
                f.write("latent embedding 不可训练" + "\n") # 参数写入log文件
            
        
        # ✅ MemoryProjector：将 latent 768 -> hidden_size=64
        self.memory_projector = MemoryProjector(latent_init_tensor.size(1), args.hidden_size)

        # Self-attention分支
        self.self_layers = nn.ModuleList([
            Layer(args) for _ in range(args.num_hidden_layers)
        ])

        # Cross-attention分支
        self.cross_layers = nn.ModuleList([
            CrossAttentionLayer_gating(args, latent_init_tensor.size(0)) for _ in range(args.num_hidden_layers)
        ])

        # 融合方式
        self.fusion_type = fusion_type.lower()
        if self.fusion_type == 'concat':
            self.fusion_dense = nn.Linear(args.hidden_size * 2, args.hidden_size)
        elif self.fusion_type == 'gate':
            self.gate_layer = nn.Linear(args.hidden_size * 2, 1)

        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12)



    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
    
        memory_tensor = self.memory_tensor  # 固定或可学习memory: (1000, 768)
        
        # ✅ 先 projector 到 (1000, 64)
        memory_tensor = self.memory_projector(memory_tensor)

        # 1. Self-attention branch
        self_branch = hidden_states
        for layer_module in self.self_layers:
            self_branch = layer_module(self_branch, attention_mask)

        # 2. Cross-attention branch
        cross_branch = hidden_states
        attention_list = []
        for layer_module in self.cross_layers:
            cross_branch, attention_probs = layer_module(cross_branch, memory_tensor)
            attention_list.append(attention_probs)

        # 3. 融合
        if self.fusion_type == 'add':
            fused_output = self_branch + cross_branch
            fused_output = self.LayerNorm(fused_output)
        elif self.fusion_type == 'concat':
            fused_output = torch.cat([self_branch, cross_branch], dim=-1)
            fused_output = self.fusion_dense(fused_output)
            fused_output = self.dropout(fused_output)
            fused_output = self.LayerNorm(fused_output)
        elif self.fusion_type == 'gate':
            concat_output = torch.cat([self_branch, cross_branch], dim=-1)
            gate = torch.sigmoid(self.gate_layer(concat_output))
            fused_output = gate * self_branch + (1 - gate) * cross_branch
            fused_output = self.dropout(fused_output)
            fused_output = self.LayerNorm(fused_output)
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        if output_all_encoded_layers: # return_latent_info: 表示计算align loss
            # 拿序列最后一个位置作为query
            query_repr = fused_output[:, -1, :]  # (batch_size, hidden_size)
            # 平均所有层attention
            attention_scores_mean = torch.stack(attention_list, dim=0).mean(dim=0)  # (batch_size, num_heads, seq_len, memory_len)
            attention_scores_mean = attention_scores_mean.mean(dim=1)  # 跨head取均值 (batch_size, seq_len, memory_len)
            attention_scores_mean = attention_scores_mean[:, -1, :]  # 只取最后一个位置query的分数

            return fused_output, memory_tensor, query_repr, attention_scores_mean # memory_tensor(1000,64), query_repr(b,64), attention_scores_mean(b,1000)
        else:
            return fused_output
        
        # fused_output = self.dropout(fused_output)
        #fused_output = self.LayerNorm(fused_output) # 没有+原来的tensor

        # if output_all_encoded_layers:
        #     return [fused_output]
        # else:
        # return fused_output



class LateFusionEncoder3(nn.Module):
    def __init__(self, args, latent_init_tensor, trainable_latent=False, fusion_type='add'):
        super(LateFusionEncoder, self).__init__()

        # 控制 latent embedding 是否可训练
        if trainable_latent:
            self.memory_tensor = nn.Parameter(latent_init_tensor.clone()) # (1000, 768)
            with open(args.log_file, "a") as f:
                f.write("latent embedding 可训练" + "\n") # 参数写入log文件
        else:
            self.register_buffer('memory_tensor', latent_init_tensor.clone())
            with open(args.log_file, "a") as f:
                f.write("latent embedding 不可训练" + "\n") # 参数写入log文件

        # ✅ MemoryProjector：将 latent 768 -> hidden_size=64
        self.memory_projector = MemoryProjector(latent_init_tensor.size(1), args.hidden_size) # 768--64
        self.sequential_projector = MemoryProjector(args.hidden_size, args.hidden_size) # 64--64

        # Self-attention分支: 学习用户序列表示
        self.self_layers = nn.ModuleList([
            Layer(args) for _ in range(args.num_hidden_layers)
        ])

        # Cross-attention分支
        self.cross_layers = nn.ModuleList([
            CrossAttentionLayer(args) for _ in range(args.num_hidden_layers)
        ])

        # 融合方式
        self.fusion_type = fusion_type.lower()
        if self.fusion_type == 'concat':
            self.fusion_dense = nn.Linear(args.hidden_size * 2, args.hidden_size)
        elif self.fusion_type == 'gate':
            self.gate_layer = nn.Linear(args.hidden_size * 2, 1)

        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12)



    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        
        memory_tensor = self.memory_tensor
        # ✅ 先 projector 到 (1000, 64)
        memory_tensor = self.memory_projector(memory_tensor)

        # 1. Self-attention branch
        self_branch = hidden_states
        for layer_module in self.self_layers:
            self_branch = layer_module(self_branch, attention_mask) # 返回(b,l,c)

        self_branch = self.sequential_projector(self_branch)
        '''这一步通过 cross-attention 将来自 review 的 intent basis 融入用户行为序列中的每个 token，赋予 token 表征更强的语义、解释和意图表达能力，为提升推荐性能和可解释性奠定基础。'''
        # 每个位置 token 与所有 intent 做相似度
        # 输出: intent_scores ∈ ℝ^{b × l × n_intent}
        intent_scores = torch.matmul(self_branch, memory_tensor.T)  # (b, l, 1000)
        intent_scores_soft = torch.softmax(intent_scores, dim=-1)  # (b, l, 1000)
        # 选取每个位置的 top-k intent（可选，便于 sparsity 和解释）
        #topk_values, topk_indices = torch.topk(intent_scores, k=5, dim=-1)  # (b, l, K)
        token_intent_repr = torch.matmul(intent_scores_soft, memory_tensor)  # (b, l, d): 将每个 token 的 intent 激活映射为一个加权表示

        '''三种方法'''
        user_intent_repr = token_intent_repr.mean(dim=1)  # (b, d)：序列中每个行为的重要性是相同的，平均可以代表整体意图。

        # # 生成位置权重（线性或指数递增）:考虑了时间的重要性，比如用户最近的点击更能代表当前意图
        # l = self_branch.size(1)
        # positions = torch.arange(l).float().to(self.args.gpu_id)  # (l,)
        # weights = positions / positions.sum()  # normalize
        # user_intent_repr = torch.sum(token_intent_repr * weights.view(1, l, 1), dim=1)  # (b, d)

        # # 假设：token_intent_repr ∈ (b, l, d):网络可以自己决定“哪些 token 更能代表 intent”，可动态适应不同用户行为模式
        # # learnable query vector: (1, 1, d):引入一个可学习的 query 向量，对每个 token 的 intent 表示打分，然后加权
        # from math import sqrt
        # query = nn.Parameter(torch.randn(1, 1, d))  # 可学习
        # scores = torch.matmul(token_intent_repr, query.transpose(-1, -2)) / sqrt(d)  # (b, l, 1)
        # weights = torch.softmax(scores, dim=1)  # (b, l, 1)
        # user_intent_repr = torch.sum(weights * token_intent_repr, dim=1)  # (b, d)




        # 2. Cross-attention branch
        cross_branch = hidden_states
        attention_list = []
        for layer_module in self.cross_layers:
            cross_branch, attention_probs = layer_module(cross_branch, memory_tensor)
            attention_list.append(attention_probs)

        # 3. 融合
        if self.fusion_type == 'add':
            fused_output = self_branch + cross_branch
            fused_output = self.LayerNorm(fused_output)
        elif self.fusion_type == 'concat':
            fused_output = torch.cat([self_branch, cross_branch], dim=-1)
            fused_output = self.fusion_dense(fused_output)
            fused_output = self.dropout(fused_output)
            fused_output = self.LayerNorm(fused_output)
        elif self.fusion_type == 'gate':
            concat_output = torch.cat([self_branch, cross_branch], dim=-1)
            gate = torch.sigmoid(self.gate_layer(concat_output))
            fused_output = gate * self_branch + (1 - gate) * cross_branch
            fused_output = self.dropout(fused_output)
            fused_output = self.LayerNorm(fused_output)
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")
        
        if output_all_encoded_layers: # return_latent_info: 表示计算align loss
            # 拿序列最后一个位置作为query
            query_repr = fused_output[:, -1, :]  # (batch_size, hidden_size)
            # 平均所有层attention
            attention_scores_mean = torch.stack(attention_list, dim=0).mean(dim=0)  # (batch_size, num_heads, seq_len, memory_len)
            attention_scores_mean = attention_scores_mean.mean(dim=1)  # 跨head取均值 (batch_size, seq_len, memory_len)
            attention_scores_mean = attention_scores_mean[:, -1, :]  # 只取最后一个位置query的分数

            return fused_output, memory_tensor, query_repr, attention_scores_mean
        else:
            return fused_output

        # fused_output = self.LayerNorm(fused_output)

        # if output_all_encoded_layers:
        #     return [fused_output]
        # else:
        # return fused_output


