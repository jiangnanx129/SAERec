import copy
import torch.nn as nn
from modules import Layer, CrossAttentionLayer

class Encoder(nn.Module):
    def __init__(self, args, intent_dim=None):
        super(Encoder, self).__init__()
        # 1. 正常 self-attention layer
        self.self_layers = nn.ModuleList([
            copy.deepcopy(Layer(args)) for _ in range(args.num_hidden_layers)
        ])
        
        # 2. intent cross-attention layer
        self.cross_layers = nn.ModuleList([
            copy.deepcopy(CrossAttentionLayer(args, intent_dim)) for _ in range(args.num_hidden_layers)
        ])

    def forward(self, hidden_states, attention_mask, intent_tensor, output_all_encoded_layers=True):
        all_encoder_layers = []

        for layer_module, cross_module in zip(self.self_layers, self.cross_layers):
            # 1. 先做 self-attention (SASRec标准操作)
            hidden_states = layer_module(hidden_states, attention_mask)
            
            # 2. 再做 cross-attention (query: 用户序列, key/value: intent memory)
            hidden_states = cross_module(hidden_states, intent_tensor)

            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)

        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)

        return all_encoder_layers



import copy
import torch
import torch.nn as nn

class ParallelEncoderLayer(nn.Module):
    def __init__(self, args, memory_dim=None, fusion_type='add'):
        super(ParallelEncoderLayer, self).__init__()
        self.self_attention_layer = Layer(args)  # 自注意
        self.cross_attention_layer = CrossAttentionLayer(args, memory_dim)  # intent跨注意

        self.fusion_type = fusion_type.lower()
        if self.fusion_type == 'concat':
            # 如果是拼接，需要一个融合映射回来 hidden_size
            self.fusion_dense = nn.Linear(args.hidden_size * 2, args.hidden_size)
        elif self.fusion_type == 'gate':
            # 如果是门控，学习一个门控权重
            self.gate_layer = nn.Linear(args.hidden_size * 2, 1)
        
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12)

    def forward(self, hidden_states, attention_mask, memory_tensor, memory_mask):
        # 1. Self-attention通道
        self_output = self.self_attention_layer(hidden_states, attention_mask)  # (b, l, h)

        # 2. Cross-attention通道
        cross_output = self.cross_attention_layer(hidden_states, memory_tensor, memory_mask)  # (b, l, h)

        # 3. 融合
        if self.fusion_type == 'add':
            fused_output = self_output + cross_output
        elif self.fusion_type == 'concat':
            fused_output = torch.cat([self_output, cross_output], dim=-1)  # (b, l, 2h)
            fused_output = self.fusion_dense(fused_output)  # (b, l, h)
        elif self.fusion_type == 'gate':
            concat_output = torch.cat([self_output, cross_output], dim=-1)  # (b, l, 2h)
            gate = torch.sigmoid(self.gate_layer(concat_output))  # (b, l, 1)
            fused_output = gate * self_output + (1 - gate) * cross_output
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        # 4. 最后一个 LayerNorm（保证稳定训练）
        fused_output = self.LayerNorm(fused_output)

        return fused_output



class ParallelEncoder(nn.Module):
    def __init__(self, args, memory_dim=None, fusion_type='add'):
        super(ParallelEncoder, self).__init__()
        self.layers = nn.ModuleList([
            ParallelEncoderLayer(args, memory_dim, fusion_type)
            for _ in range(args.num_hidden_layers)
        ])

    def forward(self, hidden_states, attention_mask, memory_tensor, memory_mask, output_all_encoded_layers=True):
        all_encoder_layers = []

        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states, attention_mask, memory_tensor, memory_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)

        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)

        return all_encoder_layers





import copy
import torch
import torch.nn as nn

class ParallelEncoderLayer(nn.Module):
    def __init__(self, args, memory_dim=None, fusion_type='add'):
        super(ParallelEncoderLayer, self).__init__()
        self.self_attention_layer = Layer(args)  # Self-Attention通道
        self.cross_attention_layer = CrossAttentionLayer(args, memory_dim)  # Cross-Attention通道

        self.fusion_type = fusion_type.lower()
        if self.fusion_type == 'concat':
            self.fusion_dense = nn.Linear(args.hidden_size * 2, args.hidden_size)
        elif self.fusion_type == 'gate':
            self.gate_layer = nn.Linear(args.hidden_size * 2, 1)
        
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12)

    def forward(self, hidden_states, attention_mask, memory_tensor, memory_mask):
        # 两条通道分别做 attention
        self_output = self.self_attention_layer(hidden_states, attention_mask)
        cross_output = self.cross_attention_layer(hidden_states, memory_tensor, memory_mask)

        # 融合策略
        if self.fusion_type == 'add':
            fused_output = self_output + cross_output
        elif self.fusion_type == 'concat':
            fused_output = torch.cat([self_output, cross_output], dim=-1)
            fused_output = self.fusion_dense(fused_output)
        elif self.fusion_type == 'gate':
            concat_output = torch.cat([self_output, cross_output], dim=-1)
            gate = torch.sigmoid(self.gate_layer(concat_output))
            fused_output = gate * self_output + (1 - gate) * cross_output
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        fused_output = self.LayerNorm(fused_output)

        return fused_output


'''Self-Attention 路 → Cross-Attention路 → 融合 → LayerNorm → 传递到下一层'''
class ParallelEncoder(nn.Module):
    def __init__(self, args, memory_dim=None, fusion_type='add'):
        super(ParallelEncoder, self).__init__()
        self.num_hidden_layers = args.num_hidden_layers
        self.layers = nn.ModuleList([
            ParallelEncoderLayer(args, memory_dim, fusion_type)
            for _ in range(self.num_hidden_layers)
        ])

    def forward(self, hidden_states, attention_mask, memory_tensor, memory_mask, output_all_encoded_layers=True):
        all_encoder_layers = []

        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states, attention_mask, memory_tensor, memory_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)

        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)

        return all_encoder_layers





'''
trainable_latent=True，latent embedding是nn.Parameter，可以随着推荐模型训练更新；

trainable_latent=False，latent embedding是register_buffer，完全固定，只forward，不反向传播；

gating是每层单独的、极小的开销（memory_dim个参数而已）；

gating乘法使用broadcast，不复制，不新开张量，非常高效；

memory_tensor只在初始化时clone一次，forward过程中不重复clone或detach；

ParallelEncoderLayer结构清晰，融合方式可选（add/concat/gate）。
'''
import copy
import torch
import torch.nn as nn

class ParallelEncoderLayer(nn.Module):
    def __init__(self, args, memory_dim=None, fusion_type='add'):
        super().__init__()
        self.self_attention_layer = Layer(args)
        self.cross_attention_layer = CrossAttentionLayer(args, memory_dim)

        self.fusion_type = fusion_type.lower()
        if self.fusion_type == 'concat':
            self.fusion_dense = nn.Linear(args.hidden_size * 2, args.hidden_size)
        elif self.fusion_type == 'gate':
            self.gate_layer = nn.Linear(args.hidden_size * 2, 1)

        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12)

        # Gating：memory_dim个可学习参数，每层一个独立gate
        self.memory_gate = nn.Parameter(torch.ones(memory_dim))

    def forward(self, hidden_states, attention_mask, memory_tensor, memory_mask):
        # 1. 高效gating：直接elementwise乘，不复制memory
        gated_memory = memory_tensor * self.memory_gate.unsqueeze(0).unsqueeze(0)  # (1, 1, memory_dim)广播

        # 2. 两条通道
        self_output = self.self_attention_layer(hidden_states, attention_mask)
        cross_output = self.cross_attention_layer(hidden_states, gated_memory, memory_mask)

        # 3. 融合
        if self.fusion_type == 'add':
            fused_output = self_output + cross_output
        elif self.fusion_type == 'concat':
            fused_output = torch.cat([self_output, cross_output], dim=-1)
            fused_output = self.fusion_dense(fused_output)
        elif self.fusion_type == 'gate':
            concat_output = torch.cat([self_output, cross_output], dim=-1)
            gate = torch.sigmoid(self.gate_layer(concat_output))
            fused_output = gate * self_output + (1 - gate) * cross_output
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        fused_output = self.LayerNorm(fused_output)

        return fused_output

class ParallelEncoder(nn.Module):
    def __init__(self, args, latent_init_tensor, trainable_latent=False, fusion_type='add'):
        super().__init__()
        self.num_hidden_layers = args.num_hidden_layers

        # 控制latent embedding是否可训练
        if trainable_latent:
            self.memory_tensor = nn.Parameter(latent_init_tensor.clone())  # 可学习
        else:
            self.register_buffer('memory_tensor', latent_init_tensor.clone())  # 固定，不更新

        self.layers = nn.ModuleList([
            ParallelEncoderLayer(args, memory_dim=latent_init_tensor.size(-1), fusion_type=fusion_type)
            for _ in range(self.num_hidden_layers)
        ])

    def forward(self, hidden_states, attention_mask, memory_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        memory_tensor = self.memory_tensor  # 保持memory高效传递，不在每一层复制

        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states, attention_mask, memory_tensor, memory_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)

        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)

        return all_encoder_layers




'''
Input hidden_states
     ↓
(1) 连续跑 num_hidden_layers次 Self-Attention → 得到 self_branch
     ↓
(2) 连续跑 num_hidden_layers次 Cross-Attention → 得到 cross_branch
     ↓
(3) 融合 self_branch 和 cross_branch
'''
import torch
import torch.nn as nn
import copy

class LateFusionEncoder(nn.Module):
    def __init__(self, args, latent_init_tensor, trainable_latent=False, fusion_type='add'):
        super().__init__()
        self.num_hidden_layers = args.num_hidden_layers
        hidden_size = args.hidden_size
        memory_dim = latent_init_tensor.size(-1)

        # 控制 latent embedding 是否可训练
        if trainable_latent:
            self.memory_tensor = nn.Parameter(latent_init_tensor.clone())
        else:
            self.register_buffer('memory_tensor', latent_init_tensor.clone())

        # Self-attention分支
        self.self_layers = nn.ModuleList([
            copy.deepcopy(Layer(args)) for _ in range(self.num_hidden_layers)
        ])

        # Cross-attention分支
        self.cross_layers = nn.ModuleList([
            copy.deepcopy(CrossAttentionLayer(args, memory_dim)) for _ in range(self.num_hidden_layers)
        ])

        # 融合方式
        self.fusion_type = fusion_type.lower()
        if self.fusion_type == 'concat':
            self.fusion_dense = nn.Linear(hidden_size * 2, hidden_size)
        elif self.fusion_type == 'gate':
            self.gate_layer = nn.Linear(hidden_size * 2, 1)

        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        memory_tensor = self.memory_tensor  # 固定或可学习memory

        # 1. Self-attention branch
        self_branch = hidden_states
        for layer_module in self.self_layers:
            self_branch = layer_module(self_branch, attention_mask)

        # 2. Cross-attention branch
        cross_branch = hidden_states
        for layer_module in self.cross_layers:
            cross_branch = layer_module(cross_branch, memory_tensor)

        # 3. 融合
        if self.fusion_type == 'add':
            fused_output = self_branch + cross_branch
        elif self.fusion_type == 'concat':
            fused_output = torch.cat([self_branch, cross_branch], dim=-1)
            fused_output = self.fusion_dense(fused_output)
        elif self.fusion_type == 'gate':
            concat_output = torch.cat([self_branch, cross_branch], dim=-1)
            gate = torch.sigmoid(self.gate_layer(concat_output))
            fused_output = gate * self_branch + (1 - gate) * cross_branch
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        fused_output = self.LayerNorm(fused_output)

        if output_all_encoded_layers:
            return [fused_output]
        else:
            return fused_output
