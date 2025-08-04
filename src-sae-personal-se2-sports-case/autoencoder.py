import os
import numpy as np
import torch as tc


'''适用于衡量归一化误差，特别是在 𝑦 值范围可能有较大波动的情况下，归一化确保了结果的鲁棒性。'''
def normalized_l2(x, y):
    return (((x - y) ** 2).mean(dim=1) / (y ** 2).mean(dim=1)).mean()

'''对输入张量 x 实现 层归一化 (Layer Normalization)，并返回归一化后的结果及相关统计数据（均值和标准差）'''
def layer_norm(x, eps=1e-8):
    avg = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True) + eps
    return (x - avg) / std, {"mu": avg, "std": std}


'''
生成一个与输入张量形状相同的 掩码张量，在每个样本的最后一维中，仅保留最大 k 个元素的位置，其他位置为零。
适用于选择性激活神经网络的部分权重或节点。
常用于稀疏化操作，或在 Attention 模块中保留部分显著值。
'''
def MaskTopK(x, k):
    val, idx = tc.topk(x, k=k, dim=-1) # 取topk及其索引，沿着-1维度
    return tc.zeros_like(x).scatter_(-1, idx ,1) # 生成一个与输入张量 x 形状相同的二值掩码，其中Top-K 最大值对应的位置置为 1，其他位置为 0。


'''
SAE: 无监督学习模型，旨在学习输入数据的稀疏表示，同时重构输入
'''
class SparseAutoencoder(tc.nn.Module):
    def __init__(self, d_inp, d_hide, device="cuda"): # 输入维度，隐藏层维度
        super().__init__()
        self.monitoring = False
        self.do_edit = False
        self.dims = (d_inp, d_hide) # ---> 768，2^15
        self.lamda = 0.1 # 用于稀疏正则化的超参数
        self.usingmean = False
        self.mask = tc.nn.Parameter(tc.ones(d_hide), requires_grad=False) # (2^15) 隐藏层的掩码，用于控制哪些单元激活
        weight = tc.nn.init.kaiming_normal_(tc.zeros(d_inp, d_hide),
                                            mode="fan_out", nonlinearity="relu")
        self.W_enc = tc.nn.Parameter(weight, requires_grad=True) # (768,2^15)
        self.b_enc = tc.nn.Parameter(tc.zeros(d_hide)) # 2^15
        self.b_dec = tc.nn.Parameter(tc.zeros(d_inp)) # 768
        self.freq = tc.zeros(d_hide).to(device) # 2^15 记录隐藏层激活的频率，用于稀疏性控制

        self.edit_set1 = None
        self.edit_set2 = None
        self.edit_weight1 = None
        self.edit_weight2 = None
        self.enf_set1 = None
        self.enf_value1 = None
        self.to(device)
    
    def reset_frequency(self):
        self.freq = tc.zeros_like(self.freq)

    @property
    def W_dec(self):
        return self.W_enc.T # (2^15, 768)

    def _decode(self, h):
        return h @ self.W_dec + self.b_dec # (l,2^15)@(2^15, 768)--->(l,768)+(768)

    def encode(self, x): # x为输入数据，应该是(b,l,d_inp) 实际(l,768)
        self.aux_loss = 0.0
        x = x - self.b_dec # (l,768), (768)
        h = x @ self.W_enc + self.b_enc # (l,768) @ (768,2^15)--->(l,2^15)+(2^15)
        a = tc.relu(h) # 负值全0，正值保持不变--->(l,2^15)
        with tc.no_grad(): # 记录激活值的频率 ∣∣𝑎∣∣_0 ，这是一种稀疏度统计方法。这段代码块中不计算梯度
            '''
            freq 是一个一维向量，长度为隐藏层维度 d_hide，记录了每个神经元的累积激活次数
            对于每一列（总共 2^15 列），统计 768 个元素中非零元素的数量。
            返回 (2^15,)
            '''
            self.freq += a.reshape(-1, a.shape[-1]).norm(p=0, dim=0) # (l,d_hidden), 计算 𝐿0-范数:即统计张量中每列的非零元素个数。相当于统计每个隐藏层神经元的非零激活次数。
        if self.alpha > 0:
            self.recons_h = self._decode(a) # (l,768) 解码器对激活值 𝑎 进行解码，得到重构的输入 recons_ℎ，这是标准的解码结果
            '''
            -self.freq: 选择激活频率最低的1024个隐藏单元
            h-(l,2^15), mask-(2^15), 每一行的元素都与向量 a 的对应元素相乘。
            '''
            aux_recons_h = self._decode(h * MaskTopK(-self.freq, 1024)) # 利用激活频率 self.freq，稀疏化隐藏层激活 ℎ. 对负频率应用 Top-K 掩码，选出激活频率最低的 1024 个神经元。将这些神经元保留，其余神经元置为零
            self.aux_loss = normalized_l2(aux_recons_h, x - self.recons_h) # 衡量稀疏化后重构结果与原始重构误差之间的差异。目标是让稀疏化后的解码结果尽量贴近原始解码误差，从而增强稀疏表示的有效性。
        return a # 已经负值置0

    def decode(self, h):
        return h @ self.W_dec + self.b_dec

    def forward(self, x):
        self.recons_h = None
        self.aux_loss = 0
        h = self.encode(x)
        if self.do_edit:
            h = h * self.mask.unsqueeze(0)
        x_ = self.decode(h)
        return h, x_


    def generate(self, X):
        assert len(X.shape) == 3 # X 是一个三维张量，（b,l,d_hidden）
        self.recons_h = None
        self.aux_loss = 0
        h = self.encode(X[:, -1]) # (b,d_hidden）) 只对最后一个时刻编码：因为它提供了上下文信息，是当前时间点的信息总结，编码整个序列的话会重复编码
        if self.do_edit:
            h = h * self.mask.unsqueeze(0)
        X = tc.cat([X[:, :-1], self.decode(h).unsqueeze(1)], dim=1)
        return X

    def compute_loss(self, inputs, lamda=0.1):
        actvs, recons = self(inputs) # 调用 self(inputs) 计算隐藏层激活值 h 和重构结果 𝑥‘
        self.actvs = actvs
        self.l2 = normalized_l2(recons, inputs) # 衡量重构误差
        self.l1 = (actvs.abs().sum(dim=1) / inputs.norm(dim=1)).mean() # 计算隐藏层激活值的 L1 范数正则化，衡量隐藏层激活的稀疏性
        self.l0 = actvs.norm(dim=-1, p=0).mean() # 计算隐藏层激活值的 L0 范数，统计隐藏层中非零激活的平均数量
        self.ttl = self.l2 + self.lamda * self.l1  
        return self.ttl, self.l2, self.l1, self.l0






    def set_edit(self, masks, op, mag=1):
        assert op in {"turnoff", "enhance"}
        magnitude = -mag if op == "turnoff" else mag
        print("Totally %d features are %s with magnitude %s." % (len(masks), op, magnitude))
        group1 = list(masks)
        if len(group1) > 0:
            edit_set, edit_weight, edit_bias, edit_group = [], [], [], []
            if self.edit_set1 is not None:
                edit_set.append(self.edit_set1)
                edit_weight.append(self.edit_weight1)
                edit_bias.append(self.edit_bias1)
                edit_group.append(self.edit_group1)
            edit_set.append(self.W_enc[:, group1]) # get指定性质的latent
            edit_weight.append(tc.zeros(len(group1)) + magnitude)
            edit_bias.append(self.b_enc[group1])
            edit_group.append(tc.tensor(group1).long())
            self.edit_set1 = tc.hstack(edit_set)
            self.edit_weight1 = tc.hstack(edit_weight)
            self.edit_bias1 = tc.hstack(edit_bias)
            self.edit_group1 = tc.hstack(edit_group)




    def enforce_actv(self, masks, magnitude=1, keep=True):
        print("Totally %d features are activated with magnitude %s." % (len(masks), magnitude))
        group1 = list(masks)
        if len(group1) > 0:
            edit_set, edit_bias = [], []
            if self.enf_set1 is not None and keep:
                edit_set.append(self.enf_set1)
                edit_bias.append(self.enf_bias1)
            edit_bias.append(self.b_enc[group1])
            edit_set.append(self.W_enc[:, group1]) # （768,2^15）-->(768,)
            self.enf_bias1 = tc.hstack(edit_bias)
            self.enf_set1 = tc.hstack(edit_set)#.mean(axis=1) # (l, len(mask))
            self.enf_weight1 = magnitude




    def edit_generate(self, x):
        print("要edit的数据维度：", x.shape)
        assert len(x.shape) == 3
        x = x.float()
        z = x 
        self.gen_step += 1
        if self.edit_set1 is not None:
            h1 = (x - self.b_dec) @ self.edit_set1 + self.edit_bias1 # 基于指定的性质的latent的表示
            h1 = tc.relu(h1) * self.edit_weight1.to(h1.device)  # 强度 可正可负 turnoff就是-mag，enhance就是mag
            z = z + h1 @ self.edit_set1.T # 如果
        if self.enf_set1 is not None:
            if self.enf_weight1 >= 1.:
                print("先删除后增强")
                h = (x - self.b_dec) @ self.enf_set1 + self.enf_bias1 # (768, l_m)
                z = z - tc.relu(h) @ self.enf_set1.T.to(h.device)
                z = z + self.enf_set1.mean(axis=1) * self.enf_weight1
            else:
                print("编辑：I am here! 平均增强")
                #z = z + (1. - self.enf_weight1) * self.enf_set1.mean(axis=1) 
                z = self.enf_weight1 * z +\
                    (1. - self.enf_weight1) * self.enf_set1.mean(axis=1) # （l,len(mask)）, 每一行的平均值
        x = z
        return x.bfloat16() 




    # 不是平均增强，而是传入的value
    def enforce_actv2(self, values, masks, magnitude=1, keep=True):
        print("Totally %d features are activated with magnitude %s." % (len(masks), magnitude))
        group1 = list(masks)
        if len(group1) > 0:
            edit_set, edit_bias = [], []
            if self.enf_set1 is not None and keep:
                edit_set.append(self.enf_set1)
                edit_bias.append(self.enf_bias1)
            edit_bias.append(self.b_enc[group1])
            edit_set.append(self.W_enc[:, group1])
            self.enf_bias1 = tc.hstack(edit_bias)
            self.enf_set1 = tc.hstack(edit_set)#.mean(axis=1) # (768*len(mask),)
            self.enf_weight1 = magnitude
            self.enf_value1 = tc.tensor(values).float()  # values 是你传入的强度list，长度为 a



    # 加权平均而不是mean
    def edit_generate2(self, x):
        print("要edit的数据维度：", x.shape)
        assert len(x.shape) == 3
        x = x.float()
        z = x 
        self.gen_step += 1
        if self.edit_set1 is not None:
            h1 = (x - self.b_dec) @ self.edit_set1 + self.edit_bias1 # 基于指定的性质的latent的表示
            h1 = tc.relu(h1) * self.edit_weight1.to(h1.device)  # 强度 可正可负 turnoff就是-mag，enhance就是mag
            z = z + h1 @ self.edit_set1.T # 如果
        if self.enf_set1 is not None:
            if self.enf_weight1 >= 1.:
                # print("原始：", z,z.shape)
                h = (x - self.b_dec) @ self.enf_set1 + self.enf_bias1
                z = z - tc.relu(h) @ self.enf_set1.T.to(h.device)
                z = z + self.enf_set1.mean(axis=1) * self.enf_weight1
                # print("后面：", z,z.shape)
                # pirnt()
            else:
                print("编辑：I am here 加权的增强! 归一化value")
                print("self.enf_set1:", self.enf_set1.shape)
                # print("z:", z, z.shape)
                # (1. - self.enf_weight1) * self.enf_set1.mean(axis=1)
                # ===== 加权平均代替 mean(axis=1) =====
                #weighted_mean = self.enf_set1 @ self.enf_value1.to(self.enf_set1.device)  # shape: (l,)
                # 归一化+加权平均
                weights = self.enf_value1 / self.enf_value1.sum()
                weighted_mean = self.enf_set1 @ weights.to(self.enf_set1.device)  # shape: (l,)
                z = self.enf_weight1 * z + (1. - self.enf_weight1) * weighted_mean
                # print("weighted_mean:", weighted_mean, weighted_mean.shape, weights.shape)
                # z = z + (1. - self.enf_weight1) * weighted_mean
                # print("z2:", z,z.shape)
                # pirnt()
                
        x = z
        return x.bfloat16()
    

    # 加权：z: 增加index数据的影响，
    def edit_generate3(self, x):
        print("要edit的数据维度：", x.shape)
        assert len(x.shape) == 3
        x = x.float()
        z = x 
        self.gen_step += 1
        if self.edit_set1 is not None:
            h1 = (x - self.b_dec) @ self.edit_set1 + self.edit_bias1 # 基于指定的性质的latent的表示
            h1 = tc.relu(h1) * self.edit_weight1.to(h1.device)  # 强度 可正可负 turnoff就是-mag，enhance就是mag
            z = z + h1 @ self.edit_set1.T # 如果
        if self.enf_set1 is not None:
            weights = self.enf_value1 / self.enf_value1.sum()
            weighted_mean = self.enf_set1 @ weights.to(self.enf_set1.device)  # shape: (l,)
            z = z*0.8 + self.enf_weight1 * weighted_mean
            # if self.enf_weight1 >= 1.:
            #     h = (x - self.b_dec) @ self.enf_set1 + self.enf_bias1
            #     z = z - tc.relu(h) @ self.enf_set1.T.to(h.device)
            #     z = z + self.enf_set1.mean(axis=1) * self.enf_weight1
            # else:
            #     print("编辑：I am here 加权的增强! 归一化value")
            #     print("self.enf_set1:", self.enf_set1.shape)
            #     print("z:", z, z.shape)
            #     # ===== 加权平均代替 mean(axis=1) =====
            #     #weighted_mean = self.enf_set1 @ self.enf_value1.to(self.enf_set1.device)  # shape: (l,)
            #     # 归一化+加权平均
            #     weights = self.enf_value1 / self.enf_value1.sum()
            #     weighted_mean = self.enf_set1 @ weights.to(self.enf_set1.device)  # shape: (l,)
            #     # z = self.enf_weight1 * z + (1. - self.enf_weight1) * weighted_mean
            #     print("weighted_mean:", weighted_mean, weighted_mean.shape, weights.shape)
            #     z = z + (1. - self.enf_weight1) * weighted_mean
            #     print("z2:", z,z.shape)
            #     pirnt()
                
        x = z
        return x.bfloat16()
    
    


    '''针对logitscore'''
    def enforce_actv_logitscore(self, latent_i, magnitude=1, keep=True):
        print("Totally %d features are activated with magnitude %s." % (1, magnitude))
        enhance_vec = self.W_enc[:, latent_i]  # (768,), 单个增强的latent
        self.enf_set1 = enhance_vec
        self.enf_weight1 = magnitude

    def edit_generate_logitscore(self, x): # x是输入的hidden state （b,l,768）
        print("要edit的数据维度：", x.shape)
        assert len(x.shape) == 3
        x = x.float()
        z = x 
        enhance_vec_i = self.enf_set1.unsqueeze(0).expand(x.shape[1], -1).unsqueeze(0) # data_x(b,l,c), (1,768)-->(l,768)
        z_aug = self.enf_weight1 * z + (1 - self.enf_weight1) * enhance_vec_i
        print(enhance_vec_i.shape, z.shape, self.enf_set1.shape)
        
        return z_aug.bfloat16()




    @classmethod
    def from_disk(cls, fpath, device="cuda"):
        print("Loading SAE from %s." % fpath)
        states = tc.load(fpath)
        model = cls(**states["config"], device="cpu") # 根据保存的配置初始化模型对象，暂时设置为cpu设备
        model.load_state_dict(states['weight'], strict=True) # 将保存的权重加载到模型中。True确保模型的参数结构与加载的权重完全匹配
        return model.to(device)

    def dump_disk(self, fpath):
        os.makedirs(os.path.split(fpath)[0], exist_ok=True)
        tc.save({"weight": self.state_dict(), 
                 "config": {"d_inp": self.dims[0], 
                            "d_hide": self.dims[1],}
                            },
                 fpath)
        print("SAE is dumped at %s." % fpath)




class TopKSAE(SparseAutoencoder):
    def __init__(self, d_inp, d_hide, topK=20, device="cuda"):
        super().__init__(d_inp, d_hide, device)
        self.topk = topK
        print("检查topk：", self.topk)
        self.lamda = 0
        self.MaskTopK = True
        self.disabled = False
        self.logging = True
        self.epsilon = 1e-6
        self.recons_h = None
        self.freq = tc.zeros(d_hide).to(device)
    
    def reset_frequency(self):
        self.freq = tc.zeros_like(self.freq)

    def _encode(self, x):
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def _decode(self, h):
        return h @ self.W_dec + self.b_dec



    # 计算激活状态
    def caculate_act(self, x):  # 输入 shape: (b, 2^15)
        print(f"Input Shape: {x.shape}")  # 打印输入 x 的形状
        
        # 计算每行（样本）的统计信息
        feature_min = x.min(dim=1).values  # 每行最小值 (shape: [b])
        feature_max = x.max(dim=1).values  # 每行最大值 (shape: [b])
        feature_mean = x.mean(dim=1)  # 每行均值 (shape: [b])
        feature_std = x.std(dim=1, unbiased=False)  # 每行标准差 (shape: [b])

        # 计算每行（样本）中大于 0 的数据个数
        positive_count = (x > 0).sum(dim=1)  # 统计每行中大于 0 的元素个数 (shape: [b])

        # 打印统计信息
        print("Feature Min:", feature_min)
        print("Feature Max:", feature_max)
        print("Feature Mean:", feature_mean)
        print("Feature Std Dev:", feature_std)
        print("Positive Count:", positive_count)

    # 假设 input_embedding 是 (batch_size, seq_length, hidden_dim) 形状的词向量
    def check_embedding_stats(self, embedding):
        min_val = embedding.min().item()
        max_val = embedding.max().item()
        mean_val = embedding.mean().item()
        std_val = embedding.std().item()

        print(f"Embedding Stats:")
        print(f"  Min: {min_val:.4f}")
        print(f"  Max: {max_val:.4f}")
        print(f"  Mean: {mean_val:.4f}")
        print(f"  Std: {std_val:.4f}")

        # 归一化建议
        if abs(mean_val) > 5 or std_val > 10 or min_val < -100 or max_val > 100:
            print("⚠️ 建议归一化！可能会影响 Transformer 训练稳定性。")
        else:
            print("✅ 不需要归一化，数据分布正常。")



    def encode(self, x):
        h = self._encode(x) # h--(l,2^15)
        # print("-----------原始hidden state 768--------------")
        # print(x)
        # self.caculate_act(x)
        # print("-----------所有激活值（2^15）数值--------------")
        # print(h)
        # self.caculate_act(h) 

        mask = MaskTopK(h, self.topk) # (l,2^15)沿着2^15找最大，每一行找2^15中的topk。选取每个样本中最大的 topk 个激活值，并生成一个掩码矩阵（0 或 1）
        with tc.no_grad():
            # 统计掩码中每个神经元的激活频率，存储在 self.freq 中
            self.freq += mask.reshape(-1, mask.shape[-1]).sum(axis=0).to(self.freq.device) # sae的2^15中的每个feature激活了多少输入向量 一个sae feature在l中的激活状况
        if self.alpha > 0: 
            self.recons_h = self._decode(h * mask) # 用稀疏化后的激活值解码，得到重构结果 recons_h
            aux_recons_h = self._decode(h * MaskTopK(-self.freq, 1024)) # 通过激活频率最低的 1024 个神经元进行辅助重构
            self.aux_loss = normalized_l2(aux_recons_h, x - self.recons_h)  # 计算稀疏化后的重构误差，作为辅助损失
        h = h * mask
        return tc.relu(h)
    

    def encode2(self, x):
        h = self._encode(x) # h--(l,2^15)
        return tc.relu(h)
    
    def decode(self, h):
        if self.recons_h is None:
            self.recons_h = self._decode(h)
        return self.recons_h
    
    # def forward(self, x):
    #     self.recons_h = None
    #     self.aux_loss = 0
    #     h = self.encode(x)
    #     if self.do_edit:
    #         h = h * self.mask.unsqueeze(0)
    #     x_ = self.decode(h)
    #     return h, x_


    def compute_review_act(self, inputs, lamda=0.1): # 输入的就是hidde state
        if len(inputs.shape) == 3:
            inputs = inputs.squeeze(0) # 如果输入是 3 维，去掉第 0 维（当第0维是大小是1）
        
        latent_review = self.encode2(inputs.float()) # (l c)
        '''取最后一个维度topk的激活的维度'''
        #latent_review_actindex = tc.topk(latent_review[-1], 15).indices.tolist() # (l,c)最后一层l中c的那些位置
        #self.latent_review_actindex = latent_review_actindex

        '''
        1. 归一化/不归一化
        2. 每列统一求均值，只对其中非0的位置求均值
        '''
        '''考虑所有维度的激活'''
        # 1. 找出被激活的维度
        activated_mask = (latent_review != 0).any(dim=0) # 保留dim1 返回true false列表（c,）
        activated_means = latent_review[:, activated_mask != 0].mean(axis=0) # 均值针对l而不是l中有数据的-可能是小于l
        dim_strength = dict((a,b.item()) for (a,b) in zip(tc.nonzero(activated_mask, as_tuple=True)[0].tolist(), activated_means))
    
        # 3. 排序并返回维度索引列表（按激活强度从大到小）,只返回了index，没有返回激活的强度，数值
        sorted_dims = sorted(dim_strength.keys(), key=lambda d: dim_strength[d], reverse=True)
        self.latent_review_actindex = dim_strength
        return self.latent_review_actindex
        

        '''只取最后一层'''
        # # 只取最后一层
        # last_layer = latent_review[-1]  # shape: (c,)

        # # 找出非零激活的维度
        # activated_mask = last_layer != 0
        # activated_dims = tc.nonzero(activated_mask, as_tuple=True)[0]

        # # 统计每个维度的激活强度（就是该维度的值）
        # dim_strength = {dim.item(): last_layer[dim].item() for dim in activated_dims} # 字典

        # # 按激活强度排序
        # sorted_dims = sorted(dim_strength.keys(), key=lambda d: dim_strength[d], reverse=True) # 根据字典的value排序index，index的list

        # self.latent_review_actindex = dim_strength
        # return self.latent_review_actindex
    
    '''找sequential数据的激活情况'''
    def compute_review_act_sequential(self, inputs, lamda=0.1): # 输入的就是hidde state
        if len(inputs.shape) == 3:
            inputs = inputs.squeeze(0) # 如果输入是 3 维，去掉第 0 维（当第0维是大小是1）
        
        latent_review = self.encode2(inputs.float()) # (l c)
        '''取最后一个维度topk的激活的维度'''
        #latent_review_actindex = tc.topk(latent_review[-1], 15).indices.tolist() # (l,c)最后一层l中c的那些位置
        #self.latent_review_actindex = latent_review_actindex

        '''
        1. 归一化/不归一化
        2. 每列统一求均值，只对其中非0的位置求均值
        '''
        '''考虑所有维度的激活'''
        # 1. 找出被激活的维度
        activated_mask = (latent_review != 0).any(dim=0) # 保留dim1 返回true false列表（c,）
        activated_means = latent_review[:, activated_mask != 0].mean(axis=0) # 均值针对l而不是l中有数据的-可能是小于l
        dim_strength = dict((a,b.item()) for (a,b) in zip(tc.nonzero(activated_mask, as_tuple=True)[0].tolist(), activated_means))
    
        # 3. 排序并返回维度索引列表（按激活强度从大到小）,只返回了index，没有返回激活的强度，数值
        sorted_dims = sorted(dim_strength.items(), key=lambda item: item[1], reverse=True)
        self.latent_review_actindex = sorted_dims
        return self.latent_review_actindex


    '''找sequential数据的激活情况'''
    def compute_review_act_sequential_last(self, inputs, lamda=0.1): # 输入的就是hidde state
        if len(inputs.shape) == 3:
            inputs = inputs.squeeze(0) # 如果输入是 3 维，去掉第 0 维（当第0维是大小是1）
        
        latent_review = self.encode2(inputs.float()) # (l c)
        '''取最后一个维度topk的激活的维度'''
        #latent_review_actindex = tc.topk(latent_review[-1], 15).indices.tolist() # (l,c)最后一层l中c的那些位置
        #self.latent_review_actindex = latent_review_actindex
        '''只取最后一层'''
        last_layer = latent_review[-1]  # shape: (c,)

        # 找出非零激活的维度
        activated_mask = last_layer != 0
        activated_dims = tc.nonzero(activated_mask, as_tuple=True)[0]

        # 统计每个维度的激活强度（就是该维度的值）
        dim_strength = {dim.item(): last_layer[dim].item() for dim in activated_dims} # 字典
        sorted_dims_with_values = sorted(dim_strength.items(), key=lambda d: d[1], reverse=True)

        # 按激活强度排序
        #sorted_dims = sorted(dim_strength.keys(), key=lambda d: dim_strength[d], reverse=True) # 根据字典的value排序index，index的list

        self.latent_review_actindex = sorted_dims_with_values
        return self.latent_review_actindex










    def compute_loss(self, inputs, lamda=0.1): # 输入的就是hidde state
        actvs, recons = self(inputs) # actvs: encoder的输出-(l,2*15)
        self.actvs = actvs
        self.l2 = normalized_l2(recons, inputs)
        self.l1 = (actvs.abs().sum(dim=1) / inputs.norm(dim=1)).mean() # actvs(l,2^15), input(l,768)-->(l,)/(l,)--->mean
        self.l0 = actvs.norm(dim=-1, p=0).mean() # 非0个数--(l,)
        self.ttl = self.l2 + self.alpha * self.aux_loss
        return self.ttl, self.l2, self.l1, self.l0
    
    def compute_loss_ab(self, inputs, a,b, lamda=0.1):
        actvs, recons = self(inputs) # actvs: encoder的输出-(l,2*15)
        self.actvs = actvs
        #print("ab长度：", len(inputs),a,b, len(inputs[a:b])) # 有时候len(inputs[a:b])=0
        self.l2 = normalized_l2(recons[a:b], inputs[a:b])
        #print("ceshi:", self.l2) #len(inputs[a:b])==0---》nan
        self.l1 = (actvs.abs().sum(dim=1) / inputs.norm(dim=1)).mean() # actvs(l,2^15), input(l,768)-->(l,)/(l,)--->mean
        self.l0 = actvs.norm(dim=-1, p=0).mean() # 非0个数--(l,)
        self.ttl = self.l2 + self.alpha * self.aux_loss
        return self.ttl, self.l2, self.l1, self.l0
    
    def compute_loss2(self, inputs, lamda=0.1):
        print("span的计算actvs")
        inputs = inputs.squeeze()
        actvs, recons = self(inputs)
        self.l2 = normalized_l2(recons, inputs)
        self.actvs = actvs
        print("abc:", self.l2, actvs.shape, inputs.shape)
        return self.actvs
    
    
    def generate_encoder(self, inputs, lamda=0.1):
        print("计算actvs，判断哪几个维度激活了！") # (b,l,c)
        assert len(inputs.shape) == 3
        #actvs = self.encode(inputs) # (b,l,2^15)
        print("----------------原始数据输入的均值方差 （blc）----------------")
        self.check_embedding_stats(inputs)

        self.actvs_eos = self.encode(inputs[:, -1].float()) # 一般最后一个是eos-1（b,c），取这句话的语义token
        # 统计每行非零元素的数量
        self.non_zero_counts = tc.count_nonzero(self.actvs_eos, dim=1)
        print("16条数据，每条数据的激活情况：", self.non_zero_counts)
        self.non_zero_indices = tc.nonzero(self.actvs_eos, as_tuple=False)
        self.result = [[] for _ in range(self.actvs_eos.shape[0])]
        for row_idx, col_idx in self.non_zero_indices:
            self.result[row_idx.item()].append(col_idx.item())

        # # 假设 hidden_states 是 (batch_size, l, c)
        # eos_mask = (input_ids == tokenizer.eos_token_id)  # 生成 bool mask
        # eos_hidden_state = actvs[:,-1]        # 取出 <eos> 对应的向量 (c,)
        # eos_hidden_state = eos_hidden_state.unsqueeze(0)  # 变为 (1, c)
        
        
        return self.actvs_eos, self.non_zero_counts, self.non_zero_indices, self.result
    

    '''为了sequential—review'''
    def generate_encoder2(self, inputs, lamda=0.1):
        print("计算actvs，判断哪几个维度激活了！") # (b,l,c)
        assert len(inputs.shape) == 3 # (1,l,c)
        latent_review = self.encode2(inputs.squeeze(0).float()) # (l c)
        latent_review_actindex = tc.topk(latent_review[-1], 15).indices.tolist() 
        
        return latent_review_actindex
    




    # 用重构的数据
    def generate_encoder_recon(self, inputs, lamda=0.1):
        print("计算actvs，判断哪几个维度激活了！") # (b,l,c)
        assert len(inputs.shape) == 3
        #actvs = self.encode(inputs) # (b,l,2^15)
        #print("inputs dtype: float还是bfloat16：", inputs[:, -1].dtype)  # 查看数据类型 torch.float32
        self.actvs_eos = self.encode(inputs[:, -1].float()) # 一般最后一个是eos-1（b,c）
        # 统计每行非零元素的数量
        self.non_zero_counts = tc.count_nonzero(self.actvs_eos, dim=1)
        print("16条数据，每条数据的激活情况：", self.non_zero_counts)
        self.non_zero_indices = tc.nonzero(self.actvs_eos, as_tuple=False)
        self.result = [[] for _ in range(self.actvs_eos.shape[0])]
        for row_idx, col_idx in self.non_zero_indices:
            self.result[row_idx.item()].append(col_idx.item())

        if self.do_edit:
            h = self.actvs_eos * self.mask.unsqueeze(0)
        X = tc.cat([inputs[:, :-1], self.decode(self.actvs_eos).unsqueeze(1)], dim=1)
        # print("原始数据：", inputs,inputs.shape)
        # print("重构数据：", X, X.shape)
        # print("解码数据：", self.decode(self.actvs_eos), self.decode(self.actvs_eos).shape)
        # if tc.equal(X, self.decode(self.actvs_eos)):
        #     print("全等！")
        
        return X, self.actvs_eos, self.non_zero_counts, self.non_zero_indices, self.result


    
    '''为了计算谁激活'''
    def generate(self, X):
        actvs, recons = self(X)
        print("激活值actvs的维度是：", actvs.shape) # 激活值actvs的维度是： torch.Size([16, 1, 65536]) （b,l,c）

        assert len(X.shape) == 3 # X 是一个三维张量，（b,l,d_hidden）
        self.recons_h = None
        self.aux_loss = 0
        h = self.encode(X[:, -1]) # (b,d_hidden）) 只对最后一个时刻编码：因为它提供了上下文信息，是当前时间点的信息总结，编码整个序列的话会重复编码
        
        if self.do_edit:
            h = h * self.mask.unsqueeze(0)
        X = tc.cat([X[:, :-1], self.decode(h).unsqueeze(1)], dim=1)
        return X
    
    def dump_disk(self, fpath):
        os.makedirs(os.path.split(fpath)[0], exist_ok=True)
        tc.save({"weight": self.state_dict(), 
                 "config": {"d_inp": self.dims[0], 
                            "d_hide": self.dims[1], 
                            "topK": self.topk}
                            },
                 fpath)
        print("SAE is dumped at %s." % fpath)


# "ours": DisSAE, 
SAEs = {"sae": SparseAutoencoder, "ae": SparseAutoencoder, "topk": TopKSAE, "topk2": TopKSAE}

def load_pretrained(fpath, device="cuda"):
    return SAEs["topk"].from_disk(fpath, device)
    # assert os.path.exists(fpath)
    # name = os.path.split(fpath)[-1].rsplit(".pth")[0]
    # cls = name.split("_l", 1)[0].lower()
    # layer = int(name.split("_l", 1)[1].split("_", 1)[0])
    # return name, layer, SAEs[cls].from_disk(fpath, device)


    
    # assert os.path.exists(fpath), f"文件路径 {fpath} 不存在！"
    
    # # 提取文件名并去掉扩展名
    # name = os.path.split(fpath)[-1].rsplit(".pth")[0]
    
    # # 解析文件名，提取关键信息
    # parts = name.split("_")
    # model_name = parts[0]  # 模型名称，例如 "TopK5"
    # layer = int(parts[1][1:])  # 提取 "_l7" 中的数字 7
    # hidden_dim = int(parts[2][1:-1])  # 提取 "_h131k" 中的 131000
    # epoch = int(parts[3][5:])  # 提取 "epoch5" 中的 5
    
    # # 加载模型 (这里假设 P5 模型在 `SAEs` 字典中注册)
    # assert model_name.lower() in SAEs, f"未知模型类别：{model_name.lower()}"
    # model_instance = SAEs[model_name.lower()].from_disk(fpath, device)
    
    # return name, layer, hidden_dim, epoch, model_instance
