import torch
import random

'''
测试读取训练好的sae
sae训练后的保存利用dump_disk函数：
tc.save({"weight": self.state_dict(), "config": {"d_inp": self.dims[0], "d_hide": self.dims[1],}}, fpath)
'''
def load_sae(file_path, gpu_i):
    if torch.cuda.is_available():
        device_i = int(0)  # 🔥 只要设备号是int
        map_location = lambda storage, loc: storage.cuda(device_i)
    else:
        map_location = "cpu"

    W = torch.load(file_path, map_location=map_location)
    #W = torch.load(file_path)
    if "weight" in W: # topksae
        W = W["weight"] # 读取sae的权重参数：W 是 `state_dict()`，其中包含 `W_enc`
    
    #print(W.keys())# 查看有哪些参数 # odict_keys(['mask', 'W_enc', 'b_enc', 'b_dec'])
    W = W["W_enc"].T
    
    # 🔥 保证最终的W一定在正确的GPU上
    if torch.cuda.is_available():
        W = W.to(f"cuda:{device_i}")
    
    return W


# 从原始的txt文件中找到所有的"Prediction: No Confidence: 0.95"作为intent
def select_intent(intent_file_path, W):
    indices = []
    with open(intent_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            #if "Prediction: Yes Confidence: 0.95" in line:
            # if "Intent: N/A" not in line:
            if "Prediction: Yes" in line:
                # 提取行开头的 index（假设 index 和后面内容是用tab \t隔开的）
                index = line.split('\t')[0].strip()
                if index.isdigit():
                    indices.append(int(index))

    select_W = W[indices,:]

    return indices, select_W


# 随机选择一部分latent
def select_intent_random(intent_file_path, W):
    num_latents = W.shape[0]  # latent总数，比如16384
    indices = []
    with open(intent_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if "Prediction: Yes Confidence: 0.95" in line:
                # 提取行开头的 index（假设 index 和后面内容是用tab \t隔开的）
                index = line.split('\t')[0].strip()
                if index.isdigit():
                    indices.append(int(index))

    # 随机选择
    # 设置随机种子，保证每次一样
    random.seed(2022)
    num_to_select = len(indices)  # 要选的个数，比如1208 
    random_indices = random.sample(range(num_latents), num_to_select) # 🔥 随机选择 num_to_select 个索引（不重复）
    select_W = W[random_indices, :] # 🔥 选取对应的 latent


    return random_indices, select_W


def select_intent_random_no(intent_file_path, W):
    num_latents = W.shape[0]  # latent总数，比如16384
    indices = []
    indices_no = []
    with open(intent_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if "Prediction: Yes Confidence: 0.95" in line:
                # 提取行开头的 index（假设 index 和后面内容是用tab \t隔开的）
                index = line.split('\t')[0].strip()
                if index.isdigit():
                    indices.append(int(index))
            if "Prediction: No Confidence: 1.0" in line:
                # 提取行开头的 index（假设 index 和后面内容是用tab \t隔开的）
                index_no = line.split('\t')[0].strip()
                if index_no.isdigit():
                    indices_no.append(int(index_no))

    # 随机选择
    # 设置随机种子，保证每次一样
    random.seed(2022)
    num_to_select = len(indices)  # 要选的个数，比如1208 
    num_to_select_no = len(indices_no)
    print(num_to_select, num_to_select_no)

    selected = random.sample(indices_no, num_to_select)
    print(len(selected))
    select_W = W[selected, :] # 🔥 选取对应的 latent


    return selected, select_W




import os
from typing import List

def flatten_pt_lists(folder_path: str, file_pattern: str, num_files: int) -> List:
    """
    加载多个 .pt 文件中的 list 嵌套 list，并将所有内部元素拼接成一个一维列表。
    
    参数：
        folder_path (str): 文件夹路径
        file_pattern (str): 文件名模板，例如 "encoder_hidden_states_l11_{}_item.pt"
        num_files (int): 文件数量
    
    返回：
        List: 所有子列表拼接后的扁平化列表
    """
    all_elements = []
    for i in range(num_files):
        file_path = os.path.join(folder_path, file_pattern.format(i))
        print(f"加载文件: {file_path}")
        nested_lists = torch.load(file_path, map_location='cpu')
        for sublist in nested_lists:
            all_elements.extend(sublist)  # 扁平化每个子列表
    return all_elements



import torch
from typing import List, Literal

def pool_result(result: List[torch.Tensor], method: Literal['mean', 'last'] = 'mean') -> torch.Tensor:
    """
    将 List[(l, 768)] → Tensor[(len(result), 768)]
    
    参数：
        result: List of (l, 768) tensors
        method: 'mean' or 'last'
        
    返回：
        Tensor of shape (len(result), 768)
    """
    if method == 'mean':
        pooled = [r.mean(dim=0) for r in result]
    elif method == 'last':
        pooled = [r[-1] for r in result]
    else:
        raise ValueError("method must be 'mean' or 'last'")
    
    return torch.stack(pooled, dim=0)



if __name__ == "__main__":

    file_path = "/data2/jx39280/ICSRec-main/trained_sae/sae_model/TopK15_l8_epoch50_encoder_16384_beauty.pth"
    W = load_sae(file_path, 0)
    print(W.shape, W.device, type(W)) # torch.Size([16384, 768]), cuda:2

    intent_file_path = "/data2/jx39280/ICSRec-main/trained_sae/intent_select/tunelens_beauty_TopK15_l8_epoch50_encoder_16384_keep15_mistral_prompt3_5000_intent.txt"
    indices, select_W = select_intent(intent_file_path, W)
    print(len(indices), select_W.shape, select_W.device) # 1208 torch.Size([1208, 768]) cuda:2

    indices_random, select_W_random = select_intent_random(intent_file_path, W)
    print(len(indices_random), select_W_random.shape, select_W_random.device) 

    indices_random_no, select_W_random_no = select_intent_random_no(intent_file_path, W)
    print(len(indices_random_no), select_W_random_no.shape)
    pirnt()
    
    if indices_random == indices:
        print("1")
    common_elements = list(set(indices) & set(indices_random))
    print(common_elements, len(common_elements))


    folder = "/data2/jx39280/ICSRec-main/items/beauty/"
    pattern = "encoder_hidden_states_l11_{}_item.pt"
    result = flatten_pt_lists(folder, pattern, num_files=6)

    print("总长度：", len(result), type(result[0]), result[0].shape, result[1].shape, result[-1].shape) # 总长度： 12101 <class 'torch.Tensor'> <class 'torch.Tensor'>
    # print("前两项：", result[:2])

    # result: List[Tensor] with shape (l, 768)
    tensor_mean = pool_result(result, method='mean')  # shape: (len(result), 768)
    tensor_last = pool_result(result, method='last')  # shape: (len(result), 768)
    print(tensor_mean.shape, tensor_last.shape)
    mean_path = f"/data2/jx39280/ICSRec-main/items/beauty/l11_tenosr_mean.pt"
    last_path = f"/data2/jx39280/ICSRec-main/items/beauty/l11_tenosr_last.pt"
    torch.save(tensor_mean, mean_path)
    torch.save(tensor_last, last_path)
