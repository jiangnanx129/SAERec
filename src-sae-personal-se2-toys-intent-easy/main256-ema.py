import os
import numpy as np
import torch
import argparse

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import RecWithContrastiveLearningDataset,DS, DS2, DS3

from trainers import ICSRecTrainer
from models import SASRecModel,GRUEncoder, SASRecModel_ema
from utils import EarlyStopping, get_user_seqs, check_path, set_seed
from test import load_sae, select_intent
'''
batchsize =1, 每条数据维护一个 user id，根据user id去查找tfidf，保留topk个激活的intent，每个用户激活的情况不一样
topk为5？10？15？20？
不是根据review的激活筛选personal intent：统计每个用户序列的review，看他们基于p5在sae中的激活，筛选出topk个激活（tong）
有的用户可能没有review：用用户序列的embedding为输入，看激活情况
1. 用户embedding * SAE'encoder看激活
2. 用户embedding * SAE prompt后剩下的intent，相似度计算
3. 用户embedding * SAE 所有的intent
'''
def show_args_info(args):
    print(f"--------------------Configure Info:------------")
    for arg in vars(args):
        print(f"{arg:<30} : {getattr(args, arg):>35}")


def main():
    parser = argparse.ArgumentParser()
    # system args
    parser.add_argument("--data_dir", default="../data/", type=str)
    parser.add_argument("--output_dir", default="../output_sae2_se_toys_intent_easy/", type=str)
    parser.add_argument("--data_name", default="Toys_and_Games", type=str)
    parser.add_argument("--encoder",default="SAS",type=str) # {"SAS":SASRec,"GRU":GRU4Rec}
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--model_idx", default=0, type=int, help="model idenfier 10, 20, 30...")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    # robustness experiments
    parser.add_argument(
        "--noise_ratio",
        default=0.0,
        type=float,
        help="percentage of negative interactions in a sequence - robustness analysis",
    )

    ## intent args (选择topk)
    parser.add_argument(
        "--intent_num",default=512,type=int,help="the multi intent nums!."
    )


    # model args
    parser.add_argument("--model_name", default="ICSRec_sae", type=str)
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--max_seq_length", default=50, type=int)
    parser.add_argument("--sae_file_path", type=str, default="../trained_sae/sae_model/toys/TopK15_l8_epoch45_encoder_16384_toys.pth", help="trained sae path")
    #parser.add_argument("--intent_file_path", type=str, default="../trained_sae/intent_select/tunelens_beauty_TopK15_l8_epoch50_encoder_16384_keep15_mistral_prompt3_5000_intent2.txt", help="trained sae path")
    parser.add_argument("--intent_file_path", type=str, default="/data2/jx39280/ICSRec-main/case_study/toys/case_data/toys_TopK15_l8_epoch45_encoder_16384_keep15_5000_conceptsummary_easy_intent.txt", help="trained sae path")
    
    parser.add_argument("--fusion_type", type=str, default="add") # add concat gate
    parser.add_argument("--trainable_latent", action="store_true") #type=str, default="False") # True
    parser.add_argument("--tfidf_path", type=str, default="../trained_sae/tfidf/edit_TopK15_l8_encoder_16384_nozero_mask_weight_mean_last_beauty_tfidf.pkl", help="trained sae path")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--global_k", type=int, default=5)
    
    
    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=2022, type=int)
    # loss weight # 暂时没有，先不考虑
    parser.add_argument("--rec_weight", type=float, default=1, help="weight of contrastive learning task")
    parser.add_argument("--latent_weight", type=float, default=0, help="weight of contrastive learning task")
    parser.add_argument("--align_weight", type=float, default=0, help="weight of contrastive learning task")
    parser.add_argument("--attention_weight", type=float, default=0, help="weight of contrastive learning task")
    
    # ablation experiments
    parser.add_argument("--cl_mode",type=str,default='cf',help="contrastive mode")
    # {'cf':coarse-grain+fine-grain,'c':only coarse-grain,'f':only fine-grain}
    parser.add_argument("--f_neg", action="store_true", help="delete the FNM component (both in cicl and ficl)")

    # learning related
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="weight_decay of adam")

    args = parser.parse_args()
    set_seed(args.seed) # utils.py中设置随机种子
    check_path(args.output_dir) # utils.py中设置输出路径

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    print("Using Cuda:", torch.cuda.is_available())
    args.data_file = args.data_dir + args.data_name + ".txt" # "../data/Beauty.txt"
    args.train_data_file = args.data_dir + args.data_name + "_1.txt" # "../data/Beauty_1.txt", 不用对比学习可以不用

    # construct supervisory signals via DS(·) operation, 如果不存在Beauty_1.txt, 就基于Beauty.txt生成分割后的新文件_1.txt
    if not os.path.exists(args.train_data_file):
        DS(args.data_file,args.train_data_file,args.max_seq_length)

    # training data：train_user_id为用户id的list
    train_user_id,train_user_seq, _, train_valid_rating_matrix, train_test_rating_matrix = get_user_seqs(args.train_data_file) # list套list，一条序列
    # valid and test data
    '''我们不用分割数据的话，实际user_seq就是我们的train_user_seq'''
    user_id,user_seq, max_item, valid_rating_matrix, test_rating_matrix = get_user_seqs(args.data_file) # 没有分割过的数据，user_seq小于train_user_seq，没有分割

    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    # save model args
    args_str = f"{args.model_name}-{args.encoder}-{args.data_name}-{args.model_idx}" # 保存的模型名称: "ICSRec_sae-SAS-Beauty-0"
    args.log_file = os.path.join(args.output_dir, args_str + ".txt") # 输出的log文件地址: "../output_sae/"


    # # ✅ 这一段：让所有 print 同时输出到 log 文件
    # class Logger(object):
    #     def __init__(self, filename):
    #         self.terminal = sys.stdout
    #         self.log = open(filename, "a")

    #     def write(self, message):
    #         self.terminal.write(message)
    #         self.log.write(message)

    #     def flush(self):
    #         self.terminal.flush()
    #         self.log.flush()

    # sys.stdout = Logger(args.log_file)



    show_args_info(args) # 打印所有相关参数

    '''测试！！！1'''
    # '''重要！！！！筛选intent不是na的！'''
    # sae_w = load_sae(args.sae_file_path, args.gpu_id)
    # indices, latent_init_tensor = select_intent(args.intent_file_path, sae_w)
    # print(len(indices)) # 6017
    # pirnt()

    with open(args.log_file, "a") as f:
        f.write(str(args) + "\n") # 参数写入log文件
        f.write("分割的数据, 不可训练sae, 两边并行后add, 有layernorm" + "\n") # 参数写入log文件

    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix # 作用？valid_rating_matrix维度: (user_num, args.item_size)

    # save model
    checkpoint = args_str + ".pt" # 保存的模型: "ICSRec_sae-SAS-Beauty-0.pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint) # 模型保存的路径: "../output_sae/ICSRec_sae-SAS-Beauty-0.pt"

    '''加载数据'''
    import pickle
    with open(args.tfidf_path, "rb") as f:
        tfidf_list = pickle.load(f) # list嵌套dict，每个用户1个dict, {index:tfidf,

    cluster_dataloader = 0 # 便于ICSRecTrainer的参数传入，实际没有作用，cluster_dataloader是为了intent聚类
    
    train_dataset = RecWithContrastiveLearningDataset(args, train_user_seq, train_user_id, tfidf_list, data_type="train") # 来自未分割的数据集
    #train_dataset = RecWithContrastiveLearningDataset(args, user_seq, user_id, tfidf_list, data_type="train") # 来自未分割的数据集
    train_sampler = RandomSampler(train_dataset) # 每次随机打乱顺序
    # train_sampler = SequentialSampler(train_dataset) # 每次随机打乱顺序
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    eval_dataset = RecWithContrastiveLearningDataset(args, user_seq, user_id, tfidf_list, data_type="valid") # 来自未分割的数据集
    eval_sampler = SequentialSampler(eval_dataset) 
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    test_dataset = RecWithContrastiveLearningDataset(args, user_seq, user_id, tfidf_list, data_type="test") # 来自未分割的数据集
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    # read trained sae: get sae whole latent space
    sae_w = load_sae(args.sae_file_path, args.gpu_id) # (16394, 768)
    # latent_init_tensor = sae_w
    # # indices, latent_init_tensor = select_intent(args.intent_file_path, sae_w) # 1208 torch.Size([1208, 768]) cuda:2
    # # print("latnet数目：", len(indices))
    # # with open(args.log_file, "a") as f:
    # #     f.write("根据prompt选中的intent数目：" +str(len(indices))+ "\n")
    # #     f.write(str(latent_init_tensor.size(0)))
    # #     f.write(str(args) + "\n") # 参数写入log文件
    # indices = [0,0,0]

    '''重要！！！！筛选intent不是na的！'''
    indices, latent_init_tensor = select_intent(args.intent_file_path, sae_w)
    # print(len(indices))
    # pirnt()

    if args.encoder=="SAS":
        model = SASRecModel_ema(args=args, latent_init_tensor=latent_init_tensor, indices = indices) # latent_init_tensor (1000,768)
        # mean_path = f"/data2/jx39280/ICSRec-main/items/beauty/l11_tenosr_mean.pt"
        # last_path = f"/data2/jx39280/ICSRec-main/items/beauty/l11_tenosr_last.pt"
        # item_embedding = torch.load(last_path) #, map_location='cuda')
        # model = SASRecModel_item3(args=args, latent_init_tensor=latent_init_tensor, item_embedding= item_embedding)
    elif args.encoder=="GRU":
        model=GRUEncoder(args=args)
    trainer = ICSRecTrainer(model, train_dataloader,cluster_dataloader, eval_dataloader, test_dataloader, args)

    # 测试，加载模型测试
    if args.do_eval:
        trainer.args.train_matrix = test_rating_matrix
        trainer.load(args.checkpoint_path)

        print(f"Load model from {args.checkpoint_path} for test!")
        scores, result_info = trainer.test(0, full_sort=True) # 0表示epoch

    else:
        print(f"Train ICSRec_sae")
        early_stopping = EarlyStopping(args.checkpoint_path, patience=40, verbose=True) # 设置早停, 连续40次loss不下降就结束
        for epoch in range(args.epochs): # 循环300个epoch
            trainer.train(epoch)
            # evaluate on NDCG@20
            scores, _ = trainer.valid(epoch, full_sort=True)
            #scores, _ = trainer.test(epoch, full_sort=True)
            early_stopping(np.array(scores[-1:]), trainer.model) # scores[-1:]
            if early_stopping.early_stop:
                print("Early stopping")
                break
        trainer.args.train_matrix = test_rating_matrix
        print("---------------Change to test_rating_matrix!-------------------")
        # load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0, full_sort=True)

    print(args_str)
    print(result_info)
    with open(args.log_file, "a") as f:
        f.write(args_str + "\n")
        f.write(result_info + "\n")


main()

#trainable_latent,fusion_type