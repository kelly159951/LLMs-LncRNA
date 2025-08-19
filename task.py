import torch
import numpy as np
import os
import random
import argparse
import time
from util.runner import *
import datetime

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_num', type=int, default=1,help='1 for rna classification; 2 for RNA-PRO interaction prediction')
    parser.add_argument('--lr', type=float, default=1e-4,help='the learning rate for model training')
    parser.add_argument('--bs', type=int, default=32,help='the batch size for training')
    parser.add_argument('--base_log', type=str, default='log',help='the path of folder for logging')
    parser.add_argument('--log_dir', type=str, default=None,help='the path of log, if None we automately set with time')
    parser.add_argument('--inbalance_frac', type=float, default=4,help='the max frac between the max size class and min size class')
    parser.add_argument('--dropout', type=float, default=0,help='the dropout of model')
    parser.add_argument('--epochs', type=int, default=100,help='the number of epochs')
    parser.add_argument('--disable_layer', type=int, default=-1,help='the number of freeze layers of rnafm,default mean freeze all')
    parser.add_argument('--device', type=int, default=None,help='specify the device for training')
    parser.add_argument('--inference', action='store_true',help='inference mode')
    parser.add_argument('--baseline', action='store_true',help='baseline model')

    # 任务 1 的特定参数
    parser.add_argument('--rna_seq_file', type=str, help='fasta,the path of fasta file for task 1')
    parser.add_argument('--label_file', type=str, help='csv,the path of label file for task 1')

    # 任务 2 的特定参数
    parser.add_argument('--pair_file', type=str, help='txt,the path of pair file for task 2')
    parser.add_argument('--rna_sequence_file', type=str, help='fasta,the path of rna sequence file for task 2')
    parser.add_argument('--pro_sequence_file', type=str, help='fasta,the path of protein sequence file for task 2')
    parser.add_argument('--non_pair_file', type=str, help='txt,the path of non pair file for task 2')
    parser.add_argument('--non_rna_sequence_file', type=str, help='fasta,the path of non rna sequence file for task 2')
    parser.add_argument('--non_pro_sequence_file', type=str, help='fasta,the path of non protein sequence file for task 2')

    #baseline 特定参数
    parser.add_argument('--alpha', type=float, default=0.4,help='loss = alpha * loss_ae + loss_task')
    parser.add_argument('--encode',action='store_true',help='whether to encode and save rna, use for the first time to run baseline')
    parser.add_argument('--encoded_rna_file', type=str, default=None,help='csv,the path encoded rna file')
    parser.add_argument('--encoded_non_rna_file', type=str, default=None,help='csv,the path encoded rna file for non pair')
    parser.add_argument('--encoded_pro_file', type=str, default=None,help='csv,the path encoded protein file')
    parser.add_argument('--encoded_non_pro_file', type=str, default=None,help='csv,the path encoded protein file for non pair')


    parser.add_argument('--as_dna', action='store_true',help='treat the data as dna or not')
    parser.add_argument('--num_worker', type=int, default=4,help='the num of worker for dataloader')
    parser.add_argument('--checkpoint', type=str, default='',help='the path of pretrained checkpoint')
    parser.add_argument('--use_pool', action='store_true',help='use pooling instead of global feature')
    parser.add_argument('--random', action='store_true',help='randomly training without using seed')
    parser.add_argument('--seed', type=int, default=2023,help='the random seed for training')
    parser.add_argument('--sample', type=int, default=-1,help='the number of samples to use')
    parser.add_argument('--skip', type=int, default=0,help='the number of epoch to skip')
    parser.add_argument('--distilation_rna_fm', action='store_true',help='to distilate rna fm')

    #代码自动填充
    parser.add_argument('--whole_data_num', type=int, default=0,help='the number of whole data')
    parser.add_argument('--encode_time', type=float, default=0,help='the time baseline to encode, use when you have encoded and know the time')
    
    args = parser.parse_args()
    # 根据 task_num 设置条件检查
    if args.task_num == 1:
        # 任务 1 需要 rna_seq_file 和 label_file
        if not args.rna_seq_file or not args.label_file:
            parser.print_help()
            raise ValueError('For task 1, both rna_seq_file and label_file are required.')

        # 任务 1 不需要 task 2 特定的参数
        if args.pair_file or args.rna_sequence_file or args.pro_sequence_file or args.non_pair_file or args.non_rna_sequence_file or args.non_pro_sequence_file:
            parser.print_help()
            raise ValueError('For task 1, task 2 specific arguments should not be provided.')

    elif args.task_num == 2:
        # 任务 2 需要 pair_file, rna_sequence_file, pro_sequence_file, 等等
        if not args.pair_file or not args.rna_sequence_file or not args.pro_sequence_file:
            parser.print_help()
            raise ValueError('For task 2, pair_file, rna_sequence_file, and pro_sequence_file are required.')

        # 任务 2 不需要 task 1 特定的参数
        if args.rna_seq_file or args.label_file:
            parser.print_help()
            raise ValueError('For task 2, task 1 specific arguments should not be provided.')

    else:
        parser.print_help()
        raise ValueError('Invalid task_num. Please specify either 1 or 2.')
    if torch.cuda.is_available() :
        if args.device!=None:
            args.device = torch.device(f'cuda:{args.device}')
        else:
            args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    args.log_dir= create_log_model(args) if args.log_dir is None else args.log_dir
    
    return args

# set random seed
def fix_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

def create_log_model(args):
    timestamp = time.time()
    if args.baseline:
        log_dir = os.path.join(args.base_log, f'baseline_task_{args.task_num}_{timestamp}')
    else:
        log_dir = os.path.join(args.base_log, f'task_{args.task_num}_{timestamp}')
    
    return log_dir


def ddp_setup():
    if "RANK" in os.environ and "LOCAL_RANK" in os.environ:#分布式Training
        try:
            torch.distributed.init_process_group(backend='nccl', timeout=datetime.timedelta(hours=2))
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            print(f'Initialized global rank {os.environ["RANK"]}, local rank {local_rank}')
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to initialize distributed training: {e}")
    else:#单机Training
        print("Non-distributed mode. Running on single process.")
        return False


if __name__ == '__main__':
    args=create_parser()
    print(args)
    
    # 检测分布式环境
    if "RANK" in os.environ:
        args.global_rank = int(os.environ["RANK"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
    else:
        args.global_rank = 0
        args.local_rank = 0
    args.multygpu=ddp_setup()
    if not args.random:
        fix_seed(args.seed)
    if args.baseline:
        runner = base_Runner(args) 
    else:
        runner = task1_Runner(args) if args.task_num == 1 else task2_Runner(args)
    runner.pipeline()




