import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from Bio import SeqIO
import pandas as pd
import fm
from collections import Counter
from task1_final_config import *
_, Alphabet = fm.pretrained.rna_fm_t12()
# 设置随机种子，确保实验的可重复性
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class RF(nn.Module):
    def __init__(self, rna_fm_model):
        super(RF, self).__init__()
        self.rna_fm = rna_fm_model
    def forward(self, input_ids):
        outputs = self.rna_fm(input_ids, need_head_weights=False, repr_layers=[12], return_contacts=False)
        sequence_output = outputs['representations'][12][:,0,:]
        return sequence_output
    
class MLP(nn.Module):
    def __init__(self,input_dim, hidden_dim_1, hidden_dim_2, output_dim):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(drop_out_1),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(drop_out_2),
            nn.Linear(hidden_dim_2, output_dim)
        )
    def forward(self, input_ids):
        logits = self.mlp(input_ids)
        return logits


class RNADataset(Dataset):
    def __init__(self, sequences, labels, alphabet):
        self.sequences = sequences
        self.alphabet = alphabet

        #label映射为数字
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(labels)
        #获取类别编码
        res = {}
        for cl in self.label_encoder.classes_:
            res.update({cl:self.label_encoder.transform([cl])[0]})
        print('类别编码：',res)
        self.tokens = []
        for seq in self.sequences:
            token = [self.alphabet.cls_idx] + [self.alphabet.get_idx(s) for s in seq] + [self.alphabet.eos_idx]
            self.tokens.append(token)
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        tokens = [self.alphabet.cls_idx] + [self.alphabet.get_idx(s) for s in seq] + [self.alphabet.eos_idx]
        label = self.labels[idx]
        return torch.tensor(tokens, dtype=torch.long), torch.tensor( label, dtype=torch.long)

# 定义 collate_fn 函数，用于 DataLoader 的 batch 拼接
def collate_fn(batch):
    input_ids, label = zip(*batch)
    padded_inputs = []
    alphabet = Alphabet
    for i in range(len(input_ids)):
        padded_input = torch.cat([input_ids[i], torch.tensor([alphabet.padding_idx] * (max_len - len(input_ids[i])))])
        padded_inputs.append(padded_input)

    return torch.stack(padded_inputs), torch.tensor( label, dtype=torch.long)



def load_split_data(fasta_file,label_file,alphabet,batch_size=32,collate_fn=collate_fn):
    sequences = []
    labels = []
    label_fa=[]
    with open(fasta_file) as f:
        for record in SeqIO.parse(f, "fasta"):
            seq = str(record.seq).replace("A", "U")
            seq = str(record.seq).replace("T", "A")
            seq = str(record.seq).replace("G", "C")
            seq = str(record.seq).replace("C", "G")
            sequences.append(seq)
            label_fa.append(record.id)
    
    label_df = pd.read_csv(label_file, header=0, names=['Seqname', 'Label'])
    label_df['Seqname'] = label_df['Seqname'].apply(lambda x: x[1:])  # 去掉'>'符号,并去掉第一行的标题
    labels = label_df['Label']
    for i in range(len(label_fa)):
        assert label_fa[i]==label_df['Seqname'][i]
    print('name check sucessfully')
    # 划分Training、验证和测试集：确保测试集和验证集样本数量均衡
    X = sequences
    y = labels  
    # 统计每个类别的样本数
    class_counts=pd.Series(labels).value_counts()
    print("总数据中每个类别的RNA个数:\n", class_counts)
    # 确定从每个类别中抽取的样本数
    min_class_samples = min(class_counts)  
    num_samples_per_class = max(1, min_class_samples//3)  
    print('每个类别抽取的样本数:', num_samples_per_class)
    # 初始化索引列表
    temp_indices = []  
    train_indices = []  
    
    # 为每个类别随机选择样本索引
    for label, count in class_counts.items():  
        class_indices = np.where(labels == label)[0]  
        np.random.shuffle(class_indices)  # 随机打乱索引
        temp_indices.extend(class_indices[:num_samples_per_class])  
        train_indices.extend(class_indices[num_samples_per_class:])  

    # 分离出对应的X_temp, y_temp, X_train, y_train
    X_temp = [X[i] for i in temp_indices]
    y_temp = [y[i] for i in temp_indices]
    X_train = [X[i] for i in train_indices]
    y_train = [y[i] for i in train_indices]
    
    # 接下来，将temp集合进一步划分为验证集和测试集
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  

    # 统计总数据中每个类别的RNA的个数
    total_counts = labels.value_counts()  
    print("总数据中每个类别的RNA个数:", total_counts)  
    print('训练集大小：', len(X_train), len(y_train))
    print('验证集大小：', len(X_val),  len(y_val))
    print('测试集大小：', len(X_test), len(y_test))
    train_counts = pd.Series(y_train).value_counts()
    val_counts = pd.Series(y_val).value_counts()
    test_counts = pd.Series(y_test).value_counts()
    print("训练集中每个类别的RNA个数:\n", train_counts)  
    print("验证集中每个类别的RNA个数:\n", val_counts)  
    print("测试集中每个类别的RNA个数:\n", test_counts)

    # 创建数据集
    train_dataset = RNADataset(X_train, y_train, alphabet)  
    val_dataset = RNADataset(X_val, y_val, alphabet)  
    test_dataset = RNADataset(X_test, y_test, alphabet)  

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,collate_fn=collate_fn)
    
    return train_loader,val_loader,test_loader, labels
    
#获取每个类别的样本索引并随机打乱
def get_class_indices(labels, num_samples_per_class):
    class_indices = []
    class_counts = np.bincount(labels)
    for label in range(len(class_counts)):
        indices = np.where(labels == label)[0]
        np.random.shuffle(indices)
        class_indices.extend(indices[:num_samples_per_class])
    return class_indices

#在每个 epoch 之前构建一个新的 DataLoader,确保每个类别采样相同数目的样本
def create_balanced_loader(dataloader, num_samples_per_class, batch_size):
    labels = []
    tokens = []

    # 获取所有标签和tokens
    for token, label in dataloader:
        labels.extend(label.numpy())
        tokens.extend(token.numpy())

    labels = np.array(labels)
    tokens = np.array(tokens)

    # 获取平衡的类别索引
    class_indices = get_class_indices(labels, num_samples_per_class)
    
    # 从tokens和labels中提取平衡的样本
    sampled_tokens = tokens[class_indices]
    sampled_labels = labels[class_indices]

    
    # 创建TensorDataset
    train_dataset = TensorDataset(torch.tensor(sampled_tokens, dtype=torch.float32), torch.tensor(sampled_labels, dtype=torch.long))
    
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#获取data_loader里最少种类的数目
def calculate_min_samples_per_class(data_loader,num=2):
    # 初始化计数器
    class_counts = Counter()
    labellist=[]
    if(num==2):
        # 遍历数据加载器，统计每个类别的数量
        for _, labels in data_loader:
            if(str(labels.type())=='torch.LongTensor'):
                labels=labels.tolist()
            labellist.extend(labels)
    if(num==3):
        # 遍历数据加载器，统计每个类别的数量
        for _, _, labels in data_loader:
            if(str(labels.type())=='torch.LongTensor'):
                labels=labels.tolist()
            labellist.extend(labels)
    # 获取最少样本数的类别的样本数
    # print(labellist)
    class_counts=pd.Series(labellist).value_counts()
    print(class_counts)
    num_samples_per_class = min(class_counts)

    return num_samples_per_class
