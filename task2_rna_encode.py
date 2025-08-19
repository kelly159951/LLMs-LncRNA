import csv
import pandas as pd
import os
from corain.utility import RNAcoding_01
import numpy as np
import os

input_files = ['dataset/task2_rna_protein_pairs.txt','dataset/task2_nonrna_protein_pairs.txt']
output_files = ['dataset/task2_rna_protein_pairs.csv','dataset/task2_nonrna_protein_pairs.csv']
Afastapaths=["dataset/task2_rna_sequences.fasta","dataset/task2_nonrna_sequences.fasta"]
resultpath=['./task2_rna_encode/pair','./task2_rna_encode/nonpair']
for s in [1]:
    input_file =input_files[s]
    output_file = output_files[s]
    Afastapath=Afastapaths[s]
    resultpath=resultpath[s]

    # 存储转换后的数据
    converted_data = []

    # 打开并读取文件
    with open(input_file, 'r') as file:
        for line in file:
            # 去掉行尾的换行符并按制表符拆分
            columns = line.strip().split()
            
            # 获取 seqname 和 label
            seqname = columns[1]
            label = columns[2]
            
            # 构造新的格式：">seqname,label"
            formatted_line = f">{seqname},{label};"
            converted_data.append([seqname, label])  # 添加 seqname 和 label

    # 将转换后的数据写入新的 CSV 文件
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # 写入标题行
        csv_writer.writerow(['Seqname', ' Label'])
        
        # 写入数据行
        csv_writer.writerows(converted_data)

    print(f"转换完成，数据已保存到 {output_file}")


    



    for i in range(1,11):
        if(i!=1 and i!=5 and i!=9):
            dimension="1"
            savetype='csvnpy'
            n_select=str(i)
            RNAcoding_01.RNA_coding(Afastapath, output_file, resultpath, dimension = dimension, savetype = savetype,n_select = n_select)
    



# 加载并拼接特征
def load_and_concatenate_features(file_paths):
    features = []
    for file in file_paths:
        df = pd.read_csv(file)
        feature_array = df.values[:, 2:]  # 取编码
        features.append(feature_array)
        # print(features)
        names=df.values[:, 1]
    return names,np.hstack(features)

# 保存特征和名称到CSV文件
def save_to_csv(names, features, file_path):
    df = pd.DataFrame(features)
    df.insert(0, 'Seqname', names)
    df.to_csv(file_path, index=False)

# 保存特征和名称到NPY文件
def save_to_npy(names, features, file_path):
    combined_data = np.column_stack((names, features))
    np.save(file_path, combined_data)


# 主函数
data_set=['pair','non_pair']
for set in data_set:
    base_path = './task2_rna_encode/'+set+'/encoding_features/'
    file_names = [
        'Entropy density of transcript (1D).csv', 'Global descriptor (1D).csv', 'K-mer (1D).csv',
        'Pseudo protein related (1D).csv', 'Guanine-cytosine related (1D).csv', 'Nucleotide related (1D).csv', 'EIIP based spectrum (1D).csv',
    ]
    file_paths = [base_path + file_name for file_name in file_names]


    # 加载并拼接特征
    names,features = load_and_concatenate_features(file_paths)


    # 设置保存路径
    save_base_path = './task2_rna_encode/'+set+'/combined_features'
    if not os.path.exists(save_base_path):
        os.makedirs(save_base_path)

    csv_path = os.path.join(save_base_path, 'combined_features.csv')
    npy_path = os.path.join(save_base_path, 'combined_features.npy')

    # 保存到不同文件格式
    save_to_csv(names, features, csv_path)
    save_to_npy(names, features, npy_path)

    print("Features and names have been successfully saved to CSV, NPY, and PKL files.")


