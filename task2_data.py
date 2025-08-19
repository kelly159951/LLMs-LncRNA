import os
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

# 将输入字符串中的所有字母转换为大写，并返回转换后的字符串。
def convert_to_uppercase(input_str):
    # 逐个字符检查并转换为大写
    result = ''
    for char in input_str:
        # 如果字符是小写字母，转换为大写
        if 'a' <= char <= 'z':
            result += chr(ord(char) - 32)
        else:
            result += char
    return result

# 定义文件路径
rna_protein_file = './dataset/RNA-protein_lnc_filter.txt'  # rna-protein关系文件
noncode_source_file = './dataset/NONCODEv5_source.txt'     # rna 名称-ID
ncrna_fasta_file = './dataset/ncrna_NONCODEv5.fasta'       # rna ID-序列
pro_sec_file = './dataset/output_protein.fasta'            # pro 名称-序列

# 加载RNA-蛋白质关系文件
rna_protein_df = pd.read_csv(rna_protein_file, sep='\t', header=None)
rna_protein_df.columns = ['id', 'rna_name', 'noncode_id', 'rna_type', 'protein_name', 'protein_id', 'protein_type', 'description', 'method', 'pubmed_id', 'species', 'cell_line', 'interaction_type', 'interaction_mode', 'interaction_subject', 'source']

# 加载NONCODE源文件
noncode_df = pd.read_csv(noncode_source_file, sep='\t', header=None)
noncode_df.columns = ['noncode_id', 'source', 'source_id']

# 加载FASTA文件
source_rna_sequences = SeqIO.to_dict(SeqIO.parse(ncrna_fasta_file, "fasta"))
source_pro_sequences = SeqIO.to_dict(SeqIO.parse(pro_sec_file, "fasta"))

# 初始化结果列表
triplets = []
rna_sequences = []
pro_sequences = []
missing_info = []

# 遍历RNA-蛋白质关系文件
for index, row in tqdm(rna_protein_df.iterrows(), total=rna_protein_df.shape[0], desc="Processing RNA-Protein interactions"):
    rna_name = row['rna_name']
    protein_id = row['protein_id']
    
    # 确保蛋白质存在
    if protein_id == '-':
        missing_info.append(f"{rna_name}\t{protein_id}")
        continue
    
    # 在NONCODE源文件中查找RNA的ID
    rna_id = None
    matched_rows = noncode_df[(noncode_df['source'] == rna_name) | (noncode_df['source_id'] == rna_name)]
    
    if not matched_rows.empty:
        rna_id = matched_rows.iloc[0]['noncode_id']
    else:
        missing_info.append(f"{rna_name}\t{protein_id}")
        continue
    
    # 在FASTA文件中查找RNA的序列
    if rna_id in source_rna_sequences and protein_id in source_pro_sequences:
        rna_sequence = str(source_rna_sequences[rna_id].seq)
        protein_sequence = str(source_pro_sequences[protein_id].seq)
        rna_sequence = convert_to_uppercase(rna_sequence)
        protein_sequence = convert_to_uppercase(protein_sequence)
    else:
        missing_info.append(f"{rna_name}\t{protein_id}")
        continue
    
    # 保存三元组
    triplets.append((protein_id, rna_id, 1))
    rna_sequences.append((rna_id, rna_sequence))
    pro_sequences.append((protein_id, protein_sequence))

# 保存结果到文件
triplet_file = './dataset/task2_rna_protein_pairs.txt'
rna_sequence_file = './dataset/task2_rna_sequences.fasta'  # 修改为FASTA格式
pro_sequence_file = './dataset/task2_pro_sequences.fasta'  # 修改为FASTA格式
missing_info_file = './dataset/task2_missing_info.txt'

# 保存三元组
with open(triplet_file, 'w') as f:
    for triplet in triplets:
        f.write(f"{triplet[0]}\t{triplet[1]}\t{triplet[2]}\n")

# 保存RNAid和序列 (FASTA格式)
with open(rna_sequence_file, 'w') as f:
    for rna_name, rna_sequence in rna_sequences:
        f.write(f">{rna_name}\n{rna_sequence}\n")

# 保存蛋白质名字和序列 (FASTA格式)
with open(pro_sequence_file, 'w') as f:
    for protein_id, protein_sequence in pro_sequences:
        f.write(f">{protein_id}\n{protein_sequence}\n")

# 保存缺失信息
with open(missing_info_file, 'w') as f:
    for info in missing_info:
        f.write(f"{info}\n")

print("Processing completed and files saved in FASTA format.")
