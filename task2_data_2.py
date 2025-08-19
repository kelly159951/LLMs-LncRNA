import random
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
def convert_to_uppercase(input_str):
    return input_str.upper()

# Define file paths
triplet_file = './dataset/task2_rna_protein_pairs.txt'  
rna_protein_file = './dataset/RNA-protein_lnc_filter.txt'  
noncode_source_file = './dataset/NONCODEv5_source.txt'  
ncrna_fasta_file = './dataset/ncrna_NONCODEv5.fasta'    
pro_sec_file = './dataset/output_protein.fasta'         

# Load RNA-protein relationship file
rna_protein_df = pd.read_csv(rna_protein_file, sep='\t', header=None)
rna_protein_df.columns = ['id', 'rna_name', 'noncode_id', 'rna_type', 'protein_name', 'protein_id', 'protein_type', 'description', 'method', 'pubmed_id', 'species', 'cell_line', 'interaction_type', 'interaction_mode', 'interaction_subject', 'source']

# 加载NONCODE源文件（RNA名称和ID的对应关系），并转换为字典
noncode_df = pd.read_csv(noncode_source_file, sep='\t', header=None)
noncode_df.columns = ['rna_id', 'rna_name_1', 'rna_name_2']

# 创建一个字典，将 RNA 名称 (source) 和 RNA ID (noncode_id) 的对应关系存储
rna_name_to_id = {row['rna_name_1']: row['rna_id'] for _, row in noncode_df.iterrows()}
rna_name_to_id.update({row['rna_name_2']: row['rna_id'] for _, row in noncode_df.iterrows()})

# 加载FASTA文件
source_rna_sequences = SeqIO.to_dict(SeqIO.parse(ncrna_fasta_file, "fasta"))
source_pro_sequences = SeqIO.to_dict(SeqIO.parse(pro_sec_file, "fasta"))

# 获取集合A（RNA）和集合B（Protein）
rna_set = list(set(rna_protein_df['rna_name'].unique()) ) # RNA集合 A
protein_set = list(set(rna_protein_df['protein_id'].unique()))  # Protein集合 B

# 提取已有的RNA-Protein正反应对
existing_pairs = set(zip(rna_protein_df['rna_name'], rna_protein_df['protein_id']))

# 初始化非反应对结果列表
non_pairs = set()  # 使用集合来避免重复
non_triplets = []
non_rna_sequences = []
non_pro_sequences = []

# 计算要生成的非反应对数量
num_non_pairs = len(existing_pairs) 

# 生成非反应对
print(f"Generating {num_non_pairs} non-reaction pairs...")

# 使用 tqdm 添加进度条
for _ in tqdm(range(num_non_pairs), desc="Generating non-reaction pairs"):
    while len(non_pairs) < num_non_pairs:
        # 随机选择一个 RNA 和一个蛋白质
        rna_name = random.choice(rna_set)
        protein_id = random.choice(protein_set)
        # print(rna_name, protein_id)
        
        # 获取 RNA 的 ID 和序列
        if rna_name in rna_name_to_id:
            rna_id = rna_name_to_id[rna_name]
            if rna_id in source_rna_sequences :
                if protein_id in source_pro_sequences:
                    # 确保 RNA 和蛋白质都有对应的序列，并且它们不在正反应对中，且非反应对不重复
                    if (rna_name, protein_id) not in existing_pairs and (rna_id, protein_id) not in non_pairs:
                        rna_sequence = str(source_rna_sequences[rna_id].seq)
                        protein_sequence = str(source_pro_sequences[protein_id].seq)
                        rna_sequence = convert_to_uppercase(rna_sequence)
                        protein_sequence = convert_to_uppercase(protein_sequence)

                        non_pairs.add((rna_id, protein_id))
                        non_rna_sequences.append((rna_id, rna_sequence))
                        non_pro_sequences.append((protein_id, protein_sequence))
                        non_triplets.append((rna_id, protein_id))
                else:
                    protein_set.remove(protein_id)
            else:
                rna_set.remove(rna_name)
        else:
            rna_set.remove(rna_name)

non_pair_file = './dataset/task2_nonrna_protein_pairs.txt'
non_rna_sequence_file = './dataset/task2_nonrna_sequences.fasta'  # 保存为FASTA格式
non_pro_sequence_file = './dataset/task2_nonpro_sequences.fasta'  # 保存为FASTA格式

# 保存非反应对到文件
with open(non_pair_file, 'w') as f:
    for rna_name, protein_id in non_triplets:
        f.write(f"{protein_id}\t{rna_name}\t0\n")

# 保存RNA名字和序列到FASTA文件
with open(non_rna_sequence_file, 'w') as f:
    for rna_name, rna_sequence in non_rna_sequences:
        f.write(f">{rna_name}\n{rna_sequence}\n")

# 保存蛋白质名字和序列到FASTA文件
with open(non_pro_sequence_file, 'w') as f:
    for protein_id, protein_sequence in non_pro_sequences:
        f.write(f">{protein_id}\n{protein_sequence}\n")

print("Non-reaction pairs generated and saved in FASTA format.")
