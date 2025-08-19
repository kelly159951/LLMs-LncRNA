import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

# 定义文件路径
rna_file = './dataset/lnc_class_new.fasta'
label_file = './dataset/lnc_class_new_label.csv'
filtered_rna_file = './dataset/task1_sample.fasta'
filtered_label_file = './dataset/task1_sample_label.csv'

# 读取FASTA文件和标签文件
rna_sequences = list(SeqIO.parse(rna_file, "fasta"))
labels_df = pd.read_csv(label_file, header=None, names=["Seqname", "Label"])

filtered_rna_sequences = []
filtered_labels = []

# 过滤长度超过1024的序列
num = 0
for rna_record, label_row in tqdm(zip(rna_sequences, labels_df.iterrows()), desc="Filtering RNA sequences", total=len(rna_sequences)):
    if len(rna_record.seq) <= 1022:
        filtered_rna_sequences.append(rna_record)
        filtered_labels.append(label_row[1])
        num += 1
    if num >= 10000:
        break

# 将过滤后的标签拼接成DataFrame
filtered_labels_df = pd.DataFrame(filtered_labels)

# 写入过滤后的序列到新的FASTA文件
SeqIO.write(filtered_rna_sequences, filtered_rna_file, "fasta")

# 写入过滤后的标签到新的CSV文件，不包含标题行
filtered_labels_df.to_csv(filtered_label_file, index=False, header=False)

print(f"Filtered RNA sequences saved to {filtered_rna_file}")
print(f"Filtered RNA labels saved to {filtered_label_file}")
