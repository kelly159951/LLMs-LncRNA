# import pandas as pd
# from Bio import SeqIO

# # 定义文件路径
# lnc_fasta_file = './dataset/lnc_class_new_label.fasta'

# # 初始化类别计数字典
# category_counts = {
#     "Linc": 0,
#     "Sense No Exonic": 0,
#     "Antisense": 0,
#     "Exonic": 0
# }

# # 读取FASTA文件并统计类别数量
# for record in SeqIO.parse(lnc_fasta_file, "fasta"):
#     # print(record)
#     # 分割描述部分，提取标签

#     # 提取并去除末尾的分号
#     label = record.description.split(" ", 1)[1].strip().rstrip(';')
#     # print(label)
    
#     # 更新类别计数
#     if label in category_counts:
#         category_counts[label] += 1
#     else:
#         print(label)
#         category_counts["Others"] += 1

# # 打印每种类别的RNA数量
# print("各类别的RNA数量统计：")
# for category, count in category_counts.items():
#     print(f"{category}: {count}")

# #Linc: 382832
# # Sense No Exonic: 13555
# # Antisense: 23677
# # Exonic: 38040




import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

# 定义文件路径
rna_file = './dataset/lnc_class_new.fasta'
label_file = './dataset/lnc_class_new_label.csv'
filtered_rna_file = './dataset/task1_rna.fasta'
filtered_label_file = './dataset/task1_rna_label.csv'

# 读取FASTA文件和标签文件

rna_sequences = list(SeqIO.parse(rna_file, "fasta"))
labels_df = pd.read_csv(label_file,header=0, names=["Seqname", "Label"])
# print(rna_sequences[0])
# print(labels_df.head())
filtered_rna_sequences = []
filtered_labels = []

# 过滤长度超过1024的序列
for rna_record, label_row in tqdm(zip(rna_sequences, labels_df.iterrows()), desc="Filtering RNA sequences", total=len(rna_sequences)):
    if len(rna_record.seq) <= 1022:
        filtered_rna_sequences.append(rna_record)
        filtered_labels.append(label_row[1])

# 将过滤后的标签拼接成DataFrame
filtered_labels_df = pd.DataFrame(filtered_labels)

# 写入过滤后的序列到新的FASTA文件
SeqIO.write(filtered_rna_sequences, filtered_rna_file, "fasta")

# 写入过滤后的标签到新的CSV文件
filtered_labels_df.to_csv(filtered_label_file, index=False, header=["Seqname", "Label"])

print(f"Filtered RNA sequences saved to {filtered_rna_file}")
print(f"Filtered RNA labels saved to {filtered_label_file}")
