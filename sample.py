import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

# Define file paths
rna_file = './dataset/lnc_class_new.fasta'
label_file = './dataset/lnc_class_new_label.csv'
filtered_rna_file = './dataset/task1_sample.fasta'
filtered_label_file = './dataset/task1_sample_label.csv'

# Read FASTA file and label file
rna_sequences = list(SeqIO.parse(rna_file, "fasta"))
labels_df = pd.read_csv(label_file, header=None, names=["Seqname", "Label"])

filtered_rna_sequences = []
filtered_labels = []

# Filter sequences longer than 1024
num = 0
for rna_record, label_row in tqdm(zip(rna_sequences, labels_df.iterrows()), desc="Filtering RNA sequences", total=len(rna_sequences)):
    if len(rna_record.seq) <= 1022:
        filtered_rna_sequences.append(rna_record)
        filtered_labels.append(label_row[1])
        num += 1
    if num >= 10000:
        break

# Concatenate filtered labels into DataFrame
filtered_labels_df = pd.DataFrame(filtered_labels)

# Write filtered sequences to new FASTA file
SeqIO.write(filtered_rna_sequences, filtered_rna_file, "fasta")

# Write filtered labels to new CSV file, without header row
filtered_labels_df.to_csv(filtered_label_file, index=False, header=False)

print(f"Filtered RNA sequences saved to {filtered_rna_file}")
print(f"Filtered RNA labels saved to {filtered_label_file}")
