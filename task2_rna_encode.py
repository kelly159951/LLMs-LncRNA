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

    # Store converted data
    converted_data = []

    # Open and read file
    with open(input_file, 'r') as file:
        for line in file:
            # Remove newline characters and split by tabs
            columns = line.strip().split()
            
            # Get seqname and label
            seqname = columns[1]
            label = columns[2]
            
            # Construct new format: ">seqname,label"
            formatted_line = f">{seqname},{label};"
            converted_data.append([seqname, label])  # Add seqname and label

    # Write converted data to new CSV file
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write header row
        csv_writer.writerow(['Seqname', ' Label'])
        
        # Write data rows
        csv_writer.writerows(converted_data)

    print(f"Conversion completed, data saved to {output_file}")


    



    for i in range(1,11):
        if(i!=1 and i!=5 and i!=9):
            dimension="1"
            savetype='csvnpy'
            n_select=str(i)
            RNAcoding_01.RNA_coding(Afastapath, output_file, resultpath, dimension = dimension, savetype = savetype,n_select = n_select)
    



# Load and concatenate features
def load_and_concatenate_features(file_paths):
    features = []
    for file in file_paths:
        df = pd.read_csv(file)
        feature_array = df.values[:, 2:]  # Extract encoding
        features.append(feature_array)
        # print(features)
        names=df.values[:, 1]
    return names,np.hstack(features)

# Save features and names to CSV file
def save_to_csv(names, features, file_path):
    df = pd.DataFrame(features)
    df.insert(0, 'Seqname', names)
    df.to_csv(file_path, index=False)

# Save features and names to NPY file
def save_to_npy(names, features, file_path):
    combined_data = np.column_stack((names, features))
    np.save(file_path, combined_data)


# Main function
data_set=['pair','non_pair']
for set in data_set:
    base_path = './task2_rna_encode/'+set+'/encoding_features/'
    file_names = [
        'Entropy density of transcript (1D).csv', 'Global descriptor (1D).csv', 'K-mer (1D).csv',
        'Pseudo protein related (1D).csv', 'Guanine-cytosine related (1D).csv', 'Nucleotide related (1D).csv', 'EIIP based spectrum (1D).csv',
    ]
    file_paths = [base_path + file_name for file_name in file_names]


    # Load and concatenate features
    names,features = load_and_concatenate_features(file_paths)


    # Set save path
    save_base_path = './task2_rna_encode/'+set+'/combined_features'
    if not os.path.exists(save_base_path):
        os.makedirs(save_base_path)

    csv_path = os.path.join(save_base_path, 'combined_features.csv')
    npy_path = os.path.join(save_base_path, 'combined_features.npy')

    # Save to different file formats
    save_to_csv(names, features, csv_path)
    save_to_npy(names, features, npy_path)

    print("Features and names have been successfully saved to CSV, NPY, and PKL files.")


