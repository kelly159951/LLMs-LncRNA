import pandas as pd
import random
import torch
from Bio import SeqIO
import numpy as np
import csv
import os
from sklearn.preprocessing import LabelEncoder
import time
# transform DNA to RNA
def transform_dna_rna(seq):
    mapper = {'A': 'U', 'T': 'A', 'C': "G", 'G': 'C'}
    seq = [mapper[x] for x in seq]
    return ''.join(seq)
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

def baseline_task1_encode_rna(rna_seq_file,label_file,resultpath,final):
    from corain.utility import RNAcoding_01
    for i in range(1,11):
        if(i!=1 and i!=5 and i!=9):
            dimension="1"
            savetype='csvnpy'
            n_select=str(i)
            RNAcoding_01.RNA_coding(rna_seq_file, label_file, resultpath, dimension = dimension, savetype = savetype,n_select = n_select)

    base_path = resultpath+'/encoding_features/'
    file_names = [
        'Entropy density of transcript (1D).csv', 'Global descriptor (1D).csv', 'K-mer (1D).csv',
        'Pseudo protein related (1D).csv', 'Guanine-cytosine related (1D).csv', 'Nucleotide related (1D).csv', 'EIIP based spectrum (1D).csv',
    ]
    file_paths = [base_path + file_name for file_name in file_names]
    # 加载并拼接特征
    names,features = load_and_concatenate_features(file_paths)
    #确保path存在
    dir_path = os.path.dirname(final)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    
    save_to_csv(names, features, final)

    print("Features and names have been successfully saved to ",final)
    return len(names)

def baseline_task2_encode_rna(input_files,output_files,Afastapaths,Bfastapaths,resultpaths,rnafinals,profinals):
    from corain.utility import RNAcoding_01
    #inputfiles: the txt label file
    #output_files: the csv label file
    #Afastapaths: the fasta file for rna seq
    #Bfastapaths: the fasta file for protein seq
    #resultpath: the path to save the middle result for rna
    #rnafinal: the path to save the final result for rna
    #profinal: the path to save the final result for protein
    for s in [0,1]:
        input_file =input_files[s]
        output_file = output_files[s]
        Afastapath=Afastapaths[s]
        Bfastapath=Bfastapaths[s]
        resultpath=resultpaths[s]
        profinal=profinals[s]
        converted_data = []

        with open(input_file, 'r') as file:
            for line in file:
                columns = line.strip().split()
                
                converted_data.append(columns)  

        # 将转换后的数据写入新的 CSV 文件
        with open(output_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['proSeqname', 'Seqname', 'Label'])
            csv_writer.writerows(converted_data)

        print(f"转换完成，数据已保存到 {output_file}")

        RNAcoding_01.protein_coding(Bfastapath, output_file, profinal)

        for i in range(1,11):
            if(i!=1 and i!=5 and i!=9):
                dimension="1"
                savetype='csv'
                n_select=str(i)
                RNAcoding_01.RNA_coding(Afastapath, output_file, resultpath, dimension = dimension, savetype = savetype,n_select = n_select)
                
    print('concate protein features')
    for result, final in zip(resultpaths,rnafinals):
        base_path = result+'/encoding_features/'
        file_names = [
            'Entropy density of transcript (1D).csv', 'Global descriptor (1D).csv', 'K-mer (1D).csv',
            'Pseudo protein related (1D).csv', 'Guanine-cytosine related (1D).csv', 'Nucleotide related (1D).csv', 'EIIP based spectrum (1D).csv',
        ]
        file_paths = [base_path + file_name for file_name in file_names]
        # 加载并拼接特征
        names,features = load_and_concatenate_features(file_paths)
        #确保path存在
        dir_path = os.path.dirname(final)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        save_to_csv(names, features, final)
        print("Features and names have been successfully saved to ",final)

    


    return 
# load data: devide the data into training, validation, and testing sets
def load_data(args):
    if args.baseline and args.encode and args.global_rank == 0:
        print('encode raw data')
        starttime=time.time()
        if(args.task_num==1):
            resultpath='./baseline/task1_rna_encode'
            baseline_task1_encode_rna(args.rna_seq_file,args.label_file,resultpath, args.encoded_rna_file)
        else:
            input_files=[args.pair_file,args.non_pair_file]
            output_files=[args.pair_file[:-3]+'csv',args.non_pair_file[:-3]+'csv']
            Afastapaths=[args.rna_sequence_file,args.non_rna_sequence_file]
            Bfastapaths=[args.pro_sequence_file,args.non_pro_sequence_file]
            resultpath=['./baseline/task2_rna_encode/pair','./baseline/task2_rna_encode/non_pair']
            rnafinal=[args.encoded_rna_file,args.encoded_non_rna_file]
            profinal=[args.encoded_pro_file,args.encoded_non_pro_file]
            baseline_task2_encode_rna(input_files,output_files,Afastapaths,Bfastapaths,resultpath, rnafinal,profinal)

            df = pd.read_csv(args.encoded_pro_file)
            df = df.drop(df.columns[0], axis=1)
            df.to_csv(args.encoded_pro_file, index=False)
            df = pd.read_csv(args.encoded_non_pro_file)
            df = df.drop(df.columns[0], axis=1)
            df.to_csv(args.encoded_non_pro_file, index=False)
        endtime=time.time()
        encodetime=endtime-starttime
        print('time:',encodetime)
        args.encode_time=encodetime

    #for task 1
    if(args.task_num==1):
        rna_seq_file=args.rna_seq_file
        label_file=args.label_file
        as_dna=args.as_dna
        
        if args.baseline:
            rna_sequences = pd.read_csv(args.encoded_rna_file).iloc[:, 1:].values.tolist()
        else:
            rna_sequences = list(SeqIO.parse(rna_seq_file, "fasta"))
        labels_df = pd.read_csv(label_file, header=0)

        sequences = []
        labels=[]
        num=0
        length=len(labels_df['Label'])
        assert  length== len(rna_sequences), "UnMatched Data, label {} and rna_sequences {}".format(length, len(rna_sequences))
        args.whole_data_num=length
        for rna_record, label_row in zip(rna_sequences, labels_df.iterrows()):
            if args.baseline:
                sequences.append(rna_record)
                lab=label_row[1]['Label'].rstrip(';')
                labels.append(lab)
                num += 1
            elif len(rna_record.seq) <= 1022:
                if not as_dna:
                    seq = str(rna_record.seq).replace("T", "U")
                else:
                    seq = transform_dna_rna(str(rna_record.seq))
                sequences.append(seq)
                lab=label_row[1]['Label'].rstrip(';')
                labels.append(lab)
                num += 1
            if args.sample>0 and num >=  args.sample:
                break


        label_encoder = LabelEncoder()
        labels=label_encoder.fit_transform(labels)
        raw_data = list(zip(labels, sequences))

        
        label2idx = {}
        for idx, (x, _) in enumerate(raw_data):
            if x not in label2idx:
                label2idx[x] = []
            label2idx[x].append(idx)
        label2num = {k: len(v) for k, v in label2idx.items()}
        min_class_num = min(label2num.values())
        val_num = test_num = min_class_num // 6
        for k in label2idx:
            random.shuffle(label2idx[k])

        test_idx = {k: v[:test_num] for k, v in label2idx.items()}
        val_idx = {
            k: v[test_num: test_num + val_num] for k, v in label2idx.items()
        }
        train_idx = {k: v[test_num + val_num:] for k, v in label2idx.items()}

        train_data, val_data, test_data = [], [], []
        for k, v in train_idx.items():
            train_data.extend(raw_data[x] for x in v)

        for k, v in val_idx.items():
            val_data.extend(raw_data[x] for x in v)

        for k, v in test_idx.items():
            test_data.extend(raw_data[x] for x in v)

        return RNADataset(train_data), RNADataset(val_data), RNADataset(test_data)
    
    #for task 2
    else:
        pair_file=args.pair_file[:-3]+'csv'
        non_pair_file=args.non_pair_file[:-3]+'csv'
        rna_sequence_file=args.encoded_rna_file
        pro_sequence_file=args.encoded_pro_file
        non_rna_sequence_file=args.encoded_non_rna_file
        non_pro_sequence_file=args.encoded_non_pro_file
        
        if args.baseline:
            rna_seq = pd.read_csv(rna_sequence_file).iloc[:, 1:].values.tolist()
            non_rna_seq = pd.read_csv(non_rna_sequence_file).iloc[:, 1:].values.tolist()
            pro_seq = pd.read_csv(pro_sequence_file).iloc[:, 1:].values.tolist()
            non_pro_seq = pd.read_csv(non_pro_sequence_file).iloc[:, 1:].values.tolist()
        else:
            rna_seq = list(SeqIO.parse(args.rna_sequence_file, "fasta"))
            non_rna_seq = list(SeqIO.parse(args.non_rna_sequence_file, "fasta"))
            pro_seq = list(SeqIO.parse(args.pro_sequence_file, "fasta"))
            non_pro_seq = list(SeqIO.parse(args.non_pro_sequence_file, "fasta"))

        pair_df = pd.read_csv(pair_file, sep='\t', header=None)
        # print(pair_df)
        pair_df.columns = ['protein_id', 'rna_name', 'label']

        non_pair_df = pd.read_csv(non_pair_file,  sep='\t',header=None)
        non_pair_df.columns = ['protein_id', 'rna_name', 'label']

        labels_1= pair_df['label'].values
        labels_0= non_pair_df['label'].values
        length=len(labels_1)+len(labels_0)
        args.whole_data_num=length
        # print(labels_1[0])
        # print(rna_seq[0])
        # print(pro_seq[0])
        # print(labels_1[0])
        # print(rna_seq[0])
        # print(len(labels_1))
        # print(len(rna_seq))
        assert len(labels_1) == len(rna_seq), "UnMatched Data: rna_sequences"
        assert len(labels_1) == len(pro_seq), "UnMatched Data: pro_sequences"
        assert len(labels_0) == len(non_rna_seq), "UnMatched Data: non_rna_sequences"
        assert len(labels_0) == len(non_pro_seq), "UnMatched Data: non_pro_sequences"
        raw_data = [] # rna, pro, label
        num=0
        for rna_seq,pro_seq, label in zip(rna_seq,pro_seq,labels_1):
            if args.baseline:
                raw_data.append((rna_seq, pro_seq, label))
                num += 1
            elif len(rna_seq.seq) <= 1022 and len(pro_seq.seq) <= 1022:
                if not args.as_dna:
                    rna_seq = str(rna_seq.seq).replace("T", "U")
                else:
                    rna_seq = transform_dna_rna(str(rna_seq.seq))
                pro_seq = str(pro_seq.seq)
                raw_data.append((rna_seq, pro_seq, label))
                num += 1
            # else:
            #     print('rna:',len(rna_seq.seq))
            #     print('pro:',len(pro_seq.seq))
            if args.sample>0 and num >= args.sample:
                break
        num=0
        for rna_seq,pro_seq, label in zip(non_rna_seq,non_pro_seq,labels_0):
            if args.baseline:
                raw_data.append((rna_seq, pro_seq, label))
                num += 1
            elif len(rna_seq.seq) <= 1022 and len(pro_seq.seq) <= 1022:
                if not args.as_dna:
                    rna_seq = str(rna_seq.seq).replace("T", "U")
                else:
                    rna_seq = transform_dna_rna(str(rna_seq.seq))
                pro_seq = str(pro_seq.seq)
                raw_data.append((rna_seq, pro_seq, label))
                num += 1
            if args.sample>0 and num >= args.sample:
                break

        lengh=len(raw_data)
        count_1=0
        count_0=0
        for data in raw_data:
            if data[2]==1:
                count_1+=1
            else:
                count_0+=1
        print('positive:',count_1)
        print('negative:',count_0)
        val_num = test_num =lengh // 6
        print('train_num:',lengh - val_num - test_num)
        print('val_num:',val_num)
        print('test_num:',test_num)
        random.shuffle(raw_data)
        train_data = raw_data[:lengh - val_num - test_num]
        val_data = raw_data[lengh - val_num - test_num: lengh - test_num]
        test_data = raw_data[lengh - test_num:]
        return RNADataset(train_data), RNADataset(val_data), RNADataset(test_data)



# RNA dataset , assert balance
class RNADataset(torch.utils.data.Dataset):
    def __init__(self, data, seed=2023):
        super(RNADataset, self).__init__()
        self.data = data
        random.shuffle(self.data)
        if(len(data)>0 and len(data[0])==2): #对于task2不需要进行balance，也就不需要进行统计
            self.label2idx = {}
            for idx, (x, _) in enumerate(self.data):
                if x not in self.label2idx:
                    self.label2idx[x] = []
                self.label2idx[x].append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def create_balance_subset(self, max_frac=4):
        min_class_num = min(len(x) for x in self.label2idx.values())
        out_data = []
        for k, v in self.label2idx.items():
            if len(v) <= max_frac * min_class_num:
                out_data.extend(self.data[x] for x in v)
            else:
                all_idx = list(range(len(v)))
                random.shuffle(all_idx)
                max_l = int(max_frac * min_class_num)
                out_data.extend(self.data[v[x]] for x in all_idx[:max_l])

        return RNADataset(out_data)
