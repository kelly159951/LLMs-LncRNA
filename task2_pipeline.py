import json
import torch
import numpy as np
import os
import random
import pandas
from Bio import SeqIO
import argparse
from tqdm import tqdm
import time
import fm
import pandas as pd
import esm  
import matplotlib.pyplot as plt

# set random seed
def fix_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

# transform DNA to RNA
def transform_dna_rna(seq):
    mapper = {'A': 'U', 'T': 'A', 'C': "G", 'G': 'C'}
    seq = [mapper[x] for x in seq]
    return ''.join(seq)

# load data: devide the data into training, validation, and testing sets

def load_data(pair_file, rna_sequence_file, pro_sequence_file, non_pair_file, non_rna_sequence_file, non_pro_sequence_file, as_dna=False):
    rna_sequences = []
    pro_sequences = []
    
    with open(rna_sequence_file) as f:
        for record in SeqIO.parse(f, "fasta"):
            if not as_dna:
                seq = str(record.seq).replace("T", "U")
            else:
                seq = transform_dna_rna(str(record.seq))
            rna_sequences.append(seq)
    with open(non_rna_sequence_file) as f:
        for record in SeqIO.parse(f, "fasta"):
            if not as_dna:
                seq = str(record.seq).replace("T", "U")
            else:
                seq = transform_dna_rna(str(record.seq))
            rna_sequences.append(seq)

    with open(pro_sequence_file) as f:
        for record in SeqIO.parse(f, "fasta"):
            pro_sequences.append(str(record.seq))
    with open(non_pro_sequence_file) as f:
        for record in SeqIO.parse(f, "fasta"):
            pro_sequences.append(str(record.seq))

    pair_df = pd.read_csv(pair_file, sep='\t', header=None)
    pair_df.columns = ['protein_id', 'rna_name', 'label']

    non_pair_df = pd.read_csv(non_pair_file, sep='\t', header=None)
    non_pair_df.columns = ['protein_id', 'rna_name', 'label']

    labels_1= pair_df['label'].values
    labels_0= non_pair_df['label'].values
    labels=labels_1.tolist()+labels_0.tolist()
    rawlengh=len(labels)
    assert rawlengh == len(rna_sequences), "UnMatched Data: rna_sequences"
    assert rawlengh == len(pro_sequences), "UnMatched Data: pro_sequences"
    raw_data = [] # rna, pro, label
    for i in range(rawlengh):
        if(len(rna_sequences[i])<=1022 and len(pro_sequences[i])<=1022):
            raw_data.append((rna_sequences[i], pro_sequences[i], labels[i]))
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
    def __init__(self, data):
        super(RNADataset, self).__init__()
        self.data = data
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# model
class Model(torch.nn.Module):
    def __init__(self, rna_model, pro_model, dropout=0.0, attn_pool=False):
        super(Model, self).__init__()
        self.rna_model = rna_model
        self.pro_model = pro_model
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(1920, 640),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(640, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 2)
        )
        if attn_pool:#Optional attention pooling layer
            self.rna_pool_query = torch.nn.Parameter(torch.randn(1, 1, 640))
            self.pro_pool_query = torch.nn.Parameter(torch.randn(1, 1, 1280))
            self.rna_pooler = torch.nn.MultiheadAttention(
                640, 8, dropout=dropout, batch_first=True
            )
            self.pro_pooler = torch.nn.MultiheadAttention(
                1280, 8, dropout=dropout, batch_first=True
            )
        self.attn_pool = attn_pool

    def forward(self, batch_tokens):
        rna_tokens, pro_tokens = batch_tokens
        rna_hidden = self.rna_model(rna_tokens, repr_layers=[12])["representations"][12] # [batch, seq_len+2, 640]
        pro_hidden = self.pro_model(pro_tokens, repr_layers=[33])["representations"][33]  # [batch, seq_len+2, 1280]
        # print('rna_hidden:',rna_hidden.shape)
        # print('pro_hidden:',pro_hidden.shape)

        if self.attn_pool:
            # Apply attention pooling to rna_hidden
            batch_size = rna_hidden.shape[0]
            rna_padding_mask = rna_tokens == self.rna_model.padding_idx
            rna_pk = self.rna_pool_query.repeat(batch_size, 1, 1)
            rna_feat, _ = self.rna_pooler(
                query=rna_pk, value=rna_hidden, key=rna_hidden,
                key_padding_mask=rna_padding_mask
            )
            rna_feat=rna_feat.squeeze(dim=1)
            # print('rna_feat:',rna_feat.shape)
            # Apply attention pooling to pro_hidden
            pro_padding_mask = pro_tokens == self.pro_model.padding_idx
            pro_pk = self.pro_pool_query.repeat(batch_size, 1, 1)
            # print(f"pro_pk shape: {pro_pk.shape}")
            # print(f"pro_hidden shape: {pro_hidden.shape}")
            pro_feat, _ = self.pro_pooler(
                query=pro_pk, value=pro_hidden, key=pro_hidden,
                key_padding_mask=pro_padding_mask
            )
            pro_feat=pro_feat.squeeze(dim=1)
            # print('pro_feat:',pro_feat.shape)

            feat=torch.cat([rna_feat, pro_feat], dim=-1)
            return self.mlp(feat)
        else:
            hidden = torch.cat([rna_hidden[0], pro_hidden[0]], dim=-1)  # [batch, seq_len+2, 1920]
            return self.mlp(hidden[:, 0])  # Use only the first token representation
        
def train(loader, model, optimizer, rna_converter, pro_converter, device):
    model = model.train()
    losses = []
    for data in tqdm(loader):
        # Extract RNA and Protein sequences
        rna_strs = [d[0] for d in data]
        pro_strs = [d[1] for d in data]

        # Use corresponding converters for transformation
        _, _, rna_tokens = rna_converter([(None, rna_str) for rna_str in rna_strs])
        _, _, pro_tokens = pro_converter([(None, pro_str) for pro_str in pro_strs])

        rna_tokens = rna_tokens.to(device)
        pro_tokens = pro_tokens.to(device)
        batch_labels = torch.LongTensor([d[2] for d in data]).to(device)  # Get labels
        # Use RNA and Protein tokens as model input
        results = model((rna_tokens, pro_tokens))
        
        loss = torch.nn.functional.cross_entropy(results, batch_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    return np.mean(losses)


def evaluate(loader, model, rna_converter, pro_converter, device):
    model = model.eval()
    y_pred, y_true = [], []
    for data in tqdm(loader):
        # Extract RNA and Protein sequences
        rna_strs = [d[0] for d in data]
        pro_strs = [d[1] for d in data]

        # Use corresponding converters for transformation
        _, _, rna_tokens = rna_converter([(None, rna_str) for rna_str in rna_strs])
        _, _, pro_tokens = pro_converter([(None, pro_str) for pro_str in pro_strs])

        rna_tokens = rna_tokens.to(device)
        pro_tokens = pro_tokens.to(device)

        # Use RNA and Protein tokens as model input
        with torch.no_grad():
            results = model((rna_tokens, pro_tokens))
            # print('results:',results)
            results=results.argmax(dim=-1)
            y_pred.append(results.cpu())
        y_true.extend([d[2] for d in data])  # Get labels

    y_true = torch.LongTensor(y_true)
    y_pred = torch.cat(y_pred, dim=0)
    acc = (y_true == y_pred).float().mean()
    return acc.item()



def create_log_model(args):
    timestamp = time.time()
    if not os.path.exists(args.base_log):
        os.makedirs(args.base_log)
    if not os.path.exists(args.base_figure):
        os.makedirs(args.base_figure)
    detail_log_dir = os.path.join(args.base_log, f'task2_log-{timestamp}.json')
    detail_model_dir = os.path.join(args.base_log, f'task2_mod-{timestamp}.pth')
    data_figure_dir=os.path.join(args.base_figure, f'task2_train-{timestamp}-figure.png')
    return detail_log_dir, detail_model_dir, data_figure_dir




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4,help='the learning rate for model training')
    parser.add_argument('--bs', type=int, default=32,help='the batch size for training')
    parser.add_argument('--base_log', type=str, default='log',help='the path of folder for logging')
    parser.add_argument('--base_figure', type=str, default='figure',help='the path of folder for figure')
    parser.add_argument('--inbalance_frac', type=float, default=4,help='the max frac between the max size class and min size class')
    parser.add_argument('--dropout', type=float, default=0,help='the dropout of model')
    parser.add_argument('--epochs', type=int, default=100,help='the number of epochs')
    parser.add_argument('--disable_layer_rna', type=int, default=10,help='the number of freeze layers of rnafm')
    parser.add_argument('--pair_file', type=str, required=True,help='the path of pair file')
    parser.add_argument('--rna_sequence_file', type=str, required=True,help='the path of rna sequence file')
    parser.add_argument('--pro_sequence_file', type=str, required=True,help='the path of protein sequence file')
    parser.add_argument('--non_pair_file', type=str, required=True,help='the path of non pair file')
    parser.add_argument('--non_rna_sequence_file', type=str, required=True,help='the path of non rna sequence file')
    parser.add_argument('--non_pro_sequence_file', type=str, required=True,help='the path of non protein sequence file')
    parser.add_argument('--as_dna', action='store_true',help='treat the data as dna or not')
    parser.add_argument('--num_worker', type=int, default=4,help='the num of worker for dataloader')
    parser.add_argument('--device', type=int, default=0,help='the cuda id, minus for cpu')
    parser.add_argument('--checkpoint', type=str, default='',help='the path of pretrained checkpoint')
    parser.add_argument('--use_pool', action='store_true',help='use pooling instead of global feature')
    parser.add_argument('--seed', type=int, default=2023,help='the random seed for training')

    args = parser.parse_args()
    print(args)

    fix_seed(args.seed)

    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')
    # # Positive interactions
    # pair_file = './dataset/task2_rna_protein_pairs.txt' #proid-rnaname-label
    # rna_sequence_file = './dataset/task2_rna_sequences.txt' #rnaname-rnaseq
    # pro_sequence_file = './dataset/task2_pro_sequences.txt' #proid-proseq

    # # Negative interactions
    # non_pair_file = './dataset/task2_nonrna_protein_pairs.txt' #proid-rnaname-label
    # non_rna_sequence_file = './dataset/task2_nonrna_sequences.txt' #rnaname-rnaseq
    # non_pro_sequence_file = './dataset/task2_nonpro_sequences.txt' #proid-proseq

    log_dir, model_dir, figure_dir = create_log_model(args)

    train_set, valid_set, test_set = load_data(
        args.pair_file, args.rna_sequence_file, args.pro_sequence_file,
        args.non_pair_file, args.non_rna_sequence_file, args.non_pro_sequence_file,
        as_dna=args.as_dna
    )
    rna_model, rna_alphabet = fm.pretrained.rna_fm_t12()
    pro_model, pro_alphabet = esm.pretrained.esm2_t33_650M_UR50D()  

    rna_batch_converter = rna_alphabet.get_batch_converter()
    pro_batch_converter = pro_alphabet.get_batch_converter()  

    allmodel = Model(rna_model,pro_model, args.dropout, args.use_pool).to(device)
    if args.checkpoint != '':
        weight = torch.load(args.checkpoint, map_location=device)
        allmodel.load_state_dict(weight)

    x_paras = []
    for k, v in allmodel.named_parameters():
        if k.startswith('pro_model'): # Freeze all protein model parameters
            v.requires_grad = False
        if args.disable_layer_rna > 0 and k.startswith('rna_model.embed_tokens'):
            v.requires_grad = False
        elif args.disable_layer_rna > 0 and k.startswith('rna_model.embed_positions'):
            v.requires_grad = False
        elif args.disable_layer_rna > 0 and k.startswith('rna_model.emb_layer_norm_before'):
            v.requires_grad = False
        elif k.startswith('rna_model.layers'):
            if int(k.split('.')[2]) < args.disable_layer_rna:
                v.requires_grad = False
            else:
                print('enabled', k)
                x_paras.append(v)
        else:
            print('enabled', k)
            x_paras.append(v)

    optimizer = torch.optim.Adam(x_paras, lr=args.lr)
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=args.bs, num_workers=args.num_worker,
        collate_fn=lambda x: x, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.bs, num_workers=args.num_worker,
        collate_fn=lambda x: x, shuffle=False
    )

    losses, val_acc, test_acc = [], [], []
    best_perf, best_ep = None, None
    for ep in range(args.epochs):
        print(f'[Epoch] -----------{ep}----------')
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.bs, num_workers=args.num_worker,
            collate_fn=lambda x: x, shuffle=True
        )

        train_loss = train(
            train_loader, allmodel, optimizer, rna_batch_converter, pro_batch_converter, device
        )

        valid_acc = evaluate(valid_loader, allmodel, rna_batch_converter, pro_batch_converter, device)
        test_res = evaluate(test_loader, allmodel, rna_batch_converter, pro_batch_converter, device)

        losses.append(train_loss)
        val_acc.append(valid_acc)
        test_acc.append(test_res)

        print('[train]', train_loss)
        print('[valid]', valid_acc)
        print('[test]', test_res)



        if best_perf is None or valid_acc > best_perf:
            best_perf, best_ep = valid_acc, ep
            torch.save(allmodel.state_dict(), model_dir)

    print(f'[INFO] best acc epoch: {best_ep}')
    print(f'[INFO] best valid loss: {val_acc[best_ep]}')
    print(f'[INFO] best test loss: {test_acc[best_ep]}')

    with open(log_dir, 'w') as Fout:
        json.dump({
            'args': args.__dict__, 'train_loss': losses,
            'best acc epoch': best_ep,'best valid acc':val_acc[best_ep],'bast test acc':test_acc[best_ep], 
            'valid_metric': val_acc, 'test_metric': test_acc
        }, Fout, indent=4)


# Plotting function
def plot_training_results(losses, val_acc, test_acc, save_path):
    epochs = range(1, len(losses) + 1)

    # Plot training loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()

    # Plot validation and test accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_acc, label='Validation Accuracy', color='blue')
    plt.plot(epochs, test_acc, label='Test Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation and Test Accuracy over Epochs')
    plt.legend()

    # Save plot
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

plot_training_results(losses, val_acc, test_acc, figure_dir)



#python task2_pipeline.py --pair_file='./dataset/task2_rna_protein_pairs.txt' --rna_sequence_file='./dataset/task2_rna_sequences.fasta' --pro_sequence_file='./dataset/task2_pro_sequences.fasta' --non_pair_file='./dataset/task2_nonrna_protein_pairs.txt' --non_rna_sequence_file='./dataset/task2_nonrna_sequences.fasta' --non_pro_sequence_file='./dataset/task2_nonpro_sequences.fasta' --lr=0.0001 --bs=32 --dropout=0.1 --epochs=30 --disable_layer_rna=12 --base_log='./log_pipeline' --num_worker=4 --device=2 --use_pool --seed=2023


#python task2_pipeline.py --pair_file='./dataset/task2_sample_rna_protein_pairs.txt' --rna_sequence_file='./dataset/task2_sample_rna_sequences.fasta' --pro_sequence_file='./dataset/task2_sample_pro_sequences.fasta' --non_pair_file='./dataset/task2_sample_nonrna_protein_pairs.txt' --non_rna_sequence_file='./dataset/task2_sample_nonrna_sequences.fasta' --non_pro_sequence_file='./dataset/task2_sample_nonpro_sequences.fasta' --lr=0.0001 --bs=32 --dropout=0.1 --epochs=30 --disable_layer_rna=12 --base_log='./log_pipeline' --num_worker=4 --device=2 --use_pool --seed=2023
