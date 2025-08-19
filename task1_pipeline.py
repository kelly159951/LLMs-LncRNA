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
def load_data(fasta_file, label_file, as_dna=False):
    sequences = []
    label_mapper = {
        'Antisense;': 0, 'Exonic;': 1, 'Linc;': 2,
        'Sense No Exonic;': 3
    }
    with open(fasta_file) as f:
        for record in SeqIO.parse(f, "fasta"):
            if not as_dna:
                seq = str(record.seq).replace("T", "U")
            else:
                seq = transform_dna_rna(str(record.seq))
            sequences.append(seq)

    label_df = pandas.read_csv(label_file)
    # print(len(label_df['Label']), len(sequences))
    # print(label_df['Label'][len(label_df['Label']) - 1])
    assert len(label_df['Label']) == len(sequences), "UnMatched Data"
    raw_data = []
    label2idx = {i: [] for i in range(4)}
    for idx, p in enumerate(label_df['Label']):
        raw_data.append((label_mapper[p], sequences[idx]))
        label2idx[label_mapper[p]].append(idx)

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

# RNA dataset , assert balance
class RNADataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super(RNADataset, self).__init__()
        self.data = data
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

# model
class Model(torch.nn.Module):
    def __init__(self, rna_model, dropout=0.0, attn_pool=False):
        super(Model, self).__init__()
        self.rna_model = rna_model
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(640, 640),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(640, 320),
            torch.nn.GELU(),
            torch.nn.Linear(320, 4)
        )
        if attn_pool:#Optional attention pooling layer
            self.pool_query = torch.nn.Parameter(torch.randn(1, 1, 640))
            self.pooler = torch.nn.MultiheadAttention(
                640, 8, dropout=dropout, batch_first=True
            )
        self.attn_pool = attn_pool

    def forward(self, batch_tokens):
        hidden = self.rna_model(batch_tokens, repr_layers=[12])["representations"][12]
        if self.attn_pool:
            batch_size = hidden.shape[0]
            padding_mask = batch_tokens == self.rna_model.padding_idx
            pk = self.pool_query.repeat(batch_size, 1, 1)
            feat, _ = self.pooler(
                query=pk, value=hidden, key=hidden,
                key_padding_mask=padding_mask
            )
            return self.mlp(feat.squeeze(dim=1))
        else:
            return self.mlp(hidden[:, 0])


def train(loader, model, optimizer, converter, device):
    model = model.train()
    losses = []
    for data in tqdm(loader):
        batch_labels, batch_strs, batch_tokens = converter(data)
        batch_tokens = batch_tokens.to(device)
        batch_labels = torch.LongTensor(batch_labels).to(device)
        results = model(batch_tokens)
        loss = torch.nn.functional.cross_entropy(results, batch_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    return np.mean(losses)


def evaluate(loader, model, converter, device):
    model = model.eval()
    y_pred, y_true = [], []
    for data in tqdm(loader):
        batch_labels, batch_strs, batch_tokens = converter(data)
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            resutls = model(batch_tokens).argmax(dim=-1)
            y_pred.append(resutls.cpu())
        y_true.extend(batch_labels)

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
    detail_log_dir = os.path.join(args.base_log, f'task1_log-{timestamp}.json')
    detail_model_dir = os.path.join(args.base_log, f'task1_mod-{timestamp}.pth')
    data_figure_dir=os.path.join(args.base_figure, f'task1_train-{timestamp}-figure.png')
    return detail_log_dir, detail_model_dir, data_figure_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--lr', type=float, default=1e-4,
        help='the learning rate for model training'
    )
    parser.add_argument(
        '--bs', type=int, default=32,
        help='the batch size for training'
    )

    parser.add_argument(
        '--base_log', type=str, default='log',
        help='the path of folder for logging'
    )
    parser.add_argument(
        '--base_figure', type=str, default='figure',
        help='the path of folder for logging'
    )
    parser.add_argument(
        '--inbalance_frac', type=float, default=4,
        help='the max frac between the max size class and min size class'
    )

    parser.add_argument(
        '--dropout', type=float, default=0,
        help='the dropout of model'
    )

    parser.add_argument(
        '--epochs', type=int, default=100,
        help='the number of epochs'
    )
    parser.add_argument(
        '--disable_layer', type=int, default=10,
        help='the number of freeze layers of rnafm'
    )

    parser.add_argument(
        '--fasta_file', type=str, required=True,
        help='the path of fasta file'
    )
    parser.add_argument(
        '--label_file', type=str, required=True,
        help='the path of label file'
    )
    parser.add_argument(
        '--as_dna', action='store_true',
        help='treat the data as dna or not'
    )
    parser.add_argument(
        '--num_worker', type=int, default=4,
        help='the num of worker for dataloader'
    )
    parser.add_argument(
        '--device', type=int, default=0,
        help='the cuda id, minus for cpu'
    )
    parser.add_argument(
        '--checkpoint', type=str, default='',
        help='the path of pretrained checkpoint'
    )
    parser.add_argument(
        '--use_pool', action='store_true',
        help='use pooling instead of global feature'
    )
    parser.add_argument(
        '--seed', type=int, default=2023,
        help='the random seed for training'
    )

    args = parser.parse_args()
    print(args)

    fix_seed(args.seed)

    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')

    log_dir, model_dir, figure_dir = create_log_model(args)

    train_all_set, valid_set, test_set = load_data(
        args.fasta_file, args.label_file, args.as_dna
    )

    model, alphabet = fm.pretrained.rna_fm_t12()
    batch_converter = alphabet.get_batch_converter()

    allmodel = Model(model, args.dropout, args.use_pool).to(device)
    if args.checkpoint != '':
        weight = torch.load(args.checkpoint, map_location=device)
        allmodel.load_state_dict(weight)

    x_paras = []
    for k, v in allmodel.named_parameters():
        if args.disable_layer > 0 and k.startswith('rna_model.embed_tokens'):
            v.requires_grad = False
        elif args.disable_layer > 0 and k.startswith('rna_model.embed_positions'):
            v.requires_grad = False
        elif args.disable_layer > 0 and k.startswith('rna_model.emb_layer_norm_before'):
            v.requires_grad = False
        elif k.startswith('rna_model.layers'):
            if int(k.split('.')[2]) < args.disable_layer:
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
        train_set = train_all_set.create_balance_subset(args.inbalance_frac)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.bs, num_workers=args.num_worker,
            collate_fn=lambda x: x, shuffle=True
        )

        train_loss = train(
            train_loader, allmodel, optimizer, batch_converter, device
        )

        valid_acc = evaluate(valid_loader, allmodel, batch_converter, device)
        test_res = evaluate(test_loader, allmodel, batch_converter, device)

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
    print(f'[INFO] best valid acc: {val_acc[best_ep]}')
    print(f'[INFO] best test acc: {test_acc[best_ep]}')
    with open(log_dir, 'w') as Fout:
        json.dump({
            'args': args.__dict__,
            'best acc epoch': best_ep,'best valid acc':val_acc[best_ep],'bast test acc':test_acc[best_ep], 
            'train_loss': losses,'valid_metric': val_acc, 'test_metric': test_acc
        }, Fout, indent=4)



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

# Call plotting function and save
plot_training_results(losses, val_acc, test_acc, figure_dir)




#示例代码:python task1_pipeline.py --fasta_file='./dataset/task1_rna.fasta' --label_file='./dataset/task1_rna_label.csv' --lr=0.0001 --bs=32 --base_log='./log_pipeline' --inbalance_frac=4.0 --dropout=0.1 --epochs=30 --disable_layer=12 --num_worker=4 --device=4 --use_pool --seed=2023
