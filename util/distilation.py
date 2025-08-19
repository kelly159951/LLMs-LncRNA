import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
from Bio import SeqIO
import fm

class DistilledRNAModel(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.embed_tokens = original_model.embed_tokens
        self.layers = self.create_pruned_layers(original_model)
        self.contact_head = original_model.contact_head
        self.embed_positions = original_model.embed_positions
        self.emb_layer_norm_before = original_model.emb_layer_norm_before
        self.emb_layer_norm_after = original_model.emb_layer_norm_after
        self.lm_head = original_model.lm_head
        self.load_original_weights(original_model)

    def create_pruned_layers(self, original):
        keep_layers = [original.layers[i] for i in range(1, 12, 4)]
        return nn.ModuleList(keep_layers)

    def load_original_weights(self, original):
        """手动复制所有可训练参数"""
        self.embed_tokens.load_state_dict(original.embed_tokens.state_dict())
        self.embed_positions.load_state_dict(original.embed_positions.state_dict())
        self.emb_layer_norm_before.load_state_dict(original.emb_layer_norm_before.state_dict())
        self.emb_layer_norm_after.load_state_dict(original.emb_layer_norm_after.state_dict())
        self.contact_head.load_state_dict(original.contact_head.state_dict())
        self.lm_head.load_state_dict(original.lm_head.state_dict())

    def forward(self, input_ids, positions=None):
        """前向传播"""
        x = self.embed_tokens(input_ids)
        if positions is not None:
            pos_emb = self.embed_positions(positions)
            print(pos_emb.size())
            x += pos_emb
        x = self.emb_layer_norm_before(x)
        for layer in self.layers:
            x = layer(x)[0]
        x = self.emb_layer_norm_after(x)
        lm_logits = self.lm_head(x)
        print(lm_logits.size())
        return lm_logits

def distillation_loss(student_output, teacher_output, T=2.0, alpha=0.7):
    """蒸馏损失函数"""
    soft_target_loss = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(student_output / T, dim=-1),
        F.softmax(teacher_output / T, dim=-1)
    )
    hard_target_loss = nn.MSELoss()(student_output, teacher_output.detach())
    return alpha * soft_target_loss + (1. - alpha) * hard_target_loss

class RNADataset(Dataset):
    """RNA 数据集"""
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long)

def train_distillation(student_model, teacher_model, dataloader, optimizer, logger, num_epochs=10, device='cuda'):
    """蒸馏训练过程"""
    teacher_model.eval()
    student_model.train()
    best_loss = 1e9

    for epoch in range(num_epochs):
        total_loss = 0
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for batch in pbar:
                optimizer.zero_grad()
                inputs = batch.to(device)
                with torch.no_grad():
                    teacher_output = teacher_model(inputs, repr_layers=[12])["representations"][12]
                student_output = student_model(inputs)
                loss = distillation_loss(student_output, teacher_output)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                if total_loss < best_loss:
                    best_loss = total_loss
                    torch.save(student_model.state_dict(), './log/distillation/best_model.pth')

                pbar.set_postfix(loss=total_loss / (pbar.n + 1))

        # 记录蒸馏训练指标
        logger.log_distillation(
            epoch=epoch + 1,
            train_loss=total_loss / len(dataloader),
            lr=optimizer.param_groups[0]['lr'],  # 当前学习率
            best_loss=best_loss
        )

def distillate(data_file, logger, num_epochs=50, batch_size=64, lr=1e-4, device='cuda'):

    # 初始化模型
    print('Initializing models')
    rna_model, alphabet = fm.pretrained.rna_fm_t12()
    batch_converter = alphabet.get_batch_converter()
    teacher_model = rna_model.to(device)
    student_model = DistilledRNAModel(rna_model).to(device)

    # 准备数据
    print('Loading data')
    rna_raw_sequences = list(SeqIO.parse(data_file, "fasta"))
    rna_sequences = []
    for rna_record in rna_raw_sequences:
        if len(rna_record.seq) <= 1022:
            seq = str(rna_record.seq).replace("T", "U")
            rna_sequences.append(seq)
    _, _, rna_tokens = batch_converter([(None, seq) for seq in rna_sequences])
    print(rna_tokens)

    # 数据集和数据加载器
    dataset = RNADataset(rna_tokens)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 设置优化器
    optimizer = optim.Adam(student_model.parameters(), lr=lr)

    # 训练
    print('Starting distillation training')
    train_distillation(student_model, teacher_model, dataloader, optimizer, logger, num_epochs=num_epochs, device=device)

    # 返回训练好的学生模型
    return student_model