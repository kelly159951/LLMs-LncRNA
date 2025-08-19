import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import fm
import matplotlib.pyplot as plt
from task1_final_model import *
from task1_final_config import *

def evaluate(rna_fm_model, mlp_model, data_loader, criterion, device, state='valid'):
    rna_fm_model.eval()
    mlp_model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"validing"):
            inputs, labels = batch
            inputs, labels = inputs.to(device).long() , labels.to(device).long() 
            encodes = rna_fm_model(inputs)
            # print(encodes.size())
            outputs = mlp_model(encodes)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    val_loss /= len(data_loader)
    print(state+f" Accuracy: {accuracy * 100:.2f}%, Loss: {val_loss:.4f}")
    return val_loss, accuracy

# Training步骤
def train(rna_fm_model, mlp_model, train_loader, val_loader, criterion, device):
    rna_fm_model.train()
    mlp_model.train()
    optimizer_1 = optim.Adam(mlp_model.parameters(), lr=lr_m1)
    optimizer_2 = optim.Adam(rna_fm_model.parameters(), lr=lr_fm)
    optimizer_3 = optim.Adam(mlp_model.parameters(), lr=lr_m2)

    #第一阶段Training：只Trainingmlp
    # 冻结 rna_fm 模型的所有参数
    for param in rna_fm_model.parameters():
        param.requires_grad = False
    for param in mlp_model.parameters():
        param.requires_grad = True

    # 计算类别中最少的样本数
    num_samples_per_class=calculate_min_samples_per_class(train_loader,2)
    print('最少样本数的类别的样本数: ', num_samples_per_class)
    result['num_samples_per_class']=num_samples_per_class
    best_val_loss=float('inf')
    for epoch in range(num_epochs_1):
        print('epoch:',epoch)
        #随机获取均衡的样本
        balenced_train_loader = create_balanced_loader(train_loader, num_samples_per_class=num_samples_per_class, batch_size=batch_size)
        train_loss = 0
        for batch in tqdm(balenced_train_loader, desc="Training"):
            tokens,labels= batch
            tokens= tokens.to(device).long() 
            labels= labels.to(device).long()
            optimizer_1.zero_grad()
            encodes = rna_fm_model(tokens)
            outputs = mlp_model(encodes)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_1.step()
            train_loss += loss.item()
        train_loss /= len(balenced_train_loader)
        print(f"Epoch {epoch+1}/{num_epochs_1} completed, loss: {train_loss}.")
        val_loss, val_accuracy = evaluate(rna_fm_model, mlp_model, val_loader, criterion, device)
        result['s1_train_losses'].append(train_loss)
        result['s1_val_losses'].append(val_loss)
        result['s1_val_accuracies'].append(val_accuracy)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(mlp_model.state_dict(), mlp_model_save_path)
            print(f"update Model saved to {mlp_model_save_path}")

    #第二阶段Training：同时Trainingrna_fm和mlp
    # 冻结 rna_fm 模型的所有参数
    for param in rna_fm_model.parameters():
        param.requires_grad = True
    for param in mlp_model.parameters():
        param.requires_grad = True
    # 计算类别中最少的样本数
    best_val_loss=float('inf')
    for epoch in range(num_epochs_2):
        print('epoch:',epoch)
        #随机获取均衡的样本
        balenced_train_loader = create_balanced_loader(train_loader, num_samples_per_class=num_samples_per_class, batch_size=batch_size)
        train_loss = 0
        for batch in tqdm(balenced_train_loader, desc="Training"):
            tokens,labels= batch
            tokens= tokens.to(device).long() 
            labels= labels.to(device).long()
            optimizer_3.zero_grad()
            optimizer_2.zero_grad()
            encodes = rna_fm_model(tokens)
            outputs = mlp_model(encodes)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_3.step()
            optimizer_2.step()
            train_loss += loss.item()
        train_loss /= len(balenced_train_loader)
        print(f" 1, Epoch {epoch+1}/{num_epochs_2} completed, loss: {train_loss}.")
        val_loss, val_accuracy = evaluate(rna_fm_model, mlp_model, val_loader, criterion, device)
        result['s2_train_losses'].append(train_loss)
        result['s2_val_losses'].append(val_loss)
        result['s2_val_accuracies'].append(val_accuracy)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(mlp_model.state_dict(), mlp_model_save_path)
            torch.save(rna_fm_model.state_dict(), rna_fm_save_path)
            print(f"update Model saved to {mlp_model_save_path}, {rna_fm_save_path}")
            





seed_torch(2021)

# 加载 RNA-FM 模型,初始化 MLP 模型
rna_fm_model, alphabet = fm.pretrained.rna_fm_t12()
rna_fm_model = RF(rna_fm_model).to(device)
mlp_model=MLP(input_dim, hidden_dim_1, hidden_dim_2, output_dim).to(device)
print("create model sucessfully")

# Load data集
train_loader,val_loader,test_loader,labels=load_split_data(fasta_file,label_file,alphabet, batch_size,collate_fn)
print('load data sucessfully')

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=alphabet.padding_idx)  # 忽略填充标记的损失

# Train model
print('start train')
s1_train_losses = []
s1_val_losses = []
s1_val_accuracies = []
train(rna_fm_model,mlp_model, train_loader, val_loader, criterion, device)
print('end train')

print('start test')
test_loss, test_accuracy = evaluate(rna_fm_model, mlp_model ,test_loader, criterion, device,'Test ')
result['test_loss']=test_loss
result['test_accuracy']=test_accuracy
print('end test')









#绘图保存
# 创建一个包含两个子图的图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# 绘制左图 - 阶段1的Training损失和验证损失
color = 'tab:red'
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color=color)
ln1 = ax1.plot(range(1, len(result['s1_train_losses']) + 1), result['s1_train_losses'], label='Train Loss', color=color)
ln2 = ax1.plot(range(1, len(result['s1_val_losses']) + 1), result['s1_val_losses'], label='Validation Loss', color='tab:orange')
ax1.tick_params(axis='y', labelcolor=color)

# 创建一个共享x轴的第二个y轴来绘制准确率
ax1_2 = ax1.twinx()  
color = 'tab:blue'
ax1_2.set_ylabel('Accuracy', color=color)
ln3 = ax1_2.plot(range(1, len(result['s1_val_accuracies']) + 1), result['s1_val_accuracies'], label='Validation Accuracy', color=color)
ax1_2.tick_params(axis='y', labelcolor=color)

# 合并图例
lns1 = ln1 + ln2 + ln3
labs1 = [l.get_label() for l in lns1]
ax1.legend(lns1, labs1, loc='upper left')

ax1.set_title('Stage 1: Training and Validation Loss and Accuracy')

# 绘制右图 - 阶段2的Training损失和验证损失
color = 'tab:red'
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss', color=color)
ln4 = ax2.plot(range(1, len(result['s2_train_losses']) + 1), result['s2_train_losses'], label='Train Loss', color=color)
ln5 = ax2.plot(range(1, len(result['s2_val_losses']) + 1), result['s2_val_losses'], label='Validation Loss', color='tab:orange')
ax2.tick_params(axis='y', labelcolor=color)

# 创建一个共享x轴的第二个y轴来绘制准确率
ax2_2 = ax2.twinx()  
color = 'tab:blue'
ax2_2.set_ylabel('Accuracy', color=color)
ln6 = ax2_2.plot(range(1, len(result['s2_val_accuracies']) + 1), result['s2_val_accuracies'], label='Validation Accuracy', color=color)
ax2_2.tick_params(axis='y', labelcolor=color)

# 合并图例
lns2 = ln4 + ln5 + ln6
labs2 = [l.get_label() for l in lns2]
ax2.legend(lns2, labs2, loc='upper left')

ax2.set_title('Stage 2: Training and Validation Loss and Accuracy')

# 保存图形
plt.tight_layout()
plt.savefig(figure_save_path)
plt.close()
#保存结果
import json
with open(result_save_path, 'w') as f:
    json.dump(result, f, indent=4)