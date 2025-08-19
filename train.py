import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from model import *
from config import *

# # 加载数据
# data = pd.read_csv(data_path)
# # 加载标签数据
# labels_df = pd.read_csv(label_path)
# labels_df.columns = labels_df.columns.str.strip()  # 去除列名的前空格
# labels_df['Label'] = labels_df['Label'].str.strip()  # 去除Label列空格
# labels_df['Label'] = labels_df['Label'].str.replace(';', '')  # 去除；

# # 数据标准化
# scaler = StandardScaler()
# data_scaled = scaler.fit_transform(data.iloc[:, 1:])
# # 数据转换为 PyTorch 张量
# data_tensor = torch.tensor(data_scaled, dtype=torch.float32)
# data_tensor = data_tensor.unsqueeze(1).unsqueeze(3)  # 增加一个维度以匹配 Conv2d 的输入要求

# # 将标签映射为数字
# label_mapping = {'Linc': 0, 'Sense No Exonic': 1, 'Antisense': 2, 'Exonic': 3}
# labels = labels_df['Label'].map(label_mapping).values
# # 转换为 PyTorch 张量
# labels = torch.tensor(labels, dtype=torch.long)

# # 划分训练集和验证集
# train_size = int(rate_t_v * len(data_tensor))
# valid_size = len(data_tensor) - train_size
# train_dataset, valid_dataset = random_split(TensorDataset(data_tensor, labels), [train_size, valid_size])

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers)




# 加载数据
train_loader,valid_loader,test_loader, labels=load_split_data(data_path,label_path,batch_size=batch_size)
# 模型初始化
autoencoder = RNAAutoencoder()
classifier = MLP()
autoencoder = nn.DataParallel(autoencoder)
classifier = nn.DataParallel(classifier)
autoencoder = autoencoder.cuda()
classifier = classifier.cuda()

# 定义损失函数和优化器
criterion_ae = nn.MSELoss()
criterion_cls = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(autoencoder.parameters()) + list(classifier.parameters()), lr=lr)

# 训练模型
best_valid_loss = float('inf')
best_model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 先只训练ae，之后同时训练ae和mlp
for epoch in range(num_epochs_ae):
    autoencoder.train()
    train_loss = 0
    balenced_train_loader = create_balanced_loader(train_loader, num_samples_per_class=num_samples_per_class, batch_size=batch_size)

    for inputs, labels in tqdm(balenced_train_loader, desc=f'Training Epoch {epoch + 1}/{num_epochs_ae}'):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()  # 清空梯度
        outputs, latent_feature = autoencoder(inputs)  # 前向传播
        loss= criterion_ae(outputs, inputs)  # 损失函数
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数
        train_loss += loss.item()
    
    autoencoder.eval()
    valid_loss = 0.0
    valid_preds = []
    valid_targets = []
    with torch.no_grad():
        for inputs, labels in tqdm(valid_loader, desc=f'Validation Epoch {epoch + 1}/{num_epochs_ae}'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs, latent_feature = autoencoder(inputs)
            loss = criterion_ae(outputs, inputs)
            valid_loss += loss.item()
    
    train_loss /= len(train_loader)
    valid_loss /= len(valid_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs_ae}], Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}')

for epoch in range(num_epochs_aemlp):
    autoencoder.train()
    classifier.train()
    train_loss = 0
    balenced_train_loader = create_balanced_loader(train_loader, num_samples_per_class=10000, batch_size=batch_size)
    for inputs, labels in tqdm(balenced_train_loader, desc=f'Training Epoch {epoch + 1}/{num_epochs_aemlp}'):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()  # 清空梯度

        outputs, latent_feature = autoencoder(inputs)  # 前向传播

        loss_ae = criterion_ae(outputs, inputs)  # 损失函数
        # print(outputs)
        predict_outputs = classifier(latent_feature)
        loss_cls = criterion_cls(predict_outputs, labels)
        # print(predict_outputs)

        loss = alpha * loss_ae + (1 - alpha) * loss_cls
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数

        train_loss += loss.item()
    
    autoencoder.eval()
    classifier.eval()
    valid_loss = 0.0
    valid_preds = []
    valid_targets = []
    with torch.no_grad():
        for inputs, labels in tqdm(valid_loader, desc=f'Validation Epoch {epoch + 1}/{num_epochs_aemlp}'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs, latent_feature = autoencoder(inputs)
            loss_ae = criterion_ae(outputs, inputs)
            predict_outputs = classifier(latent_feature)
            loss_cls = criterion_cls(predict_outputs, labels)

            loss = alpha * loss_ae + (1 - alpha) * loss_cls
            valid_loss += loss.item()

            valid_preds.extend(torch.argmax(predict_outputs, dim=1).cpu().numpy())
            valid_targets.extend(labels.cpu().numpy())
    
    train_loss /= len(train_loader)
    valid_loss /= len(valid_loader)

    valid_accuracy = accuracy_score(valid_targets, valid_preds)
    print(f'Epoch [{epoch + 1}/{num_epochs_aemlp}], Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}')

    if valid_loss < best_valid_loss:  # 更新最优模型
        best_valid_loss = valid_loss
        best_model = {'autoencoder': autoencoder.state_dict(), 'classifier': classifier.state_dict()}

# 保存最佳模型
os.makedirs(root, exist_ok=True)
torch.save(best_model, root + '/best_autoencoder_classifier.pth')

# 提取中间嵌入特征
autoencoder.load_state_dict(best_model['autoencoder'])
autoencoder.eval()
embeddings = []
with torch.no_grad():
    for inputs, _ in tqdm(valid_loader, desc='Extracting Features'):
        inputs = inputs.cuda()
        _, encoded = autoencoder(inputs)
        embeddings.append(encoded.cpu().numpy())

embeddings = np.concatenate(embeddings, axis=0)
np.save(root + '/embedded_features.npy', embeddings)

print("Training complete and embeddings saved.")

# 加载最佳模型并测试
autoencoder.load_state_dict(best_model['autoencoder'])
classifier.load_state_dict(best_model['classifier'])
autoencoder.eval()
classifier.eval()

test_ae_loss = 0
test_cls_loss = 0
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc='Testing Model'):
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs, latent_feature = autoencoder(inputs)
        loss_ae = criterion_ae(outputs, inputs)
        test_ae_loss += loss_ae.item()

        predict_outputs = classifier(latent_feature)
        loss_cls = criterion_cls(predict_outputs, labels)
        test_cls_loss += loss_cls.item()

        _, predicted = torch.max(predict_outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_ae_loss /= len(valid_loader)
test_cls_loss /= len(valid_loader)
accuracy = correct / total

print(f'Test Autoencoder Loss: {test_ae_loss:.4f}')
print(f'Test Classifier Loss: {test_cls_loss:.4f}')
print(f'Test Classification Accuracy: {accuracy:.4f}')



# import os
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset, random_split
# from sklearn.preprocessing import StandardScaler

# from model import *
# from config import *
# # 加载数据
# data = pd.read_csv(data_path)
# # 加载标签数据
# labels_df = pd.read_csv(label_path)
# labels_df.columns = labels_df.columns.str.strip()  # 去除列名的前空格
# labels_df['Label'] = labels_df['Label'].str.strip()  # 去除Label列空格
# labels_df['Label'] = labels_df['Label'].str.replace(';', '')  # 去除；

# # 数据标准化
# scaler = StandardScaler()
# data_scaled = scaler.fit_transform(data.iloc[:, 1:])
# # 数据转换为 PyTorch 张量
# data_tensor = torch.tensor(data_scaled, dtype=torch.float32)
# data_tensor = data_tensor.unsqueeze(1).unsqueeze(3)  # 增加一个维度以匹配 Conv2d 的输入要求

# # 将标签映射为数字
# label_mapping = {'Linc': 0, 'Sense No Exonic': 1, 'Antisense': 2, 'Exonic': 3}
# labels = labels_df['Label'].map(label_mapping).values
# # 转换为 PyTorch 张量
# labels = torch.tensor(labels, dtype=torch.long)

# # 划分训练集和验证集
# train_size = int(rate_t_v * len(data_tensor))
# valid_size = len(data_tensor) - train_size
# train_dataset, valid_dataset = random_split(TensorDataset(data_tensor, labels), [train_size, valid_size])

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers)

# # 模型初始化
# autoencoder = RNAAutoencoder()
# classifier = MLP()
# autoencoder = nn.DataParallel(autoencoder)
# classifier = nn.DataParallel(classifier)
# autoencoder = autoencoder.cuda()
# classifier = classifier.cuda()

# # 定义损失函数和优化器
# criterion_ae = nn.MSELoss()
# criterion_cls = nn.CrossEntropyLoss()
# optimizer = optim.Adam(list(autoencoder.parameters()) + list(classifier.parameters()), lr=lr)

# # 训练模型
# best_valid_loss = float('inf')
# best_model = None
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #第一个epoch只训练ae，之后同时训练ae和mlp
# for epoch in range(num_epochs):
#     autoencoder.train()
#     classifier.train()
#     train_loss = 0
#     for inputs, labels in tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{num_epochs}'):
#         inputs = inputs.to(device)
#         labels = labels.to(device)
#         optimizer.zero_grad()# 清空梯度

#         outputs, latent_feature = autoencoder(inputs)# 前向传播

#         loss_ae = criterion_ae(outputs, inputs)# 损失函数
#         if epoch > 0:
#             predict_outputs = classifier(latent_feature)
#             loss_cls = criterion_cls(predict_outputs, labels)
#         else:
#             loss_cls = 0
#         loss = alpha * loss_ae + (1 - alpha) * loss_cls

#         loss.backward()# 反向传播
#         optimizer.step() # 更新模型参数

#         train_loss += loss.item()
    
#     autoencoder.eval()
#     classifier.eval()
#     valid_loss = 0.0
#     with torch.no_grad():
#         for inputs, labels in tqdm(valid_loader, desc=f'Validation Epoch {epoch + 1}/{num_epochs}'):
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             outputs, latent_feature = autoencoder(inputs)
#             loss_ae = criterion_ae(outputs, inputs)

#             if epoch > 0:
#                 predict_outputs = classifier(latent_feature)
#                 loss_cls = criterion_cls(predict_outputs, labels)
#             else:
#                 loss_cls = 0

#             loss = alpha * loss_ae + (1 - alpha) * loss_cls
#             valid_loss += loss.item()
    
#     train_loss /= len(train_loader)
#     valid_loss /= len(valid_loader)
#     print(outputs)
#     print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}')

#     if epoch > 0 and valid_loss < best_valid_loss:# 更新最优模型
#         best_valid_loss = valid_loss
#         best_model = {'autoencoder': autoencoder.state_dict(), 'classifier': classifier.state_dict()}

# # 保存最佳模型
# os.makedirs(root, exist_ok=True)
# torch.save(best_model, root + '/best_autoencoder_classifier.pth')

# # 提取中间嵌入特征
# autoencoder.load_state_dict(best_model['autoencoder'])
# autoencoder.eval()
# embeddings = []
# with torch.no_grad():
#     for inputs, _ in tqdm(valid_loader, desc='Extracting Features'):
#         inputs = inputs.cuda()
#         _, encoded = autoencoder(inputs)
#         embeddings.append(encoded.cpu().numpy())

# embeddings = np.concatenate(embeddings, axis=0)
# np.save(root + '/embedded_features.npy', embeddings)

# print("Training complete and embeddings saved.")

# # 加载最佳模型并测试
# autoencoder.load_state_dict(best_model['autoencoder'])
# classifier.load_state_dict(best_model['classifier'])
# autoencoder.eval()
# classifier.eval()

# test_ae_loss = 0
# test_cls_loss = 0
# correct = 0
# total = 0

# with torch.no_grad():
#     for inputs, labels in tqdm(valid_loader, desc='Testing Model'):
#         inputs = inputs.cuda()
#         labels = labels.cuda()
#         outputs, latent_feature = autoencoder(inputs)
#         loss_ae = criterion_ae(outputs, inputs)
#         test_ae_loss += loss_ae.item()

#         predict_outputs = classifier(latent_feature)
#         loss_cls = criterion_cls(predict_outputs, labels)
#         test_cls_loss += loss_cls.item()

#         _, predicted = torch.max(predict_outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# test_ae_loss /= len(valid_loader)
# test_cls_loss /= len(valid_loader)
# accuracy = correct / total

# print(f'Test Autoencoder Loss: {test_ae_loss:.4f}')
# print(f'Test Classifier Loss: {test_cls_loss:.4f}')
# print(f'Test Classification Accuracy: {accuracy:.4f}')