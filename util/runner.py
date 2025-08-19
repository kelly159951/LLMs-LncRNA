from util.dataset import load_data
import torch
from tqdm import tqdm
import numpy as np
from util.logger import Logger
from util.model import create_model
from util.visualize import plot_training_results
import json
import os
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, confusion_matrix
import time

def normalize_softmax(softmax_results):
    prefix_sums = softmax_results[..., :-1].sum(dim=-1, keepdim=True)
    softmax_results[..., -1] = 1.0 - prefix_sums.squeeze(dim=-1)
    return softmax_results

class  task1_Runner():
    def __init__(self, args):
        self.args = args
        if self.args.global_rank == 0:
            self.logger = Logger(self.args.log_dir)
        else:
            self.logger = None

        self.rank = self.args.local_rank
        self.device= args.device
        self.train_all_set, self.valid_set, self.test_set = load_data(args)

        self.model=create_model(args, self.logger)
        self.batch_converter = self.model.batch_converter
        x_paras = []
        for k, v in self.model.named_parameters():
            if args.disable_layer < 0 and k.startswith('rna_model'):
                v.requires_grad = False
            elif k.startswith('rna_model.embed_tokens'):
                v.requires_grad = False
            elif k.startswith('rna_model.embed_positions'):
                v.requires_grad = False
            elif k.startswith('rna_model.emb_layer_norm_before'):
                v.requires_grad = False
            elif k.startswith('rna_model.layers'):
                if int(k.split('.')[2]) < args.disable_layer:
                    v.requires_grad = False
                else:
                    v.requires_grad = True
                    print('enabled', k)
                    x_paras.append(v)
            else:
                print('enabled', k)
                x_paras.append(v)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True ) if self.args.multygpu else self.model
        self.optimizer = torch.optim.AdamW(x_paras, lr=args.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.scaler = torch.cuda.amp.GradScaler()



    def train(self):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = [] 
        all_results = []
        for data in tqdm(self.train_loader, disable=(self.rank != 0)):
            batch_labels, _, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(self.device)
            batch_labels = torch.as_tensor(batch_labels, dtype=torch.long, device=self.device)

            result = self.model(batch_tokens) 
            loss = self.criterion(result, batch_labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()

            all_results.append(normalize_softmax(result.softmax(dim=-1))) #make sure the sum of each row is 1
            all_preds.append(result.argmax(dim=-1))
            all_labels.append(batch_labels)

            torch.cuda.empty_cache()

        # Concatenate all predictions and labels
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_results = torch.cat(all_results)

        # Calculate mean loss
        mean_loss = total_loss / len(self.train_loader)

        # Convert predictions and labels to numpy for sklearn metrics
        all_preds = all_preds.cpu().detach().numpy()
        all_labels = all_labels.cpu().detach().numpy()
        all_results = all_results.cpu().detach().numpy()

        # Calculate classification metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')  # Use macro average for multi-class
        recall = recall_score(all_labels, all_preds, average='macro')  # Use macro average for multi-class
        auc = roc_auc_score(all_labels, all_results, multi_class='ovr', average='macro')

        return mean_loss, accuracy, f1, recall, auc


    def evaluate(self, test=False):
        self.model.eval()
        dataloader = self.test_loader if test else self.valid_loader

        all_preds = []
        all_labels = []
        all_results = []
        whole_time=[]
        with torch.no_grad():
            for data in tqdm(dataloader, disable=(self.rank != 0)):
                batch_labels, _, batch_tokens = self.batch_converter(data)
                batch_tokens = batch_tokens.to(self.device)
                batch_labels = torch.as_tensor(batch_labels, dtype=torch.long, device=self.device)

                starttimer = time.time()
                result = self.model(batch_tokens)  # model output: logits of shape [batch_size, 4]
                whole_time.append(time.time() - starttimer)

                all_preds.append(result.argmax(dim=-1))
                all_labels.append(batch_labels)
                all_results.append( normalize_softmax(result.softmax(dim=-1))) 

        avg_time = np.mean(whole_time)
        # Concatenate all predictions and labels
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_results = torch.cat(all_results)

        # Convert predictions and labels to numpy for sklearn metrics
        all_preds = all_preds.cpu().detach().numpy()
        all_labels = all_labels.cpu().detach().numpy()
        all_results = all_results.cpu().detach().numpy()

        # Calculate classification metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')  # Use macro average for multi-class
        recall = recall_score(all_labels, all_preds, average='macro')  # Use macro average for multi-class
        auc = roc_auc_score(all_labels, all_results, multi_class='ovr', average='macro')

        return accuracy, f1, recall, auc, avg_time

    def pipeline(self):
        self.losses,self.val_acc, self.test_acc, self.val_f1, self.test_f1, self.val_recall, self.test_recall, self.val_auc, self.test_auc, self.val_batch_time , self.test_batch_time = [], [], [], [], [], [], [], [], [], [], []
        self.best_perf, self.best_ep = None, None

        log_dir = self.args.log_dir  
        result_path = os.path.join(log_dir, 'result.json')
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        output_data = {
            'args': {k: (v if not isinstance(v, torch.device) else str(v)) for k, v in vars(self.args).items()},
        }
        with open(result_path, 'w') as f:
            json.dump(output_data, f, indent=4)
            
        self.valid_loader = torch.utils.data.DataLoader(
            self.valid_set, batch_size=self.args.bs, num_workers=self.args.num_worker,
            collate_fn=lambda x: x, shuffle=False, pin_memory=True, persistent_workers=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_set, batch_size=self.args.bs, num_workers=self.args.num_worker,
            collate_fn=lambda x: x, shuffle=False, pin_memory=True, persistent_workers=True
        )
        assert len(self.train_all_set) > 0, 'Training set is empty'
        assert len(self.valid_set) > 0, 'Validation set is empty'
        assert len(self.test_set) > 0, 'Test set is empty'

        print('start training')
        for ep in range(self.args.skip,self.args.epochs):
            if self.args.global_rank == 0:
                print(f'[Epoch] -----------{ep}----------')
            train_set = self.train_all_set.create_balance_subset(self.args.inbalance_frac)
            self.train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=self.args.bs, num_workers=self.args.num_worker,
                collate_fn=lambda x: x, shuffle=True, pin_memory=True, persistent_workers=True
            )

            train_loss, train_acc, train_f1, train_recall, train_auc = (0,0,0,0,0) if self.args.inference else self.train()
            valid_acc, valid_f1, valid_recall, valid_auc, val_time = self.evaluate()
            test_acc, test_f1, test_recall, test_auc , test_time= self.evaluate(True)
            
            self.losses.append(train_loss)
            self.val_acc.append(valid_acc)
            self.test_acc.append(test_acc)
            self.val_f1.append(valid_f1)
            self.test_f1.append(test_f1)
            self.val_recall.append(valid_recall)
            self.test_recall.append(test_recall)
            self.val_auc.append(valid_auc)
            self.test_auc.append(test_auc)
            self.val_batch_time.append(val_time)
            self.test_batch_time.append(test_time)

            if self.args.global_rank == 0:
                self.logger.log_epoch(ep, train_loss, train_acc, valid_acc, test_acc,train_f1, valid_f1, test_f1,train_recall, valid_recall, test_recall,train_auc, valid_auc, test_auc, val_time, test_time)

                print('[train]', train_loss,train_acc,train_f1,train_recall,train_auc)
                print('[valid]', valid_acc,valid_f1,valid_recall,valid_auc,val_time)
                print('[test]', test_acc,test_f1,test_recall,test_auc,test_time)
            
                if self.best_perf is None or valid_acc > self.best_perf:
                    self.best_perf, self.best_ep = valid_acc, ep
                    if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                        state_dict = self.model.module.state_dict()
                    else:
                        state_dict = self.model.state_dict()
                    torch.save(state_dict, self.args.log_dir + '/best_model.pth')
                    print(f'[INFO] best acc epoch: {self.best_ep}')
                    print(f'[INFO] best valid acc: {self.val_acc[self.best_ep]}')
                    print(f'[INFO] best test acc: {self.test_acc[self.best_ep]}')

                    avg_val_batch_time=np.mean(self.val_batch_time)
                    avg_test_batch_time=np.mean(self.test_batch_time)
                    avg_encode_time=self.args.encode_time/self.args.whole_data_num
                    avg_forward_time=(avg_val_batch_time+avg_test_batch_time)/2
                    output_data = {
                        'args': {k: (v if not isinstance(v, torch.device) else str(v)) for k, v in vars(self.args).items()},
                        'best_ep': self.best_ep,
                        'best_valid_acc': self.val_acc[self.best_ep],
                        'best_test_acc': self.test_acc[self.best_ep],
                        'best_valid_f1': self.val_f1[self.best_ep],
                        'best_test_f1': self.test_f1[self.best_ep],
                        'best_valid_recall': self.val_recall[self.best_ep],
                        'best_test_recall': self.test_recall[self.best_ep],
                        'best_valid_auc': self.val_auc[self.best_ep],
                        'best_test_auc': self.test_auc[self.best_ep],
                        'avg_val_batch_time': avg_val_batch_time,
                        'avg_test_batch_time': avg_test_batch_time,
                        'avg_encode_time': avg_encode_time,
                        'avg_forward_time': avg_forward_time,
                        'avg_whole_time': avg_encode_time+avg_forward_time
                    }

                    with open(result_path, 'w') as f:
                        json.dump(output_data, f, indent=4)




class task2_Runner():
    def __init__(self, args):
        self.args = args
        self.rank = self.args.local_rank
        self.device= args.device
        self.train_set, self.valid_set, self.test_set = load_data(args)

        self.model=create_model(args)
        self.rna_batch_converter = self.model.rna_batch_converter
        self.pro_batch_converter = self.model.pro_batch_converter
        x_paras = []
        for k, v in self.model.named_parameters():
            if k.startswith('pro_model'):
                v.requires_grad = False
            elif args.disable_layer < 0 and k.startswith('rna_model'):
                v.requires_grad = False
            elif k.startswith('rna_model.embed_tokens'):
                v.requires_grad = False
            elif k.startswith('rna_model.embed_positions'):
                v.requires_grad = False
            elif k.startswith('rna_model.emb_layer_norm_before'):
                v.requires_grad = False
            elif k.startswith('rna_model.layers'):
                if int(k.split('.')[2]) < args.disable_layer:
                    v.requires_grad = False
                else:
                    v.requires_grad = True
                    print('enabled', k)
                    x_paras.append(v)
            else:
                print('enabled', k)
                x_paras.append(v)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True ) if self.args.multygpu else self.model
        self.optimizer = torch.optim.AdamW(x_paras, lr=args.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.scaler = torch.cuda.amp.GradScaler()
        if self.args.global_rank == 0:
            self.logger = Logger(self.args.log_dir)


    def train(self):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = [] 
        all_results = []
        for data in tqdm(self.train_loader, disable=(self.rank != 0)):
            rna_strs = [d[0] for d in data]
            pro_strs = [d[1] for d in data]
            _, _, rna_tokens = self.rna_batch_converter([(None, rna_str) for rna_str in rna_strs])
            _, _, pro_tokens = self.pro_batch_converter([(None, pro_str) for pro_str in pro_strs])

            rna_tokens = rna_tokens.to(self.device)
            pro_tokens = pro_tokens.to(self.device)
            batch_labels = torch.LongTensor([d[2] for d in data]).to(self.device) 
            # with torch.cuda.amp.autocast():
            result = self.model((rna_tokens, pro_tokens))
            loss = self.criterion(result, batch_labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()

            all_preds.append(result.argmax(dim=-1))
            all_labels.append(batch_labels)
            all_results.append( normalize_softmax(result.softmax(dim=-1))) 

            torch.cuda.empty_cache()

        mean_loss = total_loss / len(self.train_loader)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_results = torch.cat(all_results)

        all_preds = all_preds.cpu().detach().numpy()
        all_labels = all_labels.cpu().detach().numpy()
        all_results = all_results.cpu().detach().numpy()

        # Calculate classification metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')  # Use macro average for multi-class
        recall = recall_score(all_labels, all_preds, average='macro')  # Use macro average for multi-class
        auc = roc_auc_score(all_labels, all_results[:, 1], average='macro')  # Use probabilities for positive class (index 1)

        return mean_loss, accuracy, f1, recall, auc




    def evaluate(self, test=False):
        self.model.eval()
        dataloader = self.test_loader if test else self.valid_loader

        all_preds = []
        all_labels = []
        all_results = []
        whole_time=[]
        with torch.no_grad():
            for data in tqdm(dataloader, disable=(self.rank != 0)):
                rna_strs = [d[0] for d in data]
                pro_strs = [d[1] for d in data]

                # 使用对应的 converter 进行转换
                _, _, rna_tokens = self.rna_batch_converter([(None, rna_str) for rna_str in rna_strs])
                _, _, pro_tokens = self.pro_batch_converter([(None, pro_str) for pro_str in pro_strs])

                rna_tokens = rna_tokens.to(self.device)
                pro_tokens = pro_tokens.to(self.device)

                batch_labels = torch.LongTensor([d[2] for d in data]).to(self.device)

                starttimer = time.time()
                result = self.model((rna_tokens, pro_tokens))
                whole_time.append(time.time() - starttimer)

                all_preds.append(result.argmax(dim=-1))
                all_labels.append(batch_labels)
                all_results.append(normalize_softmax(result.softmax(dim=-1))) 
        avg_time = np.mean(whole_time)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_results = torch.cat(all_results)

        all_preds = all_preds.cpu().detach().numpy()
        all_labels = all_labels.cpu().detach().numpy()
        all_results = all_results.cpu().detach().numpy()

        # Calculate classification metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')  # Use macro average for multi-class
        recall = recall_score(all_labels, all_preds, average='macro')  # Use macro average for multi-class
        auc = roc_auc_score(all_labels, all_results[:, 1], average='macro')  # Use probabilities for positive class (index 1)

        return accuracy, f1, recall, auc, avg_time




    def pipeline(self):
        self.losses,self.val_acc, self.test_acc, self.val_f1, self.test_f1, self.val_recall, self.test_recall, self.val_auc, self.test_auc, self.val_batch_time , self.test_batch_time = [], [], [], [], [], [], [], [], [], [], []
        self.best_perf, self.best_ep = None, None
        log_dir = self.args.log_dir  
        result_path = os.path.join(log_dir, 'result.json')
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        output_data = {
            'args': {k: (v if not isinstance(v, torch.device) else str(v)) for k, v in vars(self.args).items()},
        }
        with open(result_path, 'w') as f:
            json.dump(output_data, f, indent=4)

        self.valid_loader = torch.utils.data.DataLoader(
            self.valid_set, batch_size=self.args.bs, num_workers=self.args.num_worker,
            collate_fn=lambda x: x, shuffle=False, pin_memory=True, persistent_workers=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_set, batch_size=self.args.bs, num_workers=self.args.num_worker,
            collate_fn=lambda x: x, shuffle=False, pin_memory=True, persistent_workers=True
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_set, batch_size=self.args.bs, num_workers=self.args.num_worker,
            collate_fn=lambda x: x, shuffle=True, pin_memory=True, persistent_workers=True
        )
        assert len(self.train_set) > 0, 'Training set is empty'
        assert len(self.valid_set) > 0, 'Validation set is empty'
        assert len(self.test_set) > 0, 'Test set is empty'

        print('start training')
        for ep in range(self.args.skip,self.args.epochs):
            print(f'[Epoch] -----------{ep}----------')

            train_loss, train_acc, train_f1, train_recall, train_auc = (0,0,0,0,0) if self.args.inference else self.train()
            valid_acc, valid_f1, valid_recall, valid_auc, val_time = self.evaluate()
            test_acc, test_f1, test_recall, test_auc, test_time = self.evaluate(True)
               
            self.losses.append(train_loss)
            self.val_acc.append(valid_acc)
            self.test_acc.append(test_acc)
            self.val_f1.append(valid_f1)
            self.test_f1.append(test_f1)
            self.val_recall.append(valid_recall)
            self.test_recall.append(test_recall)
            self.val_auc.append(valid_auc)
            self.test_auc.append(test_auc)
            self.val_batch_time.append(val_time)
            self.test_batch_time.append(test_time)

            if self.args.global_rank == 0:
                self.logger.log_epoch(ep, train_loss, train_acc, valid_acc, test_acc,train_f1, valid_f1, test_f1,train_recall, valid_recall, test_recall,train_auc, valid_auc, test_auc)

                print('[train]', train_loss,train_acc,train_f1,train_recall,train_auc)
                print('[valid]', valid_acc,valid_f1,valid_recall,valid_auc,val_time)
                print('[test]', test_acc,test_f1,test_recall,test_auc,test_time)

                if self.best_perf is None or valid_acc > self.best_perf and self.args.global_rank == 0:
                    self.best_perf, self.best_ep = valid_acc, ep
                    if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                        state_dict = self.model.module.state_dict()
                    else:
                        state_dict = self.model.state_dict()
                    torch.save(state_dict, self.args.log_dir + '/best_model.pth')

                    print(f'[INFO] best acc epoch: {self.best_ep}')
                    print(f'[INFO] best valid acc: {self.val_acc[self.best_ep]}')
                    print(f'[INFO] best test acc: {self.test_acc[self.best_ep]}')
                    avg_val_batch_time=np.mean(self.val_batch_time)
                    avg_test_batch_time=np.mean(self.test_batch_time)
                    avg_encode_time=self.args.encode_time/self.args.whole_data_num
                    avg_forward_time=(avg_val_batch_time+avg_test_batch_time)/2
                    output_data = {
                        'args': {k: (v if not isinstance(v, torch.device) else str(v)) for k, v in vars(self.args).items()},
                        'best_ep': self.best_ep,
                        'best_valid_acc': self.val_acc[self.best_ep],
                        'best_test_acc': self.test_acc[self.best_ep],
                        'best_valid_f1': self.val_f1[self.best_ep],
                        'best_test_f1': self.test_f1[self.best_ep],
                        'best_valid_recall': self.val_recall[self.best_ep],
                        'best_test_recall': self.test_recall[self.best_ep],
                        'best_valid_auc': self.val_auc[self.best_ep],
                        'best_test_auc': self.test_auc[self.best_ep],
                        'avg_val_batch_time': avg_val_batch_time,
                        'avg_test_batch_time': avg_test_batch_time,
                        'avg_encode_time': avg_encode_time,
                        'avg_forward_time': avg_forward_time,
                        'avg_whole_time': avg_encode_time+avg_forward_time
                    }

                    with open(result_path, 'w') as f:
                        json.dump(output_data, f, indent=4)

def col_fn(batch):
    # 假设每个元素是一个元组，长度可以变化。处理前两列是数据，最后一列是标签的情况。
    
    if len(batch[0]) == 3:
        # 如果每个元素有3列，第一列和第二列为数据，第三列为标签
        data1 = torch.FloatTensor(np.array([x[0] for x in batch]))  # 提取第一列作为数据
        data2 = torch.FloatTensor(np.array([x[1] for x in batch]))  # 提取第二列作为数据
        labels = torch.LongTensor([x[2] for x in batch])  # 提取第三列作为标签
        return data1.unsqueeze(dim=1), data2.unsqueeze(dim=1), labels  # 返回数据1、数据2和标签
    elif len(batch[0]) == 2:
        # 如果每个元素有2列，第一列为数据，第二列为标签
        data = torch.FloatTensor(np.array([x[0] for x in batch]))  # 提取第一列作为数据
        labels = torch.LongTensor([x[1] for x in batch])  # 提取第二列作为标签
        return data.unsqueeze(dim=1), labels  # 返回数据和标签

class base_Runner():
    def __init__(self, args):
        self.args = args
        self.rank = self.args.local_rank
        self.device= args.device
        self.alpha = args.alpha
        self.task= args.task_num
        self.train_all_set, self.valid_set, self.test_set = load_data(args)
        self.model = create_model(args)
        
        x_paras = []
        for k, v in self.model.named_parameters():
            v.requires_grad = True
            print('enabled', k)
            x_paras.append(v)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True ) if self.args.multygpu else self.model
        
        self.optimizer = torch.optim.Adam(x_paras, lr=args.lr)
        self.criterion_cls = torch.nn.CrossEntropyLoss()
        self.criterion_ae = torch.nn.MSELoss()
        if self.args.global_rank == 0:
            self.logger = Logger(self.args.log_dir)

    def train(self):
        self.model.train()
        
        losses = []
        all_preds = []
        all_labels = [] 
        all_results = []
        for data in tqdm(self.train_loader, disable=(self.rank != 0)):
            if self.task == 1:
                rnainputs, labels = data
                rnainputs = rnainputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()  # 清空梯度

                pred, rna_output = self.model(rnainputs)  # 前向传播

                loss_ae = self.criterion_ae(rna_output, rnainputs)  # 损失函数
                loss_cls = self.criterion_cls(pred, labels)
                loss = self.alpha * loss_ae + (1 - self.alpha) * loss_cls
            else:
                rnainputs, proinputs, labels = data  # 解包 data
                rnainputs = rnainputs.to(self.device)
                proinputs = proinputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()

                pred, rna_output, pro_output = self.model(rnainputs, proinputs)  # 前向传播

                loss_rna_ae = self.criterion_ae(rna_output, rnainputs)  # 损失函数
                loss_pro_ae = self.criterion_ae(pro_output, proinputs)
                loss_cls = self.criterion_cls(pred, labels)
                loss = self.alpha * (loss_rna_ae + loss_pro_ae) + (1 - self.alpha) * loss_cls
                
            loss.backward()  # 反向传播
            self.optimizer.step()  # 更新模型参数

            losses.append(loss.item())
            all_results.append(normalize_softmax(pred.softmax(dim=-1))) #make sure the sum of each row is 1
            all_preds.append(pred.argmax(dim=-1))
            all_labels.append(labels)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_results = torch.cat(all_results)
         # Calculate mean loss
        mean_loss = np.mean(losses)

        # Convert predictions and labels to numpy for sklearn metrics
        all_preds = all_preds.cpu().detach().numpy()
        all_labels = all_labels.cpu().detach().numpy()
        all_results = all_results.cpu().detach().numpy()
        if self.task == 2:
            all_results=all_results[:, 1]

        # Calculate classification metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')  # Use macro average for multi-class
        recall = recall_score(all_labels, all_preds, average='macro')  # Use macro average for multi-class
        # print(all_labels.shape,all_results.shape)
        auc = roc_auc_score(all_labels, all_results, multi_class='ovr', average='macro')

        return mean_loss, accuracy, f1, recall, auc
    
    def evaluate(self,test=False):
        self.model.eval()
        dataloader = self.test_loader if test else self.valid_loader

        all_preds = []
        all_labels = []
        all_results = []
        whole_time=[]
        with torch.no_grad():
            for data in tqdm(dataloader, disable=(self.rank != 0)):
                if self.task == 1:
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    starttimer = time.time()
                    pred, _ = self.model(inputs)
                    whole_time.append( time.time() - starttimer)
                else:
                    rnainputs, proinputs, labels = data
                    rnainputs = rnainputs.to(self.device)
                    proinputs = proinputs.to(self.device)
                    labels = labels.to(self.device)
                    starttimer = time.time()
                    pred, _, _ = self.model(rnainputs, proinputs)
                    whole_time.append( time.time() - starttimer)
                all_preds.append(pred.argmax(dim=-1))
                all_labels.append(labels)
                all_results.append(normalize_softmax(pred.softmax(dim=-1)))
            avg_time=np.mean(whole_time)
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            all_results = torch.cat(all_results)
            all_preds = all_preds.cpu().detach().numpy()
            all_labels = all_labels.cpu().detach().numpy()
            all_results = all_results.cpu().detach().numpy()
            if self.task == 2:
                all_results=all_results[:, 1]
            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='macro')
            recall = recall_score(all_labels, all_preds, average='macro')
            auc = roc_auc_score(all_labels, all_results, multi_class='ovr', average='macro')
        return accuracy, f1, recall, auc, avg_time
    
    def pipeline(self):
        self.losses,self.val_acc, self.test_acc, self.val_f1, self.test_f1, self.val_recall, self.test_recall, self.val_auc, self.test_auc, self.val_batch_time , self.test_batch_time = [], [], [], [], [], [], [], [], [], [], []
        self.best_perf, self.best_ep = None, None
        log_dir = self.args.log_dir  
        result_path = os.path.join(log_dir, 'result.json')
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        output_data = {
            'args': {k: (v if not isinstance(v, torch.device) else str(v)) for k, v in vars(self.args).items()},
        }
        with open(result_path, 'w') as f:
            json.dump(output_data, f, indent=4)

        self.valid_loader = torch.utils.data.DataLoader(
            self.valid_set, batch_size=self.args.bs, num_workers=self.args.num_worker,
            collate_fn=col_fn, shuffle=False
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_set, batch_size=self.args.bs, num_workers=self.args.num_worker,
            collate_fn=col_fn, shuffle=False
        )
        if self.args.task_num == 2:
            self.train_loader = torch.utils.data.DataLoader(
                self.train_all_set, batch_size=self.args.bs, num_workers=self.args.num_worker,
                collate_fn= col_fn, shuffle=True
            )
        assert len(self.train_all_set) > 0, 'Training set is empty'
        assert len(self.valid_set) > 0, 'Validation set is empty'
        assert len(self.test_set) > 0, 'Test set is empty'
        
        print('start training')
        for ep in range(self.args.skip,self.args.epochs):
            if self.args.global_rank == 0:
                print(f'[Epoch] -----------{ep}----------')
            if self.args.task_num == 1:
                train_set = self.train_all_set.create_balance_subset(self.args.inbalance_frac)
                self.train_loader = torch.utils.data.DataLoader(
                    train_set, batch_size=self.args.bs, num_workers=self.args.num_worker,
                    collate_fn= col_fn, shuffle=True
                )

            train_loss, train_acc, train_f1, train_recall, train_auc = (0,0,0,0,0) if self.args.inference else self.train()
            valid_acc, valid_f1, valid_recall, valid_auc, val_time = self.evaluate()
            test_acc, test_f1, test_recall, test_auc , test_time= self.evaluate(True)
            
            self.losses.append(train_loss)
            self.val_acc.append(valid_acc)
            self.test_acc.append(test_acc)
            self.val_f1.append(valid_f1)
            self.test_f1.append(test_f1)
            self.val_recall.append(valid_recall)
            self.test_recall.append(test_recall)
            self.val_auc.append(valid_auc)
            self.test_auc.append(test_auc)
            self.val_batch_time.append(val_time)
            self.test_batch_time.append(test_time)

            if self.args.global_rank == 0:
                self.logger.log_epoch(ep, train_loss, train_acc, valid_acc, test_acc,train_f1, valid_f1, test_f1,train_recall, valid_recall, test_recall,train_auc, valid_auc, test_auc, val_time, test_time)

                print('[train]', train_loss,train_acc,train_f1,train_recall,train_auc)
                print('[valid]', valid_acc,valid_f1,valid_recall,valid_auc,val_time)
                print('[test]', test_acc,test_f1,test_recall,test_auc,test_time)
            
                if self.best_perf is None or valid_acc > self.best_perf:
                    self.best_perf, self.best_ep = valid_acc, ep
                    if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                        state_dict = self.model.module.state_dict()
                    else:
                        state_dict = self.model.state_dict()
                    torch.save(state_dict, self.args.log_dir + '/best_model.pth')
                    print(f'[INFO] best acc epoch: {self.best_ep}')
                    print(f'[INFO] best valid acc: {self.val_acc[self.best_ep]}')
                    print(f'[INFO] best test acc: {self.test_acc[self.best_ep]}')

                    avg_val_batch_time=np.mean(self.val_batch_time)
                    avg_test_batch_time=np.mean(self.test_batch_time)
                    avg_encode_time=self.args.encode_time/self.args.whole_data_num
                    avg_forward_time=(avg_val_batch_time+avg_test_batch_time)/2
                    output_data = {
                        'args': {k: (v if not isinstance(v, torch.device) else str(v)) for k, v in vars(self.args).items()},
                        'best_ep': self.best_ep,
                        'best_valid_acc': self.val_acc[self.best_ep],
                        'best_test_acc': self.test_acc[self.best_ep],
                        'best_valid_f1': self.val_f1[self.best_ep],
                        'best_test_f1': self.test_f1[self.best_ep],
                        'best_valid_recall': self.val_recall[self.best_ep],
                        'best_test_recall': self.test_recall[self.best_ep],
                        'best_valid_auc': self.val_auc[self.best_ep],
                        'best_test_auc': self.test_auc[self.best_ep],
                        'avg_val_batch_time': avg_val_batch_time,
                        'avg_test_batch_time': avg_test_batch_time,
                        'avg_encode_time': avg_encode_time,
                        'avg_forward_time': avg_forward_time,
                        'avg_whole_time': avg_encode_time+avg_forward_time
                    }

                    with open(result_path, 'w') as f:
                        json.dump(output_data, f, indent=4)
        
