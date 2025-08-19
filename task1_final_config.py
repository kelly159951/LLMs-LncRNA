import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_len = 1024

chose=['rna','sample']
ch=chose[0]
num='2'
fasta_file = './dataset/task1_'+ch+'.fasta'
label_file = './dataset/task1_'+ch+'_label.csv'

batch_size=8
# 4-class MLP model
input_dim = 640  # RNA sequence encoding size
hidden_dim_1 = 160
hidden_dim_2 = 32
output_dim = 4  # Number of label categories
drop_out_1=0.1
drop_out_2=0.1

#Training parameters
lr_m1=0.0001 # MLP learning rate, stage 1
lr_m2=0.00001 # MLP learning rate, stage 2
lr_fm=0.000005 # RNA FM learning rate
num_epochs_1 = 6 # Train MLP only
num_epochs_2 = 10  # Train both RNA FM and MLP

mlp_model_save_path = './model/task1_mlp_model_'+num+'.pth'
rna_fm_save_path = './model/task1_fm_model_'+num+'.pth'
figure_save_path = './figure/task1_figure_'+num+'.png'
result_save_path = './result/task1_result_'+num+'.json'

result={
    'batch_size':batch_size,
    'input_dim':input_dim,
    'hidden_dim_1':hidden_dim_1,
    'hidden_dim_2':hidden_dim_2,
    'output_dim':output_dim,
    'drop_out_1':drop_out_1,
    'drop_out_2':drop_out_2,
    'lr_m1':lr_m1,
    'lr_m2':lr_m2,
    'lr_fm':lr_fm,
    'num_epochs_1':num_epochs_1,
    'num_epochs_2':num_epochs_2,
    'mlp_model_save_path':mlp_model_save_path,
    'rna_fm_save_path':rna_fm_save_path,
    'figure_save_path':figure_save_path,
    'result_save_path':result_save_path,
    'num_samples_per_class':-1,
    'test_loss':0,
    'test_accuracy':0,
    's1_train_losses':[],
    's1_val_losses':[],
    's1_val_accuracies':[],
    's2_train_losses':[],
    's2_val_losses':[],
    's2_val_accuracies':[]
}


